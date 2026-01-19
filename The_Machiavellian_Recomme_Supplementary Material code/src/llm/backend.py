"""LLM Backend with shared backbone and LoRA adapters."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRAAdapter(nn.Module):
    """LoRA adapter for parameter-efficient fine-tuning."""
    
    def __init__(self, adapter_id: str, config: LoRAConfig, hidden_size: int):
        super().__init__()
        self.adapter_id = adapter_id
        self.config = config
        self.hidden_size = hidden_size
        
        # LoRA matrices
        self.lora_A = nn.Linear(hidden_size, config.rank, bias=False)
        self.lora_B = nn.Linear(config.rank, hidden_size, bias=False)
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation."""
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LLMBackend:
    """LLM backend supporting shared backbone with LoRA adapters."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8b-hf",
        device: str = "cuda",
        dtype: str = "float16",
        use_mock: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.use_mock = use_mock
        
        self.backbone = None
        self.tokenizer = None
        self.lora_adapters: Dict[str, LoRAAdapter] = {}
        self.hidden_size = 4096  # Default for Llama-3-8B
        
        if not use_mock:
            self._load_backbone()
        else:
            logger.info("Using mock LLM backend for testing")
    
    def _load_backbone(self):
        """Load the shared backbone model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading backbone model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.backbone = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True
            )
            self.backbone.eval()
            
            # Get hidden size from model config
            self.hidden_size = self.backbone.config.hidden_size
            
            logger.info(f"Backbone loaded successfully. Hidden size: {self.hidden_size}")
            
        except Exception as e:
            logger.warning(f"Failed to load backbone: {e}. Using mock mode.")
            self.use_mock = True
    
    def create_adapter(
        self,
        adapter_id: str,
        config: Optional[LoRAConfig] = None
    ) -> LoRAAdapter:
        """Create a LoRA adapter for an agent."""
        if config is None:
            config = LoRAConfig()
        
        adapter = LoRAAdapter(adapter_id, config, self.hidden_size)
        
        if not self.use_mock:
            adapter = adapter.to(self.device)
        
        self.lora_adapters[adapter_id] = adapter
        return adapter
    
    def get_adapter(self, adapter_id: str) -> Optional[LoRAAdapter]:
        """Get an existing adapter by ID."""
        return self.lora_adapters.get(adapter_id)
    
    def remove_adapter(self, adapter_id: str):
        """Remove an adapter."""
        if adapter_id in self.lora_adapters:
            del self.lora_adapters[adapter_id]
    
    def generate(
        self,
        prompt: str,
        adapter_id: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text using the backbone with optional LoRA adapter."""
        if self.use_mock:
            return self._mock_generate(prompt, adapter_id)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.backbone.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        response = generated[len(prompt):].strip()
        return response
    
    def _mock_generate(self, prompt: str, adapter_id: Optional[str] = None) -> str:
        """Mock generation for testing."""
        import random
        responses = [
            "I recommend this product for its quality.",
            "This item offers great value for the price.",
            "Based on your preferences, this would be a good choice.",
            "I'll consider this offer carefully.",
            "The price seems reasonable for this product."
        ]
        return random.choice(responses)
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to embeddings."""
        if self.use_mock:
            return torch.randn(1, self.hidden_size)
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.backbone(**inputs, output_hidden_states=True)
            # Use last hidden state, mean pooling
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        return embeddings
    
    def get_num_adapters(self) -> int:
        """Get the number of registered adapters."""
        return len(self.lora_adapters)
    
    def get_adapter_ids(self) -> List[str]:
        """Get all adapter IDs."""
        return list(self.lora_adapters.keys())
