"""Base agent class for all agents in the system."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import torch

from src.models.data_models import Observation, Action, ActionType
from src.llm.backend import LLMBackend, LoRAAdapter


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        agent_id: str,
        llm_backend: Optional[LLMBackend] = None,
        adapter: Optional[LoRAAdapter] = None
    ):
        self.agent_id = agent_id
        self.llm_backend = llm_backend
        self.adapter = adapter
        self._policy = None
    
    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """
        Generate an action based on the current observation.
        
        Args:
            observation: Current observation for this agent
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def communicate(self, context: str, receiver_id: str) -> str:
        """
        Generate a natural language message.
        
        Args:
            context: Context for the communication
            receiver_id: ID of the message recipient
            
        Returns:
            Generated message content
        """
        pass
    
    def get_policy_parameters(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get policy parameters for training."""
        if self.adapter:
            return {
                name: param for name, param in self.adapter.named_parameters()
            }
        return None
    
    def set_policy_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set policy parameters."""
        if self.adapter:
            for name, param in self.adapter.named_parameters():
                if name in parameters:
                    param.data.copy_(parameters[name])
    
    def _generate_with_llm(self, prompt: str, max_tokens: int = 128) -> str:
        """Generate text using the LLM backend."""
        if self.llm_backend is None:
            return self._default_response(prompt)
        
        return self.llm_backend.generate(
            prompt=prompt,
            adapter_id=self.agent_id,
            max_new_tokens=max_tokens
        )
    
    def _default_response(self, prompt: str) -> str:
        """Default response when LLM is not available."""
        return "I acknowledge your message."
    
    def reset(self):
        """Reset agent state for a new episode."""
        pass


class PolicyNetwork(torch.nn.Module):
    """Neural network policy for agent decision making."""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        
        layers = []
        input_dim = observation_dim
        
        for _ in range(num_layers):
            layers.extend([
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        self.encoder = torch.nn.Sequential(*layers)
        self.action_head = torch.nn.Linear(hidden_dim, action_dim)
        self.value_head = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, observation: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Returns:
            action_logits: Logits for action distribution
            value: State value estimate
        """
        features = self.encoder(observation)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        return action_logits, value
    
    def get_action(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample an action from the policy."""
        action_logits, _ = self.forward(observation)
        
        if deterministic:
            return action_logits.argmax(dim=-1)
        
        probs = torch.softmax(action_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
