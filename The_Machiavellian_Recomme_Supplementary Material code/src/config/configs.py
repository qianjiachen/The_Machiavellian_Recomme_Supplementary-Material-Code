"""Configuration classes for Machiavellian Recommender experiments."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ArenaConfig:
    """Configuration for RecGame-Arena environment."""
    num_users: int = 1000
    num_vendors: int = 100
    num_items_per_vendor: int = 50
    communication_budget: int = 5
    max_tokens: int = 128
    discount_factor: float = 0.99
    embedding_dim: int = 128
    
    def __post_init__(self):
        assert self.num_users >= 1, "num_users must be >= 1"
        assert self.num_vendors >= 1, "num_vendors must be >= 1"
        assert self.communication_budget >= 1, "communication_budget must be >= 1"
        assert 0 < self.discount_factor < 1, "discount_factor must be in (0, 1)"


@dataclass
class SWAConfig:
    """Configuration for Social Welfare Alignment training."""
    lambda_welfare: float = 0.5
    lambda_equity: float = 0.3
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_steps: int = 500000
    warmup_steps: int = 10000
    kl_coef: float = 0.01
    gradient_clip_norm: float = 1.0
    target_update_tau: float = 0.005
    
    def __post_init__(self):
        assert self.lambda_welfare >= 0, "lambda_welfare must be >= 0"
        assert self.lambda_equity >= 0, "lambda_equity must be >= 0"
        assert self.learning_rate > 0, "learning_rate must be > 0"


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    model_name: str = "meta-llama/Llama-3-8b-hf"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    device: str = "cuda"
    dtype: str = "float16"
    
    def __post_init__(self):
        assert self.lora_rank > 0, "lora_rank must be > 0"
        assert self.lora_alpha > 0, "lora_alpha must be > 0"


@dataclass
class DetectionConfig:
    """Configuration for collusion detection."""
    num_paraphrases: int = 5
    threshold_tau: float = 0.3
    threshold_mi: float = 0.1
    mine_hidden_dim: int = 128
    mine_num_layers: int = 3
    
    def __post_init__(self):
        assert self.num_paraphrases >= 1, "num_paraphrases must be >= 1"
        assert 0 < self.threshold_tau < 1, "threshold_tau must be in (0, 1)"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    num_seeds: int = 5
    eval_interval: int = 10000
    checkpoint_interval: int = 50000


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    use_tensorboard: bool = True
    use_wandb: bool = False
    log_interval: int = 1000
    save_dir: str = "outputs"


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    swa: SWAConfig = field(default_factory=SWAConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        arena = ArenaConfig(**config_dict.get("arena", {}))
        swa = SWAConfig(**config_dict.get("swa", {}))
        llm = LLMConfig(**config_dict.get("llm", {}))
        detection = DetectionConfig(**config_dict.get("detection", {}))
        evaluation = EvaluationConfig(**config_dict.get("evaluation", {}))
        logging = LoggingConfig(**config_dict.get("logging", {}))
        seed = config_dict.get("seed", 42)
        
        return cls(
            arena=arena, swa=swa, llm=llm, detection=detection,
            evaluation=evaluation, logging=logging, seed=seed
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    return TrainingConfig.from_yaml(path)
