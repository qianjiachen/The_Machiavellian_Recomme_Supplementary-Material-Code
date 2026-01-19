"""Ablation study variants for SWA training."""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum

from src.config.configs import SWAConfig


class AblationVariant(Enum):
    """Ablation study variants."""
    FULL_SWA = "full_swa"
    WITHOUT_EQUITY = "wo_equity"
    WITHOUT_WELFARE = "wo_welfare"
    WITHOUT_BOTH = "wo_both"


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    variant: AblationVariant
    lambda_welfare: float
    lambda_equity: float
    description: str


def get_ablation_config(variant: AblationVariant, base_config: SWAConfig) -> SWAConfig:
    """
    Get SWA config for a specific ablation variant.
    
    Args:
        variant: Ablation variant to use
        base_config: Base SWA configuration
        
    Returns:
        Modified SWA config for the ablation variant
    """
    # Create a copy of base config
    config = SWAConfig(
        lambda_welfare=base_config.lambda_welfare,
        lambda_equity=base_config.lambda_equity,
        learning_rate=base_config.learning_rate,
        batch_size=base_config.batch_size,
        num_steps=base_config.num_steps,
        warmup_steps=base_config.warmup_steps,
        kl_coef=base_config.kl_coef,
        gradient_clip_norm=base_config.gradient_clip_norm,
        target_update_tau=base_config.target_update_tau
    )
    
    if variant == AblationVariant.FULL_SWA:
        # Keep original values
        pass
    elif variant == AblationVariant.WITHOUT_EQUITY:
        config.lambda_equity = 0.0
    elif variant == AblationVariant.WITHOUT_WELFARE:
        config.lambda_welfare = 0.0
    elif variant == AblationVariant.WITHOUT_BOTH:
        config.lambda_welfare = 0.0
        config.lambda_equity = 0.0
    
    return config


def get_all_ablation_configs(base_config: SWAConfig) -> Dict[str, SWAConfig]:
    """
    Get all ablation variant configurations.
    
    Args:
        base_config: Base SWA configuration
        
    Returns:
        Dict mapping variant names to configs
    """
    return {
        variant.value: get_ablation_config(variant, base_config)
        for variant in AblationVariant
    }


ABLATION_DESCRIPTIONS = {
    AblationVariant.FULL_SWA: "Full SWA with both welfare and equity losses",
    AblationVariant.WITHOUT_EQUITY: "SWA without equity loss (λ_equity = 0)",
    AblationVariant.WITHOUT_WELFARE: "SWA without welfare loss (λ_welfare = 0)",
    AblationVariant.WITHOUT_BOTH: "SWA without both losses (task loss only)"
}


def validate_ablation_config(config: SWAConfig, variant: AblationVariant) -> bool:
    """
    Validate that config matches expected ablation variant.
    
    Args:
        config: SWA configuration to validate
        variant: Expected ablation variant
        
    Returns:
        True if config matches variant requirements
    """
    if variant == AblationVariant.FULL_SWA:
        return config.lambda_welfare > 0 and config.lambda_equity > 0
    elif variant == AblationVariant.WITHOUT_EQUITY:
        return config.lambda_equity == 0 and config.lambda_welfare > 0
    elif variant == AblationVariant.WITHOUT_WELFARE:
        return config.lambda_welfare == 0 and config.lambda_equity > 0
    elif variant == AblationVariant.WITHOUT_BOTH:
        return config.lambda_welfare == 0 and config.lambda_equity == 0
    return False
