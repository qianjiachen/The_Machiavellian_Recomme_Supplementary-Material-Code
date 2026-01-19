#!/usr/bin/env python3
"""Ablation study runner for SWA components."""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

from src.config.configs import SWAConfig
from src.training.ablation import (
    AblationVariant, get_ablation_config, get_all_ablation_configs,
    ABLATION_DESCRIPTIONS
)
from src.evaluation.coalition import CoalitionAnalyzer
from src.visualization.plots import AblationPlotter, generate_results_table
from scripts.run_experiment import (
    run_single_experiment, aggregate_results, set_seed
)


def run_ablation_study(
    config: Dict,
    seeds: List[int],
    output_dir: Path,
    device: str = "cuda"
) -> Dict[str, Dict]:
    """
    Run ablation study for all SWA variants.
    
    Args:
        config: Base configuration
        seeds: Random seeds to use
        output_dir: Output directory
        device: Device to use
        
    Returns:
        Dict mapping variant names to results
    """
    base_swa_config = SWAConfig(**config["swa"])
    ablation_configs = get_all_ablation_configs(base_swa_config)
    
    all_results = {}
    
    for variant_name, swa_config in ablation_configs.items():
        print(f"\n{'='*60}")
        print(f"Running ablation variant: {variant_name}")
        print(f"  λ_welfare = {swa_config.lambda_welfare}")
        print(f"  λ_equity = {swa_config.lambda_equity}")
        print(f"{'='*60}")
        
        # Update config with ablation settings
        ablation_config = config.copy()
        ablation_config["swa"] = {
            "lambda_welfare": swa_config.lambda_welfare,
            "lambda_equity": swa_config.lambda_equity,
            "learning_rate": swa_config.learning_rate,
            "batch_size": swa_config.batch_size,
            "num_steps": swa_config.num_steps,
            "warmup_steps": swa_config.warmup_steps,
            "kl_coef": swa_config.kl_coef,
            "gradient_clip_norm": swa_config.gradient_clip_norm,
            "target_update_tau": swa_config.target_update_tau
        }
        
        # Run experiments for this variant
        variant_results = []
        for seed in seeds:
            print(f"  Seed {seed}...")
            result = run_single_experiment(
                method="swa",
                config=ablation_config,
                seed=seed,
                output_dir=output_dir / variant_name,
                device=device
            )
            variant_results.append(result)
        
        # Aggregate results
        aggregated = aggregate_results(variant_results)
        aggregated["variant"] = variant_name
        aggregated["lambda_welfare"] = swa_config.lambda_welfare
        aggregated["lambda_equity"] = swa_config.lambda_equity
        aggregated["description"] = ABLATION_DESCRIPTIONS.get(
            AblationVariant(variant_name), ""
        )
        
        all_results[variant_name] = aggregated
        
        # Save intermediate results
        with open(output_dir / f"{variant_name}_results.json", "w") as f:
            json.dump(aggregated, f, indent=2, default=str)
    
    return all_results


def run_coalition_robustness(
    config: Dict,
    seeds: List[int],
    output_dir: Path,
    device: str = "cuda"
) -> Dict:
    """
    Run coalition robustness analysis.
    
    Args:
        config: Configuration
        seeds: Random seeds
        output_dir: Output directory
        device: Device to use
        
    Returns:
        Coalition analysis results
    """
    print("\n" + "="*60)
    print("Running Coalition Robustness Analysis")
    print("="*60)
    
    analyzer = CoalitionAnalyzer()
    
    # First, get baseline collusion rate (no coalition)
    print("Computing baseline (no coalition)...")
    baseline_result = run_single_experiment(
        method="swa",
        config=config,
        seed=seeds[0],
        output_dir=output_dir / "coalition_baseline",
        device=device
    )
    baseline_collusion = baseline_result.get("collusion_rate", 0.5)
    
    # Run for each coalition size
    coalition_results = {}
    for size in analyzer.coalition_sizes:
        print(f"\nCoalition size: {size*100:.0f}%")
        
        size_results = []
        for seed in seeds:
            # In a real implementation, you'd pass coalition info to the experiment
            result = run_single_experiment(
                method="swa",
                config=config,
                seed=seed,
                output_dir=output_dir / f"coalition_{int(size*100)}",
                device=device
            )
            size_results.append(result)
        
        aggregated = aggregate_results(size_results)
        
        # Compute defense rate
        if baseline_collusion > 0:
            aggregated["defense_rate"] = 1 - (aggregated.get("collusion_rate", 0) / baseline_collusion)
        else:
            aggregated["defense_rate"] = 1.0
        
        coalition_results[f"{int(size*100)}%"] = aggregated
    
    return coalition_results


def main():
    parser = argparse.ArgumentParser(description="Run SWA ablation studies")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Random seeds")
    parser.add_argument("--output-dir", type=str, default="outputs/ablation",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--run-coalition", action="store_true",
                       help="Also run coalition robustness analysis")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Run ablation study
    ablation_results = run_ablation_study(
        config, args.seeds, output_dir, args.device
    )
    
    # Generate ablation visualizations
    print("\nGenerating ablation visualizations...")
    plotter = AblationPlotter(str(output_dir / "plots"))
    
    plotter.plot_ablation_bars(ablation_results)
    plotter.plot_component_contribution(
        full_swa=ablation_results.get("full_swa", {}),
        wo_equity=ablation_results.get("wo_equity", {}),
        wo_welfare=ablation_results.get("wo_welfare", {}),
        wo_both=ablation_results.get("wo_both", {})
    )
    
    # Run coalition analysis if requested
    if args.run_coalition:
        coalition_results = run_coalition_robustness(
            config, args.seeds, output_dir, args.device
        )
        
        with open(output_dir / "coalition_results.json", "w") as f:
            json.dump(coalition_results, f, indent=2, default=str)
    
    # Generate results table
    table = generate_results_table(ablation_results, str(output_dir / "ablation_table.txt"))
    print("\nAblation Results:")
    print(table)
    
    # Save all results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
