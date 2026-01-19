#!/usr/bin/env python3
"""Semi-real validation runner using Amazon Reviews dataset."""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

from src.config.configs import ArenaConfig
from src.data.amazon_reviews import AmazonReviewsLoader
from src.environment.arena import RecGameArena
from src.evaluation.evaluator import Evaluator
from src.detection.detector import CollusionDetector
from src.detection.paraphrase import ParaphraseGenerator
from src.detection.mine import MINEEstimator
from src.visualization.plots import TrainingPlotter, generate_results_table
from scripts.run_experiment import run_single_experiment, aggregate_results


def setup_semi_real_arena(
    loader: AmazonReviewsLoader,
    config: Dict,
    num_users: int = 100,
    num_vendors: int = 20
) -> RecGameArena:
    """
    Setup arena with Amazon Reviews data.
    
    Args:
        loader: Amazon reviews data loader
        config: Configuration dict
        num_users: Number of users to sample
        num_vendors: Number of vendors (brands) to use
        
    Returns:
        Configured RecGameArena
    """
    rng = np.random.RandomState(42)
    
    # Get active users and popular products
    active_users = loader.get_active_users(min_reviews=5)[:num_users]
    popular_products = loader.get_popular_products(min_reviews=10)
    
    # Create arena config
    arena_config = ArenaConfig(
        num_users=len(active_users),
        num_vendors=num_vendors,
        num_items_per_vendor=len(popular_products) // num_vendors,
        embedding_dim=config["arena"]["embedding_dim"]
    )
    
    arena = RecGameArena(arena_config)
    return arena


def run_semi_real_validation(
    config: Dict,
    data_dir: str,
    category: str,
    seeds: List[int],
    output_dir: Path,
    device: str = "cuda"
) -> Dict:
    """
    Run semi-real validation experiment.
    
    Args:
        config: Configuration
        data_dir: Path to Amazon Reviews data
        category: Product category to use
        seeds: Random seeds
        output_dir: Output directory
        device: Device to use
        
    Returns:
        Experiment results
    """
    print(f"\nLoading Amazon Reviews data from {data_dir}...")
    loader = AmazonReviewsLoader(
        data_dir=data_dir,
        category=category,
        embedding_dim=config["arena"]["embedding_dim"]
    )
    
    # Load data
    num_reviews = loader.load_reviews(max_reviews=50000)
    num_products = loader.load_products()
    print(f"Loaded {num_reviews} reviews and {num_products} products")
    
    if num_reviews == 0:
        print("No data loaded. Please download Amazon Reviews dataset.")
        print("Download from: https://nijianmo.github.io/amazon/index.html")
        return {}
    
    # Extract communication templates
    templates = loader.extract_communication_templates(n_templates=100)
    print(f"Extracted {len(templates)} communication templates")
    
    # Run experiments
    all_results = []
    for seed in seeds:
        print(f"\nRunning seed {seed}...")
        
        # Modify config for semi-real setting
        semi_real_config = config.copy()
        semi_real_config["arena"]["num_users"] = min(100, len(loader.get_active_users()))
        semi_real_config["arena"]["num_vendors"] = 20
        
        result = run_single_experiment(
            method="swa",
            config=semi_real_config,
            seed=seed,
            output_dir=output_dir,
            device=device
        )
        
        # Add semi-real specific metrics
        result["data_source"] = "amazon_reviews"
        result["category"] = category
        result["num_reviews_used"] = num_reviews
        
        all_results.append(result)
    
    # Aggregate
    aggregated = aggregate_results(all_results)
    aggregated["data_source"] = "amazon_reviews"
    aggregated["category"] = category
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run semi-real validation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data/amazon")
    parser.add_argument("--category", type=str, default="Electronics")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--output-dir", type=str, default="outputs/semi_real")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_semi_real_validation(
        config, args.data_dir, args.category, args.seeds, output_dir, args.device
    )
    
    with open(output_dir / "semi_real_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
