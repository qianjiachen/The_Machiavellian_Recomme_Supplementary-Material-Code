#!/usr/bin/env python3
"""Main experiment runner for Machiavellian Recommender experiments."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

from src.config.configs import (
    ArenaConfig, SWAConfig, LLMConfig, DetectionConfig, 
    EvaluationConfig, LoggingConfig, load_config
)
from src.environment.arena import RecGameArena
from src.agents.user_agent import UserAgent
from src.agents.vendor_agent import VendorAgent
from src.agents.platform_agent import PlatformAgent
from src.training.swa_trainer import SWATrainer
from src.baselines.independent_ppo import IndependentPPO
from src.baselines.maddpg import MADDPG
from src.baselines.qmix import QMIX
from src.baselines.soto import SOTO
from src.baselines.aga import AgA
from src.baselines.agent_mixer import AgentMixer
from src.baselines.rlhf import RLHF
from src.detection.detector import CollusionDetector
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import MetricsCalculator
from src.visualization.plots import TrainingPlotter, ParetoPlotter, generate_results_table
from src.utils.logging import ExperimentLogger
from src.utils.checkpoint import CheckpointManager


METHODS = {
    "swa": "Social Welfare Alignment",
    "independent_ppo": "Independent PPO",
    "maddpg": "MADDPG",
    "qmix": "QMIX",
    "soto": "SOTO",
    "aga": "AgA",
    "agent_mixer": "AgentMixer",
    "rlhf": "RLHF"
}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_trainer(method: str, config: Dict, arena: RecGameArena, device: str):
    """Create trainer for specified method."""
    state_dim = config["arena"]["embedding_dim"] * 2
    action_dim = 10
    num_agents = config["arena"]["num_users"] + config["arena"]["num_vendors"]
    
    if method == "swa":
        swa_config = SWAConfig(**config["swa"])
        return SWATrainer(swa_config, state_dim, action_dim, num_agents, device)
    elif method == "independent_ppo":
        return IndependentPPO(state_dim, action_dim, num_agents, device=device)
    elif method == "maddpg":
        return MADDPG(state_dim, action_dim, num_agents, device=device)
    elif method == "qmix":
        return QMIX(state_dim, action_dim, num_agents, device=device)
    elif method == "soto":
        return SOTO(state_dim, action_dim, num_agents, device=device)
    elif method == "aga":
        return AgA(state_dim, action_dim, num_agents, device=device)
    elif method == "agent_mixer":
        return AgentMixer(state_dim, action_dim, num_agents, device=device)
    elif method == "rlhf":
        return RLHF(state_dim, action_dim, num_agents, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_single_experiment(
    method: str,
    config: Dict,
    seed: int,
    output_dir: Path,
    device: str = "cuda"
) -> Dict:
    """Run a single experiment with specified method and seed."""
    set_seed(seed)
    
    # Create arena
    arena_config = ArenaConfig(**config["arena"])
    arena = RecGameArena(arena_config)
    
    # Create trainer
    trainer = create_trainer(method, config, arena, device)
    
    # Create detector and evaluator
    detection_config = DetectionConfig(**config["detection"])
    detector = CollusionDetector(detection_config)
    evaluator = Evaluator(detector)
    
    # Create logger
    logger = ExperimentLogger(
        experiment_name=f"{method}_seed{seed}",
        log_dir=str(output_dir / "logs"),
        use_tensorboard=config["logging"]["use_tensorboard"],
        use_wandb=config["logging"]["use_wandb"]
    )
    
    # Training parameters
    num_steps = config["swa"]["num_steps"]
    eval_interval = config["evaluation"]["eval_interval"]
    rollout_length = config.get("training", {}).get("rollout_length", 128)
    batch_size = config.get("training", {}).get("batch_size", 64)
    
    training_curves = {
        "collusion_rate": [],
        "user_utility": [],
        "gini_coefficient": [],
        "defense_rate": []
    }
    
    # Create environment wrapper for baseline trainers
    class EnvWrapper:
        """Wrapper to provide consistent interface for baseline trainers."""
        def __init__(self, arena, seed):
            self.arena = arena
            self.seed = seed
            self._step_count = 0
        
        def reset(self):
            self._step_count = 0
            obs = self.arena.reset(seed=self.seed + self._step_count)
            # Convert observations to numpy arrays
            return {k: np.array(v.preference_embedding if hasattr(v, 'preference_embedding') else [0.0] * 128) 
                    for k, v in obs.items()} if obs else {}
        
        def step(self, actions):
            self._step_count += 1
            # Convert actions to proper format
            formatted_actions = {}
            for agent_id, action in actions.items():
                formatted_actions[agent_id] = _sample_action(agent_id, None)
            
            obs, rewards, dones, infos = self.arena.step(formatted_actions)
            
            # Convert to numpy
            obs_np = {k: np.array(v.preference_embedding if hasattr(v, 'preference_embedding') else [0.0] * 128) 
                      for k, v in obs.items()} if obs else {}
            rewards_np = {k: float(v) for k, v in rewards.items()} if rewards else {}
            dones_np = {k: bool(v) for k, v in dones.items()} if dones else {}
            
            return obs_np, rewards_np, dones_np, infos
    
    env_wrapper = EnvWrapper(arena, seed)
    
    # Use trainer's train method if available and method is not SWA
    if method != "swa" and hasattr(trainer, 'train'):
        print(f"[{method}] Starting training with {num_steps} steps...")
        
        # Run training with periodic evaluation
        eval_steps = num_steps // 10
        for epoch in range(10):
            # Train for eval_steps
            trainer.train(env_wrapper, eval_steps)
            
            # Evaluate
            state = arena.get_state()
            transactions = state.transaction_history[-1000:] if state.transaction_history else []
            
            metrics = _compute_metrics(detector, transactions, state, arena)
            
            training_curves["collusion_rate"].append(metrics.get("collusion_rate", 0))
            training_curves["user_utility"].append(metrics.get("user_utility", 0))
            training_curves["gini_coefficient"].append(metrics.get("gini_coefficient", 0))
            training_curves["defense_rate"].append(metrics.get("defense_rate", 0))
            
            logger.log_metrics(metrics, (epoch + 1) * eval_steps)
            
            print(f"[{method}] Epoch {epoch+1}/10: collusion={metrics.get('collusion_rate', 0):.3f}, "
                  f"gini={metrics.get('gini_coefficient', 0):.3f}")
    else:
        # SWA or fallback to step-by-step training
        observations = arena.reset(seed=seed)
        
        for step in range(num_steps):
            # Generate actions using trainer's policy if available
            actions = {}
            for agent_id in arena.get_all_agent_ids():
                if method == "swa" and hasattr(trainer, 'get_action'):
                    obs = observations.get(agent_id)
                    if obs is not None:
                        obs_tensor = torch.FloatTensor(
                            obs.preference_embedding if hasattr(obs, 'preference_embedding') else [0.0] * 128
                        ).unsqueeze(0).to(device)
                        action = trainer.get_action(agent_id, obs_tensor)
                        actions[agent_id] = _sample_action(agent_id, obs)
                    else:
                        actions[agent_id] = _sample_action(agent_id, None)
                else:
                    actions[agent_id] = _sample_action(agent_id, observations.get(agent_id))
            
            # Environment step
            observations, rewards, dones, infos = arena.step(actions)
            
            # Training step for SWA
            if method == "swa" and hasattr(trainer, 'train_step'):
                trainer.train_step(observations, actions, rewards, dones)
            
            # Evaluation
            if step % eval_interval == 0:
                state = arena.get_state()
                transactions = state.transaction_history[-1000:] if state.transaction_history else []
                
                metrics = _compute_metrics(detector, transactions, state, arena)
                
                training_curves["collusion_rate"].append(metrics.get("collusion_rate", 0))
                training_curves["user_utility"].append(metrics.get("user_utility", 0))
                training_curves["gini_coefficient"].append(metrics.get("gini_coefficient", 0))
                training_curves["defense_rate"].append(metrics.get("defense_rate", 0))
                
                logger.log_metrics(metrics, step)
                
                if step % (eval_interval * 10) == 0:
                    print(f"[{method}] Step {step}: collusion={metrics.get('collusion_rate', 0):.3f}, "
                          f"utility={metrics.get('user_utility', 0):.3f}")
            
            if all(dones.values()):
                observations = arena.reset(seed=seed + step)
    
    # Final evaluation
    state = arena.get_state()
    transactions = state.transaction_history
    final_metrics = _compute_metrics(detector, transactions, state, arena, full=True)
    
    final_metrics["training_curves"] = training_curves
    final_metrics["seed"] = seed
    final_metrics["method"] = method
    
    logger.close()
    
    return final_metrics


def _compute_metrics(detector, transactions, state, arena, full=False):
    """Compute evaluation metrics."""
    metrics = {
        "collusion_rate": detector.compute_collusion_rate(transactions) if transactions else 0.0,
        "user_utility": 0.0,
        "gini_coefficient": 0.0,
        "defense_rate": 0.0
    }
    
    if full:
        metrics.update({
            "price_margin": 0.0,
            "hhi": 0.0,
            "consumer_surplus": 0.0
        })
    
    if transactions:
        calc = MetricsCalculator()
        rewards = np.array(list(state.agent_rewards.values())) if state.agent_rewards else np.array([0.0])
        metrics["gini_coefficient"] = calc.compute_gini_efficient(rewards)
        metrics["defense_rate"] = calc.compute_defense_rate(metrics["collusion_rate"], 0.42)
        
        if full:
            metrics["price_margin"] = calc.compute_price_margin(transactions)
            metrics["consumer_surplus"] = calc.compute_consumer_surplus(transactions)
            market_shares = calc.compute_market_shares(transactions, arena.vendor_ids)
            metrics["hhi"] = calc.compute_hhi(market_shares)
    
    return metrics


def _sample_action(agent_id: str, observation):
    """Sample a random action (placeholder)."""
    from src.models.data_models import Action, ActionType
    
    if agent_id.startswith("user_"):
        return Action(
            agent_id=agent_id,
            action_type=ActionType.PURCHASE,
            target_id="item_vendor_0_0",
            parameters={}
        )
    else:
        return Action(
            agent_id=agent_id,
            action_type=ActionType.SET_PRICE,
            target_id="item_" + agent_id + "_0",
            parameters={"price": np.random.uniform(10, 100)}
        )


def run_multi_seed_experiment(
    method: str,
    config: Dict,
    seeds: List[int],
    output_dir: Path,
    device: str = "cuda"
) -> Dict:
    """Run experiment across multiple seeds and aggregate results."""
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Running {method} with seed {seed}")
        print(f"{'='*50}")
        
        result = run_single_experiment(method, config, seed, output_dir, device)
        all_results.append(result)
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    aggregated["method"] = method
    aggregated["seeds"] = seeds
    
    return aggregated


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results from multiple seeds."""
    metrics_to_aggregate = [
        "collusion_rate", "defense_rate", "user_utility",
        "gini_coefficient", "price_margin", "hhi", "consumer_surplus"
    ]
    
    aggregated = {}
    for metric in metrics_to_aggregate:
        values = [r.get(metric, 0) for r in results if metric in r]
        if values:
            aggregated[metric] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run Machiavellian Recommender experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--method", type=str, default="all",
                       choices=list(METHODS.keys()) + ["all"],
                       help="Method to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024],
                       help="Random seeds")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Determine methods to run
    methods = list(METHODS.keys()) if args.method == "all" else [args.method]
    
    all_results = {}
    
    for method in methods:
        print(f"\n{'#'*60}")
        print(f"# Running method: {METHODS[method]}")
        print(f"{'#'*60}")
        
        results = run_multi_seed_experiment(
            method, config, args.seeds, output_dir, args.device
        )
        all_results[method] = results
        
        # Save intermediate results
        with open(output_dir / f"{method}_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plotter = TrainingPlotter(str(output_dir / "plots"))
    
    # Training curves
    curves = {m: r.get("training_curves", {}) for m, r in all_results.items()}
    plotter.plot_training_curves(curves)
    
    # Pareto frontier
    pareto_plotter = ParetoPlotter(str(output_dir / "plots"))
    pareto_data = {
        m: (r.get("user_utility", 0), 1 - r.get("gini_coefficient", 0))
        for m, r in all_results.items()
    }
    pareto_plotter.plot_pareto_frontier(pareto_data)
    
    # Results table
    table = generate_results_table(all_results, str(output_dir / "results_table.txt"))
    print("\nResults:")
    print(table)
    
    # Save final results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
