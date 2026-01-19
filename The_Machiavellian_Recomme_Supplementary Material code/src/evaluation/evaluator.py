"""Evaluator module for comprehensive experiment evaluation."""

from typing import List, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass, field

from src.evaluation.metrics import MetricsCalculator
from src.detection.detector import CollusionDetector
from src.models.data_models import Transaction, GlobalState, ExperimentResult


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    collusion_rate: float
    defense_rate: float
    user_utility: float
    gini_coefficient: float
    price_margin: float
    hhi: float
    consumer_surplus: float
    additional_metrics: Dict[str, float] = field(default_factory=dict)


class Evaluator:
    """
    Comprehensive evaluator for experiment results.
    """
    
    def __init__(
        self,
        detector: Optional[CollusionDetector] = None,
        baseline_collusion_rate: float = 0.42  # From paper: Independent PPO
    ):
        self.detector = detector or CollusionDetector()
        self.baseline_collusion_rate = baseline_collusion_rate
        self.metrics = MetricsCalculator()
    
    def evaluate(
        self,
        state: GlobalState,
        transactions: List[Transaction],
        user_preferences: Dict[str, np.ndarray],
        vendor_ids: List[str],
        get_response_fn=None
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation.
        
        Args:
            state: Current global state
            transactions: List of transactions to evaluate
            user_preferences: User preference vectors
            vendor_ids: List of vendor IDs
            get_response_fn: Function to get agent responses (for detection)
            
        Returns:
            Complete evaluation result
        """
        # Compute collusion rate
        collusion_rate = self.detector.compute_collusion_rate(
            transactions, get_response_fn
        )
        
        # Compute defense rate
        defense_rate = self.metrics.compute_defense_rate(
            collusion_rate, self.baseline_collusion_rate
        )
        
        # Compute user utility (average across users)
        user_utilities = []
        for user_id, pref in user_preferences.items():
            user_purchases = [
                txn.item.embedding for txn in transactions
                if txn.user_id == user_id
            ]
            if user_purchases:
                utility = self.metrics.compute_user_utility(pref, user_purchases)
                user_utilities.append(utility)
        
        avg_user_utility = np.mean(user_utilities) if user_utilities else 0.0
        
        # Compute Gini coefficient
        rewards = np.array(list(state.agent_rewards.values()))
        gini = self.metrics.compute_gini_efficient(rewards)
        
        # Compute price margin
        price_margin = self.metrics.compute_price_margin(transactions)
        
        # Compute HHI
        market_shares = self.metrics.compute_market_shares(transactions, vendor_ids)
        hhi = self.metrics.compute_hhi(market_shares)
        
        # Compute consumer surplus
        consumer_surplus = self.metrics.compute_consumer_surplus(transactions)
        
        return EvaluationResult(
            collusion_rate=collusion_rate,
            defense_rate=defense_rate,
            user_utility=avg_user_utility,
            gini_coefficient=gini,
            price_margin=price_margin,
            hhi=hhi,
            consumer_surplus=consumer_surplus
        )
    
    def evaluate_multiple_seeds(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate evaluation results across multiple seeds.
        
        Args:
            results: List of evaluation results from different seeds
            
        Returns:
            Aggregated statistics (mean, std) for each metric
        """
        if not results:
            return {}
        
        metrics_dict = [
            {
                "collusion_rate": r.collusion_rate,
                "defense_rate": r.defense_rate,
                "user_utility": r.user_utility,
                "gini_coefficient": r.gini_coefficient,
                "price_margin": r.price_margin,
                "hhi": r.hhi,
                "consumer_surplus": r.consumer_surplus
            }
            for r in results
        ]
        
        return self.metrics.aggregate_results(metrics_dict)
    
    def create_experiment_result(
        self,
        method_name: str,
        eval_result: EvaluationResult,
        training_curves: Dict[str, List[float]],
        seed: int,
        config: Dict[str, Any]
    ) -> ExperimentResult:
        """
        Create an ExperimentResult from evaluation.
        
        Args:
            method_name: Name of the method
            eval_result: Evaluation result
            training_curves: Training curves data
            seed: Random seed used
            config: Configuration used
            
        Returns:
            Complete experiment result
        """
        return ExperimentResult(
            method_name=method_name,
            collusion_rate=eval_result.collusion_rate,
            defense_rate=eval_result.defense_rate,
            user_utility=eval_result.user_utility,
            gini_coefficient=eval_result.gini_coefficient,
            price_margin=eval_result.price_margin,
            hhi=eval_result.hhi,
            consumer_surplus=eval_result.consumer_surplus,
            training_curves=training_curves,
            seed=seed,
            config=config
        )
    
    def compare_methods(
        self,
        results: Dict[str, List[EvaluationResult]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare multiple methods.
        
        Args:
            results: Dictionary mapping method names to their results
            
        Returns:
            Comparison statistics for each method
        """
        comparison = {}
        for method_name, method_results in results.items():
            comparison[method_name] = self.evaluate_multiple_seeds(method_results)
        return comparison
    
    def format_results_table(
        self,
        comparison: Dict[str, Dict[str, Dict[str, float]]]
    ) -> str:
        """
        Format comparison results as a table string.
        
        Args:
            comparison: Comparison results from compare_methods
            
        Returns:
            Formatted table string
        """
        headers = ["Method", "Defense Rate", "User Util.", "Gini", "Margin", "HHI", "Surplus"]
        
        rows = []
        for method, metrics in comparison.items():
            row = [method]
            for key in ["defense_rate", "user_utility", "gini_coefficient", 
                       "price_margin", "hhi", "consumer_surplus"]:
                if key in metrics:
                    mean = metrics[key]["mean"]
                    std = metrics[key]["std"]
                    row.append(f"{mean:.3f}±{std:.3f}")
                else:
                    row.append("N/A")
            rows.append(row)
        
        # Format as table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) 
                      for i in range(len(headers))]
        
        lines = []
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        for row in rows:
            lines.append(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))
        
        return "\n".join(lines)
