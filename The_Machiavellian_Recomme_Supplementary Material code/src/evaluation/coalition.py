"""Coalition robustness analysis for collusion detection."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.models.data_models import Transaction


@dataclass
class CoalitionResult:
    """Result of coalition robustness analysis."""
    coalition_size: float  # Percentage of colluding agents
    collusion_rate: float
    defense_rate: float
    user_utility: float
    gini_coefficient: float


class CoalitionAnalyzer:
    """
    Analyzer for coalition robustness.
    
    Tests how well the defense mechanism performs when
    different percentages of agents form coalitions.
    """
    
    DEFAULT_COALITION_SIZES = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    def __init__(
        self,
        coalition_sizes: Optional[List[float]] = None,
        seed: int = 42
    ):
        self.coalition_sizes = coalition_sizes or self.DEFAULT_COALITION_SIZES
        self.rng = np.random.RandomState(seed)
    
    def select_coalition_members(
        self,
        agent_ids: List[str],
        coalition_size: float
    ) -> List[str]:
        """
        Select agents to form a coalition.
        
        Args:
            agent_ids: List of all agent IDs
            coalition_size: Fraction of agents to include (0.0 to 1.0)
            
        Returns:
            List of agent IDs in the coalition
        """
        n_members = int(len(agent_ids) * coalition_size)
        if n_members == 0:
            return []
        
        indices = self.rng.choice(len(agent_ids), size=n_members, replace=False)
        return [agent_ids[i] for i in indices]
    
    def compute_coalition_metrics(
        self,
        transactions: List[Transaction],
        coalition_members: List[str],
        all_agent_rewards: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute metrics for a specific coalition configuration.
        
        Args:
            transactions: List of transactions
            coalition_members: IDs of agents in the coalition
            all_agent_rewards: Rewards for all agents
            
        Returns:
            Dict of computed metrics
        """
        coalition_set = set(coalition_members)
        
        # Collusion rate: fraction of transactions involving coalition members
        if not transactions:
            collusion_rate = 0.0
        else:
            collusive_txns = sum(
                1 for t in transactions
                if t.vendor_id in coalition_set
            )
            collusion_rate = collusive_txns / len(transactions)
        
        # User utility: average utility from non-coalition transactions
        non_coalition_utilities = []
        for t in transactions:
            if t.vendor_id not in coalition_set:
                utility = t.willingness_to_pay - t.actual_price
                non_coalition_utilities.append(utility)
        
        user_utility = np.mean(non_coalition_utilities) if non_coalition_utilities else 0.0
        
        # Gini coefficient of rewards
        rewards = np.array(list(all_agent_rewards.values()))
        gini = self._compute_gini(rewards)
        
        return {
            "collusion_rate": collusion_rate,
            "user_utility": user_utility,
            "gini_coefficient": gini
        }
    
    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient."""
        if len(values) == 0 or values.sum() == 0:
            return 0.0
        
        n = len(values)
        sorted_values = np.sort(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values)) / (n * cumsum[-1])) - (n + 1) / n
    
    def run_coalition_analysis(
        self,
        run_experiment_fn,
        agent_ids: List[str],
        baseline_collusion_rate: float
    ) -> List[CoalitionResult]:
        """
        Run coalition robustness analysis across different coalition sizes.
        
        Args:
            run_experiment_fn: Function that runs experiment and returns metrics
            agent_ids: List of all agent IDs
            baseline_collusion_rate: Baseline collusion rate for defense rate calculation
            
        Returns:
            List of results for each coalition size
        """
        results = []
        
        for size in self.coalition_sizes:
            coalition = self.select_coalition_members(agent_ids, size)
            
            # Run experiment with this coalition
            metrics = run_experiment_fn(coalition_members=coalition)
            
            # Compute defense rate
            if baseline_collusion_rate > 0:
                defense_rate = 1 - (metrics["collusion_rate"] / baseline_collusion_rate)
            else:
                defense_rate = 1.0
            
            results.append(CoalitionResult(
                coalition_size=size,
                collusion_rate=metrics["collusion_rate"],
                defense_rate=defense_rate,
                user_utility=metrics["user_utility"],
                gini_coefficient=metrics["gini_coefficient"]
            ))
        
        return results
    
    def summarize_results(self, results: List[CoalitionResult]) -> Dict[str, List[float]]:
        """Summarize coalition analysis results."""
        return {
            "coalition_sizes": [r.coalition_size for r in results],
            "collusion_rates": [r.collusion_rate for r in results],
            "defense_rates": [r.defense_rate for r in results],
            "user_utilities": [r.user_utility for r in results],
            "gini_coefficients": [r.gini_coefficient for r in results]
        }
