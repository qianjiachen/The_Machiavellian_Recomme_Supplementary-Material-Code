"""Metrics calculation module."""

import numpy as np
from typing import List, Dict, Optional
from src.models.data_models import Transaction, GlobalState


class MetricsCalculator:
    """Calculator for all evaluation metrics."""
    
    @staticmethod
    def compute_collusion_rate(
        collusive_transactions: int,
        total_transactions: int
    ) -> float:
        """
        Compute collusion rate.
        
        Collusion Rate = |collusive transactions| / |total transactions|
        """
        if total_transactions == 0:
            return 0.0
        return collusive_transactions / total_transactions
    
    @staticmethod
    def compute_defense_rate(
        method_collusion_rate: float,
        baseline_collusion_rate: float
    ) -> float:
        """
        Compute defense rate relative to baseline.
        
        Defense Rate = 1 - (method_collusion_rate / baseline_collusion_rate)
        """
        if baseline_collusion_rate == 0:
            return 1.0 if method_collusion_rate == 0 else 0.0
        return 1.0 - (method_collusion_rate / baseline_collusion_rate)
    
    @staticmethod
    def compute_gini_coefficient(rewards: np.ndarray) -> float:
        """
        Compute Gini coefficient for reward distribution.
        
        G(R) = Σᵢ Σⱼ |Rᵢ - Rⱼ| / (2n × Σᵢ Rᵢ)
        
        Args:
            rewards: Array of rewards for each agent
            
        Returns:
            Gini coefficient in [0, 1], where 0 is perfect equality
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        n = len(rewards)
        
        if n == 0:
            return 0.0
        
        total_reward = np.sum(rewards)
        if total_reward == 0:
            return 0.0
        
        # Compute sum of absolute differences
        diff_sum = 0.0
        for i in range(n):
            for j in range(n):
                diff_sum += abs(rewards[i] - rewards[j])
        
        gini = diff_sum / (2 * n * total_reward)
        return gini
    
    @staticmethod
    def compute_gini_efficient(rewards: np.ndarray) -> float:
        """
        Compute Gini coefficient efficiently using sorted array.
        
        More efficient O(n log n) implementation.
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        n = len(rewards)
        
        if n == 0:
            return 0.0
        
        total_reward = np.sum(rewards)
        if total_reward == 0:
            return 0.0
        
        sorted_rewards = np.sort(rewards)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * sorted_rewards) / (n * total_reward)) - (n + 1) / n
        return gini
    
    @staticmethod
    def compute_hhi(market_shares: np.ndarray) -> float:
        """
        Compute Herfindahl-Hirschman Index (HHI) for market concentration.
        
        HHI = Σᵢ (market_shareᵢ)²
        
        Args:
            market_shares: Array of market shares (should sum to 1)
            
        Returns:
            HHI in [0, 1], where higher values indicate more concentration
        """
        market_shares = np.asarray(market_shares, dtype=np.float64)
        
        if len(market_shares) == 0:
            return 0.0
        
        # Normalize if not already
        total = np.sum(market_shares)
        if total > 0:
            market_shares = market_shares / total
        
        return float(np.sum(market_shares ** 2))
    
    @staticmethod
    def compute_consumer_surplus(transactions: List[Transaction]) -> float:
        """
        Compute average consumer surplus.
        
        Consumer Surplus = mean(willingness_to_pay - actual_price)
        """
        if not transactions:
            return 0.0
        
        surpluses = [
            txn.willingness_to_pay - txn.actual_price
            for txn in transactions
        ]
        return float(np.mean(surpluses))
    
    @staticmethod
    def compute_price_margin(transactions: List[Transaction]) -> float:
        """
        Compute average price-cost margin.
        
        Margin = mean((price - base_price) / base_price)
        """
        if not transactions:
            return 0.0
        
        margins = []
        for txn in transactions:
            if txn.item.base_price > 0:
                margin = (txn.actual_price - txn.item.base_price) / txn.item.base_price
                margins.append(margin)
        
        return float(np.mean(margins)) if margins else 0.0
    
    @staticmethod
    def compute_user_utility(
        user_preference: np.ndarray,
        purchased_embeddings: List[np.ndarray]
    ) -> float:
        """
        Compute ground-truth user utility.
        
        Utility = cosine_similarity(mean(purchased_embeddings), preference)
        """
        if not purchased_embeddings:
            return 0.0
        
        embeddings = np.stack(purchased_embeddings)
        mean_embedding = embeddings.mean(axis=0)
        
        # Cosine similarity
        norm_pref = np.linalg.norm(user_preference)
        norm_emb = np.linalg.norm(mean_embedding)
        
        if norm_pref < 1e-8 or norm_emb < 1e-8:
            return 0.0
        
        return float(np.dot(user_preference, mean_embedding) / (norm_pref * norm_emb))
    
    @staticmethod
    def compute_market_shares(
        transactions: List[Transaction],
        vendor_ids: List[str]
    ) -> np.ndarray:
        """
        Compute market shares for each vendor.
        
        Args:
            transactions: List of transactions
            vendor_ids: List of all vendor IDs
            
        Returns:
            Array of market shares
        """
        if not transactions:
            n = len(vendor_ids)
            return np.ones(n) / n if n > 0 else np.array([])
        
        # Count transactions per vendor
        vendor_counts = {vid: 0 for vid in vendor_ids}
        for txn in transactions:
            if txn.vendor_id in vendor_counts:
                vendor_counts[txn.vendor_id] += 1
        
        total = sum(vendor_counts.values())
        if total == 0:
            n = len(vendor_ids)
            return np.ones(n) / n
        
        shares = np.array([vendor_counts[vid] / total for vid in vendor_ids])
        return shares
    
    @staticmethod
    def aggregate_results(
        results: List[Dict[str, float]],
        keys: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results across multiple seeds.
        
        Args:
            results: List of result dictionaries
            keys: Keys to aggregate (default: all keys)
            
        Returns:
            Dictionary with mean and std for each metric
        """
        if not results:
            return {}
        
        if keys is None:
            keys = list(results[0].keys())
        
        aggregated = {}
        for key in keys:
            values = [r[key] for r in results if key in r]
            if values:
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        return aggregated
