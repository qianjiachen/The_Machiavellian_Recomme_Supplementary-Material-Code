"""Platform agent implementation."""

import numpy as np
from typing import List, Dict, Optional
from src.models.data_models import Item, UserPersona
from src.environment.utility import UtilityCalculator


class PlatformAgent:
    """
    Platform agent that implements recommendation algorithms.
    """
    
    def __init__(
        self,
        recommendation_strategy: str = "utility_based",
        num_recommendations: int = 10
    ):
        self.recommendation_strategy = recommendation_strategy
        self.num_recommendations = num_recommendations
        self.utility_calculator = UtilityCalculator()
        
        # Track platform metrics
        self.total_transactions = 0
        self.total_revenue = 0.0
        self.platform_fee_rate = 0.05
    
    def generate_recommendations(
        self,
        user_persona: UserPersona,
        available_items: List[Item],
        exclude_items: Optional[List[str]] = None
    ) -> List[Item]:
        """
        Generate personalized recommendations for a user.
        
        Args:
            user_persona: User's persona with preferences
            available_items: List of available items
            exclude_items: Item IDs to exclude (e.g., already purchased)
            
        Returns:
            List of recommended items
        """
        exclude_set = set(exclude_items or [])
        
        # Filter available items
        candidates = [
            item for item in available_items
            if item.item_id not in exclude_set and item.stock > 0
        ]
        
        if not candidates:
            return []
        
        if self.recommendation_strategy == "utility_based":
            return self._utility_based_recommendations(user_persona, candidates)
        elif self.recommendation_strategy == "popularity":
            return self._popularity_based_recommendations(candidates)
        elif self.recommendation_strategy == "random":
            return self._random_recommendations(candidates)
        else:
            return self._utility_based_recommendations(user_persona, candidates)
    
    def _utility_based_recommendations(
        self,
        user_persona: UserPersona,
        items: List[Item]
    ) -> List[Item]:
        """Generate recommendations based on utility matching."""
        scored_items = []
        
        for item in items:
            # Compute utility score
            utility = self.utility_calculator.compute_item_utility(
                user_persona.preference_vector,
                item.embedding
            )
            
            # Adjust for price (prefer items within budget)
            price_factor = 1.0
            if item.current_price > user_persona.budget:
                price_factor = 0.1
            elif item.current_price > user_persona.budget * 0.8:
                price_factor = 0.7
            
            # Compute final score
            score = utility * price_factor
            scored_items.append((score, item))
        
        # Sort by score and return top N
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:self.num_recommendations]]
    
    def _popularity_based_recommendations(self, items: List[Item]) -> List[Item]:
        """Generate recommendations based on popularity (stock as proxy)."""
        # Use inverse of stock as popularity proxy (lower stock = more popular)
        scored_items = [(100 - item.stock, item) for item in items]
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:self.num_recommendations]]
    
    def _random_recommendations(self, items: List[Item]) -> List[Item]:
        """Generate random recommendations."""
        import random
        return random.sample(items, min(len(items), self.num_recommendations))
    
    def compute_platform_fee(self, transaction_value: float) -> float:
        """Compute platform fee for a transaction."""
        return transaction_value * self.platform_fee_rate
    
    def record_transaction(self, transaction_value: float):
        """Record a transaction for platform metrics."""
        self.total_transactions += 1
        self.total_revenue += self.compute_platform_fee(transaction_value)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get platform metrics."""
        return {
            "total_transactions": self.total_transactions,
            "total_revenue": self.total_revenue,
            "avg_fee_per_transaction": (
                self.total_revenue / self.total_transactions
                if self.total_transactions > 0 else 0.0
            )
        }
    
    def reset(self):
        """Reset platform state."""
        self.total_transactions = 0
        self.total_revenue = 0.0
