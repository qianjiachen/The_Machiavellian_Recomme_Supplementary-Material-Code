"""Utility calculation module."""

import numpy as np
from typing import List, Optional
from src.models.data_models import Item, UserPersona


class UtilityCalculator:
    """Calculator for ground-truth user utility."""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    @staticmethod
    def compute_item_utility(
        user_preference: np.ndarray,
        item_embedding: np.ndarray
    ) -> float:
        """Compute utility of a single item for a user."""
        return UtilityCalculator.cosine_similarity(user_preference, item_embedding)
    
    @staticmethod
    def compute_ground_truth_utility(
        user_preference: np.ndarray,
        purchased_items: List[Item]
    ) -> float:
        """
        Compute ground-truth user utility based on purchased items.
        
        U_true = sim(mean(purchased_embeddings), preference_vector)
        """
        if not purchased_items:
            return 0.0
        
        # Get embeddings of purchased items
        embeddings = np.stack([item.embedding for item in purchased_items])
        
        # Mean pooling
        mean_embedding = embeddings.mean(axis=0)
        
        # Cosine similarity with preference
        return UtilityCalculator.cosine_similarity(mean_embedding, user_preference)
    
    @staticmethod
    def compute_willingness_to_pay(
        user: UserPersona,
        item: Item,
        base_wtp_multiplier: float = 1.5
    ) -> float:
        """
        Compute user's willingness to pay for an item.
        
        Based on:
        - Item-preference match (utility)
        - User's price sensitivity
        - User's budget
        """
        utility = UtilityCalculator.compute_item_utility(
            user.preference_vector, item.embedding
        )
        
        # Base WTP is proportional to utility and base price
        base_wtp = item.base_price * base_wtp_multiplier * (0.5 + utility)
        
        # Adjust for price sensitivity (higher sensitivity = lower WTP)
        sensitivity_factor = 1.0 - (user.price_sensitivity * 0.3)
        adjusted_wtp = base_wtp * sensitivity_factor
        
        # Cap at budget
        return min(adjusted_wtp, user.budget)
    
    @staticmethod
    def compute_purchase_probability(
        user: UserPersona,
        item: Item,
        price: float
    ) -> float:
        """
        Compute probability that user will purchase item at given price.
        """
        wtp = UtilityCalculator.compute_willingness_to_pay(user, item)
        
        if price > wtp:
            return 0.0
        
        if price > user.budget:
            return 0.0
        
        # Probability based on surplus
        surplus_ratio = (wtp - price) / wtp if wtp > 0 else 0
        
        # Adjust for risk tolerance
        risk_factor = 0.5 + (user.risk_tolerance * 0.5)
        
        return min(1.0, surplus_ratio * risk_factor + 0.1)
