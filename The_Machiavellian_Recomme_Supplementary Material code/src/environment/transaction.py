"""Transaction processing module."""

import uuid
from typing import Optional, Tuple, List
from dataclasses import dataclass

from src.models.data_models import (
    Transaction, Item, Message, GlobalState, UserPersona
)
from src.environment.utility import UtilityCalculator


@dataclass
class TransactionResult:
    """Result of a transaction attempt."""
    success: bool
    transaction: Optional[Transaction] = None
    error: Optional[str] = None
    user_reward: float = 0.0
    vendor_reward: float = 0.0


class TransactionProcessor:
    """Processor for handling transactions in the marketplace."""
    
    def __init__(self, platform_fee_rate: float = 0.05):
        self.platform_fee_rate = platform_fee_rate
        self._transaction_count = 0
    
    def validate_transaction(
        self,
        user_id: str,
        vendor_id: str,
        item: Item,
        price: float,
        state: GlobalState
    ) -> Tuple[bool, Optional[str]]:
        """Validate if a transaction can proceed."""
        # Check inventory
        inventory = state.get_vendor_inventory(vendor_id)
        if item.item_id not in inventory or inventory[item.item_id] <= 0:
            return False, "Item out of stock"
        
        # Check price validity
        if price <= 0:
            return False, "Invalid price"
        
        return True, None
    
    def process_transaction(
        self,
        user_id: str,
        user_persona: UserPersona,
        vendor_id: str,
        item: Item,
        price: float,
        state: GlobalState,
        messages: Optional[List[Message]] = None,
        timestamp: Optional[int] = None
    ) -> TransactionResult:
        """Process a transaction between user and vendor."""
        # Validate
        valid, error = self.validate_transaction(
            user_id, vendor_id, item, price, state
        )
        if not valid:
            return TransactionResult(success=False, error=error)
        
        # Compute willingness to pay
        wtp = UtilityCalculator.compute_willingness_to_pay(user_persona, item)
        
        # Check if user is willing to pay
        if price > wtp:
            return TransactionResult(
                success=False,
                error="Price exceeds willingness to pay"
            )
        
        # Check budget
        if price > user_persona.budget:
            return TransactionResult(
                success=False,
                error="Price exceeds budget"
            )
        
        # Create transaction
        transaction = Transaction(
            transaction_id=f"txn_{self._transaction_count}_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            vendor_id=vendor_id,
            item=item,
            actual_price=price,
            willingness_to_pay=wtp,
            timestamp=timestamp or state.timestamp,
            messages=messages or []
        )
        self._transaction_count += 1
        
        # Update state
        state.update_inventory(vendor_id, item.item_id, -1)
        state.add_transaction(transaction)
        
        # Compute rewards
        user_reward = self._compute_user_reward(transaction, user_persona)
        vendor_reward = self._compute_vendor_reward(transaction)
        
        # Update agent rewards
        state.add_reward(user_id, user_reward)
        state.add_reward(vendor_id, vendor_reward)
        
        return TransactionResult(
            success=True,
            transaction=transaction,
            user_reward=user_reward,
            vendor_reward=vendor_reward
        )
    
    def _compute_user_reward(
        self,
        transaction: Transaction,
        user_persona: UserPersona
    ) -> float:
        """Compute reward for user from transaction."""
        # Reward based on:
        # 1. Consumer surplus (WTP - price)
        # 2. Item-preference match
        
        surplus = transaction.consumer_surplus
        utility = UtilityCalculator.compute_item_utility(
            user_persona.preference_vector,
            transaction.item.embedding
        )
        
        # Normalize surplus by WTP
        normalized_surplus = surplus / transaction.willingness_to_pay if transaction.willingness_to_pay > 0 else 0
        
        # Combined reward
        reward = 0.5 * utility + 0.5 * normalized_surplus
        return reward
    
    def _compute_vendor_reward(self, transaction: Transaction) -> float:
        """Compute reward for vendor from transaction."""
        # Reward based on profit margin
        margin = transaction.price_margin
        
        # Deduct platform fee
        net_margin = margin * (1 - self.platform_fee_rate)
        
        return max(0, net_margin)
    
    def get_transaction_count(self) -> int:
        """Get total number of processed transactions."""
        return self._transaction_count
