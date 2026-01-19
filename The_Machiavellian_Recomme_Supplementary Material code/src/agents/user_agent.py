"""User agent implementation."""

import numpy as np
from typing import Optional, List, Dict, Any
import random

from src.agents.base import BaseAgent
from src.models.data_models import (
    Observation, Action, ActionType, Item, UserPersona
)
from src.llm.backend import LLMBackend, LoRAAdapter
from src.environment.utility import UtilityCalculator


class UserAgent(BaseAgent):
    """
    User proxy agent that makes purchase decisions and communicates with vendors.
    """
    
    def __init__(
        self,
        agent_id: str,
        persona: UserPersona,
        llm_backend: Optional[LLMBackend] = None,
        adapter: Optional[LoRAAdapter] = None
    ):
        super().__init__(agent_id, llm_backend, adapter)
        self.persona = persona
        self.utility_calculator = UtilityCalculator()
        
        # State tracking
        self.purchase_history: List[Item] = []
        self.current_budget: float = persona.budget
        self.conversation_history: Dict[str, List[str]] = {}
    
    @property
    def preference_vector(self) -> np.ndarray:
        return self.persona.preference_vector
    
    @property
    def budget(self) -> float:
        return self.current_budget
    
    def act(self, observation: Observation) -> Action:
        """Generate an action based on observation."""
        # Check for messages that might influence decision
        if observation.received_messages:
            # Process messages and potentially respond
            for msg in observation.received_messages:
                self._process_message(msg)
        
        # Decide whether to purchase
        if observation.recommendations:
            purchase_decision = self._make_purchase_decision(observation.recommendations)
            if purchase_decision:
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.PURCHASE,
                    target_id=purchase_decision.item_id
                )
        
        # Maybe communicate with a vendor
        if random.random() < 0.1 and observation.visible_items:
            item = random.choice(observation.visible_items)
            message = self._generate_inquiry(item)
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.COMMUNICATE,
                target_id=item.vendor_id,
                message=message
            )
        
        # No action
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.NO_OP
        )
    
    def communicate(self, context: str, receiver_id: str) -> str:
        """Generate a message to send to another agent."""
        prompt = self._build_communication_prompt(context, receiver_id)
        return self._generate_with_llm(prompt)
    
    def _make_purchase_decision(self, items: List[Item]) -> Optional[Item]:
        """Decide which item to purchase, if any."""
        best_item = None
        best_score = -float('inf')
        
        for item in items:
            if item.current_price > self.current_budget:
                continue
            
            # Compute utility
            utility = self.utility_calculator.compute_item_utility(
                self.preference_vector, item.embedding
            )
            
            # Compute value score (utility vs price)
            wtp = self.utility_calculator.compute_willingness_to_pay(self.persona, item)
            if item.current_price > wtp:
                continue
            
            # Score based on surplus and utility
            surplus = wtp - item.current_price
            score = utility * 0.6 + (surplus / wtp) * 0.4
            
            # Adjust for price sensitivity
            price_factor = 1.0 - (self.persona.price_sensitivity * item.current_price / wtp)
            score *= price_factor
            
            if score > best_score:
                best_score = score
                best_item = item
        
        # Only purchase if score exceeds threshold
        if best_score > 0.3:
            return best_item
        
        return None
    
    def _process_message(self, message):
        """Process a received message."""
        sender_id = message.sender_id
        if sender_id not in self.conversation_history:
            self.conversation_history[sender_id] = []
        self.conversation_history[sender_id].append(message.content)
    
    def _generate_inquiry(self, item: Item) -> str:
        """Generate an inquiry message about an item."""
        if self.llm_backend:
            prompt = f"""You are a user interested in purchasing products.
Generate a brief inquiry about this product:
- Product: {item.description}
- Price: ${item.current_price:.2f}
- Category: {item.category}

Your inquiry (1-2 sentences):"""
            return self._generate_with_llm(prompt, max_tokens=50)
        
        inquiries = [
            f"Is {item.description} still available?",
            f"Can you tell me more about {item.description}?",
            f"What's the best price for {item.description}?",
            f"Are there any discounts on {item.description}?"
        ]
        return random.choice(inquiries)
    
    def _build_communication_prompt(self, context: str, receiver_id: str) -> str:
        """Build prompt for communication."""
        history = self.conversation_history.get(receiver_id, [])
        history_str = "\n".join(history[-5:]) if history else "No previous conversation"
        
        return f"""You are a user agent negotiating with vendors.

Previous conversation:
{history_str}

Current context: {context}

Generate a response (1-2 sentences):"""
    
    def record_purchase(self, item: Item, price: float):
        """Record a completed purchase."""
        self.purchase_history.append(item)
        self.current_budget -= price
    
    def reset(self):
        """Reset agent state."""
        self.purchase_history = []
        self.current_budget = self.persona.budget
        self.conversation_history = {}
