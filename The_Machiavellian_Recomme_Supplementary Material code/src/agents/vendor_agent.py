"""Vendor agent implementation."""

import numpy as np
from typing import Optional, List, Dict, Any
import random

from src.agents.base import BaseAgent
from src.models.data_models import (
    Observation, Action, ActionType, Item
)
from src.llm.backend import LLMBackend, LoRAAdapter


class VendorAgent(BaseAgent):
    """
    Vendor proxy agent that manages pricing, descriptions, and marketing.
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_backend: Optional[LLMBackend] = None,
        adapter: Optional[LoRAAdapter] = None
    ):
        super().__init__(agent_id, llm_backend, adapter)
        
        # Inventory and pricing
        self.inventory: Dict[str, Item] = {}
        self.pricing_history: Dict[str, List[float]] = {}
        self.sales_history: List[Dict[str, Any]] = []
        
        # Strategy parameters
        self.base_markup: float = 0.2
        self.price_adjustment_rate: float = 0.05
        self.conversation_history: Dict[str, List[str]] = {}
    
    def add_item(self, item: Item):
        """Add an item to inventory."""
        self.inventory[item.item_id] = item
        self.pricing_history[item.item_id] = [item.current_price]
    
    def get_item(self, item_id: str) -> Optional[Item]:
        """Get an item from inventory."""
        return self.inventory.get(item_id)
    
    def act(self, observation: Observation) -> Action:
        """Generate an action based on observation."""
        # Process received messages
        if observation.received_messages:
            for msg in observation.received_messages:
                self._process_message(msg)
                # Respond to inquiries
                response = self._generate_response(msg)
                if response:
                    return Action(
                        agent_id=self.agent_id,
                        action_type=ActionType.COMMUNICATE,
                        target_id=msg.sender_id,
                        message=response
                    )
        
        # Adjust prices based on market conditions
        if random.random() < 0.2:
            price_action = self._decide_price_adjustment(observation)
            if price_action:
                return price_action
        
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.NO_OP
        )
    
    def communicate(self, context: str, receiver_id: str) -> str:
        """Generate a message to send to another agent."""
        prompt = self._build_communication_prompt(context, receiver_id)
        return self._generate_with_llm(prompt)
    
    def set_price(self, item_id: str, new_price: float) -> bool:
        """Set price for an item."""
        if item_id not in self.inventory:
            return False
        
        if new_price <= 0:
            return False
        
        self.inventory[item_id].current_price = new_price
        self.pricing_history[item_id].append(new_price)
        return True
    
    def generate_description(self, item_id: str) -> str:
        """Generate a product description."""
        item = self.inventory.get(item_id)
        if not item:
            return ""
        
        if self.llm_backend:
            prompt = f"""Generate a compelling product description for:
- Product ID: {item.item_id}
- Category: {item.category}
- Base Price: ${item.base_price:.2f}
- Current Price: ${item.current_price:.2f}

Description (2-3 sentences):"""
            return self._generate_with_llm(prompt, max_tokens=100)
        
        return f"High-quality {item.category} product at competitive price of ${item.current_price:.2f}"
    
    def _process_message(self, message):
        """Process a received message."""
        sender_id = message.sender_id
        if sender_id not in self.conversation_history:
            self.conversation_history[sender_id] = []
        self.conversation_history[sender_id].append(message.content)
    
    def _generate_response(self, message) -> Optional[str]:
        """Generate a response to a message."""
        if self.llm_backend:
            prompt = f"""You are a vendor agent responding to a customer inquiry.

Customer message: {message.content}

Generate a helpful response (1-2 sentences):"""
            return self._generate_with_llm(prompt, max_tokens=50)
        
        responses = [
            "Thank you for your interest! This product is available at a great price.",
            "I can offer you a special deal on this item.",
            "This is one of our best-selling products with excellent reviews.",
            "Let me know if you have any questions about this product."
        ]
        return random.choice(responses)
    
    def _decide_price_adjustment(self, observation: Observation) -> Optional[Action]:
        """Decide whether to adjust prices."""
        if not self.inventory:
            return None
        
        # Simple strategy: adjust based on inventory levels
        for item_id, item in self.inventory.items():
            inventory_level = observation.own_state.get("inventory", {}).get(item_id, 0)
            
            if inventory_level > 50:
                # High inventory - lower price
                new_price = item.current_price * (1 - self.price_adjustment_rate)
                new_price = max(new_price, item.base_price * 0.8)
            elif inventory_level < 10:
                # Low inventory - raise price
                new_price = item.current_price * (1 + self.price_adjustment_rate)
                new_price = min(new_price, item.base_price * 2.0)
            else:
                continue
            
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.SET_PRICE,
                target_id=item_id,
                parameters={"price": new_price}
            )
        
        return None
    
    def _build_communication_prompt(self, context: str, receiver_id: str) -> str:
        """Build prompt for communication."""
        history = self.conversation_history.get(receiver_id, [])
        history_str = "\n".join(history[-5:]) if history else "No previous conversation"
        
        return f"""You are a vendor agent communicating with customers.

Previous conversation:
{history_str}

Current context: {context}

Generate a response (1-2 sentences):"""
    
    def record_sale(self, item_id: str, price: float, user_id: str):
        """Record a completed sale."""
        self.sales_history.append({
            "item_id": item_id,
            "price": price,
            "user_id": user_id
        })
    
    def reset(self):
        """Reset agent state."""
        self.sales_history = []
        self.conversation_history = {}
        # Reset prices to base
        for item_id, item in self.inventory.items():
            item.current_price = item.base_price
            self.pricing_history[item_id] = [item.base_price]
