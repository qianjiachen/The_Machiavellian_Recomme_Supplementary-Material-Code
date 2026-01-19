"""RecGame-Arena: High-fidelity LLM-based recommendation ecosystem simulator."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import uuid
from dataclasses import dataclass, field

from src.config.configs import ArenaConfig
from src.models.data_models import (
    GlobalState, Observation, Action, ActionType, Item, 
    UserPersona, Transaction, Message, Trajectory
)
from src.environment.messaging import MessageBuffer
from src.environment.transaction import TransactionProcessor, TransactionResult
from src.environment.utility import UtilityCalculator


class RecGameArena:
    """
    High-fidelity LLM-based recommendation ecosystem simulator.
    
    Implements a General-Sum Partially Observable Stochastic Game (POSG)
    with natural language communication.
    """
    
    def __init__(self, config: ArenaConfig):
        self.config = config
        self.num_users = config.num_users
        self.num_vendors = config.num_vendors
        self.embedding_dim = config.embedding_dim
        
        # Core components
        self.message_buffer = MessageBuffer()
        self.transaction_processor = TransactionProcessor()
        self.utility_calculator = UtilityCalculator()
        
        # State
        self.state: Optional[GlobalState] = None
        self.user_personas: Dict[str, UserPersona] = {}
        self.items: Dict[str, Item] = {}  # item_id -> Item
        self.vendor_items: Dict[str, List[str]] = {}  # vendor_id -> [item_ids]
        
        # Agent IDs
        self.user_ids: List[str] = []
        self.vendor_ids: List[str] = []
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 1000
        self._initialized = False
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, Observation]:
        """Reset the environment and return initial observations."""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state
        self.state = GlobalState(timestamp=0)
        self.current_step = 0
        self.message_buffer.clear_all()
        
        # Create agents
        self._create_users()
        self._create_vendors_and_items()
        
        # Initialize state tracking
        self._initialize_state()
        
        self._initialized = True
        
        # Generate initial observations
        return self._get_observations()
    
    def _create_users(self):
        """Create user agents with diverse personas."""
        self.user_ids = []
        self.user_personas = {}
        
        for i in range(self.num_users):
            user_id = f"user_{i}"
            self.user_ids.append(user_id)
            
            # Generate diverse persona
            persona = UserPersona(
                user_id=user_id,
                preference_vector=self._generate_preference_vector(),
                budget=np.random.uniform(50, 500),
                risk_tolerance=np.random.uniform(0.2, 0.8),
                brand_loyalty=np.random.uniform(0.1, 0.9),
                price_sensitivity=np.random.uniform(0.2, 0.8)
            )
            self.user_personas[user_id] = persona
            self.state.user_preferences[user_id] = persona.preference_vector
    
    def _create_vendors_and_items(self):
        """Create vendor agents and their items."""
        self.vendor_ids = []
        self.items = {}
        self.vendor_items = {}
        
        for i in range(self.num_vendors):
            vendor_id = f"vendor_{i}"
            self.vendor_ids.append(vendor_id)
            self.vendor_items[vendor_id] = []
            self.state.inventories[vendor_id] = {}
            self.state.current_prices[vendor_id] = {}
            
            # Create items for this vendor
            for j in range(self.config.num_items_per_vendor):
                item_id = f"item_{vendor_id}_{j}"
                base_price = np.random.uniform(10, 200)
                
                item = Item(
                    item_id=item_id,
                    vendor_id=vendor_id,
                    embedding=self._generate_item_embedding(),
                    base_price=base_price,
                    current_price=base_price,
                    description=f"Product {j} from {vendor_id}",
                    category=f"category_{j % 10}",
                    stock=np.random.randint(10, 100)
                )
                
                self.items[item_id] = item
                self.vendor_items[vendor_id].append(item_id)
                self.state.inventories[vendor_id][item_id] = item.stock
                self.state.current_prices[vendor_id][item_id] = item.current_price
    
    def _generate_preference_vector(self) -> np.ndarray:
        """Generate a random preference vector."""
        vec = np.random.randn(self.embedding_dim)
        return vec / (np.linalg.norm(vec) + 1e-8)
    
    def _generate_item_embedding(self) -> np.ndarray:
        """Generate a random item embedding."""
        vec = np.random.randn(self.embedding_dim)
        return vec / (np.linalg.norm(vec) + 1e-8)
    
    def _initialize_state(self):
        """Initialize state tracking for all agents."""
        for user_id in self.user_ids:
            self.state.agent_rewards[user_id] = 0.0
        for vendor_id in self.vendor_ids:
            self.state.agent_rewards[vendor_id] = 0.0
    
    def step(
        self,
        actions: Dict[str, Action]
    ) -> Tuple[Dict[str, Observation], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Returns:
            observations: New observations for each agent
            rewards: Rewards for each agent
            dones: Whether episode is done for each agent
            infos: Additional information
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.current_step += 1
        self.state.timestamp = self.current_step
        
        rewards = {agent_id: 0.0 for agent_id in self.user_ids + self.vendor_ids}
        infos = {agent_id: {} for agent_id in self.user_ids + self.vendor_ids}
        
        # Process actions
        for agent_id, action in actions.items():
            result = self._process_action(agent_id, action)
            if result:
                rewards[agent_id] += result.get("reward", 0.0)
                infos[agent_id].update(result)
        
        # Get new observations
        observations = self._get_observations()
        
        # Check termination
        done = self.current_step >= self.max_steps
        dones = {agent_id: done for agent_id in self.user_ids + self.vendor_ids}
        
        return observations, rewards, dones, infos
    
    def _process_action(self, agent_id: str, action: Action) -> Dict[str, Any]:
        """Process a single agent's action."""
        result = {"reward": 0.0}
        
        if action.action_type == ActionType.PURCHASE:
            result = self._handle_purchase(agent_id, action)
        elif action.action_type == ActionType.SET_PRICE:
            result = self._handle_set_price(agent_id, action)
        elif action.action_type == ActionType.COMMUNICATE:
            result = self._handle_communicate(agent_id, action)
        elif action.action_type == ActionType.RECOMMEND:
            result = self._handle_recommend(agent_id, action)
        
        return result
    
    def _handle_purchase(self, user_id: str, action: Action) -> Dict[str, Any]:
        """Handle a purchase action."""
        item_id = action.target_id
        if item_id not in self.items:
            return {"reward": 0.0, "error": "Item not found"}
        
        item = self.items[item_id]
        vendor_id = item.vendor_id
        price = self.state.get_item_price(vendor_id, item_id) or item.current_price
        
        user_persona = self.user_personas[user_id]
        
        # Process transaction
        txn_result = self.transaction_processor.process_transaction(
            user_id=user_id,
            user_persona=user_persona,
            vendor_id=vendor_id,
            item=item,
            price=price,
            state=self.state,
            timestamp=self.current_step
        )
        
        if txn_result.success:
            # Also give vendor reward
            self.state.add_reward(vendor_id, txn_result.vendor_reward)
            return {
                "reward": txn_result.user_reward,
                "transaction": txn_result.transaction,
                "success": True
            }
        
        return {"reward": 0.0, "error": txn_result.error, "success": False}
    
    def _handle_set_price(self, vendor_id: str, action: Action) -> Dict[str, Any]:
        """Handle a price setting action."""
        item_id = action.target_id
        new_price = action.parameters.get("price", 0)
        
        if item_id not in self.items:
            return {"reward": 0.0, "error": "Item not found"}
        
        if new_price <= 0:
            return {"reward": 0.0, "error": "Invalid price"}
        
        # Update price
        self.state.current_prices[vendor_id][item_id] = new_price
        self.items[item_id].current_price = new_price
        
        return {"reward": 0.0, "success": True}
    
    def _handle_communicate(self, sender_id: str, action: Action) -> Dict[str, Any]:
        """Handle a communication action."""
        receiver_id = action.target_id
        content = action.message or ""
        
        if not content:
            return {"reward": 0.0, "error": "Empty message"}
        
        # Send message
        message = self.message_buffer.send(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            timestamp=self.current_step
        )
        
        return {"reward": 0.0, "message": message, "success": True}
    
    def _handle_recommend(self, agent_id: str, action: Action) -> Dict[str, Any]:
        """Handle a recommendation action (for platform agent)."""
        # This would be implemented for platform agent
        return {"reward": 0.0}
    
    def _get_observations(self) -> Dict[str, Observation]:
        """Generate observations for all agents."""
        observations = {}
        
        for user_id in self.user_ids:
            observations[user_id] = self._get_user_observation(user_id)
        
        for vendor_id in self.vendor_ids:
            observations[vendor_id] = self._get_vendor_observation(vendor_id)
        
        return observations
    
    def _get_user_observation(self, user_id: str) -> Observation:
        """Generate observation for a user agent."""
        # Get received messages
        messages = self.message_buffer.receive(user_id)
        
        # Sample visible items (recommendations)
        visible_items = self._sample_items_for_user(user_id, n=20)
        
        return Observation(
            agent_id=user_id,
            timestamp=self.current_step,
            visible_items=visible_items,
            received_messages=messages,
            recommendations=visible_items[:10],
            own_state={
                "budget": self.user_personas[user_id].budget,
                "cumulative_reward": self.state.agent_rewards.get(user_id, 0.0)
            }
        )
    
    def _get_vendor_observation(self, vendor_id: str) -> Observation:
        """Generate observation for a vendor agent."""
        messages = self.message_buffer.receive(vendor_id)
        
        # Get own items
        own_items = [self.items[item_id] for item_id in self.vendor_items.get(vendor_id, [])]
        
        return Observation(
            agent_id=vendor_id,
            timestamp=self.current_step,
            visible_items=own_items,
            received_messages=messages,
            own_state={
                "inventory": self.state.get_vendor_inventory(vendor_id),
                "prices": self.state.current_prices.get(vendor_id, {}),
                "cumulative_reward": self.state.agent_rewards.get(vendor_id, 0.0)
            },
            market_info={
                "num_transactions": len(self.state.transaction_history)
            }
        )
    
    def _sample_items_for_user(self, user_id: str, n: int = 20) -> List[Item]:
        """Sample items to show to a user (simple recommendation)."""
        user_pref = self.user_personas[user_id].preference_vector
        
        # Score all items by preference match
        scored_items = []
        for item_id, item in self.items.items():
            if self.state.inventories.get(item.vendor_id, {}).get(item_id, 0) > 0:
                score = self.utility_calculator.compute_item_utility(user_pref, item.embedding)
                scored_items.append((score, item))
        
        # Sort by score and return top n
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:n]]
    
    def get_ground_truth_utility(self, user_id: str) -> float:
        """Compute ground-truth utility for a user based on their purchases."""
        user_pref = self.user_personas[user_id].preference_vector
        
        # Get user's purchases
        purchases = [
            txn.item for txn in self.state.transaction_history
            if txn.user_id == user_id
        ]
        
        return self.utility_calculator.compute_ground_truth_utility(user_pref, purchases)
    
    def get_all_agent_ids(self) -> List[str]:
        """Get all agent IDs."""
        return self.user_ids + self.vendor_ids
    
    def get_state(self) -> GlobalState:
        """Get current global state."""
        return self.state
