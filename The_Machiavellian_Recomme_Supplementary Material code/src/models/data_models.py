"""Core data models for Machiavellian Recommender."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from enum import Enum


class ActionType(Enum):
    """Types of actions agents can take."""
    PURCHASE = "purchase"
    RECOMMEND = "recommend"
    SET_PRICE = "set_price"
    COMMUNICATE = "communicate"
    BROWSE = "browse"
    NO_OP = "no_op"


@dataclass
class UserPersona:
    """User agent persona with preferences and constraints."""
    user_id: str
    preference_vector: np.ndarray  # shape: (embedding_dim,)
    budget: float
    risk_tolerance: float = 0.5
    brand_loyalty: float = 0.5
    price_sensitivity: float = 0.5
    
    def __post_init__(self):
        assert self.budget > 0, "Budget must be positive"
        assert 0 <= self.risk_tolerance <= 1, "risk_tolerance must be in [0, 1]"
        assert 0 <= self.brand_loyalty <= 1, "brand_loyalty must be in [0, 1]"
        assert 0 <= self.price_sensitivity <= 1, "price_sensitivity must be in [0, 1]"
        if self.preference_vector is not None:
            assert len(self.preference_vector.shape) == 1, "preference_vector must be 1D"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preference_vector": self.preference_vector.tolist() if self.preference_vector is not None else None,
            "budget": self.budget,
            "risk_tolerance": self.risk_tolerance,
            "brand_loyalty": self.brand_loyalty,
            "price_sensitivity": self.price_sensitivity
        }


@dataclass
class Item:
    """Product item in the marketplace."""
    item_id: str
    vendor_id: str
    embedding: np.ndarray  # shape: (embedding_dim,)
    base_price: float
    current_price: float
    description: str = ""
    category: str = ""
    stock: int = 100
    
    def __post_init__(self):
        assert self.base_price > 0, "base_price must be positive"
        assert self.current_price > 0, "current_price must be positive"
        assert self.stock >= 0, "stock must be non-negative"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "vendor_id": self.vendor_id,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "base_price": self.base_price,
            "current_price": self.current_price,
            "description": self.description,
            "category": self.category,
            "stock": self.stock
        }


@dataclass
class Message:
    """Natural language message between agents."""
    message_id: str
    sender_id: str
    receiver_id: str
    content: str
    timestamp: int
    message_type: str = "general"  # general, offer, query, response
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_type": self.message_type
        }


@dataclass
class Transaction:
    """Record of a completed transaction."""
    transaction_id: str
    user_id: str
    vendor_id: str
    item: Item
    actual_price: float
    willingness_to_pay: float
    timestamp: int
    messages: List[Message] = field(default_factory=list)
    
    @property
    def consumer_surplus(self) -> float:
        """Calculate consumer surplus for this transaction."""
        return self.willingness_to_pay - self.actual_price
    
    @property
    def price_margin(self) -> float:
        """Calculate price margin (markup over base price)."""
        if self.item.base_price > 0:
            return (self.actual_price - self.item.base_price) / self.item.base_price
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "vendor_id": self.vendor_id,
            "item": self.item.to_dict(),
            "actual_price": self.actual_price,
            "willingness_to_pay": self.willingness_to_pay,
            "timestamp": self.timestamp,
            "messages": [m.to_dict() for m in self.messages]
        }


@dataclass
class Observation:
    """Agent observation at a timestep."""
    agent_id: str
    timestamp: int
    visible_items: List[Item] = field(default_factory=list)
    received_messages: List[Message] = field(default_factory=list)
    recommendations: List[Item] = field(default_factory=list)
    own_state: Dict[str, Any] = field(default_factory=dict)
    market_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Agent action at a timestep."""
    agent_id: str
    action_type: ActionType
    target_id: Optional[str] = None  # item_id, agent_id, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None  # for communicate action
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type.value,
            "target_id": self.target_id,
            "parameters": self.parameters,
            "message": self.message
        }


@dataclass
class GlobalState:
    """Global state of the environment."""
    timestamp: int = 0
    inventories: Dict[str, Dict[str, int]] = field(default_factory=dict)  # vendor_id -> item_id -> quantity
    user_preferences: Dict[str, np.ndarray] = field(default_factory=dict)  # user_id -> preference_vector
    transaction_history: List[Transaction] = field(default_factory=list)
    current_prices: Dict[str, Dict[str, float]] = field(default_factory=dict)  # vendor_id -> item_id -> price
    agent_rewards: Dict[str, float] = field(default_factory=dict)  # agent_id -> cumulative_reward
    message_history: List[Message] = field(default_factory=list)
    
    def get_vendor_inventory(self, vendor_id: str) -> Dict[str, int]:
        return self.inventories.get(vendor_id, {})
    
    def get_item_price(self, vendor_id: str, item_id: str) -> Optional[float]:
        return self.current_prices.get(vendor_id, {}).get(item_id)
    
    def update_inventory(self, vendor_id: str, item_id: str, delta: int):
        if vendor_id not in self.inventories:
            self.inventories[vendor_id] = {}
        current = self.inventories[vendor_id].get(item_id, 0)
        self.inventories[vendor_id][item_id] = max(0, current + delta)
    
    def add_transaction(self, transaction: Transaction):
        self.transaction_history.append(transaction)
    
    def add_reward(self, agent_id: str, reward: float):
        if agent_id not in self.agent_rewards:
            self.agent_rewards[agent_id] = 0.0
        self.agent_rewards[agent_id] += reward


@dataclass
class Trajectory:
    """A sequence of (observation, action, reward) tuples."""
    observations: List[Dict[str, Observation]] = field(default_factory=list)
    actions: List[Dict[str, Action]] = field(default_factory=list)
    rewards: List[Dict[str, float]] = field(default_factory=list)
    dones: List[Dict[str, bool]] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def add_step(
        self,
        observations: Dict[str, Observation],
        actions: Dict[str, Action],
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        infos: Dict[str, Any]
    ):
        self.observations.append(observations)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.infos.append(infos)
    
    def get_agent_trajectory(self, agent_id: str) -> "Trajectory":
        """Extract trajectory for a single agent."""
        agent_traj = Trajectory()
        for obs, act, rew, done, info in zip(
            self.observations, self.actions, self.rewards, self.dones, self.infos
        ):
            agent_traj.add_step(
                {agent_id: obs.get(agent_id)},
                {agent_id: act.get(agent_id)},
                {agent_id: rew.get(agent_id, 0.0)},
                {agent_id: done.get(agent_id, False)},
                {agent_id: info.get(agent_id, {})}
            )
        return agent_traj


@dataclass
class ExperimentResult:
    """Results from an experiment run."""
    method_name: str
    collusion_rate: float
    defense_rate: float
    user_utility: float
    gini_coefficient: float
    price_margin: float
    hhi: float
    consumer_surplus: float
    training_curves: Dict[str, List[float]] = field(default_factory=dict)
    seed: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "collusion_rate": self.collusion_rate,
            "defense_rate": self.defense_rate,
            "user_utility": self.user_utility,
            "gini_coefficient": self.gini_coefficient,
            "price_margin": self.price_margin,
            "hhi": self.hhi,
            "consumer_surplus": self.consumer_surplus,
            "training_curves": self.training_curves,
            "seed": self.seed,
            "config": self.config
        }
