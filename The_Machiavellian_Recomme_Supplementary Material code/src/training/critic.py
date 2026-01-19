"""Centralized critic for CTDE architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import copy


class CentralizedCritic(nn.Module):
    """
    Centralized critic network for welfare estimation.
    
    Has access to full state and all agent actions during training.
    Used for computing welfare loss in SWA.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Action encoder (for all agents)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Combined processing
        combined_dim = hidden_dim * 2
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(combined_dim if len(layers) == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
        
        self.combined_layers = nn.Sequential(*layers)
        
        # Output heads
        self.value_head = nn.Linear(hidden_dim, 1)  # V(s)
        self.welfare_head = nn.Linear(hidden_dim, 1)  # W(s, a)
        self.individual_head = nn.Linear(hidden_dim, num_agents)  # Individual values
    
    def forward(
        self,
        global_state: torch.Tensor,
        all_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            global_state: Global state, shape (batch_size, state_dim)
            all_actions: All agent actions, shape (batch_size, num_agents * action_dim)
            
        Returns:
            value: State value V(s), shape (batch_size, 1)
            welfare: Social welfare W(s, a), shape (batch_size, 1)
            individual_values: Per-agent values, shape (batch_size, num_agents)
        """
        # Encode state and actions
        state_features = self.state_encoder(global_state)
        action_features = self.action_encoder(all_actions)
        
        # Combine
        combined = torch.cat([state_features, action_features], dim=-1)
        features = self.combined_layers(combined)
        
        # Compute outputs
        value = self.value_head(features)
        welfare = self.welfare_head(features)
        individual_values = self.individual_head(features)
        
        return value, welfare, individual_values
    
    def estimate_optimal_welfare(
        self,
        global_state: torch.Tensor,
        all_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate optimal social welfare W*.
        
        Args:
            global_state: Global state
            all_actions: All agent actions
            
        Returns:
            Estimated optimal welfare
        """
        _, welfare, individual_values = self.forward(global_state, all_actions)
        
        # W* is estimated as the sum of individual optimal values
        # This is an approximation - true W* would require optimization
        optimal_welfare = individual_values.sum(dim=-1, keepdim=True)
        
        return optimal_welfare
    
    def compute_welfare_gap(
        self,
        global_state: torch.Tensor,
        all_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gap between current and optimal welfare.
        
        Args:
            global_state: Global state
            all_actions: All agent actions
            
        Returns:
            Welfare gap (W* - W)
        """
        _, welfare, _ = self.forward(global_state, all_actions)
        optimal_welfare = self.estimate_optimal_welfare(global_state, all_actions)
        
        return optimal_welfare - welfare


class TargetCritic:
    """
    Target network wrapper for stable welfare estimation.
    """
    
    def __init__(self, critic: CentralizedCritic, tau: float = 0.005):
        self.critic = critic
        self.target = copy.deepcopy(critic)
        self.tau = tau
        
        # Freeze target parameters
        for param in self.target.parameters():
            param.requires_grad = False
    
    def soft_update(self):
        """
        Perform soft update of target network.
        
        target_param = τ * param + (1 - τ) * target_param
        """
        for param, target_param in zip(
            self.critic.parameters(),
            self.target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def hard_update(self):
        """Copy critic parameters to target."""
        self.target.load_state_dict(self.critic.state_dict())
    
    def estimate_optimal_welfare(
        self,
        global_state: torch.Tensor,
        all_actions: torch.Tensor
    ) -> torch.Tensor:
        """Estimate optimal welfare using target network."""
        with torch.no_grad():
            return self.target.estimate_optimal_welfare(global_state, all_actions)
