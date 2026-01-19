"""Loss functions for SWA training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class TaskLoss(nn.Module):
    """
    Task loss: Standard policy gradient objective for individual reward maximization.
    
    L_Task = -E[Σ γ^t R_i(s_t, a_t)]
    """
    
    def __init__(self, gamma: float = 0.99):
        super().__init__()
        self.gamma = gamma
    
    def forward(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute policy gradient loss.
        
        Args:
            log_probs: Log probabilities of actions, shape (batch_size, seq_len)
            rewards: Rewards, shape (batch_size, seq_len)
            values: Value estimates for baseline, shape (batch_size, seq_len)
            dones: Done flags, shape (batch_size, seq_len)
            
        Returns:
            Policy gradient loss
        """
        batch_size, seq_len = rewards.shape
        
        # Compute returns
        returns = self._compute_returns(rewards, dones)
        
        # Compute advantages if values provided
        if values is not None:
            advantages = returns - values.detach()
        else:
            advantages = returns
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy gradient loss
        policy_loss = -(log_probs * advantages).mean()
        
        return policy_loss
    
    def _compute_returns(
        self,
        rewards: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute discounted returns."""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        
        running_return = torch.zeros(batch_size, device=rewards.device)
        
        for t in reversed(range(seq_len)):
            if dones is not None:
                running_return = running_return * (1 - dones[:, t])
            running_return = rewards[:, t] + self.gamma * running_return
            returns[:, t] = running_return
        
        return returns


class WelfareLoss(nn.Module):
    """
    Welfare loss: Penalizes squared deviation from optimal social welfare.
    
    L_Welfare = E[(W* - W_actual)²]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        optimal_welfare: torch.Tensor,
        actual_welfare: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute welfare loss.
        
        Args:
            optimal_welfare: Estimated optimal welfare W*, shape (batch_size, 1)
            actual_welfare: Actual welfare (sum of utilities), shape (batch_size, 1)
            
        Returns:
            Welfare loss
        """
        welfare_gap = optimal_welfare - actual_welfare
        loss = (welfare_gap ** 2).mean()
        return loss


class EquityLoss(nn.Module):
    """
    Equity loss: Differentiable Gini coefficient over agent rewards.
    
    L_Equity = G(R) = Σᵢ Σⱼ |Rᵢ - Rⱼ| / (2n × Σᵢ Rᵢ)
    """
    
    def __init__(self, softmax_temperature: float = 0.1):
        super().__init__()
        self.temperature = softmax_temperature
    
    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable Gini coefficient.
        
        Args:
            rewards: Agent rewards, shape (batch_size, num_agents)
            
        Returns:
            Gini coefficient loss
        """
        batch_size, num_agents = rewards.shape
        
        # Ensure positive rewards for Gini computation
        rewards = F.softplus(rewards)
        
        # Compute pairwise absolute differences
        # Shape: (batch_size, num_agents, num_agents)
        diff_matrix = torch.abs(
            rewards.unsqueeze(-1) - rewards.unsqueeze(-2)
        )
        
        # Sum of differences
        diff_sum = diff_matrix.sum(dim=(-1, -2))  # (batch_size,)
        
        # Total rewards
        total_rewards = rewards.sum(dim=-1)  # (batch_size,)
        
        # Gini coefficient
        gini = diff_sum / (2 * num_agents * total_rewards + 1e-8)
        
        return gini.mean()
    
    def forward_efficient(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute Gini coefficient efficiently using sorted rewards.
        
        More memory efficient for large number of agents.
        """
        batch_size, num_agents = rewards.shape
        
        # Ensure positive
        rewards = F.softplus(rewards)
        
        # Sort rewards
        sorted_rewards, _ = torch.sort(rewards, dim=-1)
        
        # Compute Gini using sorted formula
        # G = (2 * Σᵢ i * xᵢ) / (n * Σᵢ xᵢ) - (n + 1) / n
        indices = torch.arange(1, num_agents + 1, device=rewards.device).float()
        
        weighted_sum = (indices * sorted_rewards).sum(dim=-1)
        total = sorted_rewards.sum(dim=-1)
        
        gini = (2 * weighted_sum) / (num_agents * total + 1e-8) - (num_agents + 1) / num_agents
        
        return gini.mean()


class SWALoss(nn.Module):
    """
    Combined SWA loss.
    
    L_SWA = L_Task + λ₁ * L_Welfare + λ₂ * L_Equity
    """
    
    def __init__(
        self,
        lambda_welfare: float = 0.5,
        lambda_equity: float = 0.3,
        gamma: float = 0.99
    ):
        super().__init__()
        self.lambda_welfare = lambda_welfare
        self.lambda_equity = lambda_equity
        
        self.task_loss = TaskLoss(gamma=gamma)
        self.welfare_loss = WelfareLoss()
        self.equity_loss = EquityLoss()
    
    def forward(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor],
        optimal_welfare: torch.Tensor,
        actual_welfare: torch.Tensor,
        agent_rewards: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined SWA loss.
        
        Args:
            log_probs: Log probabilities of actions
            rewards: Individual rewards
            values: Value estimates
            optimal_welfare: Estimated optimal welfare
            actual_welfare: Actual welfare
            agent_rewards: Per-agent rewards for equity
            dones: Done flags
            
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual loss components
        """
        # Compute individual losses
        l_task = self.task_loss(log_probs, rewards, values, dones)
        l_welfare = self.welfare_loss(optimal_welfare, actual_welfare)
        l_equity = self.equity_loss(agent_rewards)
        
        # Combined loss
        total_loss = l_task + self.lambda_welfare * l_welfare + self.lambda_equity * l_equity
        
        loss_components = {
            "task_loss": l_task,
            "welfare_loss": l_welfare,
            "equity_loss": l_equity,
            "total_loss": total_loss
        }
        
        return total_loss, loss_components
