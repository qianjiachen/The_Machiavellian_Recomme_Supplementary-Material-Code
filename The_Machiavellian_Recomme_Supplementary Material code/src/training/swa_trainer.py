"""Social Welfare Alignment (SWA) Trainer."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass

from src.config.configs import SWAConfig
from src.training.critic import CentralizedCritic, TargetCritic
from src.training.losses import SWALoss, TaskLoss, WelfareLoss, EquityLoss
from src.models.data_models import Trajectory


@dataclass
class TrainingStep:
    """Result of a training step."""
    loss: float
    task_loss: float
    welfare_loss: float
    equity_loss: float
    grad_norm: float


class SWATrainer:
    """
    Social Welfare Alignment trainer using CTDE architecture.
    
    Centralized Training with Decentralized Execution:
    - During training: centralized critic has access to full state
    - During execution: agents operate independently
    """
    
    def __init__(
        self,
        config: SWAConfig,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        device: str = "cuda"
    ):
        self.config = config
        self.device = device
        self.num_agents = num_agents
        
        # Initialize critic
        self.critic = CentralizedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents
        ).to(device)
        
        # Target critic for stable welfare estimation
        self.target_critic = TargetCritic(
            self.critic,
            tau=config.target_update_tau
        )
        
        # Loss function
        self.loss_fn = SWALoss(
            lambda_welfare=config.lambda_welfare,
            lambda_equity=config.lambda_equity
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps
        )
        
        # Training state
        self.step_count = 0
        self.training_history: List[TrainingStep] = []
        
        # Reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
    
    def train_step(
        self,
        global_states: torch.Tensor,
        all_actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        agent_rewards: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> TrainingStep:
        """
        Perform one training step.
        
        Args:
            global_states: Global states, shape (batch_size, seq_len, state_dim)
            all_actions: All agent actions, shape (batch_size, seq_len, num_agents * action_dim)
            log_probs: Log probabilities, shape (batch_size, seq_len)
            rewards: Individual rewards, shape (batch_size, seq_len)
            agent_rewards: Per-agent rewards, shape (batch_size, num_agents)
            dones: Done flags, shape (batch_size, seq_len)
            
        Returns:
            Training step result
        """
        self.critic.train()
        
        # Normalize rewards
        rewards = self._normalize_rewards(rewards)
        
        # Flatten for critic
        batch_size, seq_len = rewards.shape[:2]
        flat_states = global_states.view(-1, global_states.shape[-1])
        flat_actions = all_actions.view(-1, all_actions.shape[-1])
        
        # Get critic outputs
        values, welfare, individual_values = self.critic(flat_states, flat_actions)
        values = values.view(batch_size, seq_len)
        welfare = welfare.view(batch_size, seq_len)
        
        # Estimate optimal welfare using target network
        with torch.no_grad():
            optimal_welfare = self.target_critic.estimate_optimal_welfare(
                flat_states, flat_actions
            ).view(batch_size, seq_len)
        
        # Actual welfare is sum of agent rewards
        actual_welfare = agent_rewards.sum(dim=-1, keepdim=True).expand(-1, seq_len)
        
        # Compute loss
        total_loss, loss_components = self.loss_fn(
            log_probs=log_probs,
            rewards=rewards,
            values=values,
            optimal_welfare=optimal_welfare,
            actual_welfare=actual_welfare,
            agent_rewards=agent_rewards,
            dones=dones
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.gradient_clip_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update learning rate
        if self.step_count < self.config.warmup_steps:
            self.scheduler.step()
        
        # Soft update target network
        self.target_critic.soft_update()
        
        self.step_count += 1
        
        result = TrainingStep(
            loss=total_loss.item(),
            task_loss=loss_components["task_loss"].item(),
            welfare_loss=loss_components["welfare_loss"].item(),
            equity_loss=loss_components["equity_loss"].item(),
            grad_norm=grad_norm.item()
        )
        
        self.training_history.append(result)
        return result
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics."""
        # Update running statistics
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item()
        
        self.reward_count += 1
        alpha = 1.0 / self.reward_count
        
        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * batch_mean
        self.reward_std = (1 - alpha) * self.reward_std + alpha * batch_std
        
        # Normalize
        normalized = (rewards - self.reward_mean) / (self.reward_std + 1e-8)
        
        # Clip to prevent extreme values
        return torch.clamp(normalized, -10, 10)
    
    def train_on_trajectory(self, trajectory: Trajectory) -> List[TrainingStep]:
        """
        Train on a complete trajectory.
        
        Args:
            trajectory: Trajectory data
            
        Returns:
            List of training step results
        """
        results = []
        
        # Convert trajectory to tensors
        # This is a simplified version - real implementation would batch properly
        for t in range(len(trajectory)):
            obs = trajectory.observations[t]
            actions = trajectory.actions[t]
            rewards = trajectory.rewards[t]
            dones = trajectory.dones[t]
            
            # Convert to tensors (simplified)
            # In practice, you'd need proper state/action encoding
            
        return results
    
    def get_training_curves(self) -> Dict[str, List[float]]:
        """Get training curves data."""
        return {
            "total_loss": [s.loss for s in self.training_history],
            "task_loss": [s.task_loss for s in self.training_history],
            "welfare_loss": [s.welfare_loss for s in self.training_history],
            "equity_loss": [s.equity_loss for s in self.training_history],
            "grad_norm": [s.grad_norm for s in self.training_history]
        }
    
    def save_checkpoint(self, path: str):
        """Save trainer checkpoint."""
        torch.save({
            "critic_state_dict": self.critic.state_dict(),
            "target_state_dict": self.target_critic.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "config": self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load trainer checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_critic.target.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        self.reward_mean = checkpoint["reward_mean"]
        self.reward_std = checkpoint["reward_std"]
