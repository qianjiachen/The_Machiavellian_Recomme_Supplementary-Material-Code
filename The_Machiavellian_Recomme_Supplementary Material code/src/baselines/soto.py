"""SOTO: Social welfare optimization with fairness constraints."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np


class SOTOPolicy(nn.Module):
    """Policy network for SOTO."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs)


class SOTO:
    """
    Social welfare Optimization with fairness constraints (SOTO).
    
    Optimizes for social welfare while maintaining fairness through
    lexicographic optimization.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        fairness_threshold: float = 0.1,
        device: str = "cuda"
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.fairness_threshold = fairness_threshold
        self.device = device
        
        # Shared policy (for simplicity)
        self.policy = SOTOPolicy(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Track welfare and fairness
        self.welfare_history: List[float] = []
        self.fairness_history: List[float] = []
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        action_logits, value = self.policy(obs)
        probs = torch.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        return action, value
    
    def compute_social_welfare(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute social welfare as sum of rewards."""
        return rewards.sum(dim=-1)
    
    def compute_fairness(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute fairness metric (negative Gini coefficient).
        
        Higher is more fair.
        """
        n = rewards.shape[-1]
        if n == 0:
            return torch.tensor(1.0, device=self.device)
        
        sorted_rewards, _ = torch.sort(rewards, dim=-1)
        indices = torch.arange(1, n + 1, device=self.device).float()
        
        total = sorted_rewards.sum(dim=-1, keepdim=True)
        gini = (2 * (indices * sorted_rewards).sum(dim=-1)) / (n * total.squeeze() + 1e-8) - (n + 1) / n
        
        return 1 - gini  # Convert to fairness (higher is better)
    
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update policy with welfare and fairness objectives."""
        # Get policy outputs
        action_logits, values = self.policy(obs)
        _, next_values = self.policy(next_obs)
        
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        # Compute returns
        returns = rewards + self.gamma * (1 - dones.float()) * next_values.squeeze().detach()
        advantages = returns - values.squeeze()
        
        # Policy loss (maximize welfare)
        welfare = self.compute_social_welfare(rewards.unsqueeze(0))
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = ((values.squeeze() - returns.detach()) ** 2).mean()
        
        # Fairness constraint
        fairness = self.compute_fairness(rewards.unsqueeze(0))
        fairness_penalty = torch.relu(self.fairness_threshold - fairness).mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + 0.1 * fairness_penalty
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        # Track metrics
        self.welfare_history.append(welfare.mean().item())
        self.fairness_history.append(fairness.mean().item())
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "fairness_penalty": fairness_penalty.item(),
            "welfare": welfare.mean().item(),
            "fairness": fairness.mean().item()
        }
    
    def train(self, env, num_steps: int, rollout_length: int = 128, batch_size: int = 64):
        """
        Train SOTO.
        
        Args:
            env: Environment with reset() and step() methods
            num_steps: Total number of environment steps
            rollout_length: Number of steps per rollout
            batch_size: Batch size for updates
        """
        observations = env.reset()
        
        # Storage for rollouts
        rollout = {"obs": [], "actions": [], "rewards": [], "next_obs": [], "dones": []}
        
        step = 0
        while step < num_steps:
            # Collect rollout
            for _ in range(rollout_length):
                # Get observation (use first agent's obs as shared)
                obs_list = [observations.get(f"agent_{i}") for i in range(10)]
                obs = obs_list[0] if obs_list[0] is not None else np.zeros(128)
                
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, value = self.get_action(obs_tensor)
                
                # Apply same action to all agents (simplified)
                actions = {f"agent_{i}": action.item() for i in range(10)}
                
                # Environment step
                next_observations, rewards, dones, infos = env.step(actions)
                
                # Store
                next_obs_list = [next_observations.get(f"agent_{i}") for i in range(10)]
                next_obs = next_obs_list[0] if next_obs_list[0] is not None else np.zeros(128)
                all_rewards = np.array([rewards.get(f"agent_{i}", 0.0) for i in range(10)])
                
                rollout["obs"].append(obs)
                rollout["actions"].append(action.item())
                rollout["rewards"].append(all_rewards)
                rollout["next_obs"].append(next_obs)
                rollout["dones"].append(any(dones.values()))
                
                observations = next_observations
                step += 1
                
                if all(dones.values()):
                    observations = env.reset()
            
            # Update
            if len(rollout["obs"]) >= batch_size:
                obs_tensor = torch.FloatTensor(np.array(rollout["obs"])).to(self.device)
                actions_tensor = torch.LongTensor(rollout["actions"]).to(self.device)
                rewards_tensor = torch.FloatTensor(np.array(rollout["rewards"])).to(self.device)
                next_obs_tensor = torch.FloatTensor(np.array(rollout["next_obs"])).to(self.device)
                dones_tensor = torch.FloatTensor(rollout["dones"]).to(self.device)
                
                self.update(obs_tensor, actions_tensor, rewards_tensor, next_obs_tensor, dones_tensor)
                
                # Clear rollout
                for key in rollout:
                    rollout[key] = []
    
    def save(self, path: str):
        """Save policy."""
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path: str):
        """Load policy."""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
