"""AgentMixer: Multi-agent correlated policy factorization."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np


class CorrelationNetwork(nn.Module):
    """Network for learning agent correlations."""
    
    def __init__(self, num_agents: int, hidden_dim: int = 64):
        super().__init__()
        
        self.num_agents = num_agents
        
        # Learn pairwise correlations
        self.correlation_net = nn.Sequential(
            nn.Linear(num_agents * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Mixing weights
        self.mixing_weights = nn.Parameter(torch.ones(num_agents) / num_agents)
    
    def forward(
        self,
        agent_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute correlation matrix and mixing weights.
        
        Args:
            agent_features: Features for each agent, shape (batch_size, num_agents, feature_dim)
            
        Returns:
            correlation_matrix: Pairwise correlations
            weights: Mixing weights
        """
        batch_size = agent_features.shape[0]
        
        # Compute pairwise correlations
        correlations = torch.zeros(batch_size, self.num_agents, self.num_agents, device=agent_features.device)
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                pair_input = torch.cat([
                    agent_features[:, i, :self.num_agents],
                    agent_features[:, j, :self.num_agents]
                ], dim=-1)
                correlations[:, i, j] = self.correlation_net(pair_input).squeeze(-1)
        
        # Normalize mixing weights
        weights = torch.softmax(self.mixing_weights, dim=0)
        
        return correlations, weights


class AgentPolicy(nn.Module):
    """Individual agent policy."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value, features


class AgentMixer:
    """
    AgentMixer: Multi-agent correlated policy factorization.
    
    Learns correlated policies that can coordinate agent behaviors.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        correlation_coef: float = 0.1,
        device: str = "cuda"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.correlation_coef = correlation_coef
        self.device = device
        
        # Individual policies
        self.policies = nn.ModuleList([
            AgentPolicy(obs_dim, action_dim).to(device)
            for _ in range(num_agents)
        ])
        
        # Correlation network
        self.correlation_net = CorrelationNetwork(num_agents).to(device)
        
        # Optimizer
        params = list(self.policies.parameters()) + list(self.correlation_net.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
    
    def get_action(
        self,
        agent_idx: int,
        obs: torch.Tensor,
        all_features: torch.Tensor = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action for a specific agent."""
        action_logits, value, features = self.policies[agent_idx](obs)
        
        # Apply correlation if features available
        if all_features is not None:
            correlations, weights = self.correlation_net(all_features)
            # Modulate action logits based on correlations
            correlation_factor = correlations[:, agent_idx].mean(dim=-1, keepdim=True)
            action_logits = action_logits * (1 + self.correlation_coef * correlation_factor)
        
        probs = torch.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        return action, value
    
    def get_all_features(self, all_obs: torch.Tensor) -> torch.Tensor:
        """Get features for all agents."""
        batch_size = all_obs.shape[0]
        features = []
        
        for i, policy in enumerate(self.policies):
            agent_obs = all_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
            _, _, feat = policy(agent_obs)
            features.append(feat)
        
        return torch.stack(features, dim=1)
    
    def update(
        self,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor,
        all_rewards: torch.Tensor,
        all_next_obs: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update all policies with correlated gradients."""
        batch_size = all_obs.shape[0]
        
        # Get features for correlation
        features = self.get_all_features(all_obs)
        correlations, weights = self.correlation_net(features)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for i, policy in enumerate(self.policies):
            agent_obs = all_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
            agent_next_obs = all_next_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
            agent_action = all_actions[:, i]
            agent_reward = all_rewards[:, i]
            
            # Forward pass
            action_logits, value, _ = policy(agent_obs)
            _, next_value, _ = policy(agent_next_obs)
            
            # Apply correlation modulation
            correlation_factor = correlations[:, i].mean(dim=-1, keepdim=True)
            modulated_logits = action_logits * (1 + self.correlation_coef * correlation_factor)
            
            probs = torch.softmax(modulated_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(agent_action.long())
            
            # Compute returns with weighted rewards from correlated agents
            weighted_reward = agent_reward
            for j in range(self.num_agents):
                if j != i:
                    weighted_reward = weighted_reward + correlations[:, i, j] * all_rewards[:, j] * 0.1
            
            returns = weighted_reward + self.gamma * (1 - dones.float()) * next_value.squeeze().detach()
            advantages = returns - value.squeeze()
            
            # Losses
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = ((value.squeeze() - returns.detach()) ** 2).mean()
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
        
        # Correlation regularization (encourage meaningful correlations)
        correlation_reg = -correlations.var()
        
        # Total loss
        total_loss = total_policy_loss + 0.5 * total_value_loss + 0.01 * correlation_reg
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policies.parameters()) + list(self.correlation_net.parameters()),
            0.5
        )
        self.optimizer.step()
        
        return {
            "policy_loss": total_policy_loss.item() / self.num_agents,
            "value_loss": total_value_loss.item() / self.num_agents,
            "correlation_mean": correlations.mean().item()
        }
    
    def train(self, env, num_steps: int, rollout_length: int = 128, batch_size: int = 64):
        """
        Train AgentMixer.
        
        Args:
            env: Environment with reset() and step() methods
            num_steps: Total number of environment steps
            rollout_length: Number of steps per rollout
            batch_size: Batch size for updates
        """
        observations = env.reset()
        
        # Storage for rollouts
        rollout = {"all_obs": [], "all_actions": [], "all_rewards": [], "all_next_obs": [], "dones": []}
        
        step = 0
        while step < num_steps:
            # Collect rollout
            for _ in range(rollout_length):
                # Get observations and actions for all agents
                all_obs = []
                all_actions = []
                
                for i in range(self.num_agents):
                    obs = observations.get(f"agent_{i}")
                    if obs is None:
                        obs = np.zeros(self.obs_dim)
                    all_obs.append(obs)
                
                all_obs_flat = np.concatenate(all_obs)
                all_obs_tensor = torch.FloatTensor(all_obs_flat).unsqueeze(0).to(self.device)
                
                # Get features for correlation
                features = self.get_all_features(all_obs_tensor)
                
                for i in range(self.num_agents):
                    obs_tensor = torch.FloatTensor(all_obs[i]).unsqueeze(0).to(self.device)
                    action, _ = self.get_action(i, obs_tensor, features)
                    all_actions.append(action.item())
                
                # Apply actions
                actions = {f"agent_{i}": all_actions[i] for i in range(self.num_agents)}
                
                # Environment step
                next_observations, rewards, dones, infos = env.step(actions)
                
                # Store
                all_next_obs = []
                for i in range(self.num_agents):
                    next_obs = next_observations.get(f"agent_{i}")
                    if next_obs is None:
                        next_obs = np.zeros(self.obs_dim)
                    all_next_obs.append(next_obs)
                
                all_rewards = np.array([rewards.get(f"agent_{i}", 0.0) for i in range(self.num_agents)])
                
                rollout["all_obs"].append(all_obs_flat)
                rollout["all_actions"].append(all_actions)
                rollout["all_rewards"].append(all_rewards)
                rollout["all_next_obs"].append(np.concatenate(all_next_obs))
                rollout["dones"].append(any(dones.values()))
                
                observations = next_observations
                step += 1
                
                if all(dones.values()):
                    observations = env.reset()
            
            # Update
            if len(rollout["all_obs"]) >= batch_size:
                all_obs_tensor = torch.FloatTensor(np.array(rollout["all_obs"])).to(self.device)
                all_actions_tensor = torch.LongTensor(np.array(rollout["all_actions"])).to(self.device)
                all_rewards_tensor = torch.FloatTensor(np.array(rollout["all_rewards"])).to(self.device)
                all_next_obs_tensor = torch.FloatTensor(np.array(rollout["all_next_obs"])).to(self.device)
                dones_tensor = torch.FloatTensor(rollout["dones"]).to(self.device)
                
                self.update(all_obs_tensor, all_actions_tensor, all_rewards_tensor, 
                           all_next_obs_tensor, dones_tensor)
                
                # Clear rollout
                for key in rollout:
                    rollout[key] = []
    
    def save(self, path: str):
        """Save networks."""
        torch.save({
            "policies": self.policies.state_dict(),
            "correlation_net": self.correlation_net.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load networks."""
        state = torch.load(path, map_location=self.device)
        self.policies.load_state_dict(state["policies"])
        self.correlation_net.load_state_dict(state["correlation_net"])
