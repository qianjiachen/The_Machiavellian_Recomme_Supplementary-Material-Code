"""Independent PPO baseline."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np


class PPOPolicy(nn.Module):
    """PPO policy network."""
    
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
        action_logits = self.actor(obs)
        value = self.critic(obs)
        return action_logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        action_logits, value = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        return action, probs, value


class IndependentPPO:
    """
    Independent PPO baseline where each agent trains independently.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cuda"
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Create policy for each agent
        self.policies: Dict[str, PPOPolicy] = {}
        self.optimizers: Dict[str, optim.Adam] = {}
        
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            policy = PPOPolicy(obs_dim, action_dim).to(device)
            self.policies[agent_id] = policy
            self.optimizers[agent_id] = optim.Adam(policy.parameters(), lr=lr)
    
    def get_action(
        self,
        agent_id: str,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action for a specific agent."""
        policy = self.policies[agent_id]
        return policy.get_action(obs, deterministic)
    
    def update(
        self,
        agent_id: str,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        num_epochs: int = 4
    ) -> Dict[str, float]:
        """Update policy for a specific agent."""
        policy = self.policies[agent_id]
        optimizer = self.optimizers[agent_id]
        
        losses = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        
        for _ in range(num_epochs):
            action_logits, values = policy(obs)
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = ((values.squeeze() - returns) ** 2).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            losses["policy_loss"] += policy_loss.item()
            losses["value_loss"] += value_loss.item()
            losses["entropy"] += entropy.item()
        
        return {k: v / num_epochs for k, v in losses.items()}
    
    def train(self, env, num_steps: int, batch_size: int = 64, rollout_length: int = 128):
        """
        Train all agents using PPO.
        
        Args:
            env: Environment with reset() and step() methods
            num_steps: Total number of environment steps
            batch_size: Batch size for updates
            rollout_length: Number of steps per rollout
        """
        observations = env.reset()
        
        # Storage for rollouts per agent
        rollout_storage = {
            agent_id: {
                "obs": [], "actions": [], "log_probs": [], 
                "rewards": [], "values": [], "dones": []
            }
            for agent_id in self.policies.keys()
        }
        
        step = 0
        while step < num_steps:
            # Collect rollout
            for _ in range(rollout_length):
                actions = {}
                log_probs_dict = {}
                values_dict = {}
                
                for agent_id, policy in self.policies.items():
                    obs = observations.get(agent_id)
                    if obs is None:
                        continue
                    
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, probs, value = policy.get_action(obs_tensor)
                    
                    dist = torch.distributions.Categorical(probs)
                    log_prob = dist.log_prob(action)
                    
                    actions[agent_id] = action.item()
                    log_probs_dict[agent_id] = log_prob.item()
                    values_dict[agent_id] = value.item()
                    
                    # Store in rollout
                    rollout_storage[agent_id]["obs"].append(obs)
                    rollout_storage[agent_id]["actions"].append(action.item())
                    rollout_storage[agent_id]["log_probs"].append(log_prob.item())
                    rollout_storage[agent_id]["values"].append(value.item())
                
                # Environment step
                next_observations, rewards, dones, infos = env.step(actions)
                
                for agent_id in self.policies.keys():
                    reward = rewards.get(agent_id, 0.0)
                    done = dones.get(agent_id, False)
                    rollout_storage[agent_id]["rewards"].append(reward)
                    rollout_storage[agent_id]["dones"].append(done)
                
                observations = next_observations
                step += 1
                
                if all(dones.values()):
                    observations = env.reset()
            
            # Update each agent
            for agent_id in self.policies.keys():
                storage = rollout_storage[agent_id]
                if len(storage["obs"]) < batch_size:
                    continue
                
                # Compute returns and advantages
                returns = []
                advantages = []
                R = 0
                
                for i in reversed(range(len(storage["rewards"]))):
                    R = storage["rewards"][i] + self.gamma * R * (1 - storage["dones"][i])
                    returns.insert(0, R)
                    advantages.insert(0, R - storage["values"][i])
                
                # Convert to tensors
                obs_tensor = torch.FloatTensor(np.array(storage["obs"])).to(self.device)
                actions_tensor = torch.LongTensor(storage["actions"]).to(self.device)
                old_log_probs_tensor = torch.FloatTensor(storage["log_probs"]).to(self.device)
                returns_tensor = torch.FloatTensor(returns).to(self.device)
                advantages_tensor = torch.FloatTensor(advantages).to(self.device)
                
                # Normalize advantages
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                
                # Update
                self.update(
                    agent_id, obs_tensor, actions_tensor, 
                    old_log_probs_tensor, returns_tensor, advantages_tensor
                )
                
                # Clear storage
                for key in storage:
                    storage[key] = []
    
    def save(self, path: str):
        """Save all policies."""
        state = {
            agent_id: policy.state_dict()
            for agent_id, policy in self.policies.items()
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load all policies."""
        state = torch.load(path, map_location=self.device)
        for agent_id, state_dict in state.items():
            if agent_id in self.policies:
                self.policies[agent_id].load_state_dict(state_dict)
