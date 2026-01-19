"""MADDPG (Multi-Agent Deep Deterministic Policy Gradient) baseline."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
import copy


class Actor(nn.Module):
    """Actor network for MADDPG."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Centralized critic for MADDPG."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Input: all observations and all actions
        input_dim = obs_dim * num_agents + action_dim * num_agents
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)


class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient.
    
    Centralized training with decentralized execution.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        device: str = "cuda"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Create actors and critics for each agent
        self.actors: List[Actor] = []
        self.critics: List[Critic] = []
        self.target_actors: List[Actor] = []
        self.target_critics: List[Critic] = []
        self.actor_optimizers: List[optim.Adam] = []
        self.critic_optimizers: List[optim.Adam] = []
        
        for i in range(num_agents):
            # Actor
            actor = Actor(obs_dim, action_dim).to(device)
            target_actor = copy.deepcopy(actor)
            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr_actor))
            
            # Critic (centralized)
            critic = Critic(obs_dim, action_dim, num_agents).to(device)
            target_critic = copy.deepcopy(critic)
            self.critics.append(critic)
            self.target_critics.append(target_critic)
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr_critic))
    
    def get_action(
        self,
        agent_idx: int,
        obs: torch.Tensor,
        noise_scale: float = 0.1
    ) -> torch.Tensor:
        """Get action for a specific agent."""
        actor = self.actors[agent_idx]
        actor.eval()
        
        with torch.no_grad():
            action = actor(obs)
            if noise_scale > 0:
                noise = torch.randn_like(action) * noise_scale
                action = torch.clamp(action + noise, -1, 1)
        
        return action
    
    def update(
        self,
        agent_idx: int,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor,
        all_rewards: torch.Tensor,
        all_next_obs: torch.Tensor,
        all_dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update actor and critic for a specific agent."""
        actor = self.actors[agent_idx]
        critic = self.critics[agent_idx]
        target_actor = self.target_actors[agent_idx]
        target_critic = self.target_critics[agent_idx]
        actor_optimizer = self.actor_optimizers[agent_idx]
        critic_optimizer = self.critic_optimizers[agent_idx]
        
        # Get target actions for all agents
        target_actions = []
        for i, ta in enumerate(self.target_actors):
            agent_obs = all_next_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
            target_actions.append(ta(agent_obs))
        target_actions = torch.cat(target_actions, dim=-1)
        
        # Compute target Q value
        with torch.no_grad():
            target_q = target_critic(
                all_next_obs.view(all_next_obs.shape[0], -1),
                target_actions
            )
            target_value = all_rewards[:, agent_idx:agent_idx+1] + \
                          self.gamma * (1 - all_dones[:, agent_idx:agent_idx+1]) * target_q
        
        # Update critic
        current_q = critic(
            all_obs.view(all_obs.shape[0], -1),
            all_actions.view(all_actions.shape[0], -1)
        )
        critic_loss = F.mse_loss(current_q, target_value)
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # Update actor
        actor.train()
        
        # Get current actions with updated actor
        current_actions = []
        for i, a in enumerate(self.actors):
            agent_obs = all_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
            if i == agent_idx:
                current_actions.append(actor(agent_obs))
            else:
                current_actions.append(a(agent_obs).detach())
        current_actions = torch.cat(current_actions, dim=-1)
        
        actor_loss = -critic(
            all_obs.view(all_obs.shape[0], -1),
            current_actions
        ).mean()
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        # Soft update targets
        self._soft_update(actor, target_actor)
        self._soft_update(critic, target_critic)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def train(self, env, num_steps: int, buffer_size: int = 100000, batch_size: int = 256, warmup_steps: int = 1000):
        """
        Train all agents using MADDPG.
        
        Args:
            env: Environment with reset() and step() methods
            num_steps: Total number of environment steps
            buffer_size: Replay buffer size
            batch_size: Batch size for updates
            warmup_steps: Steps before starting training
        """
        from collections import deque
        import random
        
        # Simple replay buffer
        buffer = deque(maxlen=buffer_size)
        
        observations = env.reset()
        noise_scale = 0.3
        
        for step in range(num_steps):
            # Collect actions from all agents
            actions = {}
            for i in range(self.num_agents):
                agent_id = f"agent_{i}"
                obs = observations.get(agent_id)
                if obs is None:
                    continue
                
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.get_action(i, obs_tensor, noise_scale)
                actions[agent_id] = action.cpu().numpy().flatten()
            
            # Environment step
            next_observations, rewards, dones, infos = env.step(actions)
            
            # Store transition
            all_obs = np.concatenate([observations.get(f"agent_{i}", np.zeros(self.obs_dim)) 
                                      for i in range(self.num_agents)])
            all_actions = np.concatenate([actions.get(f"agent_{i}", np.zeros(self.action_dim)) 
                                          for i in range(self.num_agents)])
            all_rewards = np.array([rewards.get(f"agent_{i}", 0.0) for i in range(self.num_agents)])
            all_next_obs = np.concatenate([next_observations.get(f"agent_{i}", np.zeros(self.obs_dim)) 
                                           for i in range(self.num_agents)])
            all_dones = np.array([dones.get(f"agent_{i}", False) for i in range(self.num_agents)])
            
            buffer.append((all_obs, all_actions, all_rewards, all_next_obs, all_dones))
            
            observations = next_observations
            
            if all(dones.values()):
                observations = env.reset()
            
            # Training
            if step > warmup_steps and len(buffer) >= batch_size:
                # Sample batch
                batch = random.sample(buffer, batch_size)
                batch_obs = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
                batch_actions = torch.FloatTensor(np.array([t[1] for t in batch])).to(self.device)
                batch_rewards = torch.FloatTensor(np.array([t[2] for t in batch])).to(self.device)
                batch_next_obs = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
                batch_dones = torch.FloatTensor(np.array([t[4] for t in batch])).to(self.device)
                
                # Update each agent
                for i in range(self.num_agents):
                    self.update(i, batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)
            
            # Decay noise
            noise_scale = max(0.05, noise_scale * 0.9999)
    
    def save(self, path: str):
        """Save all networks."""
        state = {
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics]
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load all networks."""
        state = torch.load(path, map_location=self.device)
        for i, state_dict in enumerate(state["actors"]):
            self.actors[i].load_state_dict(state_dict)
            self.target_actors[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(state["critics"]):
            self.critics[i].load_state_dict(state_dict)
            self.target_critics[i].load_state_dict(state_dict)
