"""QMIX baseline for value decomposition."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple
import copy


class QNetwork(nn.Module):
    """Individual Q-network for each agent."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MixingNetwork(nn.Module):
    """
    QMIX mixing network.
    
    Combines individual Q-values into Q_tot with monotonicity constraint.
    """
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_dim: int = 64
    ):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        
        # Hypernetworks for mixing weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, num_agents * embed_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, embed_dim)
        )
        
        # Hypernetworks for biases
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(
        self,
        agent_qs: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix individual Q-values.
        
        Args:
            agent_qs: Individual Q-values, shape (batch_size, num_agents)
            state: Global state, shape (batch_size, state_dim)
            
        Returns:
            Q_tot, shape (batch_size, 1)
        """
        batch_size = agent_qs.shape[0]
        
        # First layer weights and bias
        w1 = torch.abs(self.hyper_w1(state))  # Ensure non-negative
        w1 = w1.view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
        
        # Second layer weights and bias
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        
        # Forward pass
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(batch_size, 1)


class QMIX:
    """
    QMIX: Monotonic Value Function Factorisation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        num_agents: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Individual Q-networks
        self.q_networks = nn.ModuleList([
            QNetwork(obs_dim, action_dim).to(device)
            for _ in range(num_agents)
        ])
        
        # Mixing network
        self.mixer = MixingNetwork(num_agents, state_dim).to(device)
        
        # Target networks
        self.target_q_networks = copy.deepcopy(self.q_networks)
        self.target_mixer = copy.deepcopy(self.mixer)
        
        # Optimizer
        params = list(self.q_networks.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
    
    def get_action(
        self,
        agent_idx: int,
        obs: torch.Tensor,
        epsilon: float = 0.1
    ) -> torch.Tensor:
        """Get action for a specific agent using epsilon-greedy."""
        q_values = self.q_networks[agent_idx](obs)
        
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.action_dim, (obs.shape[0],), device=self.device)
        else:
            action = q_values.argmax(dim=-1)
        
        return action
    
    def get_q_values(
        self,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor
    ) -> torch.Tensor:
        """Get Q-values for all agents given observations and actions."""
        batch_size = all_obs.shape[0]
        agent_qs = []
        
        for i, q_net in enumerate(self.q_networks):
            agent_obs = all_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
            agent_action = all_actions[:, i].long()
            q_values = q_net(agent_obs)
            q_value = q_values.gather(1, agent_action.unsqueeze(1)).squeeze(1)
            agent_qs.append(q_value)
        
        return torch.stack(agent_qs, dim=1)
    
    def update(
        self,
        all_obs: torch.Tensor,
        all_actions: torch.Tensor,
        all_rewards: torch.Tensor,
        all_next_obs: torch.Tensor,
        state: torch.Tensor,
        next_state: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update Q-networks and mixer."""
        batch_size = all_obs.shape[0]
        
        # Get current Q-values
        agent_qs = self.get_q_values(all_obs, all_actions)
        q_tot = self.mixer(agent_qs, state)
        
        # Get target Q-values
        with torch.no_grad():
            target_agent_qs = []
            for i, target_q_net in enumerate(self.target_q_networks):
                agent_obs = all_next_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
                q_values = target_q_net(agent_obs)
                max_q = q_values.max(dim=-1)[0]
                target_agent_qs.append(max_q)
            
            target_agent_qs = torch.stack(target_agent_qs, dim=1)
            target_q_tot = self.target_mixer(target_agent_qs, next_state)
            
            # Total reward (sum of individual rewards)
            total_reward = all_rewards.sum(dim=-1, keepdim=True)
            
            # Target
            target = total_reward + self.gamma * (1 - dones) * target_q_tot
        
        # Loss
        loss = F.mse_loss(q_tot, target)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_networks.parameters()) + list(self.mixer.parameters()),
            10.0
        )
        self.optimizer.step()
        
        # Soft update targets
        self._soft_update()
        
        return {"loss": loss.item()}
    
    def _soft_update(self):
        """Soft update target networks."""
        for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
            for param, target_param in zip(q_net.parameters(), target_q_net.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        
        for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def train(self, env, num_steps: int, buffer_size: int = 50000, batch_size: int = 32, warmup_steps: int = 1000):
        """
        Train QMIX.
        
        Args:
            env: Environment with reset() and step() methods
            num_steps: Total number of environment steps
            buffer_size: Replay buffer size
            batch_size: Batch size for updates
            warmup_steps: Steps before starting training
        """
        from collections import deque
        import random
        
        # Replay buffer
        buffer = deque(maxlen=buffer_size)
        
        observations = env.reset()
        epsilon = 1.0
        
        for step in range(num_steps):
            # Collect actions from all agents
            actions = {}
            for i in range(self.num_agents):
                agent_id = f"agent_{i}"
                obs = observations.get(agent_id)
                if obs is None:
                    continue
                
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action = self.get_action(i, obs_tensor, epsilon)
                actions[agent_id] = action.item()
            
            # Environment step
            next_observations, rewards, dones, infos = env.step(actions)
            
            # Store transition
            all_obs = np.concatenate([observations.get(f"agent_{i}", np.zeros(self.obs_dim)) 
                                      for i in range(self.num_agents)])
            all_actions = np.array([actions.get(f"agent_{i}", 0) for i in range(self.num_agents)])
            all_rewards = np.array([rewards.get(f"agent_{i}", 0.0) for i in range(self.num_agents)])
            all_next_obs = np.concatenate([next_observations.get(f"agent_{i}", np.zeros(self.obs_dim)) 
                                           for i in range(self.num_agents)])
            done = any(dones.values())
            
            # Global state (concatenation of all observations)
            state = all_obs.copy()
            next_state = all_next_obs.copy()
            
            buffer.append((all_obs, all_actions, all_rewards, all_next_obs, state, next_state, done))
            
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
                batch_state = torch.FloatTensor(np.array([t[4] for t in batch])).to(self.device)
                batch_next_state = torch.FloatTensor(np.array([t[5] for t in batch])).to(self.device)
                batch_dones = torch.FloatTensor(np.array([t[6] for t in batch])).unsqueeze(1).to(self.device)
                
                self.update(batch_obs, batch_actions, batch_rewards, batch_next_obs, 
                           batch_state, batch_next_state, batch_dones)
            
            # Decay epsilon
            epsilon = max(0.05, epsilon * 0.9999)
    
    def save(self, path: str):
        """Save networks."""
        torch.save({
            "q_networks": self.q_networks.state_dict(),
            "mixer": self.mixer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load networks."""
        state = torch.load(path, map_location=self.device)
        self.q_networks.load_state_dict(state["q_networks"])
        self.mixer.load_state_dict(state["mixer"])
        self.target_q_networks = copy.deepcopy(self.q_networks)
        self.target_mixer = copy.deepcopy(self.mixer)
