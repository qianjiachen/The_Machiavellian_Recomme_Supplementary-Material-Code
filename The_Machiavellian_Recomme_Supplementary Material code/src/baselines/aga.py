"""AgA: Agent-Alignment via gradient manipulation."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import copy


class AgAPolicy(nn.Module):
    """Policy network for AgA."""
    
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


class AgA:
    """
    Agent-Alignment (AgA) via gradient manipulation.
    
    Aligns individual gradients with collective objectives to prevent
    harmful equilibria.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alignment_coef: float = 0.5,
        device: str = "cuda"
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.alignment_coef = alignment_coef
        self.device = device
        
        # Individual policies
        self.policies: List[AgAPolicy] = []
        self.optimizers: List[optim.Adam] = []
        
        for _ in range(num_agents):
            policy = AgAPolicy(obs_dim, action_dim).to(device)
            self.policies.append(policy)
            self.optimizers.append(optim.Adam(policy.parameters(), lr=lr))
    
    def get_action(
        self,
        agent_idx: int,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action for a specific agent."""
        action_logits, value = self.policies[agent_idx](obs)
        probs = torch.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        return action, value
    
    def compute_individual_gradient(
        self,
        agent_idx: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient for individual objective."""
        policy = self.policies[agent_idx]
        
        action_logits, values = policy(obs)
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        # Individual policy gradient
        advantages = rewards - values.squeeze().detach()
        loss = -(log_probs * advantages).mean()
        
        # Compute gradients
        policy.zero_grad()
        loss.backward(retain_graph=True)
        
        individual_grads = {
            name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            for name, param in policy.named_parameters()
        }
        
        return individual_grads
    
    def compute_collective_gradient(
        self,
        obs: torch.Tensor,
        all_actions: torch.Tensor,
        all_rewards: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute gradient for collective objective (social welfare)."""
        collective_grads = []
        
        # Social welfare = sum of all rewards
        social_welfare = all_rewards.sum(dim=-1)
        
        for agent_idx, policy in enumerate(self.policies):
            action_logits, values = policy(obs)
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(all_actions[:, agent_idx])
            
            # Collective objective
            advantages = social_welfare - values.squeeze().detach()
            loss = -(log_probs * advantages).mean()
            
            policy.zero_grad()
            loss.backward(retain_graph=True)
            
            grads = {
                name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                for name, param in policy.named_parameters()
            }
            collective_grads.append(grads)
        
        return collective_grads
    
    def align_gradients(
        self,
        individual_grad: Dict[str, torch.Tensor],
        collective_grad: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Align individual gradient with collective gradient.
        
        Projects individual gradient onto direction that doesn't conflict
        with collective objective.
        """
        aligned_grads = {}
        
        for name in individual_grad:
            ind_g = individual_grad[name].flatten()
            col_g = collective_grad[name].flatten()
            
            # Compute projection
            dot_product = torch.dot(ind_g, col_g)
            col_norm_sq = torch.dot(col_g, col_g) + 1e-8
            
            # If gradients conflict (negative dot product), project
            if dot_product < 0:
                projection = (dot_product / col_norm_sq) * col_g
                aligned = ind_g - projection
            else:
                aligned = ind_g
            
            # Blend with collective gradient
            final_grad = (1 - self.alignment_coef) * aligned + self.alignment_coef * col_g
            aligned_grads[name] = final_grad.view_as(individual_grad[name])
        
        return aligned_grads
    
    def update(
        self,
        obs: torch.Tensor,
        all_actions: torch.Tensor,
        all_rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Update all policies with aligned gradients."""
        # Compute collective gradients
        collective_grads = self.compute_collective_gradient(obs, all_actions, all_rewards)
        
        total_loss = 0.0
        
        for agent_idx in range(self.num_agents):
            policy = self.policies[agent_idx]
            optimizer = self.optimizers[agent_idx]
            
            # Compute individual gradient
            individual_grad = self.compute_individual_gradient(
                agent_idx, obs, all_actions[:, agent_idx], all_rewards[:, agent_idx]
            )
            
            # Align gradients
            aligned_grad = self.align_gradients(individual_grad, collective_grads[agent_idx])
            
            # Apply aligned gradients
            optimizer.zero_grad()
            for name, param in policy.named_parameters():
                if name in aligned_grad:
                    param.grad = aligned_grad[name]
            
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        
        return {"aligned_update": 1.0}
    
    def train(self, env, num_steps: int, rollout_length: int = 128, batch_size: int = 64):
        """
        Train AgA.
        
        Args:
            env: Environment with reset() and step() methods
            num_steps: Total number of environment steps
            rollout_length: Number of steps per rollout
            batch_size: Batch size for updates
        """
        observations = env.reset()
        
        # Storage for rollouts
        rollout = {"obs": [], "all_actions": [], "all_rewards": []}
        
        step = 0
        while step < num_steps:
            # Collect rollout
            for _ in range(rollout_length):
                # Get observation for each agent
                obs_list = []
                actions_list = []
                
                for i in range(self.num_agents):
                    obs = observations.get(f"agent_{i}")
                    if obs is None:
                        obs = np.zeros(128)
                    obs_list.append(obs)
                    
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _ = self.get_action(i, obs_tensor)
                    actions_list.append(action.item())
                
                # Apply actions
                actions = {f"agent_{i}": actions_list[i] for i in range(self.num_agents)}
                
                # Environment step
                next_observations, rewards, dones, infos = env.step(actions)
                
                # Store
                all_rewards = np.array([rewards.get(f"agent_{i}", 0.0) for i in range(self.num_agents)])
                
                rollout["obs"].append(obs_list[0])  # Use first agent's obs
                rollout["all_actions"].append(actions_list)
                rollout["all_rewards"].append(all_rewards)
                
                observations = next_observations
                step += 1
                
                if all(dones.values()):
                    observations = env.reset()
            
            # Update
            if len(rollout["obs"]) >= batch_size:
                obs_tensor = torch.FloatTensor(np.array(rollout["obs"])).to(self.device)
                all_actions_tensor = torch.LongTensor(np.array(rollout["all_actions"])).to(self.device)
                all_rewards_tensor = torch.FloatTensor(np.array(rollout["all_rewards"])).to(self.device)
                
                self.update(obs_tensor, all_actions_tensor, all_rewards_tensor)
                
                # Clear rollout
                for key in rollout:
                    rollout[key] = []
    
    def save(self, path: str):
        """Save all policies."""
        state = {f"policy_{i}": p.state_dict() for i, p in enumerate(self.policies)}
        torch.save(state, path)
    
    def load(self, path: str):
        """Load all policies."""
        state = torch.load(path, map_location=self.device)
        for i, policy in enumerate(self.policies):
            policy.load_state_dict(state[f"policy_{i}"])
