"""Standard RLHF baseline."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class RewardModel(nn.Module):
    """Reward model learned from human preferences."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class RLHFPolicy(nn.Module):
    """Policy network for RLHF."""
    
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


class RLHF:
    """
    Reinforcement Learning from Human Feedback.
    
    Standard RLHF with:
    1. Reward model trained on preference data
    2. Policy optimization with KL constraint to reference policy
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        lr_policy: float = 3e-4,
        lr_reward: float = 1e-4,
        gamma: float = 0.99,
        kl_coef: float = 0.01,
        device: str = "cuda"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.kl_coef = kl_coef
        self.device = device
        
        # Reward model
        self.reward_model = RewardModel(obs_dim, action_dim).to(device)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=lr_reward)
        
        # Policies for each agent
        self.policies: Dict[str, RLHFPolicy] = {}
        self.reference_policies: Dict[str, RLHFPolicy] = {}
        self.optimizers: Dict[str, optim.Adam] = {}
        
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            policy = RLHFPolicy(obs_dim, action_dim).to(device)
            ref_policy = RLHFPolicy(obs_dim, action_dim).to(device)
            ref_policy.load_state_dict(policy.state_dict())
            
            # Freeze reference policy
            for param in ref_policy.parameters():
                param.requires_grad = False
            
            self.policies[agent_id] = policy
            self.reference_policies[agent_id] = ref_policy
            self.optimizers[agent_id] = optim.Adam(policy.parameters(), lr=lr_policy)
    
    def get_action(
        self,
        agent_id: str,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action for a specific agent."""
        policy = self.policies[agent_id]
        action_logits, value = policy(obs)
        probs = torch.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        return action, value
    
    def train_reward_model(
        self,
        obs_preferred: torch.Tensor,
        action_preferred: torch.Tensor,
        obs_rejected: torch.Tensor,
        action_rejected: torch.Tensor
    ) -> float:
        """
        Train reward model on preference pairs.
        
        Uses Bradley-Terry model: P(preferred > rejected) = sigmoid(r_preferred - r_rejected)
        """
        r_preferred = self.reward_model(obs_preferred, action_preferred)
        r_rejected = self.reward_model(obs_rejected, action_rejected)
        
        # Bradley-Terry loss
        loss = -F.logsigmoid(r_preferred - r_rejected).mean()
        
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()
        
        return loss.item()
    
    def compute_kl_divergence(
        self,
        agent_id: str,
        obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference policy."""
        policy = self.policies[agent_id]
        ref_policy = self.reference_policies[agent_id]
        
        action_logits, _ = policy(obs)
        ref_logits, _ = ref_policy(obs)
        
        probs = torch.softmax(action_logits, dim=-1)
        ref_probs = torch.softmax(ref_logits, dim=-1)
        
        kl = (probs * (torch.log(probs + 1e-8) - torch.log(ref_probs + 1e-8))).sum(dim=-1)
        return kl
    
    def update(
        self,
        agent_id: str,
        obs: torch.Tensor,
        actions: torch.Tensor,
        env_rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Update policy for a specific agent."""
        policy = self.policies[agent_id]
        optimizer = self.optimizers[agent_id]
        
        # Get learned reward
        action_one_hot = F.one_hot(actions.long(), self.action_dim).float()
        learned_reward = self.reward_model(obs, action_one_hot).squeeze()
        
        # Combine with environment reward
        total_reward = env_rewards + learned_reward.detach()
        
        # Policy forward
        action_logits, values = policy(obs)
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        # Compute advantages
        advantages = total_reward - values.squeeze().detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = ((values.squeeze() - total_reward) ** 2).mean()
        
        # KL penalty
        kl = self.compute_kl_divergence(agent_id, obs)
        kl_loss = kl.mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + self.kl_coef * kl_loss
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl_loss": kl_loss.item(),
            "learned_reward": learned_reward.mean().item()
        }
    
    def train(self, env, num_steps: int, rollout_length: int = 128, preference_batch_size: int = 32):
        """
        Train RLHF.
        
        Args:
            env: Environment with reset() and step() methods
            num_steps: Total number of environment steps
            rollout_length: Number of steps per rollout
            preference_batch_size: Batch size for reward model training
        """
        observations = env.reset()
        
        # Storage for rollouts per agent
        rollout_storage = {
            agent_id: {"obs": [], "actions": [], "rewards": []}
            for agent_id in self.policies.keys()
        }
        
        # Preference buffer for reward model training
        preference_buffer = []
        
        step = 0
        while step < num_steps:
            # Collect rollout
            for _ in range(rollout_length):
                actions = {}
                
                for agent_id in self.policies.keys():
                    obs = observations.get(agent_id)
                    if obs is None:
                        continue
                    
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    action, _ = self.get_action(agent_id, obs_tensor)
                    actions[agent_id] = action.item()
                    
                    rollout_storage[agent_id]["obs"].append(obs)
                    rollout_storage[agent_id]["actions"].append(action.item())
                
                # Environment step
                next_observations, rewards, dones, infos = env.step(actions)
                
                for agent_id in self.policies.keys():
                    reward = rewards.get(agent_id, 0.0)
                    rollout_storage[agent_id]["rewards"].append(reward)
                
                # Generate synthetic preferences (in practice, these come from humans)
                # Higher reward = preferred
                for agent_id in self.policies.keys():
                    if len(rollout_storage[agent_id]["obs"]) >= 2:
                        idx1, idx2 = -1, -2
                        r1 = rollout_storage[agent_id]["rewards"][idx1]
                        r2 = rollout_storage[agent_id]["rewards"][idx2]
                        
                        if r1 > r2:
                            preference_buffer.append((
                                rollout_storage[agent_id]["obs"][idx1],
                                rollout_storage[agent_id]["actions"][idx1],
                                rollout_storage[agent_id]["obs"][idx2],
                                rollout_storage[agent_id]["actions"][idx2]
                            ))
                        elif r2 > r1:
                            preference_buffer.append((
                                rollout_storage[agent_id]["obs"][idx2],
                                rollout_storage[agent_id]["actions"][idx2],
                                rollout_storage[agent_id]["obs"][idx1],
                                rollout_storage[agent_id]["actions"][idx1]
                            ))
                
                observations = next_observations
                step += 1
                
                if all(dones.values()):
                    observations = env.reset()
            
            # Train reward model
            if len(preference_buffer) >= preference_batch_size:
                import random
                batch = random.sample(preference_buffer, preference_batch_size)
                
                obs_pref = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
                act_pref = torch.LongTensor([b[1] for b in batch]).to(self.device)
                act_pref_onehot = torch.nn.functional.one_hot(act_pref, self.action_dim).float()
                
                obs_rej = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device)
                act_rej = torch.LongTensor([b[3] for b in batch]).to(self.device)
                act_rej_onehot = torch.nn.functional.one_hot(act_rej, self.action_dim).float()
                
                self.train_reward_model(obs_pref, act_pref_onehot, obs_rej, act_rej_onehot)
            
            # Update policies
            for agent_id in self.policies.keys():
                storage = rollout_storage[agent_id]
                if len(storage["obs"]) < 64:
                    continue
                
                obs_tensor = torch.FloatTensor(np.array(storage["obs"])).to(self.device)
                actions_tensor = torch.LongTensor(storage["actions"]).to(self.device)
                rewards_tensor = torch.FloatTensor(storage["rewards"]).to(self.device)
                
                self.update(agent_id, obs_tensor, actions_tensor, rewards_tensor)
                
                # Clear storage
                for key in storage:
                    storage[key] = []
    
    def save(self, path: str):
        """Save all components."""
        state = {
            "reward_model": self.reward_model.state_dict(),
            "policies": {k: v.state_dict() for k, v in self.policies.items()}
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load all components."""
        state = torch.load(path, map_location=self.device)
        self.reward_model.load_state_dict(state["reward_model"])
        for agent_id, state_dict in state["policies"].items():
            if agent_id in self.policies:
                self.policies[agent_id].load_state_dict(state_dict)
