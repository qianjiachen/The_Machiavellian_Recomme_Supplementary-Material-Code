"""MINE (Mutual Information Neural Estimation) for collusion detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class MINEEstimator(nn.Module):
    """
    Mutual Information Neural Estimator (MINE).
    
    Estimates mutual information I(X;Y) using the Donsker-Varadhan representation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Build network
        layers = []
        current_dim = input_dim * 2  # Concatenate X and Y
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Moving average for stability
        self.register_buffer('ma_et', torch.tensor(1.0))
        self.ma_rate = 0.01
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute T(x, y) for the MINE objective.
        
        Args:
            x: First variable, shape (batch_size, input_dim)
            y: Second variable, shape (batch_size, input_dim)
            
        Returns:
            T(x, y) values, shape (batch_size, 1)
        """
        xy = torch.cat([x, y], dim=-1)
        return self.network(xy)
    
    def estimate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_shuffle: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate mutual information I(X;Y).
        
        Args:
            x: First variable, shape (batch_size, input_dim)
            y: Second variable, shape (batch_size, input_dim)
            y_shuffle: Shuffled y for marginal (optional)
            
        Returns:
            mi_estimate: Estimated mutual information
            loss: MINE loss for training
        """
        # Joint distribution: (x, y)
        t_joint = self.forward(x, y)
        
        # Marginal distribution: (x, y_shuffled)
        if y_shuffle is None:
            # Shuffle y to break dependence
            idx = torch.randperm(y.size(0))
            y_shuffle = y[idx]
        
        t_marginal = self.forward(x, y_shuffle)
        
        # MINE objective (Donsker-Varadhan)
        # I(X;Y) >= E[T(x,y)] - log(E[exp(T(x,y'))])
        
        joint_term = t_joint.mean()
        
        # Use exponential moving average for stability
        exp_marginal = torch.exp(t_marginal)
        
        if self.training:
            # Update moving average
            with torch.no_grad():
                self.ma_et = (1 - self.ma_rate) * self.ma_et + self.ma_rate * exp_marginal.mean()
            
            # Biased gradient correction
            marginal_term = torch.log(exp_marginal.mean()) * exp_marginal.mean().detach() / self.ma_et
        else:
            marginal_term = torch.log(exp_marginal.mean())
        
        mi_estimate = joint_term - marginal_term
        
        # Loss is negative MI (we want to maximize MI estimate)
        loss = -mi_estimate
        
        return mi_estimate, loss
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Perform one training step.
        
        Args:
            x: First variable
            y: Second variable
            optimizer: Optimizer for the network
            
        Returns:
            Estimated MI value
        """
        self.train()
        optimizer.zero_grad()
        
        mi_estimate, loss = self.estimate(x, y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return mi_estimate.item()
    
    @torch.no_grad()
    def compute_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute MI estimate without training.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Estimated mutual information
        """
        self.eval()
        mi_estimate, _ = self.estimate(x, y)
        return mi_estimate.item()


class PolicyMIEstimator:
    """
    Estimator for mutual information between agent policies.
    """
    
    def __init__(
        self,
        policy_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        device: str = "cpu"
    ):
        self.device = device
        self.mine = MINEEstimator(policy_dim, hidden_dim, num_layers).to(device)
        self.optimizer = torch.optim.Adam(self.mine.parameters(), lr=1e-3)
        
        # Baseline MI from independent policies
        self.baseline_mi: Optional[float] = None
    
    def set_baseline(self, baseline_mi: float):
        """Set baseline MI from independently trained policies."""
        self.baseline_mi = baseline_mi
    
    def estimate_policy_mi(
        self,
        policy_i_outputs: np.ndarray,
        policy_j_outputs: np.ndarray,
        history_features: Optional[np.ndarray] = None,
        num_train_steps: int = 100
    ) -> float:
        """
        Estimate mutual information between two policies.
        
        Args:
            policy_i_outputs: Outputs from policy i, shape (n_samples, policy_dim)
            policy_j_outputs: Outputs from policy j, shape (n_samples, policy_dim)
            history_features: Optional conditioning features
            num_train_steps: Number of training steps for MINE
            
        Returns:
            Estimated MI between policies
        """
        # Convert to tensors
        x = torch.tensor(policy_i_outputs, dtype=torch.float32, device=self.device)
        y = torch.tensor(policy_j_outputs, dtype=torch.float32, device=self.device)
        
        # Train MINE
        for _ in range(num_train_steps):
            self.mine.train_step(x, y, self.optimizer)
        
        # Compute final estimate
        mi = self.mine.compute_mi(x, y)
        
        return mi
    
    def is_above_baseline(self, mi: float, threshold_factor: float = 1.5) -> bool:
        """
        Check if MI exceeds baseline by threshold factor.
        
        Args:
            mi: Estimated mutual information
            threshold_factor: Factor above baseline to consider collusive
            
        Returns:
            True if MI indicates potential collusion
        """
        if self.baseline_mi is None:
            return mi > 0.1  # Default threshold
        
        return mi > self.baseline_mi * threshold_factor
