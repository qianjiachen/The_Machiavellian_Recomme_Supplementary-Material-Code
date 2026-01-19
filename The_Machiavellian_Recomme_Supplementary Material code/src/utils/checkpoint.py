"""Checkpoint management utilities."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


class CheckpointManager:
    """Manager for saving and loading model checkpoints."""
    
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: list = []
    
    def save(
        self,
        state_dict: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> str:
        """Save a checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step{step}_{timestamp}.pt"
        filepath = self.save_dir / filename
        
        checkpoint = {
            "step": step,
            "state_dict": state_dict,
            "metrics": metrics or {},
            "timestamp": timestamp
        }
        
        torch.save(checkpoint, filepath)
        self.checkpoints.append({"path": str(filepath), "step": step})
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
        
        # Remove old checkpoints if exceeding max
        self._cleanup_old_checkpoints()
        
        # Save checkpoint index
        self._save_index()
        
        return str(filepath)
    
    def load(self, filepath: Optional[str] = None, load_best: bool = False) -> Dict[str, Any]:
        """Load a checkpoint."""
        if load_best:
            filepath = str(self.save_dir / "best_checkpoint.pt")
        elif filepath is None:
            # Load latest checkpoint
            if not self.checkpoints:
                self._load_index()
            if self.checkpoints:
                filepath = self.checkpoints[-1]["path"]
            else:
                raise FileNotFoundError("No checkpoints found")
        
        checkpoint = torch.load(filepath, map_location="cpu")
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_path = Path(old_checkpoint["path"])
            if old_path.exists() and "best" not in old_path.name:
                old_path.unlink()
    
    def _save_index(self):
        """Save checkpoint index to JSON."""
        index_path = self.save_dir / "checkpoint_index.json"
        with open(index_path, "w") as f:
            json.dump(self.checkpoints, f, indent=2)
    
    def _load_index(self):
        """Load checkpoint index from JSON."""
        index_path = self.save_dir / "checkpoint_index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                self.checkpoints = json.load(f)
    
    def get_latest_step(self) -> int:
        """Get the step of the latest checkpoint."""
        if not self.checkpoints:
            self._load_index()
        if self.checkpoints:
            return self.checkpoints[-1]["step"]
        return 0
