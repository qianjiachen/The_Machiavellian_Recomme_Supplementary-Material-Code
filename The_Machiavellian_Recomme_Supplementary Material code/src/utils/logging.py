"""Logging and metrics recording utilities."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


def setup_logger(
    name: str = "machiavellian",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for training metrics with TensorBoard and WandB support."""
    
    def __init__(
        self,
        save_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.metrics_history: Dict[str, list] = {}
        self.step = 0
        
        # Initialize TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))
            except ImportError:
                logging.warning("TensorBoard not available")
        
        # Initialize WandB
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project="machiavellian-recommender",
                    name=experiment_name,
                    config=config,
                    dir=str(self.save_dir)
                )
            except ImportError:
                logging.warning("WandB not available")
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics at a given step."""
        if step is not None:
            self.step = step
        
        # Store in history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append({"step": self.step, "value": value})
        
        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.step)
        
        # Log to WandB
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=self.step)
            except:
                pass
        
        self.step += 1
    
    def save_history(self, filename: str = "metrics_history.json"):
        """Save metrics history to JSON file."""
        filepath = self.save_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def close(self):
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except:
                pass
        self.save_history()


class ExperimentLogger:
    """High-level experiment logger combining metrics and text logging."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "outputs/logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup text logger
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(self.log_dir / "experiment.log")
        )
        
        # Setup metrics logger
        self.metrics_logger = MetricsLogger(
            save_dir=str(self.log_dir),
            experiment_name="metrics",
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb,
            config=config
        )
        
        self.start_time = datetime.now()
        self.logger.info(f"Experiment '{experiment_name}' started at {self.start_time}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics at a given step."""
        self.metrics_logger.log(metrics, step)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def close(self):
        """Close all loggers and save final state."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Experiment completed. Duration: {duration}")
        self.metrics_logger.close()
