"""
Shared configuration utilities for energy projects.
"""
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class BaseConfig:
    """Base configuration class for energy projects."""
    
    project_name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.safe_dump(self.__dict__, f, default_flow_style=False)

@dataclass
class RLConfig:
    """Shared RL training configuration."""
    
    algorithm: str = "PPO"
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    verbose: int = 1
    tensorboard_log: Optional[str] = None