"""
Configuration management for water leak localization.

Provides YAML-based configuration loading with dataclass defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class GraphConfig:
    """Graph generation parameters."""
    num_nodes: int = 200
    edge_probability: float = 0.08
    reservoir_nodes: int = 2
    min_conductance: float = 0.5
    max_conductance: float = 2.0


@dataclass
class SensorConfig:
    """Sensor configuration."""
    ratio: float = 0.2
    noise_std: float = 0.1


@dataclass
class DemandConfig:
    """Demand distribution parameters."""
    mean: float = 1.0
    std: float = 0.5
    reservoir_head: float = 50.0


@dataclass
class LeakConfig:
    """Leak injection parameters."""
    probability: float = 0.8
    magnitude_min: float = 0.5
    magnitude_max: float = 5.0


@dataclass
class DataConfig:
    """Dataset generation parameters."""
    train_samples: int = 2000
    val_samples: int = 400
    test_samples: int = 400
    output_dir: str = "data"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 10
    focal_gamma: float = 2.0
    no_leak_weight: float = 0.5


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    edge_dim: int = 2


@dataclass
class Stage2Config:
    """Stage 2 refinement parameters."""
    top_k: int = 5
    use_physics: bool = True


@dataclass
class EvalConfig:
    """Evaluation parameters."""
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class Config:
    """Main configuration container."""
    graph: GraphConfig = field(default_factory=GraphConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    leak: LeakConfig = field(default_factory=LeakConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "graph": self.graph.__dict__,
            "sensors": self.sensors.__dict__,
            "demand": self.demand.__dict__,
            "leak": self.leak.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "model": self.model.__dict__,
            "stage2": self.stage2.__dict__,
            "eval": self.eval.__dict__,
            "seed": self.seed,
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        Config object with loaded parameters.
    """
    config = Config()

    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        yaml_config = yaml.safe_load(f)

    if yaml_config is None:
        return config

    # Update graph config
    if "graph" in yaml_config:
        for key, value in yaml_config["graph"].items():
            if hasattr(config.graph, key):
                setattr(config.graph, key, value)

    # Update sensor config
    if "sensors" in yaml_config:
        for key, value in yaml_config["sensors"].items():
            if hasattr(config.sensors, key):
                setattr(config.sensors, key, value)

    # Update demand config
    if "demand" in yaml_config:
        for key, value in yaml_config["demand"].items():
            if hasattr(config.demand, key):
                setattr(config.demand, key, value)

    # Update leak config
    if "leak" in yaml_config:
        for key, value in yaml_config["leak"].items():
            if hasattr(config.leak, key):
                setattr(config.leak, key, value)

    # Update data config
    if "data" in yaml_config:
        for key, value in yaml_config["data"].items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)

    # Update training config
    if "training" in yaml_config:
        for key, value in yaml_config["training"].items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)

    # Update model config
    if "model" in yaml_config:
        for key, value in yaml_config["model"].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    # Update stage2 config
    if "stage2" in yaml_config:
        for key, value in yaml_config["stage2"].items():
            if hasattr(config.stage2, key):
                setattr(config.stage2, key, value)

    # Update eval config
    if "eval" in yaml_config:
        for key, value in yaml_config["eval"].items():
            if hasattr(config.eval, key):
                setattr(config.eval, key, value)

    # Update seed
    if "seed" in yaml_config:
        config.seed = yaml_config["seed"]

    return config


def save_config(config: Config, path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save.
        path: Output path for YAML file.
    """
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
