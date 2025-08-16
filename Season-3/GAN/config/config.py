import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Configuration for DCGAN+WGAN-GP model."""
    
    # Image parameters
    image_size: int = 64
    channels: int = 3
    
    # Model architecture
    latent_dim: int = 100
    gen_features: int = 64
    disc_features: int = 64
    
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate_d: float = 0.00005
    learning_rate_g: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    
    # WGAN-GP specific
    critic_iterations: int = 4
    gradient_penalty_lambda: float = 10.0
    
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    # Logging and saving
    save_interval: int = 10
    log_interval: int = 100
    sample_interval: int = 1000
    num_samples: int = 64
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Model initialization
    init_type: str = "normal"
    init_gain: float = 0.02


@dataclass
class DataConfig:
    """Configuration for dataset."""
    
    dataset_name: str = "celebrity_faces_hq"
    split: str = "train"
    streaming: bool = False
    cache_dir: str = "data/cache"
    
    # Preprocessing
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def get_config() -> ModelConfig:
    """Get the default configuration."""
    return ModelConfig()


def get_data_config() -> DataConfig:
    """Get the default data configuration."""
    return DataConfig()
