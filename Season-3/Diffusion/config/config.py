"""
Configuration for Flow Matching model training.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FlowMatchingConfig:
    """Configuration for Flow Matching model architecture."""

    # Image specifications
    image_size: int = 64
    image_channels: int = 3

    # Model architecture
    model_channels: int = 128  # Base channel count
    channel_mult: tuple = (1, 2, 2, 4)  # Channel multipliers for each resolution
    num_res_blocks: int = 2  # Number of residual blocks per resolution
    attention_resolutions: tuple = (16, 8)  # Resolutions to apply attention
    dropout: float = 0.1

    # Flow matching specific
    sigma_min: float = 1e-4  # Minimum noise level

    # Compute
    use_ema: bool = True  # Use Exponential Moving Average
    ema_decay: float = 0.9999


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Dataset
    dataset_name: str = 'celeba'
    data_dir: str = './data'

    # Training hyperparameters
    batch_size: int = 32  # Per GPU batch size
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Learning rate schedule
    lr_warmup_steps: int = 1000

    # Sampling during training
    sample_every: int = 1000  # Steps
    num_samples: int = 64
    sample_steps: int = 50  # ODE solver steps for sampling

    # Checkpointing
    save_every: int = 5000  # Steps
    checkpoint_dir: str = './checkpoints'

    # Logging
    log_every: int = 100  # Steps
    log_dir: str = './logs'

    # Multi-GPU
    num_gpus: int = 2
    distributed: bool = True

    # Other
    num_workers: int = 4
    seed: int = 42
    mixed_precision: bool = True  # Use automatic mixed precision
    compile_model: bool = False  # Use torch.compile (requires PyTorch 2.0+)

    # Resume training
    resume_from: str = None  # Path to checkpoint


def get_default_configs() -> tuple[FlowMatchingConfig, TrainingConfig]:
    """Get default configurations."""
    return FlowMatchingConfig(), TrainingConfig()
