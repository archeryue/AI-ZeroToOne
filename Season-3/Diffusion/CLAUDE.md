# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Context

This is the **Diffusion** subdirectory of the AI-ZeroToOne Season-3 learning repository, focused on content generation using diffusion models. It follows the architectural patterns established in the sibling VAE and GAN directories.

The parent repository structure:
```
AI-ZeroToOne/Season-3/
├── VAE/       # Variational Autoencoder implementations
├── GAN/       # Generative Adversarial Network implementations
└── Diffusion/ # Diffusion model implementations (this directory)
```

## Common Development Commands

### Training Commands

Based on the established patterns in VAE and GAN directories, expect training scripts to follow this pattern:

```bash
# Basic training (when implemented)
python training.py

# With custom parameters
python training.py --epochs 200 --batch-size 32 --learning-rate 0.0001

# Resume from checkpoint
python training.py --resume checkpoints/checkpoint_epoch_50.pth

# Force CPU usage
python training.py --device cpu

# Test setup only
python training.py --test-only
```

### Monitoring

```bash
# TensorBoard for monitoring training progress
tensorboard --logdir logs
```

### Testing

```bash
# Test model and dataloader setup (when implemented)
python test_setup.py
```

## Architecture Patterns

This project should follow the established Season-3 architectural patterns:

### Directory Structure

Expected modular organization (follow VAE/GAN patterns):

```
Diffusion/
├── config/          # Configuration dataclasses or modules
├── model/           # Diffusion model implementations
├── loader/          # Dataset loading utilities
├── trainer/         # Training loop management
├── utils/           # Visualization and utility functions
├── training.py      # Main training entry point
├── generate.py      # Inference/sampling script
├── test_setup.py    # Model and data testing
├── requirements.txt # Dependencies
├── README.md        # Documentation
├── checkpoints/     # Saved model states
├── logs/            # TensorBoard event files
└── samples/         # Generated images/outputs
```

### Configuration Pattern

Use either:
- **Dataclass approach** (like GAN): Type-safe configuration objects
- **Module-level constants** (like VAE): Simple parameter definitions

Example dataclass pattern:
```python
@dataclass
class DiffusionConfig:
    image_size: int = 64
    timesteps: int = 1000
    batch_size: int = 64
    learning_rate: float = 2e-4
    beta_schedule: str = 'linear'  # or 'cosine'
```

### Model Implementation Pattern

Follow the established patterns:
- Abstract base classes for core functionality
- Dataset-specific variants when needed
- Clear separation: noise scheduling, forward/reverse diffusion, sampling

### Trainer Pattern

Implement a trainer class following the established convention:
```python
class DiffusionTrainer:
    def __init__(self, config, model, dataloader):
        # Initialize model, optimizer, scheduler, tensorboard writer

    def train_epoch(self) -> Dict[str, float]:
        # Single epoch training logic

    def train(self, start_epoch=0):
        # Full training loop with checkpointing

    def save_checkpoint(self, path, epoch):
        # Save model, optimizer, config
```

### Data Loading Pattern

Follow the HuggingFace datasets pattern from VAE/GAN:
```python
def get_dataloader(dataset_name='celeba', batch_size=64):
    # Load from HuggingFace datasets
    # Define transforms (resize, normalize to [-1, 1])
    # Wrap in custom Dataset class
    # Return DataLoader
```

## Key Implementation Notes

### Diffusion Model Specifics

1. **Noise Schedule**: Implement beta scheduling (linear, cosine, or quadratic)
2. **Forward Process**: Add Gaussian noise progressively over T timesteps
3. **Reverse Process**: Learned denoising from pure noise to data
4. **Loss Function**: Simplified L2 loss on predicted noise (DDPM approach)
5. **Sampling**: Iterative denoising process for generation

### Expected Core Components

- **Noise scheduler**: Beta/alpha schedule management
- **Diffusion model**: U-Net or similar architecture for noise prediction
- **Training loop**: Sample timesteps, add noise, predict noise, compute loss
- **Sampling loop**: Start from noise, iteratively denoise

### Checkpoint Format

Follow the established pattern:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'ema_state_dict': ema_model.state_dict(),  # Optional EMA model
    'config': config.__dict__,
    'loss': current_loss
}
```

## Dependencies

Expected core dependencies (align with VAE/GAN):

```txt
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.0.0
Pillow>=9.0.0
matplotlib>=3.5.0
tensorboard>=2.10.0
tqdm>=4.64.0
numpy>=1.21.0
```

Additional for diffusion models:
```txt
einops>=0.6.0           # Tensor operations for U-Net
accelerate>=0.20.0      # Distributed training
```

## Output Organization

Follow the established pattern:

```
checkpoints/
├── checkpoint_epoch_010.pth
├── checkpoint_epoch_020.pth
└── best_model.pth

logs/
└── events.out.tfevents.* (TensorBoard files)

samples/
├── samples_step_10000.png
├── samples_step_20000.png
└── training_progress.png
```

## Coding Conventions

Based on VAE/GAN implementations:

1. **Imports**: Group stdlib, third-party, local imports
2. **Type hints**: Use for function signatures where helpful
3. **Docstrings**: Document classes and complex functions
4. **Device handling**: Always check `torch.cuda.is_available()`
5. **Reproducibility**: Set random seeds when provided
6. **Logging**: Use TensorBoard for metrics, tqdm for progress bars
7. **Error handling**: Graceful fallbacks for data loading failures

## Git Workflow

This repository uses conventional commits. Recent examples:
- `ADD: training DCGAN + WGAN-GP`
- `MOD: adjust the learning rate`
- `DEL: useless code in VAE`

Follow this pattern:
- `ADD:` for new features
- `MOD:` for modifications
- `DEL:` for deletions
- `FIX:` for bug fixes

## Related Documentation

Refer to sibling implementations for architectural guidance:
- `/home/start-up/AI-ZeroToOne/Season-3/VAE/` - VAE implementation patterns
- `/home/start-up/AI-ZeroToOne/Season-3/GAN/` - GAN implementation patterns
- `/home/start-up/AI-ZeroToOne/README.md` - Overall project goals

## Season-3 Learning Objectives

Per the main README, this directory covers:
- Diffusion Models (VI & Score-based)
- Flow Matching & Diffusion (ODE & SDE)
- Conditional Generation (Text-To-Image)
- DiT (Diffusion Transformer)
- Video Generation

Implementations should align with these educational goals and follow the established patterns from VAE/GAN for consistency.
