# Flow Matching for Image Generation

This project implements **Flow Matching** (Lipman et al., 2022) for generative image modeling using CelebA faces. Flow matching learns continuous normalizing flows through conditional optimal transport paths, offering a simpler and more efficient alternative to diffusion models.

## Features

- **Flow Matching**: Conditional flow matching with optimal transport paths
- **U-Net Architecture**: Modern U-Net with attention mechanisms and residual blocks
- **Multi-GPU Training**: Full DDP support optimized for 2x NVIDIA 4090 GPUs
- **Mixed Precision**: Automatic mixed precision (AMP) for faster training
- **EMA Model**: Exponential moving average for improved sample quality
- **Flexible ODE Solvers**: Euler and midpoint methods for sampling
- **TensorBoard Logging**: Real-time training monitoring
- **Checkpointing**: Automatic checkpoint saving and resumption

## Project Structure

```
Diffusion/
├── config/           # Configuration dataclasses
├── model/            # U-Net and Flow Matching implementations
├── loader/           # CelebA dataset loader
├── trainer/          # Multi-GPU training loop
├── utils/            # EMA and visualization utilities
├── train.py          # Main training script
├── generate.py       # Sample generation script
├── test_setup.py     # Setup verification
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 2x NVIDIA 4090 GPUs (or adapt for your hardware)

### Setup

1. Clone the repository:
```bash
cd /home/start-up/AI-ZeroToOne/Season-3/Diffusion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python test_setup.py
```

## Training

### Basic Training (Single GPU)

```bash
python train.py --batch-size 32 --num-epochs 100
```

### Multi-GPU Training (2x 4090s)

**Using torchrun (recommended):**
```bash
torchrun --nproc_per_node=2 train.py \
    --batch-size 32 \
    --num-epochs 100 \
    --mixed-precision
```

**Using torch.distributed.launch:**
```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py \
    --batch-size 32 \
    --num-epochs 100
```

### Training on RunPod

1. Launch a RunPod instance with 2x 4090 GPUs
2. Clone the repository
3. Install dependencies
4. Run training:

```bash
# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Launch training
torchrun --nproc_per_node=2 train.py \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 2e-4 \
    --sample-every 1000 \
    --save-every 5000
```

### Advanced Training Options

```bash
torchrun --nproc_per_node=2 train.py \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 2e-4 \
    --model-channels 128 \
    --channel-mult 1 2 2 4 \
    --num-res-blocks 2 \
    --attention-resolutions 16 8 \
    --dropout 0.1 \
    --use-ema \
    --ema-decay 0.9999 \
    --mixed-precision \
    --grad-clip 1.0 \
    --lr-warmup-steps 1000 \
    --sample-every 1000 \
    --save-every 5000 \
    --num-workers 4
```

### Resume Training

```bash
torchrun --nproc_per_node=2 train.py \
    --resume-from checkpoints/checkpoint_epoch_50.pth
```

## Generation

Generate samples from a trained model:

```bash
python generate.py \
    --checkpoint checkpoints/checkpoint_epoch_100.pth \
    --num-samples 64 \
    --num-steps 50 \
    --solver midpoint \
    --output-dir results
```

### Generation Options

```bash
python generate.py \
    --checkpoint checkpoints/checkpoint_epoch_100.pth \
    --num-samples 100 \
    --num-steps 100 \  # More steps = better quality but slower
    --solver midpoint \  # 'euler' or 'midpoint'
    --seed 42 \  # For reproducibility
    --save-individual  # Save individual images
```

## Configuration

### Model Configuration

The model architecture can be customized via command-line arguments:

- `--image-size`: Image resolution (default: 64)
- `--model-channels`: Base channel count (default: 128)
- `--channel-mult`: Channel multipliers per resolution (default: 1 2 2 4)
- `--num-res-blocks`: Residual blocks per resolution (default: 2)
- `--attention-resolutions`: Resolutions for attention (default: 16 8)
- `--dropout`: Dropout rate (default: 0.1)

### Training Configuration

Key training hyperparameters:

- `--batch-size`: Per-GPU batch size (default: 32)
  - Effective batch size = batch_size × num_gpus
  - For 2x 4090s: 32 × 2 = 64 effective batch size
- `--learning-rate`: Learning rate (default: 2e-4)
- `--num-epochs`: Training epochs (default: 100)
- `--grad-clip`: Gradient clipping (default: 1.0)
- `--lr-warmup-steps`: Warmup steps (default: 1000)

### Flow Matching Configuration

- `--sigma-min`: Minimum noise level (default: 1e-4)
- `--use-ema`: Enable EMA (default: True)
- `--ema-decay`: EMA decay rate (default: 0.9999)

## Monitoring

### TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir logs
```

Access at: http://localhost:6006

Metrics logged:
- Training loss
- Learning rate
- Generated samples
- Model gradients

### Checkpoints

Checkpoints are saved automatically:
- Every N steps (configured via `--save-every`)
- At the end of each epoch
- On manual interruption (Ctrl+C)

Checkpoint contents:
- Model weights
- EMA model weights
- Optimizer state
- Scheduler state
- Training step and epoch
- Configuration

## Architecture Details

### Flow Matching

Flow matching learns a velocity field `v_t(x_t)` that defines an ODE:
```
dx/dt = v_t(x_t)
```

The model is trained with conditional optimal transport paths:
```
x_t = t * x_1 + (1 - t) * x_0 + σ_min * ε
```

where:
- `x_1`: Target data sample
- `x_0`: Random noise ~ N(0, I)
- `t ~ Uniform(0, 1)`
- `ε ~ N(0, I)`: Additional small noise

Loss function:
```
L = E[||v_t(x_t) - (x_1 - x_0)||^2]
```

### U-Net Architecture

- **Encoder**: Downsampling with residual blocks
- **Middle**: Bottleneck with attention
- **Decoder**: Upsampling with skip connections
- **Time Embedding**: Sinusoidal positional encoding
- **Attention**: Multi-head self-attention at specified resolutions

Default configuration:
- Base channels: 128
- Channel multipliers: [1, 2, 2, 4] → [128, 256, 256, 512]
- Resolutions: 64 → 32 → 16 → 8
- Attention at: 16×16 and 8×8
- Parameters: ~50M (default config)

## Performance

### Expected Training Time

On 2x NVIDIA 4090 GPUs:
- **Batch size**: 32 per GPU (64 effective)
- **Time per epoch**: ~15-20 minutes (CelebA ~150K images)
- **Total training**: ~25-35 hours for 100 epochs
- **Sample quality**: Good results after 30-50 epochs

### Memory Usage

- **Per GPU**: ~12-15 GB with default settings
- **Optimizations**:
  - Mixed precision: Reduces memory by ~30-40%
  - Gradient checkpointing: Not implemented but can be added
  - Smaller batch size: Reduces memory but may affect quality

### Optimization Tips

1. **Batch Size**: Larger is better for stability (up to memory limits)
2. **Learning Rate**: 2e-4 works well; scale with batch size if needed
3. **EMA**: Essential for good sample quality
4. **Attention Resolutions**: More attention = better quality but slower
5. **ODE Steps**: 50 steps is a good balance; 100+ for best quality

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Reduce `--model-channels`
- Ensure `--mixed-precision` is enabled

### Poor Sample Quality

- Train longer (50+ epochs)
- Ensure `--use-ema` is enabled
- Increase `--num-steps` during generation
- Check that loss is decreasing

### Slow Training

- Ensure mixed precision is enabled
- Increase `--num-workers` for data loading
- Reduce `--sample-every` and `--save-every`
- Use `--compile-model` (PyTorch 2.0+)

### Multi-GPU Issues

- Check CUDA is available: `python -c "import torch; print(torch.cuda.device_count())"`
- Verify NCCL backend: `python -c "import torch.distributed as dist; print(dist.is_nccl_available())"`
- Ensure all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`

## Citation

If you use this code, please cite the original Flow Matching paper:

```bibtex
@article{lipman2022flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky TQ and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  journal={arXiv preprint arXiv:2210.02747},
  year={2022}
}
```

## License

This project is part of the AI-ZeroToOne learning series and is intended for educational purposes.

## Acknowledgments

- Flow Matching paper by Lipman et al.
- U-Net architecture inspired by DDPM (Ho et al., 2020)
- CelebA dataset by Liu et al.
