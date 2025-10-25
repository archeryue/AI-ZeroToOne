"""
Main training script for Flow Matching model with multi-GPU support.

Usage:
    # Single GPU
    python train.py

    # Multi-GPU with torchrun (recommended)
    torchrun --nproc_per_node=2 train.py

    # Multi-GPU with python -m torch.distributed.launch
    python -m torch.distributed.launch --nproc_per_node=2 train.py
"""
import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import FlowMatchingConfig, TrainingConfig
from model import UNet, FlowMatching
from loader import get_celeba_dataloader
from trainer import FlowMatchingTrainer


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        # Single GPU or CPU
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        dist.barrier()

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    """Main training function."""
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

    if is_main_process:
        print("=" * 80)
        print("Flow Matching Training")
        print("=" * 80)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    # Create configurations
    model_config = FlowMatchingConfig(
        image_size=args.image_size,
        image_channels=3,
        model_channels=args.model_channels,
        channel_mult=tuple(args.channel_mult),
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(args.attention_resolutions),
        dropout=args.dropout,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        sigma_min=args.sigma_min
    )

    training_config = TrainingConfig(
        dataset_name='celeba',
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lr_warmup_steps=args.lr_warmup_steps,
        sample_every=args.sample_every,
        num_samples=args.num_samples,
        sample_steps=args.sample_steps,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        log_dir=args.log_dir,
        num_gpus=world_size,
        distributed=(world_size > 1),
        num_workers=args.num_workers,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile_model,
        resume_from=args.resume_from
    )

    if is_main_process:
        print("\nModel Configuration:")
        for key, value in model_config.__dict__.items():
            print(f"  {key}: {value}")
        print("\nTraining Configuration:")
        for key, value in training_config.__dict__.items():
            print(f"  {key}: {value}")
        print()

    # Create data loader
    if is_main_process:
        print("Loading dataset...")

    dataloader = get_celeba_dataloader(
        data_dir=training_config.data_dir,
        split='train',
        batch_size=training_config.batch_size,
        image_size=model_config.image_size,
        num_workers=training_config.num_workers,
        distributed=training_config.distributed,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    if is_main_process:
        print(f"Dataset loaded: {len(dataloader)} batches per epoch")
        print(f"Total images per epoch: ~{len(dataloader) * training_config.batch_size * world_size}")

    # Create model
    if is_main_process:
        print("\nCreating model...")

    unet = UNet(
        image_channels=model_config.image_channels,
        model_channels=model_config.model_channels,
        channel_mult=model_config.channel_mult,
        num_res_blocks=model_config.num_res_blocks,
        attention_resolutions=model_config.attention_resolutions,
        dropout=model_config.dropout
    )

    flow_model = FlowMatching(
        model=unet,
        sigma_min=model_config.sigma_min
    )

    # Count parameters
    if is_main_process:
        total_params = sum(p.numel() for p in unet.parameters())
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Compile model (PyTorch 2.0+)
    if training_config.compile_model:
        if is_main_process:
            print("Compiling model with torch.compile...")
        unet = torch.compile(unet)

    # Create trainer
    trainer = FlowMatchingTrainer(
        flow_model=flow_model,
        dataloader=dataloader,
        model_config=model_config,
        training_config=training_config,
        rank=rank,
        world_size=world_size
    )

    # Resume from checkpoint if specified
    if training_config.resume_from is not None:
        trainer.load_checkpoint(training_config.resume_from)

    # Start training
    if is_main_process:
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_main_process:
            print("\n\nTraining interrupted by user")
            print("Saving checkpoint...")
            trainer.save_checkpoint('checkpoint_interrupted.pth')
    finally:
        cleanup_distributed()

    if is_main_process:
        print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Flow Matching model')

    # Model architecture
    parser.add_argument('--image-size', type=int, default=64, help='Image size')
    parser.add_argument('--model-channels', type=int, default=128, help='Base channel count')
    parser.add_argument('--channel-mult', type=int, nargs='+', default=[1, 2, 2, 4], help='Channel multipliers')
    parser.add_argument('--num-res-blocks', type=int, default=2, help='Residual blocks per resolution')
    parser.add_argument('--attention-resolutions', type=int, nargs='+', default=[16, 8], help='Resolutions for attention')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Flow matching
    parser.add_argument('--sigma-min', type=float, default=1e-4, help='Minimum noise level')
    parser.add_argument('--use-ema', action='store_true', default=True, help='Use EMA')
    parser.add_argument('--no-ema', dest='use_ema', action='store_false', help='Disable EMA')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay rate')

    # Training
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--lr-warmup-steps', type=int, default=1000, help='Learning rate warmup steps')

    # Sampling
    parser.add_argument('--sample-every', type=int, default=1000, help='Sample every N steps')
    parser.add_argument('--num-samples', type=int, default=64, help='Number of samples to generate')
    parser.add_argument('--sample-steps', type=int, default=50, help='ODE solver steps for sampling')

    # Checkpointing and logging
    parser.add_argument('--save-every', type=int, default=5000, help='Save checkpoint every N steps')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-every', type=int, default=100, help='Log every N steps')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')

    # Data
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mixed-precision', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--no-mixed-precision', dest='mixed_precision', action='store_false', help='Disable mixed precision')
    parser.add_argument('--compile-model', action='store_true', default=True, help='Use torch.compile for 27%% speedup (default: enabled)')
    parser.add_argument('--no-compile-model', dest='compile_model', action='store_false', help='Disable torch.compile')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    main(args)
