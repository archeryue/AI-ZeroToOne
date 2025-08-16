#!/usr/bin/env python3
"""
Main training script for DCGAN+WGAN-GP model on CelebA dataset.

This script trains a Deep Convolutional GAN with Wasserstein GAN with Gradient Penalty
loss to generate human faces using the CelebA dataset from HuggingFace.

Usage:
    python training.py [--resume CHECKPOINT_PATH] [--config CONFIG_PATH]
"""

import argparse
import os
import sys
import torch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import get_config, get_data_config
from loader.dataset import get_dataloader, test_dataloader
from model.dcgan import create_models, test_models
from trainer.trainer import WGANGPTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DCGAN+WGAN-GP on CelebA')
    
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--test-only', 
        action='store_true',
        help='Only test the models and dataloader without training'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=None,
        help='Number of epochs to train (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate-d', 
        type=float, 
        default=None,
        help='Learning rate for D (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate-g', 
        type=float, 
        default=None,
        help='Learning rate for G (overrides config)'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use for training (overrides config)'
    )
    
    parser.add_argument(
        '--checkpoint-dir', 
        type=str, 
        default=None,
        help='Directory to save checkpoints (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configurations
    config = get_config()
    data_config = get_data_config()
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate_d is not None:
        config.learning_rate_d = args.learning_rate_d
    if args.learning_rate_g is not None:
        config.learning_rate_g = args.learning_rate_g
    if args.device is not None:
        config.device = args.device
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    
    # Print configuration
    print("=" * 50)
    print("DCGAN+WGAN-GP Training Configuration")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate D: {config.learning_rate_d}")
    print(f"Learning rate G: {config.learning_rate_g}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Critic iterations: {config.critic_iterations}")
    print(f"Gradient penalty lambda: {config.gradient_penalty_lambda}")
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print("=" * 50)
    
    # Check if CUDA is available
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        config.device = 'cpu'
    
    if config.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Create dataloader
        print("\nLoading dataset...")
        dataloader = get_dataloader(data_config, config)
        print(f"Dataset loaded successfully. Number of batches: {len(dataloader)}")
        
        # Test dataloader
        print("\nTesting dataloader...")
        test_dataloader(dataloader, num_batches=2)
        
        # Create models
        print("\nCreating models...")
        generator, discriminator = create_models(config)
        
        # Test models
        print("\nTesting models...")
        test_models(config)
        
        # Count parameters
        gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"Generator: {gen_params:,}")
        print(f"Discriminator: {disc_params:,}")
        print(f"Total: {gen_params + disc_params:,}")
        
        if args.test_only:
            print("\nTest completed successfully!")
            return
        
        # Create trainer
        print("\nInitializing trainer...")
        trainer = WGANGPTrainer(config, generator, discriminator, dataloader)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            if os.path.exists(args.resume):
                trainer.load_checkpoint(args.resume)
                start_epoch = trainer.current_epoch + 1
            else:
                print(f"Checkpoint file not found: {args.resume}")
                return
        
        # Start training
        print(f"\nStarting training from epoch {start_epoch}...")
        print(f"Training logs will be saved to: {config.log_dir}")
        print(f"Checkpoints will be saved to: {config.checkpoint_dir}")
        print(f"Results will be saved to: {config.results_dir}")
        print("\nTo monitor training progress, run:")
        print(f"tensorboard --logdir {config.log_dir}")
        print("=" * 50)
        
        trainer.train(start_epoch=start_epoch)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        if 'trainer' in locals():
            trainer.save_checkpoint('interrupted_checkpoint.pth')
        print("Checkpoint saved. You can resume training with --resume interrupted_checkpoint.pth")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save checkpoint if trainer exists
        if 'trainer' in locals():
            try:
                trainer.save_checkpoint('error_checkpoint.pth')
                print("Emergency checkpoint saved.")
            except:
                print("Could not save emergency checkpoint.")
        
        sys.exit(1)


if __name__ == '__main__':
    main()
