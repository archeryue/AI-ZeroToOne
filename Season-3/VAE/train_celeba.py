#!/usr/bin/env python3
"""
Training script for VAE on CelebA dataset
"""

import torch
import os
import argparse
from torch.utils.data import random_split

# Import custom modules
from models import VAECelebA
from data import get_celeba_loader
from utils import VAETrainer, save_samples, plot_reconstruction, plot_training_curves
from configs import celeba_config as config


def main():
    parser = argparse.ArgumentParser(description='Train VAE on CelebA')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=config.BETA,
                        help='Beta parameter for β-VAE')
    parser.add_argument('--latent_dim', type=int, default=config.LATENT_DIM,
                        help='Latent dimension')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    
    # Load data
    print("Loading CelebA dataset...")
    try:
        train_loader = get_celeba_loader(
            batch_size=args.batch_size, 
            train=True, 
            download=config.DOWNLOAD_DATA,
            data_root=config.DATA_ROOT
        )
        test_loader = get_celeba_loader(
            batch_size=args.batch_size, 
            train=False, 
            download=config.DOWNLOAD_DATA,
            data_root=config.DATA_ROOT
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"Error loading CelebA dataset: {e}")
        print("Using dummy dataset for demonstration...")
        
        # Create dummy loaders for demonstration
        train_loader = get_celeba_loader(batch_size=args.batch_size, train=True, download=False)
        test_loader = get_celeba_loader(batch_size=args.batch_size, train=False, download=False)
        
        print("Note: This is using randomly generated data for demonstration.")
        print("For real training, please download the CelebA dataset manually.")
    
    # Create model
    print(f"Creating VAE model with latent dimension: {args.latent_dim}")
    model = VAECelebA(latent_dim=args.latent_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        lr=args.lr,
        beta=args.beta,
        checkpoint_dir=config.CHECKPOINT_DIR,
        log_dir=config.LOG_DIR,
        save_interval=config.SAVE_INTERVAL
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)
    
    # Generate and save samples
    print("\nGenerating samples...")
    save_samples(
        model=model,
        device=device,
        num_samples=64,
        save_dir=config.SAMPLES_DIR,
        dataset_name='CelebA'
    )
    
    # Plot training curves
    print("Plotting training curves...")
    history = trainer.get_history()
    plot_training_curves(
        losses={
            'total': history['train_loss']['total'],
            'recon': history['train_loss']['recon'],
            'kl': history['train_loss']['kl']
        },
        save_path=os.path.join(config.SAMPLES_DIR, 'training_curves.png')
    )
    
    # Show reconstructions
    print("Generating reconstructions...")
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)
        
        # Get reconstructions
        recon_data, _, _ = model(test_data)
        
        # Plot reconstructions
        plot_reconstruction(
            original=test_data,
            reconstructed=recon_data,
            n_samples=8,
            title="CelebA Reconstructions",
            save_path=os.path.join(config.SAMPLES_DIR, 'reconstructions.png')
        )
    
    print(f"\nTraining completed! Check the following directories:")
    print(f"- Checkpoints: {config.CHECKPOINT_DIR}")
    print(f"- Logs: {config.LOG_DIR}")
    print(f"- Samples: {config.SAMPLES_DIR}")
    print(f"\nTo view training logs, run: tensorboard --logdir {config.LOG_DIR}")


if __name__ == '__main__':
    main() 