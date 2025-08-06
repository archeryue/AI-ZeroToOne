#!/usr/bin/env python3
"""
Demonstration script for VAE models
This script shows how to:
1. Load and use a trained VAE model
2. Generate new samples
3. Perform reconstructions
4. Interpolate in latent space
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from models import VAEMnist, VAECelebA
from data import get_mnist_loader, get_celeba_loader
from utils import (plot_samples, plot_reconstruction, interpolate_latent, 
                   load_checkpoint, plot_latent_space)


def demo_mnist():
    """Demonstrate MNIST VAE functionality"""
    print("=== MNIST VAE Demo ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = VAEMnist(latent_dim=2)  # Use 2D latent space for visualization
    model.to(device)
    
    # Load test data
    test_loader = get_mnist_loader(batch_size=64, train=False, download=True)
    
    # If you have a trained model, load it:
    # try:
    #     load_checkpoint(model, None, './checkpoints/mnist/best_model.pth', device)
    #     print("Loaded trained model")
    # except:
    #     print("No trained model found, using randomly initialized model")
    
    print("No trained model found, using randomly initialized model for demonstration")
    
    model.eval()
    with torch.no_grad():
        # Get test batch
        test_data, test_labels = next(iter(test_loader))
        test_data = test_data.to(device)
        
        # 1. Generate samples
        print("1. Generating random samples...")
        samples = model.sample(num_samples=16, device=device)
        samples = samples.view(-1, 1, 28, 28)
        plot_samples(samples, title="MNIST Generated Samples", nrow=4, figsize=(8, 8))
        
        # 2. Show reconstructions
        print("2. Showing reconstructions...")
        recon_data, mu, logvar = model(test_data)
        recon_data = recon_data.view_as(test_data)
        plot_reconstruction(test_data, recon_data, n_samples=8, 
                          title="MNIST Original vs Reconstructed")
        
        # 3. Latent space visualization (only works for 2D latent space)
        if model.latent_dim == 2:
            print("3. Visualizing 2D latent space...")
            plot_latent_space(model, test_loader, device, n_samples=1000)
        
        # 4. Interpolation
        print("4. Latent space interpolation...")
        # Get two different digits
        img1 = test_data[0]
        img2 = test_data[1]
        interpolate_latent(model, device, img1, img2, n_steps=10)


def demo_celeba():
    """Demonstrate CelebA VAE functionality"""
    print("\n=== CelebA VAE Demo ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = VAECelebA(latent_dim=128)
    model.to(device)
    
    # Load test data (will use dummy data if CelebA not available)
    test_loader = get_celeba_loader(batch_size=32, train=False, download=False)
    
    # If you have a trained model, load it:
    # try:
    #     load_checkpoint(model, None, './checkpoints/celeba/best_model.pth', device)
    #     print("Loaded trained model")
    # except:
    #     print("No trained model found, using randomly initialized model")
    
    print("No trained model found, using randomly initialized model for demonstration")
    
    model.eval()
    with torch.no_grad():
        # Get test batch
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)
        
        # 1. Generate samples
        print("1. Generating random samples...")
        samples = model.sample(num_samples=16, device=device)
        plot_samples(samples, title="CelebA Generated Samples", nrow=4, figsize=(10, 10))
        
        # 2. Show reconstructions
        print("2. Showing reconstructions...")
        recon_data, mu, logvar = model(test_data)
        plot_reconstruction(test_data, recon_data, n_samples=6, 
                          title="CelebA Original vs Reconstructed")
        
        # 3. Interpolation
        print("3. Latent space interpolation...")
        img1 = test_data[0]
        img2 = test_data[1]
        interpolate_latent(model, device, img1, img2, n_steps=10)


def compare_models():
    """Compare different latent dimensions for MNIST"""
    print("\n=== Comparing Different Latent Dimensions ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dims = [2, 10, 20, 50]
    
    # Load test data
    test_loader = get_mnist_loader(batch_size=16, train=False, download=True)
    test_data, _ = next(iter(test_loader))
    test_data = test_data.to(device)
    
    plt.figure(figsize=(15, 10))
    
    for i, latent_dim in enumerate(latent_dims):
        # Create model
        model = VAEMnist(latent_dim=latent_dim).to(device)
        model.eval()
        
        # Generate samples
        with torch.no_grad():
            samples = model.sample(num_samples=16, device=device)
            samples = samples.view(-1, 1, 28, 28).cpu()
            
            # Create subplot
            plt.subplot(2, 2, i+1)
            
            # Create grid for display
            grid = torch.zeros(4*28, 4*28)
            for row in range(4):
                for col in range(4):
                    idx = row * 4 + col
                    grid[row*28:(row+1)*28, col*28:(col+1)*28] = samples[idx, 0]
            
            plt.imshow(grid, cmap='gray')
            plt.title(f'Latent Dim: {latent_dim}')
            plt.axis('off')
    
    plt.suptitle('Generated Samples with Different Latent Dimensions')
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration function"""
    print("VAE Demonstration Script")
    print("=" * 50)
    
    # Demo MNIST
    demo_mnist()
    
    # Demo CelebA
    demo_celeba()
    
    # Compare models
    compare_models()
    
    print("\nDemo completed!")
    print("\nTo train your own models:")
    print("- MNIST: python train_mnist.py")
    print("- CelebA: python train_celeba.py")
    print("\nAfter training, models will be saved in ./checkpoints/")
    print("You can then uncomment the checkpoint loading code in this demo to see results with trained models.")


if __name__ == '__main__':
    main() 