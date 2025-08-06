import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import os


def plot_samples(samples, title="Generated Samples", figsize=(10, 10), nrow=8, save_path=None):
    """
    Plot a grid of generated samples
    
    Args:
        samples: Tensor of shape (N, C, H, W) containing generated samples
        title: Title for the plot
        figsize: Figure size
        nrow: Number of samples per row
        save_path: Path to save the plot (optional)
    """
    with torch.no_grad():
        # Move to CPU and clamp values
        samples = samples.cpu().clamp(0, 1)
        
        # Create grid
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        
        # Convert to numpy for plotting
        grid_np = grid.permute(1, 2, 0).numpy()
        
        # Handle grayscale images
        if grid_np.shape[2] == 1:
            grid_np = grid_np.squeeze(2)
            cmap = 'gray'
        else:
            cmap = None
        
        plt.figure(figsize=figsize)
        plt.imshow(grid_np, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved plot to {save_path}")
        
        plt.show()


def plot_reconstruction(original, reconstructed, n_samples=8, title="Original vs Reconstructed", save_path=None):
    """
    Plot original images alongside their reconstructions
    
    Args:
        original: Original images tensor
        reconstructed: Reconstructed images tensor
        n_samples: Number of samples to display
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    with torch.no_grad():
        # Take first n_samples
        orig = original[:n_samples].cpu().clamp(0, 1)
        recon = reconstructed[:n_samples].cpu().clamp(0, 1)
        
        # Interleave original and reconstructed
        comparison = torch.zeros(2 * n_samples, *orig.shape[1:])
        comparison[0::2] = orig
        comparison[1::2] = recon
        
        # Create grid
        grid = make_grid(comparison, nrow=2, padding=2, normalize=False)
        grid_np = grid.permute(1, 2, 0).numpy()
        
        # Handle grayscale images
        if grid_np.shape[2] == 1:
            grid_np = grid_np.squeeze(2)
            cmap = 'gray'
        else:
            cmap = None
        
        plt.figure(figsize=(12, 6))
        plt.imshow(grid_np, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        
        # Add labels
        plt.text(0.25, 0.95, 'Original', transform=plt.gca().transAxes, 
                 ha='center', va='top', fontsize=12, weight='bold')
        plt.text(0.75, 0.95, 'Reconstructed', transform=plt.gca().transAxes, 
                 ha='center', va='top', fontsize=12, weight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved reconstruction plot to {save_path}")
        
        plt.show()


def plot_latent_space(model, data_loader, device, n_samples=1000, save_path=None):
    """
    Plot 2D visualization of latent space (only works for 2D latent space)
    
    Args:
        model: Trained VAE model
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        n_samples: Number of samples to plot
        save_path: Path to save the plot (optional)
    """
    if model.latent_dim != 2:
        print(f"Latent space visualization only works for 2D latent space, got {model.latent_dim}D")
        return
    
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if len(latents) >= n_samples:
                break
                
            data = data.to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu())
            labels.append(target)
    
    latents = torch.cat(latents, dim=0)[:n_samples]
    labels = torch.cat(labels, dim=0)[:n_samples]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Latent Space Visualization')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved latent space plot to {save_path}")
    
    plt.show()


def save_samples(model, device, num_samples=64, save_dir='./samples', epoch=None, dataset_name=''):
    """
    Generate and save samples from the model
    
    Args:
        model: Trained VAE model
        device: Device to run inference on
        num_samples: Number of samples to generate
        save_dir: Directory to save samples
        epoch: Current epoch (for filename)
        dataset_name: Name of dataset (for filename)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        samples = model.sample(num_samples, device)
        
        # Reshape MNIST samples for visualization
        if len(samples.shape) == 2:  # MNIST case
            samples = samples.view(-1, 1, 28, 28)
        
        # Create filename
        if epoch is not None:
            filename = f"{dataset_name}_samples_epoch_{epoch:03d}.png"
        else:
            filename = f"{dataset_name}_samples.png"
        
        save_path = os.path.join(save_dir, filename)
        
        plot_samples(
            samples, 
            title=f"{dataset_name} Generated Samples" + (f" (Epoch {epoch})" if epoch else ""),
            save_path=save_path
        )


def plot_training_curves(losses, save_path=None):
    """
    Plot training curves for total loss, reconstruction loss, and KL loss
    
    Args:
        losses: Dictionary containing 'total', 'recon', 'kl' loss histories
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 5))
    
    # Total loss
    plt.subplot(1, 3, 1)
    plt.plot(losses['total'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Reconstruction loss
    plt.subplot(1, 3, 2)
    plt.plot(losses['recon'])
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # KL loss
    plt.subplot(1, 3, 3)
    plt.plot(losses['kl'])
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def interpolate_latent(model, device, start_img, end_img, n_steps=10, save_path=None):
    """
    Interpolate between two images in latent space
    
    Args:
        model: Trained VAE model
        device: Device to run inference on
        start_img: Starting image tensor
        end_img: Ending image tensor
        n_steps: Number of interpolation steps
        save_path: Path to save the plot (optional)
    """
    model.eval()
    
    with torch.no_grad():
        # Encode start and end images
        start_img = start_img.unsqueeze(0).to(device)
        end_img = end_img.unsqueeze(0).to(device)
        
        start_mu, _ = model.encode(start_img)
        end_mu, _ = model.encode(end_img)
        
        # Interpolate in latent space
        interpolations = []
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            interp_mu = (1 - alpha) * start_mu + alpha * end_mu
            interp_img = model.decode(interp_mu)
            
            # Reshape for MNIST
            if len(interp_img.shape) == 2:
                interp_img = interp_img.view(-1, 1, 28, 28)
            
            interpolations.append(interp_img)
        
        # Concatenate all interpolations
        all_imgs = torch.cat(interpolations, dim=0)
        
        # Plot
        plot_samples(
            all_imgs,
            title=f"Latent Space Interpolation ({n_steps} steps)",
            nrow=n_steps,
            figsize=(n_steps * 2, 4),
            save_path=save_path
        ) 