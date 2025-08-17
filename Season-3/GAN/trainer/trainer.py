import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional
import time

from config.config import ModelConfig
from model.dcgan import Generator, Discriminator, gradient_penalty


class WGANGPTrainer:
    """Trainer for DCGAN with WGAN-GP loss."""
    
    def __init__(self, config: ModelConfig, generator: Generator, 
                 discriminator: Discriminator, dataloader: DataLoader):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        
        # Move models to device
        self.device = torch.device(config.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=(config.beta1, config.beta2)
        )
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Fixed noise for consistent samples
        self.fixed_noise = torch.randn(
            config.num_samples, config.latent_dim, 1, 1, device=self.device
        )
        
        # Loss tracking
        self.loss_history = {
            'G_loss': [],
            'D_loss': [],
            'D_real': [],
            'D_fake': [],
            'GP': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {
            'G_loss': 0.0,
            'D_loss': 0.0,
            'D_real': 0.0,
            'D_fake': 0.0,
            'GP': 0.0
        }
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch}")
        
        for i, real_images in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Train Discriminator (Critic)
            for _ in range(self.config.critic_iterations):
                self.optimizer_D.zero_grad()
                
                # Real images
                real_output = self.discriminator(real_images).view(-1)
                D_real = real_output.mean().item()
                
                # Fake images
                noise = torch.randn(batch_size, self.config.latent_dim, 1, 1, device=self.device)
                fake_images = self.generator(noise)
                fake_output = self.discriminator(fake_images.detach()).view(-1)
                D_fake = fake_output.mean().item()
                
                # Gradient penalty
                gp = gradient_penalty(self.discriminator, real_images, fake_images, self.device)
                
                # Discriminator loss (WGAN-GP)
                D_loss = -torch.mean(real_output) + torch.mean(fake_output) + self.config.gradient_penalty_lambda * gp
                
                D_loss.backward()
                self.optimizer_D.step()
            
            # Train Generator
            self.optimizer_G.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, self.config.latent_dim, 1, 1, device=self.device)
            fake_images = self.generator(noise)
            fake_output = self.discriminator(fake_images).view(-1)
            
            # Generator loss (WGAN)
            G_loss = -torch.mean(fake_output)
            
            G_loss.backward()
            self.optimizer_G.step()
            
            # Update epoch losses
            epoch_losses['G_loss'] += G_loss.item()
            epoch_losses['D_loss'] += D_loss.item()
            epoch_losses['D_real'] += D_real
            epoch_losses['D_fake'] += D_fake
            epoch_losses['GP'] += gp.item()
            
            # Log to tensorboard
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('Loss/Generator', G_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Discriminator', D_loss.item(), self.global_step)
                self.writer.add_scalar('Scores/D_real', D_real, self.global_step)
                self.writer.add_scalar('Scores/D_fake', D_fake, self.global_step)
                self.writer.add_scalar('Scores/D_margin', D_real - D_fake, self.global_step)
                self.writer.add_scalar('Loss/Gradient_Penalty', gp.item(), self.global_step)
            
            # Generate samples
            if self.global_step % self.config.sample_interval == 0:
                self.generate_samples()
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f'{G_loss.item():.4f}',
                'D_loss': f'{D_loss.item():.4f}',
                'D_real': f'{D_real:.4f}',
                'D_fake': f'{D_fake:.4f}',
                'D_margin': f'{D_real - D_fake:.4f}'
            })
            
            self.global_step += 1
        
        # Average losses over epoch
        num_batches = len(self.dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def generate_samples(self) -> None:
        """Generate and save sample images."""
        self.generator.eval()
        
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
            
            # Denormalize images (from [-1, 1] to [0, 1])
            fake_images = (fake_images + 1) / 2
            
            # Save grid of images
            grid = vutils.make_grid(
                fake_images, 
                nrow=8, 
                padding=2, 
                normalize=False
            )
            
            # Save to tensorboard
            self.writer.add_image('Generated_Images', grid, self.global_step)
            
            # Save to file
            save_path = os.path.join(
                self.config.results_dir, 
                f'samples_step_{self.global_step}.png'
            )
            vutils.save_image(grid, save_path)
        
        self.generator.train()
    
    def save_checkpoint(self, filename: Optional[str] = None) -> None:
        """Save model checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch}.pth'
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'config': self.config,
            'loss_history': self.loss_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.loss_history = checkpoint.get('loss_history', self.loss_history)
        
        print(f"Checkpoint loaded. Resuming from epoch {self.current_epoch}")
    
    def plot_losses(self) -> None:
        """Plot training losses."""
        if not self.loss_history['G_loss']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Generator and Discriminator losses
        axes[0, 0].plot(self.loss_history['G_loss'], label='Generator Loss')
        axes[0, 0].plot(self.loss_history['D_loss'], label='Discriminator Loss')
        axes[0, 0].set_title('Generator and Discriminator Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Discriminator scores
        axes[0, 1].plot(self.loss_history['D_real'], label='D(real)')
        axes[0, 1].plot(self.loss_history['D_fake'], label='D(fake)')
        axes[0, 1].set_title('Discriminator Scores')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Gradient penalty
        axes[1, 0].plot(self.loss_history['GP'], label='Gradient Penalty')
        axes[1, 0].set_title('Gradient Penalty')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('GP Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Wasserstein distance approximation
        wasserstein_dist = np.array(self.loss_history['D_real']) - np.array(self.loss_history['D_fake'])
        axes[1, 1].plot(wasserstein_dist, label='Wasserstein Distance')
        axes[1, 1].set_title('Wasserstein Distance Approximation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.results_dir, 'training_losses.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved: {plot_path}")
    
    def train(self, start_epoch: int = 0) -> None:
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate D: {self.config.learning_rate_d}")
        print(f"Learning rate G: {self.config.learning_rate_g}")
        
        # Generate initial samples
        self.generate_samples()
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train for one epoch
            epoch_losses = self.train_epoch()
            
            # Update loss history
            for key, value in epoch_losses.items():
                self.loss_history[key].append(value)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEpoch [{epoch}/{self.config.num_epochs}] completed in {epoch_time:.2f}s")
            print(f"G_loss: {epoch_losses['G_loss']:.4f}, D_loss: {epoch_losses['D_loss']:.4f}")
            print(f"D_real: {epoch_losses['D_real']:.4f}, D_fake: {epoch_losses['D_fake']:.4f}")
            print(f"GP: {epoch_losses['GP']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint()
                self.plot_losses()
        
        # Final checkpoint and plots
        self.save_checkpoint('final_checkpoint.pth')
        self.plot_losses()
        self.generate_samples()
        
        print("Training completed!")
        self.writer.close()
