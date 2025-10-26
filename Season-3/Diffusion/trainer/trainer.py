"""
Trainer for Flow Matching model with multi-GPU support.
"""
import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional
from tqdm import tqdm
import time

from config import FlowMatchingConfig, TrainingConfig
from model import FlowMatching, UNet
from utils import save_image_grid, EMA


class FlowMatchingTrainer:
    """
    Trainer for Flow Matching model with distributed training support.
    """

    def __init__(
        self,
        flow_model: FlowMatching,
        dataloader: torch.utils.data.DataLoader,
        model_config: FlowMatchingConfig,
        training_config: TrainingConfig,
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Args:
            flow_model: FlowMatching model
            dataloader: Training data loader
            model_config: Model configuration
            training_config: Training configuration
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.flow_model = flow_model
        self.dataloader = dataloader
        self.model_config = model_config
        self.training_config = training_config
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        # Device
        self.device = torch.device(f'cuda:{rank}')
        self.flow_model = self.flow_model.to(self.device)

        # Wrap with DDP if distributed
        if training_config.distributed and world_size > 1:
            self.flow_model = DDP(
                self.flow_model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False
            )
            self.model = self.flow_model.module
        else:
            self.model = self.flow_model

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Mixed precision training with conservative GradScaler settings
        self.use_amp = training_config.mixed_precision
        if self.use_amp:
            # More conservative scaler settings to prevent NaN propagation
            self.scaler = GradScaler(
                'cuda',
                init_scale=2.**10,  # Start with lower scale (default is 2^16)
                growth_factor=1.5,  # Slower growth (default is 2.0)
                backoff_factor=0.5,  # Same backoff
                growth_interval=1000  # Less frequent growth (default is 2000)
            )
        else:
            self.scaler = None

        # EMA model for better sample quality
        self.use_ema = model_config.use_ema
        if self.use_ema:
            self.ema = EMA(self.model.model, decay=model_config.ema_decay)
        else:
            self.ema = None

        # TensorBoard
        if self.is_main_process:
            os.makedirs(training_config.log_dir, exist_ok=True)
            self.writer = SummaryWriter(training_config.log_dir)
        else:
            self.writer = None

        # Tracking
        self.global_step = 0
        self.epoch = 0

        # Checkpointing
        os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    def _create_scheduler(self):
        """Create learning rate scheduler with linear warmup."""
        def lr_lambda(step):
            if step < self.training_config.lr_warmup_steps:
                return step / self.training_config.lr_warmup_steps
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: torch.Tensor) -> dict:
        """
        Single training step.

        Args:
            batch: Batch of images (B, C, H, W)

        Returns:
            Dictionary with loss and metrics
        """
        batch = batch.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(device_type='cuda', enabled=self.use_amp):
            loss, info = self.flow_model(batch)

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  WARNING: NaN or Inf loss detected at step {self.global_step}!")
            print("Skipping this batch to prevent gradient corruption...")
            info['loss'] = float('nan')
            info['skipped'] = True
            return info

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)

            # Check for NaN gradients before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.grad_clip
            )

            # Detect gradient explosion
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\n⚠️  WARNING: NaN or Inf gradients at step {self.global_step}!")
                print("Skipping optimizer step...")
                self.optimizer.zero_grad()
                info['loss'] = float('nan')
                info['grad_norm'] = float('nan')
                info['skipped'] = True
                return info

            info['grad_norm'] = grad_norm.item()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.grad_clip
            )

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\n⚠️  WARNING: NaN or Inf gradients at step {self.global_step}!")
                print("Skipping optimizer step...")
                self.optimizer.zero_grad()
                info['loss'] = float('nan')
                info['grad_norm'] = float('nan')
                info['skipped'] = True
                return info

            info['grad_norm'] = grad_norm.item()
            self.optimizer.step()

        self.scheduler.step()

        # Update EMA
        if self.use_ema:
            self.ema.update()

        info['skipped'] = False
        return info

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int) -> torch.Tensor:
        """
        Generate samples using the trained model.

        Args:
            num_samples: Number of samples to generate
            num_steps: Number of ODE solver steps

        Returns:
            Generated samples (B, C, H, W)
        """
        # Use EMA model if available
        # Note: EMA tracks the U-Net directly, not the FlowMatching wrapper
        if self.use_ema:
            unet_model = self.ema.ema_model
        else:
            unet_model = self.model.model

        # Create temporary flow model for sampling
        flow_model = FlowMatching(
            model=unet_model,
            sigma_min=self.model_config.sigma_min
        )
        flow_model.eval()

        samples = flow_model.sample(
            batch_size=num_samples,
            image_shape=(
                self.model_config.image_channels,
                self.model_config.image_size,
                self.model_config.image_size
            ),
            num_steps=num_steps,
            device=self.device,
            solver='midpoint',
            verbose=False
        )

        return samples

    def train(self):
        """Main training loop."""
        if self.is_main_process:
            print(f"Starting training for {self.training_config.num_epochs} epochs")
            print(f"Total steps per epoch: {len(self.dataloader)}")
            print(f"Training on {self.world_size} GPU(s)")

        start_time = time.time()

        for epoch in range(self.training_config.num_epochs):
            self.epoch = epoch

            # Set epoch for distributed sampler
            if hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)

            self.train_epoch()

        if self.is_main_process:
            total_time = time.time() - start_time
            print(f"\nTraining completed in {total_time/3600:.2f} hours")
            self.writer.close()

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        if self.is_main_process:
            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch {self.epoch+1}/{self.training_config.num_epochs}"
            )
        else:
            pbar = self.dataloader

        epoch_loss = 0.0
        num_batches = 0

        for batch in pbar:
            info = self.train_step(batch)
            self.global_step += 1

            epoch_loss += info['loss']
            num_batches += 1

            # Logging
            if self.is_main_process and self.global_step % self.training_config.log_every == 0:
                # Only log non-NaN losses
                if not info.get('skipped', False):
                    self.writer.add_scalar('train/loss', info['loss'], self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                    if 'grad_norm' in info:
                        self.writer.add_scalar('train/grad_norm', info['grad_norm'], self.global_step)

                    if isinstance(pbar, tqdm):
                        postfix = {
                            'loss': f"{info['loss']:.4f}",
                            'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                        }
                        if 'grad_norm' in info:
                            postfix['grad'] = f"{info['grad_norm']:.3f}"
                        pbar.set_postfix(postfix)

            # Sampling
            if self.is_main_process and self.global_step % self.training_config.sample_every == 0:
                self.log_samples()

            # Checkpointing
            if self.is_main_process and self.global_step % self.training_config.save_every == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pth')

        # Log epoch metrics
        if self.is_main_process:
            avg_loss = epoch_loss / num_batches
            print(f"\nEpoch {self.epoch+1} - Average Loss: {avg_loss:.4f}")
            self.writer.add_scalar('train/epoch_loss', avg_loss, self.epoch)

            # Save checkpoint at end of epoch
            self.save_checkpoint(f'checkpoint_epoch_{self.epoch+1}.pth')

    @torch.no_grad()
    def log_samples(self):
        """Generate and log samples to TensorBoard."""
        if not self.is_main_process:
            return

        print(f"\nGenerating samples at step {self.global_step}...")

        samples = self.sample(
            num_samples=self.training_config.num_samples,
            num_steps=self.training_config.sample_steps
        )

        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)

        # Save grid
        save_path = os.path.join(
            'samples',
            f'samples_step_{self.global_step}.png'
        )
        save_image_grid(samples, save_path, nrow=8)

        # Log to TensorBoard
        self.writer.add_images('samples', samples[:16], self.global_step)

        print(f"Samples saved to {save_path}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if not self.is_main_process:
            return

        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__
        }

        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema.ema_model.state_dict()

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        save_path = os.path.join(self.training_config.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']

        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        # DO NOT load old scaler state - it may have grown too large
        # Instead, keep our conservative scaler settings and just reset the scale
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            old_scale = checkpoint['scaler_state_dict'].get('scale', 'unknown')
            print(f"⚠️  Ignoring old GradScaler state (scale={old_scale})")
            print(f"   Using fresh conservative scaler (init_scale=2^10)")
            # Don't load: self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
