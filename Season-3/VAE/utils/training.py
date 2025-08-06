import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import time


def train_epoch(model, train_loader, optimizer, device, beta=1.0, log_interval=100):
    """
    Train the model for one epoch
    
    Args:
        model: VAE model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        beta: Beta parameter for β-VAE (weight of KL loss)
        log_interval: Interval for logging training progress
    
    Returns:
        Dictionary containing average losses for the epoch
    """
    model.train()
    train_losses = {'total': 0, 'recon': 0, 'kl': 0}
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Calculate loss
        from models.vae import vae_loss
        total_loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        batch_size = data.size(0)
        train_losses['total'] += total_loss.item()
        train_losses['recon'] += recon_loss.item()
        train_losses['kl'] += kl_loss.item()
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'Total Loss': f"{total_loss.item() / batch_size:.4f}",
                'Recon Loss': f"{recon_loss.item() / batch_size:.4f}",
                'KL Loss': f"{kl_loss.item() / batch_size:.4f}"
            })
    
    # Calculate average losses
    num_samples = len(train_loader.dataset)
    train_losses = {k: v / num_samples for k, v in train_losses.items()}
    
    return train_losses


def validate_epoch(model, val_loader, device, beta=1.0):
    """
    Validate the model for one epoch
    
    Args:
        model: VAE model to validate
        val_loader: DataLoader for validation data
        device: Device to validate on
        beta: Beta parameter for β-VAE (weight of KL loss)
    
    Returns:
        Dictionary containing average losses for the epoch
    """
    model.eval()
    val_losses = {'total': 0, 'recon': 0, 'kl': 0}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for data, _ in pbar:
            data = data.to(device)
            
            # Forward pass
            recon_batch, mu, logvar = model(data)
            
            # Calculate loss
            from models.vae import vae_loss
            total_loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            # Accumulate losses
            val_losses['total'] += total_loss.item()
            val_losses['recon'] += recon_loss.item()
            val_losses['kl'] += kl_loss.item()
            
            # Update progress bar
            batch_size = data.size(0)
            pbar.set_postfix({
                'Total Loss': f"{total_loss.item() / batch_size:.4f}",
                'Recon Loss': f"{recon_loss.item() / batch_size:.4f}",
                'KL Loss': f"{kl_loss.item() / batch_size:.4f}"
            })
    
    # Calculate average losses
    num_samples = len(val_loader.dataset)
    val_losses = {k: v / num_samples for k, v in val_losses.items()}
    
    return val_losses


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename=None):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
        filename: Custom filename (optional)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:03d}.pth"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'latent_dim': model.latent_dim,
            'model_type': type(model).__name__
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple of (epoch, loss) from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch} with loss {loss:.4f}")
    
    return epoch, loss


class VAETrainer:
    """
    Trainer class for VAE models with built-in logging and checkpointing
    """
    
    def __init__(self, model, train_loader, val_loader, device, 
                 lr=1e-3, beta=1.0, checkpoint_dir='./checkpoints', 
                 log_dir='./logs', save_interval=10):
        """
        Initialize VAE trainer
        
        Args:
            model: VAE model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            lr: Learning rate
            beta: Beta parameter for β-VAE
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            save_interval: Interval for saving checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.beta = beta
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        
        # Setup optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Setup logging
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.history = {
            'train_loss': {'total': [], 'recon': [], 'kl': []},
            'val_loss': {'total': [], 'recon': [], 'kl': []}
        }
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
    
    def train(self, num_epochs, resume_from=None):
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from (optional)
        """
        if resume_from:
            self.start_epoch, _ = load_checkpoint(
                self.model, self.optimizer, resume_from, self.device
            )
        
        print(f"Training VAE for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Beta (KL weight): {self.beta}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Latent dimension: {self.model.latent_dim}")
        print("-" * 50)
        
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            start_time = time.time()
            
            # Training
            train_losses = train_epoch(
                self.model, self.train_loader, self.optimizer, 
                self.device, self.beta
            )
            
            # Validation
            val_losses = validate_epoch(
                self.model, self.val_loader, self.device, self.beta
            )
            
            # Update history
            for key in train_losses:
                self.history['train_loss'][key].append(train_losses[key])
                self.history['val_loss'][key].append(val_losses[key])
            
            # Logging
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.start_epoch + num_epochs}")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss - Total: {train_losses['total']:.4f}, "
                  f"Recon: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f}")
            print(f"Val Loss   - Total: {val_losses['total']:.4f}, "
                  f"Recon: {val_losses['recon']:.4f}, KL: {val_losses['kl']:.4f}")
            
            # TensorBoard logging
            for key in train_losses:
                self.writer.add_scalar(f'Loss/Train_{key}', train_losses[key], epoch)
                self.writer.add_scalar(f'Loss/Val_{key}', val_losses[key], epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, 
                    val_losses['total'], self.checkpoint_dir
                )
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, 
                    val_losses['total'], self.checkpoint_dir, 
                    filename='best_model.pth'
                )
                print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
        
        self.writer.close()
        print("\nTraining completed!")
        
        # Save final checkpoint
        save_checkpoint(
            self.model, self.optimizer, self.start_epoch + num_epochs, 
            val_losses['total'], self.checkpoint_dir, 
            filename='final_model.pth'
        )
    
    def get_history(self):
        """Get training history"""
        return self.history 