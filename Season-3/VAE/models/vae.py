import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Base Variational Autoencoder class"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
    def encode(self, x):
        """Encode input to latent space parameters"""
        raise NotImplementedError
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var) using N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        raise NotImplementedError
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def sample(self, num_samples, device):
        """Generate samples from the learned distribution"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples


class VAEMnist(VAE):
    """VAE for MNIST dataset (28x28 grayscale images)"""
    
    def __init__(self, latent_dim=20):
        super(VAEMnist, self).__init__(input_dim=784, hidden_dim=400, latent_dim=latent_dim)
        
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # mu
        self.fc22 = nn.Linear(400, latent_dim)  # logvar
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)
        
    def encode(self, x):
        x = x.view(-1, 784)  # Flatten
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class VAECelebA(VAE):
    """VAE for CelebA dataset (64x64 RGB images)"""
    
    def __init__(self, latent_dim=128):
        super(VAECelebA, self).__init__(input_dim=64*64*3, hidden_dim=None, latent_dim=latent_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 32, 4, 2, 1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, 4, 2, 1),  # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 256, 4, 2, 1),  # 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(True)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # 3 x 64 x 64
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction loss and KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (β-VAE)
    """
    # Reconstruction loss (binary cross entropy for MNIST, MSE for CelebA)
    if len(x.shape) == 2 or (len(x.shape) == 4 and x.shape[1] == 1):
        # MNIST case
        recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    else:
        # CelebA case
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss 