import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config.config import ModelConfig


def weights_init(m):
    """Initialize network weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """DCGAN Generator."""
    
    def __init__(self, config: ModelConfig):
        super(Generator, self).__init__()
        self.config = config
        
        # Calculate the initial size after first conv transpose
        self.init_size = config.image_size // 8  # 8x8 for 64x64 images
        
        self.main = nn.Sequential(
            # Input is latent vector, going into a convolution
            # state size: (latent_dim) x 1 x 1
            nn.ConvTranspose2d(
                config.latent_dim, 
                config.gen_features * 8, 
                4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(config.gen_features * 8),
            nn.ReLU(True),
            # state size: (gen_features*8) x 4 x 4
            
            nn.ConvTranspose2d(
                config.gen_features * 8, 
                config.gen_features * 4, 
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.gen_features * 4),
            nn.ReLU(True),
            # state size: (gen_features*4) x 8 x 8
            
            nn.ConvTranspose2d(
                config.gen_features * 4, 
                config.gen_features * 2, 
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.gen_features * 2),
            nn.ReLU(True),
            # state size: (gen_features*2) x 16 x 16
            
            nn.ConvTranspose2d(
                config.gen_features * 2, 
                config.gen_features, 
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.gen_features),
            nn.ReLU(True),
            # state size: (gen_features) x 32 x 32
            
            nn.ConvTranspose2d(
                config.gen_features, 
                config.channels, 
                4, 2, 1, bias=False
            ),
            nn.Tanh()
            # state size: (channels) x 64 x 64
        )
        
        # Initialize weights
        self.apply(weights_init)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Forward pass of the generator.
        
        Args:
            noise: Random noise tensor of shape (batch_size, latent_dim, 1, 1)
            
        Returns:
            Generated images of shape (batch_size, channels, image_size, image_size)
        """
        return self.main(noise)


class Discriminator(nn.Module):
    """DCGAN Discriminator (Critic for WGAN-GP)."""
    
    def __init__(self, config: ModelConfig):
        super(Discriminator, self).__init__()
        self.config = config
        
        self.main = nn.Sequential(
            # Input is (channels) x 64 x 64
            nn.Conv2d(
                config.channels, 
                config.disc_features, 
                4, 2, 1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (disc_features) x 32 x 32
            
            nn.Conv2d(
                config.disc_features, 
                config.disc_features * 2, 
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (disc_features*2) x 16 x 16
            
            nn.Conv2d(
                config.disc_features * 2, 
                config.disc_features * 4, 
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (disc_features*4) x 8 x 8
            
            nn.Conv2d(
                config.disc_features * 4, 
                config.disc_features * 8, 
                4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (disc_features*8) x 4 x 4
            
            nn.Conv2d(
                config.disc_features * 8, 
                1, 
                4, 1, 0, bias=False
            ),
            # state size: 1 x 1 x 1
        )
        
        # Initialize weights
        self.apply(weights_init)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator.
        
        Args:
            img: Input images of shape (batch_size, channels, image_size, image_size)
            
        Returns:
            Critic scores of shape (batch_size, 1, 1, 1)
        """
        return self.main(img)


def gradient_penalty(discriminator: nn.Module, real_samples: torch.Tensor, 
                    fake_samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP.
    
    Args:
        discriminator: The discriminator network
        real_samples: Real images
        fake_samples: Generated images  
        device: Device to run computation on
        
    Returns:
        Gradient penalty term
    """
    batch_size = real_samples.size(0)
    
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Calculate discriminator output for interpolated samples
    d_interpolates = discriminator(interpolates)
    
    # Calculate gradients w.r.t. interpolated samples
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def create_models(config: ModelConfig) -> Tuple[Generator, Discriminator]:
    """Create and return generator and discriminator models.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (generator, discriminator)
    """
    generator = Generator(config)
    discriminator = Discriminator(config)
    
    return generator, discriminator


def test_models(config: ModelConfig) -> None:
    """Test the models with dummy data."""
    print("Testing DCGAN models...")
    
    # Create models
    generator, discriminator = create_models(config)
    
    # Test generator
    noise = torch.randn(4, config.latent_dim, 1, 1)
    fake_images = generator(noise)
    print(f"Generator output shape: {fake_images.shape}")
    
    # Test discriminator
    real_images = torch.randn(4, config.channels, config.image_size, config.image_size)
    real_output = discriminator(real_images)
    fake_output = discriminator(fake_images.detach())
    
    print(f"Discriminator real output shape: {real_output.shape}")
    print(f"Discriminator fake output shape: {fake_output.shape}")
    
    # Test gradient penalty
    gp = gradient_penalty(discriminator, real_images, fake_images.detach(), torch.device('cpu'))
    print(f"Gradient penalty: {gp.item():.4f}")
    
    print("Model testing completed!")
