#!/usr/bin/env python3
"""
Generate samples from a trained DCGAN+WGAN-GP model.

Usage:
    python generate.py --checkpoint checkpoints/final_checkpoint.pth --num-samples 64
"""

import argparse
import os
import torch
import torchvision.utils as vutils
from model.dcgan import Generator
from config.config import get_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate samples from trained DCGAN')
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to the trained model checkpoint'
    )
    
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=64,
        help='Number of samples to generate'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='generated_samples',
        help='Directory to save generated samples'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use for generation'
    )
    
    return parser.parse_args()


def load_generator(checkpoint_path, device):
    """Load generator from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create generator
    generator = Generator(config)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    
    print(f"Generator loaded successfully from epoch {checkpoint['epoch']}")
    return generator, config


def generate_samples(generator, config, num_samples, device, output_dir):
    """Generate and save samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} samples...")
    
    with torch.no_grad():
        # Generate noise
        noise = torch.randn(num_samples, config.latent_dim, 1, 1, device=device)
        
        # Generate images
        fake_images = generator(noise)
        
        # Denormalize images (from [-1, 1] to [0, 1])
        fake_images = (fake_images + 1) / 2
        
        # Save individual images
        for i, img in enumerate(fake_images):
            save_path = os.path.join(output_dir, f'sample_{i:04d}.png')
            vutils.save_image(img, save_path)
        
        # Save grid of images
        grid = vutils.make_grid(fake_images, nrow=8, padding=2, normalize=False)
        grid_path = os.path.join(output_dir, 'samples_grid.png')
        vutils.save_image(grid, grid_path)
        
        print(f"Samples saved to: {output_dir}")
        print(f"Grid saved to: {grid_path}")


def main():
    """Main generation function."""
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    try:
        # Load generator
        generator, config = load_generator(args.checkpoint, device)
        
        # Generate samples
        generate_samples(generator, config, args.num_samples, device, args.output_dir)
        
        print("Generation completed successfully!")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
