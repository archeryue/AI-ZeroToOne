"""
Generate samples from a trained Flow Matching model.

Usage:
    python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pth --num-samples 64
"""
import argparse
import torch
import os
from config import FlowMatchingConfig
from model import UNet, FlowMatching
from utils import save_image_grid


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        flow_model: Loaded FlowMatching model
        model_config: Model configuration
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model config
    model_config_dict = checkpoint['model_config']
    model_config = FlowMatchingConfig(**model_config_dict)

    # Create model
    unet = UNet(
        image_channels=model_config.image_channels,
        model_channels=model_config.model_channels,
        channel_mult=model_config.channel_mult,
        num_res_blocks=model_config.num_res_blocks,
        attention_resolutions=model_config.attention_resolutions,
        dropout=model_config.dropout
    )

    flow_model = FlowMatching(
        model=unet,
        sigma_min=model_config.sigma_min
    )

    # Load weights - handle both EMA and regular model
    if 'ema_state_dict' in checkpoint and model_config.use_ema:
        print("Loading EMA model weights...")
        unet.load_state_dict(checkpoint['ema_state_dict'])
    else:
        print("Loading model weights...")
        # Handle DDP wrapper
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        unet.load_state_dict(state_dict)

    flow_model = flow_model.to(device)
    flow_model.eval()

    print(f"Model loaded from step {checkpoint.get('global_step', 'unknown')}, "
          f"epoch {checkpoint.get('epoch', 'unknown')}")

    return flow_model, model_config


@torch.no_grad()
def generate_samples(
    flow_model: FlowMatching,
    model_config: FlowMatchingConfig,
    num_samples: int,
    num_steps: int,
    device: str,
    solver: str = 'midpoint',
    seed: int = None
):
    """
    Generate samples from the model.

    Args:
        flow_model: FlowMatching model
        model_config: Model configuration
        num_samples: Number of samples to generate
        num_steps: Number of ODE solver steps
        device: Device to use
        solver: ODE solver ('euler' or 'midpoint')
        seed: Random seed for reproducibility

    Returns:
        Generated samples tensor (B, C, H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    print(f"\nGenerating {num_samples} samples with {num_steps} steps using {solver} solver...")

    samples = flow_model.sample(
        batch_size=num_samples,
        image_shape=(
            model_config.image_channels,
            model_config.image_size,
            model_config.image_size
        ),
        num_steps=num_steps,
        device=device,
        solver=solver,
        verbose=True
    )

    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    return samples


def main(args):
    """Main generation function."""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Load model
    flow_model, model_config = load_model(args.checkpoint, device)

    # Generate samples
    samples = generate_samples(
        flow_model=flow_model,
        model_config=model_config,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device,
        solver=args.solver,
        seed=args.seed
    )

    # Save samples
    os.makedirs(args.output_dir, exist_ok=True)

    # Save as grid
    grid_path = os.path.join(args.output_dir, args.output_name)
    save_image_grid(samples, grid_path, nrow=args.nrow)
    print(f"\nSaved sample grid to: {grid_path}")

    # Optionally save individual images
    if args.save_individual:
        individual_dir = os.path.join(args.output_dir, 'individual')
        os.makedirs(individual_dir, exist_ok=True)

        for i, img in enumerate(samples):
            save_path = os.path.join(individual_dir, f'sample_{i:04d}.png')
            save_image_grid(img.unsqueeze(0), save_path, nrow=1)

        print(f"Saved {len(samples)} individual images to: {individual_dir}")

    print("\nGeneration completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples from trained Flow Matching model')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')

    # Generation parameters
    parser.add_argument('--num-samples', type=int, default=64, help='Number of samples to generate')
    parser.add_argument('--num-steps', type=int, default=50, help='Number of ODE solver steps')
    parser.add_argument('--solver', type=str, default='midpoint', choices=['euler', 'midpoint'],
                        help='ODE solver to use')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    # Output
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--output-name', type=str, default='generated_samples.png', help='Output filename')
    parser.add_argument('--nrow', type=int, default=8, help='Number of images per row in grid')
    parser.add_argument('--save-individual', action='store_true', help='Save individual images')

    # Device
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')

    args = parser.parse_args()

    main(args)
