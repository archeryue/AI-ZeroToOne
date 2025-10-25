"""
Visualization utilities for generated samples.
"""
import os
import torch
from torchvision.utils import save_image, make_grid
from typing import Optional


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = False,
    value_range: Optional[tuple] = None,
    padding: int = 2
):
    """
    Save a grid of images.

    Args:
        images: Tensor of shape (B, C, H, W)
        save_path: Path to save the image
        nrow: Number of images per row
        normalize: Whether to normalize images
        value_range: Range for normalization (min, max)
        padding: Padding between images
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save grid
    save_image(
        images,
        save_path,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=padding
    )


def make_image_grid(
    images: torch.Tensor,
    nrow: int = 8,
    normalize: bool = False,
    value_range: Optional[tuple] = None,
    padding: int = 2
) -> torch.Tensor:
    """
    Create a grid of images.

    Args:
        images: Tensor of shape (B, C, H, W)
        nrow: Number of images per row
        normalize: Whether to normalize images
        value_range: Range for normalization (min, max)
        padding: Padding between images

    Returns:
        Grid tensor of shape (C, H, W)
    """
    return make_grid(
        images,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=padding
    )
