"""
CelebA dataset loader for flow matching training.
"""
import os
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


class CelebADataset(Dataset):
    """CelebA dataset wrapper for flow matching."""

    def __init__(
        self,
        data_dir: str = './data',
        split: str = 'train',
        image_size: int = 64,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_dir: Directory to store/cache dataset
            split: 'train' or 'validation'
            image_size: Target image size (will be resized)
            cache_dir: Cache directory for HuggingFace datasets
        """
        self.image_size = image_size
        self.split = split

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(data_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        # Load dataset from HuggingFace with fallback options
        print(f"Loading CelebA dataset ({split} split)...")
        try:
            self.dataset = load_dataset(
                'nielsr/CelebA-faces',
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            print(f"Successfully loaded {len(self.dataset)} images from nielsr/CelebA-faces")
        except Exception as e:
            print(f"Failed to load from nielsr/CelebA-faces: {e}")
            print("Trying alternative source...")
            try:
                # Alternative source
                self.dataset = load_dataset(
                    'huggan/CelebA-faces',
                    split=split,
                    cache_dir=cache_dir
                )
                print(f"Successfully loaded {len(self.dataset)} images from huggan/CelebA-faces")
            except Exception as e2:
                print(f"Failed to load from huggan/CelebA-faces: {e2}")
                raise RuntimeError(
                    "Could not load CelebA dataset from any source. "
                    "Please check your internet connection and HuggingFace access."
                )

        # Define transforms: normalize to [-1, 1] for flow matching
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            Image tensor of shape (C, H, W) normalized to [-1, 1]
        """
        item = self.dataset[idx]

        # Handle different key names in dataset
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            raise KeyError(f"Could not find image in dataset item. Available keys: {item.keys()}")

        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return self.transform(image)


def get_celeba_dataloader(
    data_dir: str = './data',
    split: str = 'train',
    batch_size: int = 32,
    image_size: int = 64,
    num_workers: int = 4,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Get CelebA dataloader with optional distributed sampling.

    Args:
        data_dir: Directory to store/cache dataset
        split: 'train' or 'validation'
        batch_size: Batch size per GPU
        image_size: Target image size
        num_workers: Number of data loading workers
        distributed: Whether to use distributed sampling
        world_size: Number of processes (GPUs)
        rank: Process rank
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader for CelebA dataset
    """
    dataset = CelebADataset(
        data_dir=data_dir,
        split=split,
        image_size=image_size
    )

    # Use DistributedSampler for multi-GPU training
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last
        )
        shuffle = False  # Sampler handles shuffling

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        sampler=sampler,
        persistent_workers=num_workers > 0
    )

    return dataloader
