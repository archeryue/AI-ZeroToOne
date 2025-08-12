import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Any
from config.config import DataConfig, ModelConfig


class CelebADataset(Dataset):
    """CelebA dataset from HuggingFace."""
    
    def __init__(self, config: DataConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        
        # Load dataset from HuggingFace
        print(f"Loading {config.dataset_name} dataset from HuggingFace...")
        try:
            # Try to load CelebA-HQ first, fallback to regular CelebA
            self.dataset = load_dataset(
                "nielsr/CelebA-faces", 
                split=config.split,
                streaming=config.streaming,
                cache_dir=config.cache_dir
            )
        except Exception as e:
            print(f"Failed to load CelebA-HQ, trying regular CelebA: {e}")
            try:
                self.dataset = load_dataset(
                    "huggan/CelebA-faces",
                    split=config.split,
                    streaming=config.streaming,
                    cache_dir=config.cache_dir
                )
            except Exception as e2:
                print(f"Failed to load regular CelebA, using alternative: {e2}")
                # Fallback to another CelebA dataset
                self.dataset = load_dataset(
                    "yuvalkirstain/celeba_hq_256",
                    split="train",
                    streaming=config.streaming,
                    cache_dir=config.cache_dir
                )
        
        print(f"Dataset loaded successfully!")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((model_config.image_size, model_config.image_size)),
            transforms.CenterCrop(model_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std
            )
        ])
    
    def __len__(self) -> int:
        if hasattr(self.dataset, '__len__'):
            return len(self.dataset)
        else:
            # For streaming datasets, return a large number
            return 200000  # Approximate size of CelebA
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            # Get item from dataset
            if self.config.streaming:
                # For streaming datasets, we need to iterate
                for i, item in enumerate(self.dataset):
                    if i == idx:
                        break
            else:
                item = self.dataset[idx]
            
            # Extract image
            if 'image' in item:
                image = item['image']
            elif 'img' in item:
                image = item['img']
            else:
                # Try to get the first available image key
                image_keys = [k for k in item.keys() if 'image' in k.lower() or 'img' in k.lower()]
                if image_keys:
                    image = item[image_keys[0]]
                else:
                    raise ValueError(f"No image found in dataset item. Available keys: {item.keys()}")
            
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                # Handle other formats
                image = Image.fromarray(np.array(image))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image = self.transform(image)
            
            return image
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a random image as fallback
            random_image = torch.randn(3, self.model_config.image_size, self.model_config.image_size)
            return random_image


def get_dataloader(config: DataConfig, model_config: ModelConfig) -> DataLoader:
    """Create and return a DataLoader for the CelebA dataset."""
    
    dataset = CelebADataset(config, model_config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=model_config.num_workers,
        pin_memory=True if model_config.device == "cuda" else False,
        drop_last=True
    )
    
    return dataloader


def test_dataloader(dataloader: DataLoader, num_batches: int = 3) -> None:
    """Test the dataloader by loading a few batches."""
    print("Testing dataloader...")
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        print(f"Batch {i+1}: shape = {batch.shape}, dtype = {batch.dtype}")
        print(f"  Min value: {batch.min():.3f}, Max value: {batch.max():.3f}")
        print(f"  Mean: {batch.mean():.3f}, Std: {batch.std():.3f}")
    
    print("Dataloader test completed!")
