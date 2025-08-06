import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CelebA
import urllib.request
import zipfile
from PIL import Image
import numpy as np


def get_mnist_loader(batch_size=128, train=True, download=True):
    """
    Get MNIST dataset loader
    
    Args:
        batch_size: Batch size for training
        train: Whether to load training or test set
        download: Whether to download the dataset if not present
    
    Returns:
        DataLoader for MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Note: No normalization here as VAE expects values in [0,1]
    ])
    
    dataset = datasets.MNIST(
        root='./data/mnist',
        train=train,
        download=download,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )
    
    return loader


def get_celeba_loader(batch_size=64, train=True, download=True, data_root='./data/celeba'):
    """
    Get CelebA dataset loader
    
    Args:
        batch_size: Batch size for training
        train: Whether to load training or test split
        download: Whether to download the dataset if not present
        data_root: Root directory for CelebA data
    
    Returns:
        DataLoader for CelebA dataset
    """
    # CelebA preprocessing: resize to 64x64 and normalize to [0,1]
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        # Note: No normalization here as VAE expects values in [0,1]
    ])
    
    try:
        # Try to use torchvision's CelebA dataset
        split = 'train' if train else 'test'
        dataset = CelebA(
            root=data_root,
            split=split,
            download=download,
            transform=transform
        )
    except Exception as e:
        print(f"Error loading CelebA with torchvision: {e}")
        print("Please download CelebA dataset manually and place it in ./data/celeba/")
        print("You can download it from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        
        # Create a dummy dataset for demonstration
        print("Creating a dummy dataset for demonstration purposes...")
        dataset = DummyCelebA(transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )
    
    return loader


class DummyCelebA(torch.utils.data.Dataset):
    """Dummy CelebA dataset for demonstration when real dataset is not available"""
    
    def __init__(self, size=1000, transform=None):
        self.size = size
        self.transform = transform
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random RGB image as PIL Image (values 0-255)
        img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        if self.transform:
            img = self.transform(img)
        return img, 0  # Return dummy label


def download_celeba_manual(data_root='./data/celeba'):
    """
    Manual download function for CelebA dataset
    Note: This is a placeholder - actual CelebA requires manual download due to licensing
    """
    print("CelebA dataset requires manual download due to licensing restrictions.")
    print("Please visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("Download the following files:")
    print("- img_align_celeba.zip (images)")
    print("- list_eval_partition.txt (train/val/test split)")
    print(f"Extract to: {data_root}")
    
    os.makedirs(data_root, exist_ok=True)
    
    # Create instruction file
    with open(os.path.join(data_root, 'DOWNLOAD_INSTRUCTIONS.txt'), 'w') as f:
        f.write("CelebA Dataset Download Instructions\n")
        f.write("====================================\n\n")
        f.write("1. Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html\n")
        f.write("2. Download 'Align&Cropped Images' (img_align_celeba.zip)\n")
        f.write("3. Download 'Evaluation Partitions' (list_eval_partition.txt)\n")
        f.write("4. Extract img_align_celeba.zip to this directory\n")
        f.write("5. Place list_eval_partition.txt in this directory\n")
        f.write("\nExpected structure:\n")
        f.write("./data/celeba/\n")
        f.write("├── img_align_celeba/\n")
        f.write("│   ├── 000001.jpg\n")
        f.write("│   ├── 000002.jpg\n")
        f.write("│   └── ...\n")
        f.write("└── list_eval_partition.txt\n")


if __name__ == "__main__":
    # Test the data loaders
    print("Testing MNIST loader...")
    mnist_loader = get_mnist_loader(batch_size=32, train=True)
    for batch_idx, (data, target) in enumerate(mnist_loader):
        print(f"MNIST batch {batch_idx}: {data.shape}, min={data.min():.3f}, max={data.max():.3f}")
        if batch_idx >= 2:
            break
    
    print("\nTesting CelebA loader...")
    celeba_loader = get_celeba_loader(batch_size=32, train=True)
    for batch_idx, (data, target) in enumerate(celeba_loader):
        print(f"CelebA batch {batch_idx}: {data.shape}, min={data.min():.3f}, max={data.max():.3f}")
        if batch_idx >= 2:
            break 