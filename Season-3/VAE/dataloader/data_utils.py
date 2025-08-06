from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


def get_mnist_loader(batch_size=128, validation_split=0.1):
    """
    Get MNIST dataset loader from Hugging Face
    
    Args:
        batch_size: Batch size for the DataLoader
        train: If True, returns training data. If False, returns test data
        validation_split: Fraction of training data to use for validation (only used when train=True)
    
    Returns:
        DataLoader for MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load full training dataset and split into train/validation
    dataset = load_dataset("ylecun/mnist", split="train")
    total_size = len(dataset)
        
    # Split into train and validation
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
        
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, total_size))
        
    # Return both train and validation loaders
    train_loader = DataLoader(
        MNISTDataset(train_dataset, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
        
    val_loader = DataLoader(
        MNISTDataset(val_dataset, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
        
    return train_loader, val_loader


def get_celeba_loader(batch_size=64, validation_split=0.1, image_size=64):
    """
    Get CelebA dataset loader from Hugging Face
    
    Args:
        batch_size: Batch size for the DataLoader
        train: If True, returns training data. If False, returns test data
        validation_split: Fraction of training data to use for validation (only used when train=True)
        image_size: Size to resize images to (default 64x64)
    
    Returns:
        DataLoader for CelebA dataset
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    # Load training dataset and split into train/validation
    dataset = load_dataset("nielsr/CelebA-faces", split="train")
    total_size = len(dataset)
    
    # Split into train and validation
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, total_size))
    
    # Return both train and validation loaders
    train_loader = DataLoader(
        CelebADataset(train_dataset, transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        CelebADataset(val_dataset, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader

class MNISTDataset(Dataset):
    """Simple MNIST dataset wrapper"""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CelebADataset(Dataset):
    """Simple CelebA dataset wrapper"""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        # Return image with dummy label (VAE doesn't need labels)
        return image, 0