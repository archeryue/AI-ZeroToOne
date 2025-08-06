"""
Configuration for CelebA VAE training
"""
import torch

# Model parameters
LATENT_DIM = 128
MODEL_TYPE = 'VAECelebA'

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BETA = 1.0  # Weight for KL divergence term

# Data parameters
DOWNLOAD_DATA = True
NUM_WORKERS = 4
DATA_ROOT = './data/celeba'

# Logging and checkpointing
LOG_INTERVAL = 100
SAVE_INTERVAL = 10
CHECKPOINT_DIR = './checkpoints/celeba'
LOG_DIR = './logs/celeba'
SAMPLES_DIR = './samples/celeba'

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import torch for device check
import torch 