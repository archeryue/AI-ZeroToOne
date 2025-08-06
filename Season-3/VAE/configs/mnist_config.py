"""
Configuration for MNIST VAE training
"""
import torch

# Model parameters
LATENT_DIM = 20
MODEL_TYPE = 'VAEMnist'

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
BETA = 1.0  # Weight for KL divergence term

# Data parameters
NUM_WORKERS = 4

# Logging and checkpointing
LOG_INTERVAL = 100
SAVE_INTERVAL = 10
CHECKPOINT_DIR = './checkpoints/mnist'
LOG_DIR = './logs/mnist'
SAMPLES_DIR = './samples/mnist'

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import torch for device check
import torch 