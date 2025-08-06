# VAE Content Generation Project

This project implements Variational Autoencoders (VAE) for generating content using two different datasets:
- MNIST hand-written digits
- CelebA face images

## Features

- VAE implementation with configurable architecture
- Support for MNIST and CelebA datasets
- Training with proper VAE loss (reconstruction + KL divergence)
- Visualization utilities for generated samples
- TensorBoard logging for training monitoring

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train on MNIST:
```bash
python train_mnist.py
```

3. Train on CelebA:
```bash
python train_celeba.py
```

## Project Structure

- `models/`: VAE model implementations
- `data/`: Dataset loading and preprocessing
- `utils/`: Utility functions for training and visualization
- `configs/`: Configuration files for different experiments
- `checkpoints/`: Saved model checkpoints
- `logs/`: TensorBoard logs
- `samples/`: Generated sample images

## Model Architecture

The VAE consists of:
- Encoder: Maps input images to latent space (mean and log variance)
- Decoder: Reconstructs images from latent representations
- Reparameterization trick for backpropagation through stochastic sampling 