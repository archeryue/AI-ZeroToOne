# DCGAN + WGAN-GP for Human Face Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) with Wasserstein GAN with Gradient Penalty (WGAN-GP) loss to generate realistic human faces using the CelebA dataset from HuggingFace.

## 🎯 Features

- **DCGAN Architecture**: State-of-the-art convolutional generator and discriminator
- **WGAN-GP Loss**: Improved training stability and convergence
- **CelebA Dataset**: Automatic download and preprocessing from HuggingFace
- **Comprehensive Logging**: TensorBoard integration and loss plotting
- **Checkpoint System**: Save and resume training from any point
- **Modular Design**: Clean, extensible codebase structure

## 📁 Project Structure

```
GAN/
├── training.py              # Main training script
├── requirements.txt         # Python dependencies
├── config/
│   ├── __init__.py
│   └── config.py           # Configuration classes
├── loader/
│   ├── __init__.py
│   └── dataset.py          # Dataset loading and preprocessing
├── model/
│   ├── __init__.py
│   └── dcgan.py            # Generator, Discriminator, and WGAN-GP loss
├── trainer/
│   ├── __init__.py
│   └── trainer.py          # Training loop and utilities
├── checkpoints/            # Model checkpoints
├── logs/                   # TensorBoard logs
├── results/                # Generated samples
└── data/                   # Dataset cache
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd GAN

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Training

```bash
# Start training with default configuration
python training.py

# Monitor training progress with TensorBoard
tensorboard --logdir logs
```

### 3. Advanced Usage

```bash
# Test models and dataloader without training
python training.py --test-only

# Custom training parameters
python training.py --epochs 200 --batch-size 32 --learning-rate 0.0001

# Resume training from checkpoint
python training.py --resume checkpoints/checkpoint_epoch_50.pth

# Force CPU usage
python training.py --device cpu
```

## ⚙️ Configuration

The model configuration can be customized in `config/config.py`:

### Model Parameters
- `image_size`: Output image resolution (default: 64x64)
- `latent_dim`: Noise vector dimension (default: 100)
- `gen_features`: Generator base feature channels (default: 64)
- `disc_features`: Discriminator base feature channels (default: 64)

### Training Parameters
- `batch_size`: Training batch size (default: 64)
- `num_epochs`: Total training epochs (default: 100)
- `learning_rate`: Adam optimizer learning rate (default: 0.0002)
- `critic_iterations`: Discriminator updates per generator update (default: 5)
- `gradient_penalty_lambda`: WGAN-GP penalty weight (default: 10.0)

### Logging Parameters
- `save_interval`: Checkpoint saving frequency (default: 10 epochs)
- `log_interval`: TensorBoard logging frequency (default: 100 steps)
- `sample_interval`: Sample generation frequency (default: 500 steps)

## 🏗️ Architecture

### Generator
- Input: Random noise vector (100D)
- Architecture: 5-layer transposed convolution
- Output: RGB images (3×64×64)
- Activation: ReLU (hidden), Tanh (output)

### Discriminator (Critic)
- Input: RGB images (3×64×64)
- Architecture: 5-layer convolution
- Output: Real/fake score
- Activation: LeakyReLU
- No sigmoid (WGAN requirement)

### WGAN-GP Loss
- **Generator Loss**: `-E[D(G(z))]`
- **Discriminator Loss**: `E[D(G(z))] - E[D(x)] + λ·GP`
- **Gradient Penalty**: Enforces 1-Lipschitz constraint

## 📊 Monitoring Training

### TensorBoard Metrics
- Generator and Discriminator losses
- Real/fake discriminator scores
- Gradient penalty values
- Generated image samples

### Generated Outputs
- Sample images saved every 500 steps
- Loss plots saved every 10 epochs
- Final results in `results/` directory

## 🎮 Dataset

The project uses the CelebA dataset with automatic fallback options:

1. **Primary**: `nielsr/CelebA-faces` (CelebA-HQ)
2. **Secondary**: `huggan/CelebA-faces` (Regular CelebA)
3. **Fallback**: `yuvalkirstain/celeba_hq_256` (Alternative CelebA-HQ)

The dataset is automatically downloaded and cached on first run.

## 💡 Tips for Better Results

### Training Stability
- Start with lower learning rates (0.0001-0.0002)
- Increase critic iterations if discriminator is too weak
- Monitor Wasserstein distance for convergence

### Quality Improvements
- Train for more epochs (200-500)
- Increase image resolution gradually
- Use spectral normalization for better stability
- Experiment with different architectures

### Common Issues
- **Mode Collapse**: Reduce learning rate, increase GP weight
- **Training Instability**: Increase critic iterations
- **Poor Quality**: Train longer, check data preprocessing

## 🔧 Customization

### Adding New Datasets
1. Create new dataset class in `loader/dataset.py`
2. Update `DataConfig` in `config/config.py`
3. Modify data loading logic as needed

### Model Modifications
1. Edit architectures in `model/dcgan.py`
2. Adjust configuration parameters
3. Update weight initialization if needed

### Training Changes
1. Modify training loop in `trainer/trainer.py`
2. Add new loss functions or metrics
3. Customize logging and visualization

## 📈 Expected Results

After successful training, you should see:

- **Convergence**: Generator and discriminator losses stabilizing
- **Quality**: Recognizable human faces with good diversity
- **Stability**: Consistent sample quality across epochs

Training typically takes:
- **CPU**: 10-20 hours for 100 epochs
- **GPU**: 2-4 hours for 100 epochs (depending on GPU)

## 🐛 Troubleshooting

### Common Errors

**CUDA Out of Memory**
```bash
# Reduce batch size
python training.py --batch-size 32

# Or use CPU
python training.py --device cpu
```

**Dataset Download Issues**
- Check internet connection
- Clear cache: `rm -rf data/cache`
- Try manual download from HuggingFace

**Training Divergence**
- Reduce learning rate: `--learning-rate 0.0001`
- Increase gradient penalty: Modify `gradient_penalty_lambda` in config

### Debug Mode
```bash
# Test without training
python training.py --test-only

# Short training run
python training.py --epochs 1
```

## 📚 References

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028)
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

**Happy Training! 🎉**