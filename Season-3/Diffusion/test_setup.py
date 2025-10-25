"""
Test script to verify the setup and installation.

This script checks:
1. All dependencies are installed correctly
2. CUDA is available
3. Model can be instantiated
4. Data loader works
5. Forward pass works
6. Multi-GPU setup is correct

Usage:
    python test_setup.py
"""
import sys
import torch
import torch.nn as nn


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    required_packages = [
        ('torch', torch),
        ('torchvision', 'torchvision'),
        ('datasets', 'datasets'),
        ('PIL', 'PIL'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
        ('numpy', 'numpy'),
    ]

    all_success = True
    for name, package in required_packages:
        try:
            if isinstance(package, str):
                __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            all_success = False

    # Optional packages
    optional_packages = [
        ('tensorboard', 'tensorboard'),
        ('torchdiffeq', 'torchdiffeq'),
    ]

    for name, package in optional_packages:
        try:
            __import__(package)
            print(f"  ✓ {name} (optional)")
        except ImportError:
            print(f"  ○ {name} (optional, not installed)")

    if not all_success:
        print("\n❌ Some required packages are missing. Please run: pip install -r requirements.txt")
        return False

    print("✓ All required packages imported successfully\n")
    return True


def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")

    if not torch.cuda.is_available():
        print("  ⚠ CUDA is not available. Training will be slow on CPU.")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"  ✓ CUDA is available")
    print(f"  ✓ Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {props.name}")
        print(f"      Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"      Compute Capability: {props.major}.{props.minor}")

    if num_gpus < 2:
        print("  ⚠ Only 1 GPU detected. Multi-GPU training will not be available.")

    print()
    return True


def test_model():
    """Test model instantiation."""
    print("Testing model instantiation...")

    try:
        from config import FlowMatchingConfig
        from model import UNet, FlowMatching

        # Create config
        config = FlowMatchingConfig(
            image_size=64,
            image_channels=3,
            model_channels=64,  # Smaller for testing
            channel_mult=(1, 2, 2),
            num_res_blocks=1,
            attention_resolutions=(16,),
            dropout=0.1
        )

        # Create model
        unet = UNet(
            image_channels=config.image_channels,
            model_channels=config.model_channels,
            channel_mult=config.channel_mult,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            dropout=config.dropout
        )

        flow_model = FlowMatching(
            model=unet,
            sigma_min=config.sigma_min
        )

        # Count parameters
        total_params = sum(p.numel() for p in unet.parameters())
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Total parameters: {total_params:,}")

        print()
        return True, flow_model, config

    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False, None, None


def test_forward_pass(flow_model, config):
    """Test forward pass."""
    print("Testing forward pass...")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        flow_model = flow_model.to(device)

        # Create dummy batch
        batch_size = 4
        x = torch.randn(
            batch_size,
            config.image_channels,
            config.image_size,
            config.image_size,
            device=device
        )

        # Forward pass
        with torch.no_grad():
            loss, info = flow_model(x)

        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Loss: {loss.item():.4f}")
        print(f"  ✓ Device: {device}")

        print()
        return True

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_sampling(flow_model, config):
    """Test sampling."""
    print("Testing sampling...")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        flow_model = flow_model.to(device)
        flow_model.eval()

        # Generate a few samples
        samples = flow_model.sample(
            batch_size=2,
            image_shape=(
                config.image_channels,
                config.image_size,
                config.image_size
            ),
            num_steps=10,  # Few steps for testing
            device=device,
            solver='euler',
            verbose=False
        )

        print(f"  ✓ Sampling successful")
        print(f"  ✓ Sample shape: {samples.shape}")
        print(f"  ✓ Sample range: [{samples.min():.2f}, {samples.max():.2f}]")

        print()
        return True

    except Exception as e:
        print(f"  ✗ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_dataloader():
    """Test data loader (quick check without downloading full dataset)."""
    print("Testing data loader setup...")

    try:
        from loader import get_celeba_dataloader

        print("  ℹ Note: Full dataset download will happen during training")
        print("  ✓ Data loader module imported successfully")

        print()
        return True

    except Exception as e:
        print(f"  ✗ Failed to import data loader: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_distributed():
    """Test distributed training setup."""
    print("Testing distributed training setup...")

    try:
        import torch.distributed as dist

        if not dist.is_nccl_available():
            print("  ⚠ NCCL backend not available. Multi-GPU training may not work.")
            return False

        print("  ✓ NCCL backend is available")
        print("  ✓ Distributed training supported")

        print()
        return True

    except Exception as e:
        print(f"  ✗ Distributed setup test failed: {e}")
        print()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Flow Matching Setup Test")
    print("=" * 80)
    print()

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test CUDA
    results.append(("CUDA", test_cuda()))

    # Test model
    success, flow_model, config = test_model()
    results.append(("Model Creation", success))

    if success:
        # Test forward pass
        results.append(("Forward Pass", test_forward_pass(flow_model, config)))

        # Test sampling
        results.append(("Sampling", test_sampling(flow_model, config)))

    # Test data loader
    results.append(("Data Loader", test_dataloader()))

    # Test distributed
    results.append(("Distributed Training", test_distributed()))

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("✓ All tests passed! Your setup is ready for training.")
        print("\nTo start training:")
        print("  Single GPU:  python train.py")
        print("  Multi-GPU:   torchrun --nproc_per_node=2 train.py")
    else:
        print("✗ Some tests failed. Please fix the issues before training.")
        sys.exit(1)


if __name__ == '__main__':
    main()
