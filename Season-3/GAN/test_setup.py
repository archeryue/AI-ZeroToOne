#!/usr/bin/env python3
"""
Quick test script to verify the project setup works correctly.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config.config import get_config, get_data_config
        print("✓ Config module imported successfully")
        
        from loader.dataset import get_dataloader, CelebADataset
        print("✓ Loader module imported successfully")
        
        from model.dcgan import Generator, Discriminator, create_models
        print("✓ Model module imported successfully")
        
        from trainer.trainer import WGANGPTrainer
        print("✓ Trainer module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config.config import get_config, get_data_config
        
        config = get_config()
        data_config = get_data_config()
        
        print(f"✓ Model config loaded: {config.image_size}x{config.image_size}, latent_dim={config.latent_dim}")
        print(f"✓ Data config loaded: dataset={data_config.dataset_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False


def test_models():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from model.dcgan import test_models
        from config.config import get_config
        
        config = get_config()
        test_models(config)
        
        print("✓ Models created and tested successfully")
        return True
        
    except Exception as e:
        print(f"✗ Model test error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("DCGAN+WGAN-GP Project Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_models
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} tests passed!")
        print("\nYour project is ready for training!")
        print("Run: python training.py --test-only")
        return 0
    else:
        print(f"✗ {total - passed} out of {total} tests failed")
        print("\nPlease fix the issues before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
