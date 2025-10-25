"""
Quick test to verify CelebA dataset can be loaded.

Usage:
    python test_dataset.py
"""
import torch
from loader import get_celeba_dataloader


def test_dataset():
    """Test CelebA dataset loading."""
    print("=" * 70)
    print("Testing CelebA Dataset Loading")
    print("=" * 70)
    print()

    try:
        # Try to load dataset
        print("Creating dataloader...")
        dataloader = get_celeba_dataloader(
            data_dir='./data',
            split='train',
            batch_size=4,
            image_size=64,
            num_workers=0,
            shuffle=False
        )

        print(f"\n✓ Dataloader created successfully!")
        print(f"  Total batches: {len(dataloader):,}")
        print(f"  Images per batch: 4")
        print(f"  Total images: ~{len(dataloader) * 4:,}")
        print()

        # Test loading one batch
        print("Loading first batch...")
        batch = next(iter(dataloader))

        print(f"✓ Batch loaded successfully!")
        print(f"  Shape: {batch.shape}")
        print(f"  Data type: {batch.dtype}")
        print(f"  Value range: [{batch.min():.3f}, {batch.max():.3f}]")
        print(f"  Expected range: [-1.0, 1.0]")
        print()

        # Verify properties
        assert batch.shape == (4, 3, 64, 64), f"Unexpected shape: {batch.shape}"
        assert batch.dtype == torch.float32, f"Unexpected dtype: {batch.dtype}"
        assert batch.min() >= -1.5 and batch.max() <= 1.5, f"Values out of expected range"

        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Dataset is ready for training!")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try: huggingface-cli login")
        print("  3. Check if HuggingFace is accessible")
        print("  4. See TROUBLESHOOTING.md for manual dataset setup")
        print()

        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_dataset()
    exit(0 if success else 1)
