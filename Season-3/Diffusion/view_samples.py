"""
Simple viewer for generated samples.
"""
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

def view_samples(sample_dir='samples', max_images=10):
    """View generated sample images."""
    # Get all sample images sorted by step
    sample_files = sorted(glob.glob(os.path.join(sample_dir, 'samples_step_*.png')))

    if not sample_files:
        print(f"No samples found in {sample_dir}/")
        return

    # Limit to max_images
    sample_files = sample_files[:max_images]

    num_samples = len(sample_files)
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, sample_file in enumerate(sample_files):
        # Extract step number from filename
        step = os.path.basename(sample_file).replace('samples_step_', '').replace('.png', '')

        # Load and display image
        img = Image.open(sample_file)
        axes[idx].imshow(img)
        axes[idx].set_title(f'Step {step}')
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('training_samples_overview.png', dpi=150, bbox_inches='tight')
    print(f"Saved overview to: training_samples_overview.png")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='View generated samples')
    parser.add_argument('--sample-dir', type=str, default='samples', help='Directory with samples')
    parser.add_argument('--max-images', type=int, default=10, help='Maximum images to show')
    args = parser.parse_args()

    view_samples(args.sample_dir, args.max_images)
