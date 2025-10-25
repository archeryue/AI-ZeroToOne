"""
Plot training metrics from TensorBoard logs.
"""
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import glob
import os


def plot_training_metrics(log_dir='logs', save_path='training_metrics.png'):
    """
    Plot training loss and learning rate from TensorBoard logs.

    Args:
        log_dir: Directory containing TensorBoard event files
        save_path: Path to save the plot
    """
    # Find the most recent event file
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))

    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}/")
        return

    # Use the most recent file
    event_file = max(event_files, key=os.path.getmtime)
    print(f"Reading from: {event_file}")

    # Load the event file
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Get available tags
    print(f"Available scalar tags: {ea.Tags()['scalars']}")

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot training loss
    if 'train/loss' in ea.Tags()['scalars']:
        loss_events = ea.Scalars('train/loss')
        steps = [e.step for e in loss_events]
        losses = [e.value for e in loss_events]

        axes[0].plot(steps, losses, linewidth=2, color='#2E86AB')
        axes[0].set_xlabel('Training Step', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Add statistics
        avg_loss = np.mean(losses)
        axes[0].axhline(avg_loss, color='red', linestyle='--', alpha=0.5, label=f'Average: {avg_loss:.4f}')
        axes[0].legend()

        print(f"Loss - Min: {min(losses):.4f}, Max: {max(losses):.4f}, Avg: {avg_loss:.4f}")

    # Plot learning rate
    if 'train/lr' in ea.Tags()['scalars']:
        lr_events = ea.Scalars('train/lr')
        steps = [e.step for e in lr_events]
        lrs = [e.value for e in lr_events]

        axes[1].plot(steps, lrs, linewidth=2, color='#A23B72')
        axes[1].set_xlabel('Training Step', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        print(f"Learning Rate - Min: {min(lrs):.6f}, Max: {max(lrs):.6f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    # Also create a simple text summary
    with open('training_summary.txt', 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 50 + "\n\n")

        if 'train/loss' in ea.Tags()['scalars']:
            loss_events = ea.Scalars('train/loss')
            losses = [e.value for e in loss_events]
            f.write(f"Total training steps: {len(losses)}\n")
            f.write(f"Loss - Min: {min(losses):.4f}\n")
            f.write(f"Loss - Max: {max(losses):.4f}\n")
            f.write(f"Loss - Average: {np.mean(losses):.4f}\n")
            f.write(f"Loss - Final: {losses[-1]:.4f}\n")

        f.write("\n")

        if 'train/lr' in ea.Tags()['scalars']:
            lr_events = ea.Scalars('train/lr')
            lrs = [e.value for e in lr_events]
            f.write(f"Learning Rate - Initial: {lrs[0]:.6f}\n")
            f.write(f"Learning Rate - Final: {lrs[-1]:.6f}\n")

    print("Summary saved to: training_summary.txt")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--log-dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--save-path', type=str, default='training_metrics.png', help='Output plot path')
    args = parser.parse_args()

    plot_training_metrics(args.log_dir, args.save_path)
