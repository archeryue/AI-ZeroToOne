"""
Benchmark different U-Net implementations for Flow Matching.

Compares:
1. Our custom U-Net (model/unet.py)
2. HuggingFace Diffusers UNet2DModel (if available)
3. Our U-Net with torch.compile() optimization
"""
import torch
import time
import sys
from config import FlowMatchingConfig
from model import UNet

# Try to import diffusers
try:
    from diffusers import UNet2DModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️  diffusers library not installed")
    print("   Install with: pip install diffusers")
    print()


def benchmark_model(model, name, num_iterations=50, warmup=5):
    """
    Benchmark a model.

    Args:
        model: Model to benchmark
        name: Name of the model
        num_iterations: Number of forward passes
        warmup: Number of warmup iterations

    Returns:
        Average time per forward pass in milliseconds
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    t = torch.rand(batch_size, device=device)

    # Warmup
    print(f"Warming up {name}...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, t)

    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking {name}...")
    times = []

    with torch.no_grad():
        for i in range(num_iterations):
            start = time.perf_counter()

            output = model(x, t)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{num_iterations} iterations...")

    avg_time = sum(times) / len(times)
    std_time = torch.tensor(times).std().item()

    # Handle different output types
    if hasattr(output, 'sample'):  # Diffusers output
        output_shape = output.sample.shape
    else:
        output_shape = output.shape

    return avg_time, std_time, output_shape


def create_our_unet():
    """Create our custom U-Net."""
    config = FlowMatchingConfig(
        image_size=64,
        image_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.0  # Disable for benchmarking
    )

    unet = UNet(
        image_channels=config.image_channels,
        model_channels=config.model_channels,
        channel_mult=config.channel_mult,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout
    )

    return unet, config


def create_diffusers_unet():
    """Create HuggingFace Diffusers U-Net."""
    if not DIFFUSERS_AVAILABLE:
        return None, None

    # Configure to match our architecture as closely as possible
    unet = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,  # Similar to num_res_blocks
        block_out_channels=(128, 256, 256, 512),  # Similar to our channel_mult
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    return unet, None


def create_compiled_unet():
    """Create our U-Net with torch.compile()."""
    unet, config = create_our_unet()

    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        unet = torch.compile(unet, mode='reduce-overhead')
    else:
        print("⚠️  torch.compile() not available (requires PyTorch 2.0+)")
        return None, None

    return unet, config


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 80)
    print("U-Net Implementation Benchmark")
    print("=" * 80)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()

    results = []

    # 1. Benchmark our U-Net
    print("-" * 80)
    print("1. Our Custom U-Net")
    print("-" * 80)
    unet_ours, config = create_our_unet()
    total_params, trainable_params = count_parameters(unet_ours)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    avg_time, std_time, output_shape = benchmark_model(unet_ours, "Our U-Net")
    print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Output shape: {output_shape}")
    print()
    results.append(("Our Custom U-Net", avg_time, std_time, total_params))

    # 2. Benchmark diffusers U-Net
    if DIFFUSERS_AVAILABLE:
        print("-" * 80)
        print("2. HuggingFace Diffusers U-Net")
        print("-" * 80)
        unet_diffusers, _ = create_diffusers_unet()
        total_params, trainable_params = count_parameters(unet_diffusers)
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
        avg_time, std_time, output_shape = benchmark_model(
            unet_diffusers, "Diffusers U-Net"
        )
        print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"Output shape: {output_shape}")
        print()
        results.append(("Diffusers U-Net", avg_time, std_time, total_params))
    else:
        print("-" * 80)
        print("2. HuggingFace Diffusers U-Net - SKIPPED (not installed)")
        print("-" * 80)
        print()

    # 3. Benchmark compiled U-Net
    if hasattr(torch, 'compile'):
        print("-" * 80)
        print("3. Our U-Net with torch.compile()")
        print("-" * 80)
        unet_compiled, _ = create_compiled_unet()
        if unet_compiled is not None:
            total_params, trainable_params = count_parameters(unet_compiled)
            print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
            avg_time, std_time, output_shape = benchmark_model(
                unet_compiled, "Compiled U-Net", num_iterations=50, warmup=10
            )
            print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"Output shape: {output_shape}")
            print()
            results.append(("Our U-Net (compiled)", avg_time, std_time, total_params))
    else:
        print("-" * 80)
        print("3. torch.compile() - SKIPPED (requires PyTorch 2.0+)")
        print("-" * 80)
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Model':<30} {'Avg Time (ms)':<15} {'Params':<15} {'Speedup':<10}")
    print("-" * 80)

    baseline_time = results[0][1]
    for name, avg_time, std_time, params in results:
        speedup = baseline_time / avg_time
        print(f"{name:<30} {avg_time:>7.2f} ± {std_time:>5.2f}   {params:>12,}   {speedup:>6.2f}x")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    if len(results) == 1:
        print("✅ Our custom U-Net works well!")
        print("💡 Consider:")
        if hasattr(torch, 'compile'):
            print("   - Using torch.compile() for 1.5-2x speedup (PyTorch 2.0+)")
        print("   - Installing diffusers for comparison: pip install diffusers")
    else:
        fastest = min(results, key=lambda x: x[1])
        print(f"🏆 Fastest: {fastest[0]} ({fastest[1]:.2f} ms)")
        print()
        print("Trade-offs:")
        print()
        print("Our Custom U-Net:")
        print("  ✅ Clean, readable code")
        print("  ✅ Easy to customize for flow matching")
        print("  ✅ Full control over architecture")
        print("  ❌ May be slower without optimizations")
        print()

        if DIFFUSERS_AVAILABLE:
            print("HuggingFace Diffusers U-Net:")
            print("  ✅ Highly optimized (Flash Attention, etc.)")
            print("  ✅ Battle-tested on large models")
            print("  ✅ Maintained by HuggingFace team")
            print("  ❌ Less customizable")
            print("  ❌ Designed for diffusion (may need adaptation)")
            print()

        if hasattr(torch, 'compile'):
            print("Our U-Net + torch.compile():")
            print("  ✅ Best of both worlds")
            print("  ✅ Simple one-line addition")
            print("  ✅ 1.5-2x speedup typically")
            print("  ❌ First run is slow (compilation)")

    print()


if __name__ == '__main__':
    main()
