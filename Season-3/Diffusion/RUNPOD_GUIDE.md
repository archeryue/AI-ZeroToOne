# RunPod Quick Start Guide

This guide will help you quickly set up and train the Flow Matching model on RunPod with 2x NVIDIA 4090 GPUs.

## Step 1: Launch RunPod Instance

1. Go to [RunPod.io](https://www.runpod.io/)
2. Select a pod with **2x NVIDIA 4090** GPUs
3. Choose a PyTorch template (PyTorch 2.0+ recommended)
4. Launch the instance
5. Connect via SSH or JupyterLab

## Step 2: Setup Environment

```bash
# Clone the repository (if not already on the instance)
cd /workspace  # or your preferred directory
git clone <your-repo-url>
cd AI-ZeroToOne/Season-3/Diffusion

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

## Step 3: Start Training

### Option A: Quick Start (Default Settings)

```bash
torchrun --nproc_per_node=2 train.py
```

This will:
- Train on 2 GPUs with batch size 32 per GPU (64 total)
- Run for 100 epochs
- Save checkpoints every 5000 steps
- Generate samples every 1000 steps

### Option B: Recommended Settings for RunPod

```bash
torchrun --nproc_per_node=2 train.py \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 2e-4 \
    --mixed-precision \
    --use-ema \
    --sample-every 1000 \
    --save-every 5000 \
    --num-workers 4
```

### Option C: Fast Training (Lower Quality)

For quick testing or lower quality results:

```bash
torchrun --nproc_per_node=2 train.py \
    --batch-size 64 \
    --num-epochs 50 \
    --model-channels 96 \
    --channel-mult 1 2 2 \
    --sample-every 500
```

### Option D: High Quality (Slower)

For best quality results:

```bash
torchrun --nproc_per_node=2 train.py \
    --batch-size 24 \
    --num-epochs 150 \
    --model-channels 192 \
    --channel-mult 1 2 3 4 \
    --num-res-blocks 3 \
    --attention-resolutions 32 16 8 \
    --learning-rate 1e-4
```

## Step 4: Monitor Training

### Option A: TensorBoard (Recommended)

In a separate terminal:
```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

Then access via: `http://<your-runpod-ip>:6006`

### Option B: Check Samples

Samples are saved to `samples/` directory:
```bash
ls -lh samples/
```

Download and view them:
```bash
# From your local machine
scp -r runpod:/workspace/Diffusion/samples ./
```

### Option C: Check Logs

Training logs are printed to console and saved in `logs/`:
```bash
tail -f logs/events.out.tfevents.*
```

## Step 5: Generate Samples

After training (or during), generate samples:

```bash
python generate.py \
    --checkpoint checkpoints/checkpoint_epoch_50.pth \
    --num-samples 64 \
    --num-steps 50
```

For best quality:
```bash
python generate.py \
    --checkpoint checkpoints/checkpoint_epoch_100.pth \
    --num-samples 100 \
    --num-steps 100 \
    --solver midpoint
```

## Tips for RunPod

### 1. Use Screen/Tmux

Training takes hours, so use screen or tmux to keep it running:

```bash
# Start a new screen session
screen -S training

# Run training
torchrun --nproc_per_node=2 train.py

# Detach: Ctrl+A then D

# Reattach later
screen -r training
```

### 2. Save Checkpoints Frequently

RunPod instances can be interrupted. Save frequently:

```bash
torchrun --nproc_per_node=2 train.py --save-every 2500
```

### 3. Download Checkpoints Regularly

Download important checkpoints to your local machine:

```bash
# From your local machine
scp runpod:/workspace/Diffusion/checkpoints/checkpoint_epoch_50.pth ./
```

### 4. Resume Training

If training is interrupted:

```bash
torchrun --nproc_per_node=2 train.py \
    --resume-from checkpoints/checkpoint_epoch_50.pth
```

### 5. Monitor GPU Usage

Check GPU utilization:

```bash
watch -n 1 nvidia-smi
```

You should see ~95%+ GPU utilization when training.

### 6. Optimize for Cost

- Start with 50 epochs to check quality
- Use smaller models for testing
- Delete old checkpoints to save space:
  ```bash
  # Keep only the latest 5 checkpoints
  ls -t checkpoints/*.pth | tail -n +6 | xargs rm
  ```

## Expected Timeline

On 2x NVIDIA 4090 GPUs:

| Epochs | Time | Quality |
|--------|------|---------|
| 10     | ~3h  | Poor (noisy) |
| 30     | ~9h  | Fair (recognizable faces) |
| 50     | ~15h | Good (decent faces) |
| 100    | ~30h | Very Good (high quality faces) |
| 150+   | ~45h+ | Excellent (best quality) |

## Troubleshooting

### "RuntimeError: CUDA out of memory"

Reduce batch size:
```bash
torchrun --nproc_per_node=2 train.py --batch-size 16
```

### "Connection lost"

Use screen/tmux as described above.

### "No module named 'xxx'"

Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### "Slow training"

- Check GPU usage: `nvidia-smi`
- Ensure mixed precision is enabled: `--mixed-precision`
- Increase workers: `--num-workers 8`

### "Bad sample quality"

- Train longer (50+ epochs)
- Check that EMA is enabled: `--use-ema`
- Increase sampling steps: `--num-steps 100` in generate.py

## Cost Estimation

RunPod 2x 4090 pricing (approximate):
- ~$1.50-2.00 per hour
- 100 epochs: ~30 hours = ~$45-60
- 50 epochs: ~15 hours = ~$23-30

**Tip**: Start with 50 epochs, check quality, then continue if needed.

## Example Full Workflow

```bash
# 1. Setup
cd /workspace
git clone <repo>
cd AI-ZeroToOne/Season-3/Diffusion
pip install -r requirements.txt
python test_setup.py

# 2. Start screen session
screen -S fm_training

# 3. Start training
torchrun --nproc_per_node=2 train.py \
    --batch-size 32 \
    --num-epochs 100 \
    --mixed-precision \
    --save-every 5000

# 4. Detach (Ctrl+A then D)

# 5. Monitor in another terminal
tensorboard --logdir logs --host 0.0.0.0 --port 6006

# 6. After training, generate samples
python generate.py \
    --checkpoint checkpoints/checkpoint_epoch_100.pth \
    --num-samples 64 \
    --num-steps 50

# 7. Download results to local machine
# (from your local terminal)
scp -r runpod:/workspace/Diffusion/results ./
scp -r runpod:/workspace/Diffusion/samples ./
```

## Support

If you encounter issues:
1. Check `test_setup.py` output
2. Review training logs in `logs/`
3. Check CUDA memory with `nvidia-smi`
4. Verify data loaded successfully (check console output)

Happy training!
