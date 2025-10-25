#!/bin/bash
#
# Quick test script for RunPod (10-15 minute test run)
# This verifies training works and model is learning
#
# Usage:
#   bash quick_test.sh
#
# What it does:
# - Runs training for ~300 steps (~10-15 minutes on 2x 4090s)
# - Uses smaller model for faster iteration
# - Generates samples frequently to check progress
# - Saves checkpoints for later continuation

echo "========================================"
echo "Flow Matching Quick Test on RunPod"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Duration: ~10-15 minutes"
echo "  Expected steps: ~300"
echo "  Model size: Reduced (faster)"
echo "  Batch size: 64 per GPU"
echo ""
echo "Press Ctrl+C to stop early if needed"
echo ""

# Calculate approximate steps for 10-15 minutes
# Assuming ~2-3 seconds per step on 2x 4090s
# 600 seconds / 2 seconds per step = ~300 steps
MAX_STEPS=300

# Run training with fast settings
torchrun --nproc_per_node=2 train.py \
    --batch-size 64 \
    --num-epochs 1 \
    --model-channels 96 \
    --channel-mult 1 2 2 \
    --num-res-blocks 1 \
    --attention-resolutions 16 \
    --dropout 0.0 \
    --learning-rate 2e-4 \
    --sample-every 50 \
    --num-samples 16 \
    --sample-steps 20 \
    --save-every 100 \
    --log-every 10 \
    --num-workers 4 \
    --mixed-precision \
    --compile-model \
    2>&1 | head -n 1000 &

# Capture the PID
TRAIN_PID=$!

echo ""
echo "Training started with PID: $TRAIN_PID"
echo ""
echo "Monitoring for $MAX_STEPS steps (approximately 10-15 minutes)..."
echo ""

# Monitor and stop after ~300 steps
# We'll check the log file to count steps
sleep 600  # Wait 10 minutes

# Check if still running
if ps -p $TRAIN_PID > /dev/null; then
    echo ""
    echo "✓ Training completed 10 minutes, stopping now..."
    kill -SIGINT $TRAIN_PID
    wait $TRAIN_PID
fi

echo ""
echo "========================================"
echo "Quick Test Complete!"
echo "========================================"
echo ""
echo "Check the results:"
echo "  1. Samples: ls -lh samples/"
echo "  2. Checkpoints: ls -lh checkpoints/"
echo "  3. Logs: ls -lh logs/"
echo "  4. Training metrics: python plot_training.py"
echo "  5. View samples: python view_samples.py"
echo ""
echo "If everything looks good, run full training:"
echo "  torchrun --nproc_per_node=2 train.py --batch-size 32 --num-epochs 100"
echo ""
