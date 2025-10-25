# RunPod Quick Test Guide (10-15 Minutes)

This guide shows you how to run a **quick validation test** on RunPod to verify everything works before committing to a full training run.

## 🎯 Goal

Run training for **10-15 minutes** to verify:
- ✅ Training works on RunPod with 2x 4090s
- ✅ Model is learning (loss decreasing)
- ✅ Samples are generated correctly
- ✅ No memory/GPU issues

## ⏱️ Quick Test Strategy

### Option 1: Automated Quick Test (Recommended)

**Estimated time: ~10-15 minutes**

```bash
# 1. SSH into your RunPod instance
cd /workspace/AI-ZeroToOne/Season-3/Diffusion

# 2. Run the quick test script
bash quick_test.sh
```

This will:
- Run ~300 training steps (~10-15 min on 2x 4090s)
- Use a smaller model (96 channels instead of 128)
- Generate samples every 50 steps
- Save checkpoint every 100 steps
- Automatically stop after 10 minutes

---

### Option 2: Manual Quick Test

If you want more control:

```bash
# Start training with fast settings
torchrun --nproc_per_node=2 train.py \
    --batch-size 64 \
    --num-epochs 1 \
    --model-channels 96 \
    --channel-mult 1 2 2 \
    --sample-every 50 \
    --num-samples 16 \
    --save-every 100 \
    --log-every 10

# Press Ctrl+C after 10-15 minutes to stop
```

**Settings explained:**
- `--batch-size 64`: Larger batches = faster iteration
- `--model-channels 96`: Smaller model = faster training
- `--channel-mult 1 2 2`: 3 levels instead of 4 = faster
- `--sample-every 50`: See progress quickly
- `--num-samples 16`: Fewer samples = faster generation

---

### Option 3: Time-Limited Test

Run for exactly 15 minutes and auto-stop:

```bash
# Start training in background
timeout 900 torchrun --nproc_per_node=2 train.py \
    --batch-size 64 \
    --model-channels 96 \
    --sample-every 50 \
    --log-every 10 &

# Get the process ID
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"

# Wait and monitor
echo "Running for 15 minutes (900 seconds)..."
wait $TRAIN_PID
echo "Test complete!"
```

---

## 📊 How to Monitor Progress

### 1. **Watch Training in Real-Time**

While training is running, open a **second SSH terminal**:

```bash
# Watch the latest log
tail -f logs/events.out.tfevents.*

# Or watch GPU usage
watch -n 1 nvidia-smi
```

**What to look for:**
- GPU Utilization: Should be **90-100%**
- GPU Memory: Should be **~12-15 GB per GPU**
- Temperature: Should be **<85°C**

---

### 2. **TensorBoard (Real-Time Monitoring)**

**In a separate terminal:**

```bash
# Start TensorBoard
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

**Then access from your local machine:**
```
http://<runpod-ip>:6006
```

**What to check:**
- **Loss curve**: Should be **decreasing** (even if noisy)
- **Learning rate**: Should start from 0 and warm up
- **Samples**: Should start noisy and gradually improve

---

### 3. **Check Samples Directly**

```bash
# List generated samples
ls -lh samples/

# View the latest sample (using Python)
python view_samples.py --max-images 4

# Or download to your local machine
# From your local terminal:
scp -r runpod:/workspace/Diffusion/samples ./
```

---

### 4. **Check Training Metrics**

After training (or during):

```bash
# Generate training plot
python plot_training.py

# View summary
cat training_summary.txt
```

---

## ✅ How to Know if Model is Learning

### Good Signs ✅

1. **Loss is decreasing:**
   ```
   Step 10:  loss=1.45
   Step 50:  loss=1.32
   Step 100: loss=1.18
   Step 200: loss=0.95
   ```

2. **Samples improving:**
   - Early steps: Random noise
   - After 100 steps: Vague shapes/colors
   - After 200+ steps: Recognizable features (faces starting to form)

3. **GPU utilization high:**
   ```
   nvidia-smi shows:
   GPU 0: 95% utilization, 14GB memory
   GPU 1: 95% utilization, 14GB memory
   ```

4. **No errors in console output**

### Warning Signs ⚠️

1. **Loss not decreasing or NaN:**
   - Loss stuck around same value
   - Loss = NaN (exploding gradients)
   - Solution: Reduce learning rate

2. **Low GPU utilization (<50%):**
   - Data loading bottleneck
   - Solution: Increase `--num-workers`

3. **Out of Memory errors:**
   - Reduce `--batch-size`
   - Reduce `--model-channels`

4. **Samples still random noise after 200+ steps:**
   - Check loss is decreasing
   - May need longer training

---

## 📈 Expected Results (10-15 Minutes)

On **2x NVIDIA 4090 GPUs** with the quick test settings:

| Metric | Expected Value |
|--------|----------------|
| **Steps completed** | ~200-300 steps |
| **Time per step** | ~2-3 seconds |
| **Starting loss** | ~1.4-1.6 |
| **Ending loss** | ~0.9-1.1 (decreasing) |
| **GPU memory** | ~12-15 GB per GPU |
| **GPU utilization** | 90-100% |
| **Samples quality** | Noisy but showing basic structure |

---

## 🚀 After Quick Test Passes

If everything looks good, run **full training**:

```bash
# Full training with optimal settings
torchrun --nproc_per_node=2 train.py \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 2e-4 \
    --sample-every 1000 \
    --save-every 5000

# Estimated time: ~25-30 hours on 2x 4090s
```

---

## 📝 Checklist for RunPod Quick Test

Before starting:
- [ ] RunPod instance with 2x 4090 GPUs running
- [ ] Repository cloned
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `test_setup.py` passed

During test (10-15 min):
- [ ] Training started successfully
- [ ] GPU utilization >90%
- [ ] Loss decreasing
- [ ] Samples being generated
- [ ] No OOM errors

After test:
- [ ] Review final loss (should be lower than start)
- [ ] Check samples (should show some structure)
- [ ] Review TensorBoard plots
- [ ] Verify checkpoints saved

If all ✅, proceed to full training!

---

## 🐛 Troubleshooting

### "RuntimeError: CUDA out of memory"
```bash
# Reduce batch size
torchrun --nproc_per_node=2 train.py --batch-size 24
```

### "Connection lost" / SSH disconnected
```bash
# Use screen to keep running
screen -S training
torchrun --nproc_per_node=2 train.py ...
# Detach: Ctrl+A then D
# Reattach: screen -r training
```

### "Low GPU utilization"
```bash
# Increase data workers
torchrun --nproc_per_node=2 train.py --num-workers 8
```

### "Loss is NaN"
```bash
# Reduce learning rate
torchrun --nproc_per_node=2 train.py --learning-rate 1e-4
```

---

## 💡 Pro Tips

1. **Use screen/tmux** for long runs:
   ```bash
   screen -S quicktest
   bash quick_test.sh
   # Detach: Ctrl+A then D
   ```

2. **Monitor from local machine:**
   ```bash
   # Port forward TensorBoard
   ssh -L 6006:localhost:6006 runpod
   # Then open http://localhost:6006 in browser
   ```

3. **Save costs:**
   - Stop instance immediately after quick test if results look bad
   - Only start full training if quick test passes

4. **Download results:**
   ```bash
   # From your local machine
   scp -r runpod:/workspace/Diffusion/samples ./quicktest_samples
   scp runpod:/workspace/Diffusion/training_summary.txt ./
   ```

---

## 🎓 Summary

**Quick Test = 10-15 minutes to verify:**
- ✅ Everything works on RunPod
- ✅ Model is learning (loss ↓)
- ✅ No technical issues
- ✅ Ready for full 25-30 hour training

**Commands:**
```bash
# Quick automated test
bash quick_test.sh

# Or manual
torchrun --nproc_per_node=2 train.py \
    --batch-size 64 --model-channels 96 \
    --sample-every 50 --log-every 10
# Ctrl+C after 10-15 min
```

**Monitor:** TensorBoard, nvidia-smi, samples/, logs/

**Success criteria:** Loss ↓, GPU >90%, samples improving, no errors

Good luck with your test! 🚀
