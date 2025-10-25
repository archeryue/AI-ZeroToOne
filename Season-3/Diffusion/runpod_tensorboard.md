# Access TensorBoard on RunPod from Local Browser

## Method 1: SSH Port Forwarding (Recommended) ⭐

### Step 1: Get RunPod SSH Connection Info

1. Go to RunPod dashboard
2. Click on your pod
3. Click "Connect" → "SSH over exposed TCP"
4. Copy the connection details (will look like):
   ```
   ssh root@ssh.runpod.io -p 12345 -i ~/.ssh/your_key
   ```

### Step 2: Create SSH Tunnel with Port Forwarding

**On your LOCAL machine terminal:**

```bash
# Template:
ssh -L 6006:localhost:6006 root@<runpod-host> -p <port> -i <your-ssh-key>

# Real example:
ssh -L 6006:localhost:6006 root@ssh.runpod.io -p 12345 -i ~/.ssh/id_ed25519
```

**What the flags mean:**
- `-L 6006:localhost:6006` = Forward port 6006
- `-p 12345` = RunPod SSH port
- `-i ~/.ssh/id_ed25519` = Your SSH key (if needed)

### Step 3: Start TensorBoard on RunPod

**In the SSH session (on RunPod):**

```bash
cd /workspace/AI-ZeroToOne/Season-3/Diffusion
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

**You should see:**
```
TensorBoard 2.20.0 at http://0.0.0.0:6006/ (Press CTRL+C to quit)
```

### Step 4: Open in Your Local Browser

**On your local machine, open:**
```
http://localhost:6006
```

**✅ You should see TensorBoard!**

---

## Method 2: RunPod Exposed Ports

### Step 1: Find Your Pod's Exposed Port URL

RunPod automatically exposes certain ports. Check your pod details:

1. RunPod Dashboard → Your Pod
2. Look for "TCP Port Mappings" or similar
3. Find port 6006 mapping (might look like):
   ```
   6006 → https://xxxxx-6006.proxy.runpod.net
   ```

### Step 2: Start TensorBoard

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

### Step 3: Access the URL

Open the RunPod-provided URL in your browser:
```
https://xxxxx-6006.proxy.runpod.net
```

---

## Method 3: Quick Test (During Training)

If you just want to quickly check progress:

### Option A: Download Logs Locally

**On your LOCAL machine:**

```bash
# Download TensorBoard logs
scp -P <port> -r root@ssh.runpod.io:/workspace/Diffusion/logs ./runpod_logs

# Run TensorBoard locally
tensorboard --logdir ./runpod_logs
# Open: http://localhost:6006
```

### Option B: Download Sample Images

```bash
# Download generated samples
scp -P <port> -r root@ssh.runpod.io:/workspace/Diffusion/samples ./runpod_samples

# View locally
ls -lh runpod_samples/
```

---

## 🎯 Recommended Workflow

**Start training in screen:**
```bash
# On RunPod
screen -S training
cd /workspace/AI-ZeroToOne/Season-3/Diffusion
torchrun --nproc_per_node=2 train.py --batch-size 32 --num-epochs 100
# Detach: Ctrl+A then D
```

**Monitor with TensorBoard:**

**Terminal 1 (Local):** SSH with port forwarding
```bash
ssh -L 6006:localhost:6006 root@ssh.runpod.io -p 12345
```

**Terminal 2 (on RunPod via that SSH):** Start TensorBoard
```bash
cd /workspace/AI-ZeroToOne/Season-3/Diffusion
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

**Browser (Local):** Open http://localhost:6006

**Reattach to training:**
```bash
screen -r training
```

---

## 🐛 Troubleshooting

### "Connection refused" or "Port already in use"

**On your local machine:**
```bash
# Kill any process using port 6006
lsof -ti:6006 | xargs kill -9

# Or use a different port
ssh -L 6007:localhost:6006 root@ssh.runpod.io -p 12345
# Then open: http://localhost:6007
```

### "SSH connection drops"

Use `autossh` to maintain connection:
```bash
autossh -M 0 -L 6006:localhost:6006 root@ssh.runpod.io -p 12345
```

### Can't access TensorBoard URL

Make sure TensorBoard is binding to `0.0.0.0` not `localhost`:
```bash
# Correct:
tensorboard --logdir logs --host 0.0.0.0 --port 6006

# Wrong:
tensorboard --logdir logs --host localhost --port 6006
```

---

## 📊 Alternative: Use Plotting Scripts

If TensorBoard access is difficult, use the plotting scripts:

```bash
# On RunPod, generate plots
python plot_training.py
python view_samples.py

# Download to local machine
scp -P <port> root@ssh.runpod.io:/workspace/Diffusion/training_*.png ./
scp -P <port> root@ssh.runpod.io:/workspace/Diffusion/samples/*.png ./samples/

# View locally
open training_metrics.png
open training_samples_overview.png
```

---

## 🔒 Security Note

**SSH Port Forwarding (Method 1)** is the most secure because:
- Encrypted connection
- No public exposure
- Works through firewalls

**Exposed Ports (Method 2)** may have:
- Public access (anyone with URL can view)
- Depends on RunPod security

Always prefer **SSH port forwarding** for sensitive work!

---

## Quick Reference

| Method | Security | Ease | Requirement |
|--------|----------|------|-------------|
| SSH Forwarding | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | SSH access |
| RunPod Ports | ⭐⭐ | ⭐⭐⭐⭐⭐ | RunPod feature |
| ngrok | ⭐⭐⭐ | ⭐⭐⭐⭐ | ngrok account |
| Download logs | ⭐⭐⭐⭐⭐ | ⭐⭐ | SCP access |

**Recommendation:** Use **SSH Port Forwarding** for best security and reliability!
