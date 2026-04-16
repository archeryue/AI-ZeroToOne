#!/usr/bin/env bash
# Background gradient & weight analyzer: watches for new checkpoints
# and runs diagnostic analysis on each.
#
# Usage: run_grad_analysis.sh <ckpt_dir> <results_file>
#
# For each checkpoint, reports:
#   - Per-layer gradient norms (from one training step)
#   - Per-layer weight norms and stats
#   - Score head predictions (mean, std, range)
#   - Derived value predictions (mean, std, range)
#   - value_scale / value_bias values
#   - Policy pass probability on empty board

set -u

CKPT_DIR="${1:-checkpoints/13x13_run5}"
RESULTS="${2:-logs/grad_analysis_run5.log}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

cd "$(dirname "$0")"
mkdir -p "$(dirname "$RESULTS")"
touch "$RESULTS"

echo "[grad-analysis] starting. ckpt_dir=$CKPT_DIR results=$RESULTS"

while true; do
    for ckpt in "$CKPT_DIR"/checkpoint_*.pt; do
        [[ -f "$ckpt" ]] || continue
        iter=$(basename "$ckpt" .pt | sed 's/checkpoint_0*//')
        [[ -z "$iter" ]] && iter=0
        if grep -q "^iter=$iter " "$RESULTS" 2>/dev/null; then
            continue  # already analyzed
        fi
        echo "[grad-analysis] analyzing iter $iter from $ckpt"
        out=$(PYTHONPATH=engine AZ_COMPILE=0 timeout 120 python3 -c "
import os, sys, json
sys.path.insert(0, '.')
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import torch
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
torch.set_float32_matmul_precision('high')
import numpy as np
from model.config import CONFIGS
from model.network import AlphaZeroNet
from training.replay_buffer import ReplayBuffer
from training.trainer import Trainer

device = torch.device('cuda')
mcfg, tcfg = CONFIGS[13]
net = AlphaZeroNet(mcfg).to(device)

state = torch.load('$ckpt', map_location=device, weights_only=False)
net.load_state_dict(state['model_state_dict'])

# === Weight stats ===
weight_stats = {}
for name, param in net.named_parameters():
    d = param.data
    weight_stats[name] = {
        'norm': d.norm().item(),
        'mean': d.mean().item(),
        'std': d.std().item(),
        'absmax': d.abs().max().item(),
    }

# === Forward pass on empty board ===
import go_engine
game = go_engine.Game13(7.5)
obs_np = np.array(game.to_observation(), dtype=np.float32).reshape(1, mcfg.input_planes, 13, 13)
obs = torch.from_numpy(obs_np).to(device)
net.eval()
with torch.no_grad():
    logits, value, own_logits, score = net(obs)
    policy = torch.softmax(logits, dim=-1)

pass_prob = policy[0, 169].item()
pass_logit = logits[0, 169].item()
top5_idx = policy[0].topk(5).indices.cpu().tolist()
top5_prob = [policy[0, i].item() for i in top5_idx]

# === Forward on buffer sample ===
buf = ReplayBuffer(1000000, 13, mcfg.input_planes)
buf.load_from('$CKPT_DIR/latest_buffer.npz')
obs_s, _, _, _ = buf.sample(min(512, len(buf)))
obs_t = torch.from_numpy(obs_s.astype(np.float32)).to(device)
with torch.no_grad():
    _, vals, _, scores = net(obs_t)

# === Gradient norms (one train step) ===
net.train()
trainer = Trainer(net, tcfg, device)
trainer.load_checkpoint('$ckpt')
trainer.optimizer.zero_grad()
trainer.train_step(buf)

grad_norms = {}
for name, param in net.named_parameters():
    if param.grad is not None:
        grad_norms[name] = param.grad.norm().item()
    else:
        grad_norms[name] = 0.0

# === Summarize ===
# Group gradient norms by component
groups = {
    'trunk': [],
    'policy': [],
    'score': [],
    'ownership': [],
    'value_params': [],
}
for name, gnorm in grad_norms.items():
    if 'policy' in name:
        groups['policy'].append(gnorm)
    elif 'score' in name:
        groups['score'].append(gnorm)
    elif 'ownership' in name:
        groups['ownership'].append(gnorm)
    elif name in ('value_scale', 'value_bias'):
        groups['value_params'].append(gnorm)
    else:
        groups['trunk'].append(gnorm)

group_summary = {}
for g, norms in groups.items():
    if norms:
        group_summary[g] = f'mean={np.mean(norms):.6f},max={np.max(norms):.6f}'
    else:
        group_summary[g] = 'none'

print(f'vs={net.value_scale.item():.6f} vb={net.value_bias.item():.6f}')
print(f'score_buf: mean={scores.mean().item():.4f} std={scores.std().item():.4f} range=[{scores.min().item():.4f},{scores.max().item():.4f}]')
print(f'value_buf: mean={vals.mean().item():.4f} std={vals.std().item():.4f} range=[{vals.min().item():.4f},{vals.max().item():.4f}]')
print(f'pass_prob={pass_prob:.6f} pass_logit={pass_logit:.4f} argmax={top5_idx[0]}')
print(f'grad_trunk: {group_summary[\"trunk\"]}')
print(f'grad_policy: {group_summary[\"policy\"]}')
print(f'grad_score: {group_summary[\"score\"]}')
print(f'grad_own: {group_summary[\"ownership\"]}')
print(f'grad_vparams: {group_summary[\"value_params\"]}')

# Per-layer weight norms for key components
for name in ['score_fc1.weight', 'score_fc2.weight', 'score_conv.weight',
             'policy_fc.weight', 'ownership_conv.weight', 'ownership_conv.bias',
             'input_conv.weight']:
    if name in weight_stats:
        ws = weight_stats[name]
        print(f'w_{name}: norm={ws[\"norm\"]:.4f} std={ws[\"std\"]:.6f} absmax={ws[\"absmax\"]:.6f}')
" 2>&1)
        rc=$?
        ts=$(date +%H:%M:%S)
        if [[ $rc -eq 0 ]]; then
            # Parse key metrics from output
            vs=$(echo "$out" | grep "^vs=" | head -1)
            score_buf=$(echo "$out" | grep "^score_buf:" | head -1)
            value_buf=$(echo "$out" | grep "^value_buf:" | head -1)
            pass_info=$(echo "$out" | grep "^pass_prob=" | head -1)
            grad_trunk=$(echo "$out" | grep "^grad_trunk:" | head -1)
            grad_score=$(echo "$out" | grep "^grad_score:" | head -1)

            echo "iter=$iter ts=$ts $vs $score_buf $value_buf $pass_info $grad_trunk $grad_score" >> "$RESULTS"
            echo "[grad-analysis] iter $iter done"
            # Also dump full output to a per-iter file
            echo "$out" > "$(dirname "$RESULTS")/grad_iter${iter}.txt"
        else
            echo "iter=$iter ts=$ts FAILED rc=$rc" >> "$RESULTS"
            echo "[grad-analysis] iter $iter FAILED (rc=$rc)"
            echo "$out" > "$(dirname "$RESULTS")/grad_iter${iter}_err.txt"
        fi
    done
    sleep "$POLL_INTERVAL"
done
