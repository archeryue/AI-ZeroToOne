"""Evaluate each checkpoint vs random to find the strength-regression inflection.

Run1 attempt 3 iter 4 showed 5% win vs random, indicating strength
regression during training. This script evaluates each checkpoint_000X.pt
under the same conditions (100 games, 100 MCTS sims/move) to find which
iter the regression began.

Usage (from Season-5/AlphaZero/):
    PYTHONPATH=engine python training/_eval_checkpoints.py

Output: per-checkpoint win rate vs random + a rough bar chart.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Limit torch intra-op threads to match train.py
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_float32_matmul_precision("high")

from model.config import CONFIGS
from model.network import AlphaZeroNet
from training.trainer import Trainer
from training.train import evaluate_vs_random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cfg, train_cfg = CONFIGS[13]

net = AlphaZeroNet(model_cfg).to(device)
trainer = Trainer(net, train_cfg, device)

ckpt_dir = "checkpoints/13x13_run1"
num_games = 100

print(f"=== Checkpoint strength audit — 13x13 ===")
print(f"  device:          {device}")
print(f"  model:           {model_cfg.num_blocks}b × {model_cfg.channels}ch")
print(f"  games per ckpt:  {num_games}")
print(f"  sims per move:   {min(train_cfg.num_simulations, 100)}")
print(f"  dir:             {ckpt_dir}")
print()

results = []
for i in range(5):
    ckpt = f"{ckpt_dir}/checkpoint_{i:04d}.pt"
    if not os.path.exists(ckpt):
        print(f"iter {i}: MISSING {ckpt}")
        continue
    print(f"iter {i}: loading {ckpt} ...", flush=True)
    trainer.load_checkpoint(ckpt)
    t0 = time.time()
    wr = evaluate_vs_random(net, device, model_cfg, train_cfg,
                            num_games=num_games)
    dt = time.time() - t0
    results.append((i, wr, dt))
    print(f"iter {i}: win_rate = {wr:.1%}  ({dt:.1f}s)")
    print()

print("=== Summary ===")
print(f"{'iter':>5}  {'wr':>7}  {'time':>8}")
for i, wr, dt in results:
    bar = "█" * int(wr * 40)
    print(f"{i:>5}  {wr:>7.1%}  {dt:>6.1f}s  {bar}")

# Write summary to file for later reference
out_path = os.path.join(ckpt_dir, "strength_audit.txt")
with open(out_path, "w") as f:
    f.write(f"Checkpoint strength audit for {ckpt_dir}\n")
    f.write(f"num_games={num_games}, sims_per_move={min(train_cfg.num_simulations, 100)}\n\n")
    for i, wr, dt in results:
        f.write(f"iter {i}: {wr:.1%} ({dt:.1f}s)\n")
print(f"\nWritten to {out_path}")
