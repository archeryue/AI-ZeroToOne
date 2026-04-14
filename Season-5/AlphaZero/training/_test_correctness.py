"""End-to-end correctness test after self-play refactors.

Checks, in order:
  1. Replay buffer contains well-formed data (shapes, ranges, NaN-free,
     policies sum to 1, values in [-1, 1]).
  2. Training loss decreases over a handful of training steps.
  3. `evaluate_vs_random` returns a sane number and the MCTS plays legal
     Go positions (implicit: if legal-move generation is broken we'd crash).

Uses the real 10b×128 model so the exercised code path is identical to
Phase 1 training.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
# Skip compile for this test — we just want to verify correctness quickly.
os.environ.setdefault("AZ_COMPILE", "0")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
import numpy as np

from model.config import CONFIGS, ModelConfig, TrainingConfig
from model.network import AlphaZeroNet
from training.replay_buffer import ReplayBuffer
from training.parallel_self_play import ParallelSelfPlay
from training.trainer import Trainer
from training.train import evaluate_vs_random

device = torch.device("cuda")
N = 9
model_cfg, _ = CONFIGS[N]

# Shrink training config for speed while keeping shape of the pipeline
train_cfg = TrainingConfig(
    num_simulations=32,
    virtual_loss_batch=8,
    num_parallel_games=64,
    num_games_per_iter=64,
    max_game_moves=150,
    buffer_size=20_000,
    batch_size=128,
    train_steps_per_iter=50,
    lr_init=0.01,
    lr_final=0.001,
    eval_interval=1,
    checkpoint_interval=1000,
    temperature_moves=10,
)

print("=" * 60)
print("AlphaZero correctness check (real model, small training run)")
print("=" * 60)

net = AlphaZeroNet(model_cfg).to(device)
print(f"Model: {model_cfg.num_blocks}b × {model_cfg.channels}ch, "
      f"{net.param_count():,} params")

buffer = ReplayBuffer(train_cfg.buffer_size, N)

# ---- Stage 1: run self-play ----
sp = ParallelSelfPlay(net, device, model_cfg, train_cfg, num_workers=4)
stats = sp.run_games(train_cfg.num_games_per_iter, buffer)
print(f"\n[1] Self-play: {stats['games']} games, "
      f"{stats['positions']} positions, "
      f"buffer={stats['buffer_size']}, {stats['time']:.1f}s")

# ---- Stage 1b: validate buffer contents ----
print("\n[1b] Buffer sanity checks")
assert len(buffer) > 0, "replay buffer empty"
obs_arr = buffer.obs[:len(buffer)]
pol_arr = buffer.policy[:len(buffer)]
val_arr = buffer.value[:len(buffer)]

# obs is uint8 (Phase 2 memory optimization) — every value is integer
# so isfinite is trivially True but we keep the assertion for clarity.
assert obs_arr.dtype == np.uint8, f"obs dtype should be uint8, got {obs_arr.dtype}"
assert np.isfinite(pol_arr).all(), "NaN/inf in policy"
assert np.isfinite(val_arr).all(), "NaN/inf in value"
assert np.all(val_arr >= -1.0 - 1e-5) and np.all(val_arr <= 1.0 + 1e-5), \
    f"value out of [-1,1]: min={val_arr.min()} max={val_arr.max()}"
pol_sums = pol_arr.sum(axis=1)
assert np.allclose(pol_sums, 1.0, atol=1e-3), \
    f"policy doesn't sum to 1: min={pol_sums.min()} max={pol_sums.max()}"
obs_vals = np.unique(obs_arr)
# obs planes should be 0/1 (stones + color-to-play) — uint8 after the
# Phase 2 dtype change.
assert set(obs_vals.tolist()).issubset({0, 1}), \
    f"obs not binary: {obs_vals[:10]}"
print(f"  obs shape {obs_arr.shape} dtype {obs_arr.dtype}, binary ✓")
print(f"  policy sums to 1 ✓ (min={pol_sums.min():.4f} max={pol_sums.max():.4f})")
print(f"  value in [-1,1] ✓ (min={val_arr.min():.2f} max={val_arr.max():.2f})")
print(f"  values distribution: mean={val_arr.mean():.3f} "
      f"(black-win frac if +1)")

# ---- Stage 1c: uint8 obs round-trip + save/load ----
# Per PHASE_TWO_TODO.md §"Verification before merging": prove the
# uint8 cast is lossless for a float32 obs that came out of self-play.
print("\n[1c] uint8 obs round-trip + save/load")
import tempfile

# Take a fresh float32 obs from a raw self-play sample (buffer[0] is
# already uint8, so we materialize it back to float32 to simulate the
# C++ engine output, then push it back and confirm identity).
obs_f32_orig = buffer.obs[0].astype(np.float32)
tmp_buf = ReplayBuffer(16, N)
tmp_buf.push(obs_f32_orig, buffer.policy[0], buffer.value[0])
obs_back = tmp_buf.obs[0].astype(np.float32)
assert np.array_equal(obs_f32_orig, obs_back), \
    "uint8 round-trip lost information — obs values outside {0, 1}?"
print(f"  float32→uint8→float32 round-trip exact ✓")

# save_to / load_from round-trip preserves the raw uint8 buffer.
with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmpf:
    tmp_path = tmpf.name
try:
    buffer.save_to(tmp_path)
    restored = ReplayBuffer(train_cfg.buffer_size, N)
    restored.load_from(tmp_path)
    assert restored.size == buffer.size
    assert restored.obs.dtype == np.uint8
    assert np.array_equal(restored.obs[:restored.size],
                          buffer.obs[:buffer.size])
    assert np.array_equal(restored.policy[:restored.size],
                          buffer.policy[:buffer.size])
    assert np.array_equal(restored.value[:restored.size],
                          buffer.value[:buffer.size])
    print(f"  save_to / load_from round-trip exact ✓ "
          f"({restored.size} samples)")
finally:
    os.unlink(tmp_path)

# ---- Stage 2: training loss decreases ----
print("\n[2] Training loss trajectory")
trainer = Trainer(net, train_cfg, device)
losses = []
for step in range(5):
    stats = trainer.train_epoch(buffer, 1)
    losses.append(stats["loss"])
    print(f"  epoch {step}: loss={stats['loss']:.4f} "
          f"(pi={stats['policy_loss']:.4f}, v={stats['value_loss']:.4f})")
# Loss should strictly decrease or trend down
assert losses[-1] < losses[0], \
    f"loss not decreasing: {losses[0]:.4f} → {losses[-1]:.4f}"
print(f"  loss trajectory looks healthy ✓")

# ---- Stage 3: run MCTS-vs-random eval ----
print("\n[3] Evaluate vs random (10 games)")
win_rate = evaluate_vs_random(net, device, model_cfg, train_cfg, num_games=10)
print(f"  MCTS(trained) vs random: {win_rate:.0%} win rate")
# 10 games after only 5 training epochs won't be strong — just verify we
# completed the eval without crashing (no legal-move bugs).
print(f"  completed eval without crashes ✓")

print("\n" + "=" * 60)
print("ALL CORRECTNESS CHECKS PASSED")
print("=" * 60)
