"""End-to-end correctness test for AlphaZero 13x13 training pipeline.

PURPOSE: Catch code bugs BEFORE spending hours on real training.
This is NOT a training-quality test — it verifies plumbing, shapes,
loss scaling, gradient flow, and config consistency.

Checks, in order:
  1.  Self-play produces well-formed buffer data (13x13).
  1b. Buffer contents: shapes, ranges, NaN-free, policy sums to 1,
      ownership in {-1, 0, +1}.
  1c. uint8 obs round-trip + save/load.
  2.  Training loss decreases, all components in sane range.
  3.  MCTS-vs-random eval completes without crashes.
  4.  Score head: target scaling, loss balance, value signal for MCTS.
  5.  Pass floor: pass action zeroed in early-game policy targets.
  6.  Gradient flow: every parameter with nonzero loss weight gets
      nonzero gradients (catches frozen params like value_scale).
  7.  Engine build freshness: .so is newer than all .h source files.

Designed to run in <30s on a 4090. Uses small model (2b×32ch) with
production loss weights and config knobs.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("AZ_COMPILE", "0")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import tempfile
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
N = 13

# Use real production 13x13 model + config. Only shrink game count
# and train steps for speed — the model itself must be production
# size to catch shape/param bugs at real scale.
mcfg_prod, tcfg_prod = CONFIGS[N]
mcfg = mcfg_prod  # real 15b×128ch model
tcfg = TrainingConfig(
    num_simulations=16,
    virtual_loss_batch=8,
    num_parallel_games=8,
    num_games_per_iter=16,
    max_game_moves=250,
    buffer_size=10_000,
    batch_size=64,
    train_steps_per_iter=30,
    lr_init=tcfg_prod.lr_init,
    lr_final=tcfg_prod.lr_final,
    temperature_moves=40,
    pass_min_move=tcfg_prod.pass_min_move,
    value_loss_weight=tcfg_prod.value_loss_weight,
    score_loss_weight=tcfg_prod.score_loss_weight,
    ownership_loss_weight=tcfg_prod.ownership_loss_weight,
)

print("=" * 60)
print(f"AlphaZero {N}x{N} correctness check")
print("=" * 60)

net = AlphaZeroNet(mcfg).to(device)
print(f"Model: {mcfg.num_blocks}b × {mcfg.channels}ch, "
      f"{net.param_count():,} params")
print(f"Config: vlw={tcfg.value_loss_weight}, slw={tcfg.score_loss_weight}, "
      f"olw={tcfg.ownership_loss_weight}, pass_min={tcfg.pass_min_move}")

buffer = ReplayBuffer(tcfg.buffer_size, N)

# ---- Stage 1: run self-play ----
sp = ParallelSelfPlay(net, device, mcfg, tcfg, num_workers=2)
stats = sp.run_games(tcfg.num_games_per_iter, buffer)
print(f"\n[1] Self-play: {stats['games']} games, "
      f"{stats['positions']} positions, "
      f"buffer={stats['buffer_size']}, {stats['time']:.1f}s")

# ---- Stage 1b: validate buffer contents ----
print("\n[1b] Buffer sanity checks")
assert len(buffer) > 0, "replay buffer empty"
obs_arr = buffer.obs[:len(buffer)]
pol_arr = buffer.policy[:len(buffer)]
val_arr = buffer.value[:len(buffer)]
own_arr = buffer.ownership[:len(buffer)]

# obs
assert obs_arr.dtype == np.uint8, f"obs dtype should be uint8, got {obs_arr.dtype}"
assert obs_arr.shape[1:] == (mcfg.input_planes, N, N), \
    f"obs shape {obs_arr.shape} doesn't match ({mcfg.input_planes}, {N}, {N})"
obs_vals = np.unique(obs_arr)
assert set(obs_vals.tolist()).issubset({0, 1}), \
    f"obs not binary: {obs_vals[:10]}"
print(f"  obs shape {obs_arr.shape} dtype {obs_arr.dtype}, binary ✓")

# policy
assert np.isfinite(pol_arr).all(), "NaN/inf in policy"
pol_sums = pol_arr.sum(axis=1)
assert np.allclose(pol_sums, 1.0, atol=1e-3), \
    f"policy doesn't sum to 1: min={pol_sums.min()} max={pol_sums.max()}"
assert pol_arr.shape[1] == N * N + 1, \
    f"policy width {pol_arr.shape[1]} != {N*N+1} actions"
print(f"  policy sums to 1 ✓, {pol_arr.shape[1]} actions")

# value
assert np.isfinite(val_arr).all(), "NaN/inf in value"
assert np.all(val_arr >= -1.0 - 1e-5) and np.all(val_arr <= 1.0 + 1e-5), \
    f"value out of [-1,1]: min={val_arr.min()} max={val_arr.max()}"
print(f"  value in [-1,1] ✓ (mean={val_arr.mean():.3f})")

# ownership
own_unique = np.unique(own_arr)
assert set(own_unique.tolist()).issubset({-1, 0, 1}), \
    f"ownership not in {{-1, 0, +1}}: {own_unique}"
assert own_arr.shape[1:] == (N, N), \
    f"ownership shape {own_arr.shape} doesn't match (*, {N}, {N})"
print(f"  ownership in {{-1,0,+1}} ✓, shape {own_arr.shape}")

# ---- Stage 1c: uint8 obs round-trip + save/load ----
print("\n[1c] uint8 obs round-trip + save/load")

obs_f32_orig = buffer.obs[0].astype(np.float32)
tmp_buf = ReplayBuffer(16, N)
tmp_buf.push(obs_f32_orig, buffer.policy[0], buffer.value[0],
             buffer.ownership[0])
obs_back = tmp_buf.obs[0].astype(np.float32)
assert np.array_equal(obs_f32_orig, obs_back), \
    "uint8 round-trip lost information — obs values outside {0, 1}?"
print(f"  float32→uint8→float32 round-trip exact ✓")

with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmpf:
    tmp_path = tmpf.name
try:
    buffer.save_to(tmp_path)
    restored = ReplayBuffer(tcfg.buffer_size, N)
    restored.load_from(tmp_path)
    assert restored.size == buffer.size
    assert restored.obs.dtype == np.uint8
    assert np.array_equal(restored.obs[:restored.size],
                          buffer.obs[:buffer.size])
    assert np.array_equal(restored.policy[:restored.size],
                          buffer.policy[:buffer.size])
    assert np.array_equal(restored.value[:restored.size],
                          buffer.value[:buffer.size])
    assert np.array_equal(restored.ownership[:restored.size],
                          buffer.ownership[:buffer.size])
    print(f"  save_to / load_from round-trip exact ✓ "
          f"({restored.size} samples)")
finally:
    os.unlink(tmp_path)

# ---- Stage 2: training loss decreases, components in sane range ----
print("\n[2] Training loss trajectory")
trainer = Trainer(net, tcfg, device)
all_stats = []
for step in range(3):
    stats = trainer.train_epoch(buffer, 1)
    all_stats.append(stats)
    print(f"  epoch {step}: loss={stats['loss']:.4f} "
          f"(pi={stats['policy_loss']:.4f}, "
          f"score={stats['score_loss']:.4f}, "
          f"own={stats['ownership_loss']:.4f})")

assert all_stats[-1]["loss"] < all_stats[0]["loss"], \
    f"loss not decreasing: {all_stats[0]['loss']:.4f} → {all_stats[-1]['loss']:.4f}"
print(f"  total loss decreasing ✓")

# Score loss should be in same order as policy, not 100x+
pi0 = all_stats[0]["policy_loss"]
sc0 = all_stats[0]["score_loss"]
assert sc0 < 50 * pi0, \
    f"score_loss ({sc0:.1f}) is >50x policy_loss ({pi0:.4f}) — " \
    f"likely unnormalized targets"
print(f"  score_loss / pi_loss = {sc0 / pi0:.2f} (< 50x) ✓")

# Ownership loss should be near or below 0.693 (BCE entropy floor)
own0 = all_stats[0]["ownership_loss"]
assert own0 < 2.0, \
    f"ownership_loss ({own0:.4f}) unexpectedly high — check target mapping"
print(f"  ownership_loss = {own0:.4f} (< 2.0) ✓")

# ---- Stage 3: MCTS-vs-random eval ----
print("\n[3] Evaluate vs random (5 games)")
win_rate = evaluate_vs_random(net, device, mcfg, tcfg, num_games=5)
print(f"  MCTS(trained) vs random: {win_rate:.0%} win rate")
print(f"  completed eval without crashes ✓")

# ---- Stage 4: score head value signal for MCTS ----
print("\n[4] Score head value signal")

obs_sample, _, _, _ = buffer.sample(min(256, len(buffer)))
obs_t = torch.from_numpy(obs_sample.astype(np.float32)).to(device)
net.eval()
with torch.no_grad():
    _, values, _, scores = net(obs_t)

val_range = values.max().item() - values.min().item()
score_std = scores.std().item()
print(f"  Score predictions: std={score_std:.4f}")
print(f"  Value predictions: range={val_range:.4f}")

assert val_range > 0.05, \
    f"derived value range too narrow ({val_range:.4f}) — " \
    f"MCTS will have no signal. Check value_scale init and score normalization."
print(f"  value range > 0.05 ✓ — MCTS has value signal")

assert score_std > 0.01, \
    f"score predictions have no variance ({score_std:.6f}) — " \
    f"score head not learning"
print(f"  score std > 0.01 ✓")

# ---- Stage 5: pass floor enforcement ----
print("\n[5] Pass floor enforcement")

pass_min = tcfg.pass_min_move
assert pass_min > 0, \
    f"pass_min_move is {pass_min} on 13x13 — should be >0"
print(f"  pass_min_move = {pass_min} ✓")

PASS_ACTION = N * N  # 169
pol_all = buffer.policy[:len(buffer)]
obs_all = buffer.obs[:len(buffer)]
# Approximate move_num by total stones on board.
# Use pass_min/2 as threshold — conservative margin so captures can't
# push move_num≥60 positions below the stone threshold.
stones = obs_all[:, 0, :, :].sum(axis=(1, 2)) + \
         obs_all[:, 8, :, :].sum(axis=(1, 2))
early_threshold = pass_min // 2
early_mask = stones < early_threshold
n_early = early_mask.sum()

if n_early > 0:
    early_pass_probs = pol_all[early_mask, PASS_ACTION]
    max_early_pass = early_pass_probs.max()
    nonzero_count = (early_pass_probs > 1e-6).sum()
    print(f"  Early-game positions (stones < {early_threshold}): {n_early}")
    print(f"  Pass prob in early positions: max={max_early_pass:.6f}, "
          f"nonzero={nonzero_count}/{n_early}")
    assert max_early_pass < 1e-4, \
        f"pass prob ({max_early_pass:.6f}) in early-game policy target — " \
        f"pass floor not zeroing stored targets (v1 bug)"
    print(f"  pass zeroed in early-game targets ✓")
else:
    print(f"  WARNING: no early-game positions found (all games > "
          f"{early_threshold} stones). Test inconclusive.")

# ---- Stage 6: gradient flow ----
print("\n[6] Gradient flow")

net_grad = AlphaZeroNet(mcfg).to(device)
net_grad.train()
trainer_grad = Trainer(net_grad, tcfg, device)
trainer_grad.optimizer.zero_grad()
trainer_grad.train_step(buffer)

# With vlw=0: value_scale/value_bias get no gradient (by design).
no_grad_expected = set()
if tcfg.value_loss_weight == 0.0:
    no_grad_expected = {"value_scale", "value_bias"}

dead_params = []
alive_params = []
for name, param in net_grad.named_parameters():
    has_grad = param.grad is not None and param.grad.abs().max().item() > 0
    if not has_grad:
        if name not in no_grad_expected:
            dead_params.append(name)
        else:
            print(f"  {name}: no gradient (expected, vlw=0)")
    else:
        alive_params.append(name)

total_params = len(list(net_grad.parameters()))
print(f"  Parameters with gradient: {len(alive_params)}/{total_params}")

if dead_params:
    for name in dead_params:
        print(f"  DEAD: {name}")
    assert False, \
        f"{len(dead_params)} parameters have no gradient but should: {dead_params}"
print(f"  all expected parameters receive gradients ✓")

score_params = [n for n in alive_params if "score" in n]
assert len(score_params) > 0, \
    "score head has no gradient — score_loss_weight may be 0"
print(f"  score head receiving gradients ✓")

# ---- Stage 7: engine build freshness ----
print("\n[7] Engine build freshness")

engine_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine")
so_files = glob.glob(os.path.join(engine_dir, "*.so"))
h_files = glob.glob(os.path.join(engine_dir, "*.h"))

if so_files and h_files:
    newest_so = max(os.path.getmtime(f) for f in so_files)
    stale_headers = []
    for h in h_files:
        if os.path.getmtime(h) > newest_so:
            stale_headers.append(os.path.basename(h))

    if stale_headers:
        so_name = os.path.basename(max(so_files, key=os.path.getmtime))
        print(f"  WARNING: {stale_headers} are NEWER than {so_name}")
        print(f"  Run: cd engine && rm -rf build *.so && "
              f"python setup.py build_ext --inplace")
        assert False, \
            f"Engine headers {stale_headers} modified after .so build."
    else:
        print(f"  .so newer than all {len(h_files)} header files ✓")
else:
    if not so_files:
        print(f"  WARNING: no .so found in {engine_dir}")
    if not h_files:
        print(f"  WARNING: no .h found in {engine_dir}")

print("\n" + "=" * 60)
print("ALL CORRECTNESS CHECKS PASSED")
print("=" * 60)
