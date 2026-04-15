"""Phase 2 offline A/B for value-head fixes.

Two-phase script:
  Phase A: generate a cold 13x13 self-play buffer with seeded RNGs, save
           the cold network weights AND the buffer to disk.
  Phase B: load the cold weights, run each candidate recipe against the
           same buffer, measure post-train value loss on a held-out
           chunk and report.

The held-out value loss is the key metric: if a recipe brings post-train
held-out v_loss BELOW the cold baseline, the value head learned something
useful from the cold targets. If it goes UP, the head drifted away from
truth (the run1/run2 failure mode).

Usage:
    PYTHONPATH=engine python training/_phase2_offline_ab.py gen-buffer
    PYTHONPATH=engine python training/_phase2_offline_ab.py run-ab
"""

import argparse
import copy
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Skip torch.compile — slows the first few hundred ticks down massively
# in a short run, no benefit at this scale.
os.environ.setdefault("AZ_COMPILE", "0")
os.environ.setdefault("AZ_PROGRESS_INTERVAL", "15")

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_float32_matmul_precision("high")

from model.config import CONFIGS, ModelConfig, TrainingConfig
from model.network import AlphaZeroNet
from training.replay_buffer import ReplayBuffer
from training.parallel_self_play import ParallelSelfPlay


SEED = 42
ART_DIR = "checkpoints/_offline_ab"
COLD_WEIGHTS = f"{ART_DIR}/cold_weights.pt"
COLD_BUFFER = f"{ART_DIR}/cold_buffer.npz"


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gen_buffer():
    set_all_seeds(SEED)
    os.makedirs(ART_DIR, exist_ok=True)

    model_cfg, train_cfg = CONFIGS[13]
    # Tighten down: just one game per parallel slot. ~256 games gives
    # ~30k positions in ~5 minutes.
    train_cfg = copy.deepcopy(train_cfg)
    train_cfg.num_games_per_iter = 256
    train_cfg.num_parallel_games = 256
    # Smaller buffer to fit in memory comfortably.
    train_cfg.buffer_size = 80_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] device={device} model={model_cfg.num_blocks}b×{model_cfg.channels}ch")

    net = AlphaZeroNet(model_cfg).to(device)
    print(f"[gen] params={net.param_count():,}")

    # Save cold weights — the EXACT random init we'll reload for every
    # recipe so the experiment is controlled.
    torch.save(net.state_dict(), COLD_WEIGHTS)
    print(f"[gen] saved cold weights → {COLD_WEIGHTS}")

    buffer = ReplayBuffer(train_cfg.buffer_size, model_cfg.board_size)
    sp = ParallelSelfPlay(net, device, model_cfg, train_cfg, num_workers=5)

    print(f"[gen] running 256 cold self-play games at sims={train_cfg.num_simulations}...")
    t0 = time.time()
    stats = sp.run_games(train_cfg.num_games_per_iter, buffer)
    dt = time.time() - t0

    avg_moves = stats["positions"] / max(stats["games"], 1)
    print(f"[gen] done: {stats['games']} games, {stats['positions']} positions, "
          f"{avg_moves:.0f} avg moves, {dt:.0f}s")

    buffer.save_to(COLD_BUFFER)
    print(f"[gen] saved buffer → {COLD_BUFFER} ({len(buffer)} samples)")

    # Quick stats on the buffer
    val = buffer.value[:len(buffer)]
    pos_frac = float(np.mean(val > 0))
    print(f"[gen] value targets: +1 fraction={pos_frac:.3f}, "
          f"mean={float(val.mean()):+.3f}, std={float(val.std()):.3f}")


def forward_with_value_logit(net, obs):
    """Like net(obs) but returns the pre-tanh value logit instead of v.

    Lets us train with BCE on the logit (a numerically stable
    classification loss) without modifying AlphaZeroNet.
    """
    out = F.relu(net.input_bn(net.input_conv(obs)))
    out = net.residual_tower(out)
    p = F.relu(net.policy_bn(net.policy_conv(out)))
    p = p.flatten(1)
    policy_logits = net.policy_fc(p)
    v = F.relu(net.value_bn(net.value_conv(out)))
    v = v.flatten(1)
    v = F.relu(net.value_fc1(v))
    v_logit = net.value_fc2(v).squeeze(-1)
    return policy_logits, v_logit


def held_out_value_loss(net, buffer, indices, device, loss_fn="mse",
                        delta=1.0):
    """Compute mean policy/value loss on a fixed held-out subset."""
    net.eval()
    obs = torch.from_numpy(buffer.obs[indices].astype(np.float32)).to(device)
    pol = torch.from_numpy(buffer.policy[indices]).to(device)
    val = torch.from_numpy(buffer.value[indices]).to(device)
    with torch.no_grad():
        logits, v = net(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        p_loss = -(pol * log_probs).sum(dim=-1).mean().item()
        v_loss_mse = F.mse_loss(v, val).item()
        sat = float(((v.abs() > 0.95).float()).mean().item())
        # Resign-trigger proxy: fraction of held-out positions where the
        # raw value head output is below -0.9. In real self-play this is
        # the dominant signal for the resign loop — if a recipe pushes
        # this above ~30% the next iter's self-play will collapse to
        # short resigned games like run1/run2.
        below_resign = float(((v < -0.9).float()).mean().item())
        v_mean = float(v.mean().item())
        v_std = float(v.std().item())
    return {"p_loss": p_loss, "v_loss_mse": v_loss_mse,
            "sat_frac": sat, "v_mean": v_mean, "v_std": v_std,
            "below_resign_frac": below_resign}


def train_recipe(name, train_steps, vlw, loss_fn, lr, buffer, model_cfg,
                 device, vlw_warmup=False, value_only=False):
    """Train cold init with one recipe, return pre/post metrics."""
    net = AlphaZeroNet(model_cfg).to(device)
    net.load_state_dict(torch.load(COLD_WEIGHTS, map_location=device))

    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # Held-out indices: last 5000 samples, never sampled for training.
    n = len(buffer)
    held_out_idx = np.arange(n - 5000, n)
    train_idx_pool = np.arange(0, n - 5000)
    rng = np.random.default_rng(SEED + 1)

    pre = held_out_value_loss(net, buffer, held_out_idx, device,
                              loss_fn="mse")

    train_v_losses = []
    train_p_losses = []
    skipped = 0
    BATCH = 256
    for step in range(train_steps):
        net.train()
        idx = rng.choice(train_idx_pool, size=BATCH, replace=False)
        obs = torch.from_numpy(buffer.obs[idx].astype(np.float32)).to(device)
        pol = torch.from_numpy(buffer.policy[idx]).to(device)
        val = torch.from_numpy(buffer.value[idx]).to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            if loss_fn == "bce":
                # Bypass the tanh: train a calibrated logit head.
                # target = P(win) ∈ {0, 1}; loss is sigmoid CE.
                # Equivalent to a 2-class WDL formulation, which is
                # the cheap test of the "loss-formulation instability"
                # hypothesis vs the existing tanh+MSE recipe.
                logits, v_logit = forward_with_value_logit(net, obs)
                bce_target = (val + 1.0) / 2.0
                v_loss = F.binary_cross_entropy_with_logits(v_logit, bce_target)
            else:
                logits, v = net(obs)
                if loss_fn == "huber":
                    v_loss = F.huber_loss(v, val, delta=1.0)
                else:
                    v_loss = F.mse_loss(v, val)
            log_probs = F.log_softmax(logits, dim=-1)
            p_loss = -(pol * log_probs).sum(dim=-1).mean()
            if vlw_warmup:
                progress = step / max(train_steps - 1, 1)
                eff_vlw = vlw + (2.0 - vlw) * progress
            else:
                eff_vlw = vlw
            if value_only:
                # Train value head ONLY: zero out policy loss so the
                # trunk only sees gradients from the value objective.
                # Tests whether policy-driven trunk drift is the root
                # of the value head's instability.
                loss = eff_vlw * v_loss
            else:
                loss = p_loss + eff_vlw * v_loss

        if not torch.isfinite(loss):
            skipped += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

        train_v_losses.append(float(v_loss.item()))
        train_p_losses.append(float(p_loss.item()))

    post = held_out_value_loss(net, buffer, held_out_idx, device,
                               loss_fn="mse")

    train_v_avg = float(np.mean(train_v_losses)) if train_v_losses else float("nan")
    train_p_avg = float(np.mean(train_p_losses)) if train_p_losses else float("nan")
    train_v_first = float(np.mean(train_v_losses[:5])) if len(train_v_losses) >= 5 else float("nan")
    train_v_last = float(np.mean(train_v_losses[-5:])) if len(train_v_losses) >= 5 else float("nan")

    return {
        "name": name,
        "train_steps": train_steps,
        "vlw": vlw,
        "loss_fn": loss_fn,
        "lr": lr,
        "vlw_warmup": vlw_warmup,
        "skipped": skipped,
        "pre_v_mse_held": pre["v_loss_mse"],
        "post_v_mse_held": post["v_loss_mse"],
        "delta_v_mse": post["v_loss_mse"] - pre["v_loss_mse"],
        "pre_p_held": pre["p_loss"],
        "post_p_held": post["p_loss"],
        "train_v_avg": train_v_avg,
        "train_v_first5": train_v_first,
        "train_v_last5": train_v_last,
        "train_v_drift": train_v_last - train_v_first,
        "train_p_avg": train_p_avg,
        "post_v_mean": post["v_mean"],
        "post_v_std": post["v_std"],
        "post_sat_frac": post["sat_frac"],
        "pre_below_resign": pre["below_resign_frac"],
        "post_below_resign": post["below_resign_frac"],
    }, net


def run_ab():
    set_all_seeds(SEED)
    if not os.path.exists(COLD_WEIGHTS) or not os.path.exists(COLD_BUFFER):
        print("ERROR: run `gen-buffer` first")
        sys.exit(1)

    model_cfg, _ = CONFIGS[13]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer(80_000, model_cfg.board_size)
    buffer.load_from(COLD_BUFFER)
    print(f"[ab] loaded buffer: {len(buffer)} samples")
    val = buffer.value[:len(buffer)]
    print(f"[ab] value mean={float(val.mean()):+.3f} std={float(val.std()):.3f} "
          f"+1_frac={float(np.mean(val > 0)):.3f}")
    print()

    # Compute pre-training cold baseline once (it's deterministic from
    # weights, so we don't need to re-measure per recipe — but each
    # train_recipe call reloads cold weights and computes pre too).
    recipes = [
        # name,         steps, vlw, loss,  lr,    warmup, value_only
        # Loss-formulation tests (the most decisive). bce/bce_clip
        # bypass the tanh and use BCE-with-logits on the pre-tanh
        # output, treating value as P(win)→±1. Equivalent to a 2-class
        # WDL head with calibrated probabilities. If BCE recipes show
        # Δv ≤ 0 where MSE recipes don't, WDL escalation is justified.
        ("B1-bce100",    100, 2.0, "bce",  0.005, False, False),
        ("B2-bce30",      30, 2.0, "bce",  0.005, False, False),
        ("B3-bce-low",   100, 2.0, "bce",  0.001, False, False),
        ("B4-bce-vlw1",  100, 1.0, "bce",  0.005, False, False),
    ]

    rows = []
    for spec in recipes:
        name, steps, vlw, lossfn, lr, warmup, vonly = spec
        t0 = time.time()
        row, _ = train_recipe(name, steps, vlw, lossfn, lr, buffer,
                              model_cfg, device, vlw_warmup=warmup,
                              value_only=vonly)
        row["wall"] = time.time() - t0
        rows.append(row)
        print(f"[{name}] pre={row['pre_v_mse_held']:.4f} "
              f"post={row['post_v_mse_held']:.4f} "
              f"Δ={row['delta_v_mse']:+.4f} "
              f"resign%={row['post_below_resign']*100:.1f}% "
              f"sat={row['post_sat_frac']:.2f} "
              f"vμ={row['post_v_mean']:+.3f} vσ={row['post_v_std']:.3f} "
              f"({row['wall']:.0f}s)")

    print("\n=== Summary (held-out value MSE; lower-after = recipe converged) ===")
    print(f"{'recipe':12} {'pre_v':>7} {'post_v':>7} {'Δv':>8} "
          f"{'tr1st':>7} {'trlst':>7} {'sat':>5} {'resign%':>8}")
    for r in rows:
        print(f"{r['name']:12} {r['pre_v_mse_held']:7.4f} "
              f"{r['post_v_mse_held']:7.4f} {r['delta_v_mse']:+8.4f} "
              f"{r['train_v_first5']:7.4f} {r['train_v_last5']:7.4f} "
              f"{r['post_sat_frac']:5.2f} {r['post_below_resign']*100:7.1f}%")

    # Save raw rows to JSON for easy reference
    import json
    out_path = f"{ART_DIR}/ab_results.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["gen-buffer", "run-ab"])
    args = parser.parse_args()
    if args.phase == "gen-buffer":
        gen_buffer()
    else:
        run_ab()
