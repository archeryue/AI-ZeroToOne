"""Phase 2 run4 offline A/B for the KataGo-style ownership head.

Run3 falsified every "simple recipe" candidate (see
_phase2_offline_ab.py for the run3 narrative). Held-out value MSE
went UP for all 14 candidates; the only "least-bad" one (R2,
30 steps) failed live in iter 1 with the same +0.083 v_loss drift
and game-length collapse 146 → 56 as run1.

Run4 escalates to a per-cell ownership auxiliary head: ~169×
denser supervision per position, which forces the trunk to learn
spatial features that generalize. The decisive offline check is
whether ANY recipe gets held-out value MSE BELOW the cold floor
of ~1.00 — something no run3 recipe achieved.

Phases:
    gen-buffer  → fresh cold buffer (now WITH ownership labels) at
                  production sims/games settings.
    run-ab      → load cold weights for each recipe and train against
                  the same buffer; report held-out value MSE,
                  ownership BCE, value-head saturation, resign-trigger
                  proxy, ownership-loss trajectory.

Usage:
    PYTHONPATH=engine python training/_phase2_run4_offline_ab.py gen-buffer
    PYTHONPATH=engine python training/_phase2_run4_offline_ab.py run-ab
"""

import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
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
ART_DIR = "checkpoints/_run4_offline_ab"
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
    train_cfg = copy.deepcopy(train_cfg)
    train_cfg.num_games_per_iter = 256
    train_cfg.num_parallel_games = 256
    train_cfg.buffer_size = 80_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] device={device} model={model_cfg.num_blocks}b×{model_cfg.channels}ch")

    net = AlphaZeroNet(model_cfg).to(device)
    print(f"[gen] params={net.param_count():,}")
    torch.save(net.state_dict(), COLD_WEIGHTS)
    print(f"[gen] saved cold weights → {COLD_WEIGHTS}")

    buffer = ReplayBuffer(train_cfg.buffer_size, model_cfg.board_size)
    sp = ParallelSelfPlay(net, device, model_cfg, train_cfg, num_workers=5)

    print(f"[gen] running 256 cold games at sims={train_cfg.num_simulations}...")
    t0 = time.time()
    stats = sp.run_games(train_cfg.num_games_per_iter, buffer)
    dt = time.time() - t0

    avg_moves = stats["positions"] / max(stats["games"], 1)
    print(f"[gen] done: {stats['games']} games, {stats['positions']} positions, "
          f"{avg_moves:.0f} avg moves, {dt:.0f}s")

    buffer.save_to(COLD_BUFFER)
    print(f"[gen] saved buffer → {COLD_BUFFER} ({len(buffer)} samples)")

    val = buffer.value[:len(buffer)]
    own = buffer.ownership[:len(buffer)]
    pos_frac = float(np.mean(val > 0))
    own_pos = float(np.mean(own > 0))
    own_neg = float(np.mean(own < 0))
    own_dame = float(np.mean(own == 0))
    print(f"[gen] value targets: +1 frac={pos_frac:.3f}, mean={float(val.mean()):+.3f}")
    print(f"[gen] ownership: +1 frac={own_pos:.3f}, -1 frac={own_neg:.3f}, "
          f"dame frac={own_dame:.3f}")


def held_out_metrics(net, buffer, indices, device):
    """Compute held-out value MSE, ownership BCE, saturation, resign frac."""
    net.eval()
    obs = torch.from_numpy(buffer.obs[indices].astype(np.float32)).to(device)
    pol = torch.from_numpy(buffer.policy[indices]).to(device)
    val = torch.from_numpy(buffer.value[indices]).to(device)
    own = torch.from_numpy(buffer.ownership[indices]).to(device).float()
    own_01 = (own + 1.0) / 2.0

    with torch.no_grad():
        logits, v, own_logits = net(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        p_loss = -(pol * log_probs).sum(dim=-1).mean().item()
        v_loss_mse = F.mse_loss(v, val).item()
        own_loss_bce = F.binary_cross_entropy_with_logits(own_logits, own_01).item()
        sat = float(((v.abs() > 0.95).float()).mean().item())
        below_resign = float(((v < -0.9).float()).mean().item())
        v_mean = float(v.mean().item())
        v_std = float(v.std().item())
    return {
        "p_loss": p_loss,
        "v_loss_mse": v_loss_mse,
        "own_loss_bce": own_loss_bce,
        "sat_frac": sat,
        "below_resign_frac": below_resign,
        "v_mean": v_mean,
        "v_std": v_std,
    }


def train_recipe(name, train_steps, vlw, ow_weight, lr, buffer,
                 model_cfg, device):
    """Train cold init with one recipe, return pre/post metrics."""
    net = AlphaZeroNet(model_cfg).to(device)
    net.load_state_dict(torch.load(COLD_WEIGHTS, map_location=device))

    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    n = len(buffer)
    held_out_idx = np.arange(n - 5000, n)
    train_idx_pool = np.arange(0, n - 5000)
    rng = np.random.default_rng(SEED + 1)

    pre = held_out_metrics(net, buffer, held_out_idx, device)

    train_v_losses = []
    train_p_losses = []
    train_o_losses = []
    BATCH = 256
    for step in range(train_steps):
        net.train()
        idx = rng.choice(train_idx_pool, size=BATCH, replace=False)
        obs = torch.from_numpy(buffer.obs[idx].astype(np.float32)).to(device)
        pol = torch.from_numpy(buffer.policy[idx]).to(device)
        val = torch.from_numpy(buffer.value[idx]).to(device)
        own = torch.from_numpy(buffer.ownership[idx]).to(device).float()
        own_01 = (own + 1.0) / 2.0

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, v, own_logits = net(obs)
            log_probs = F.log_softmax(logits, dim=-1)
            p_loss = -(pol * log_probs).sum(dim=-1).mean()
            v_loss = F.mse_loss(v, val)
            if ow_weight > 0.0:
                o_loss = F.binary_cross_entropy_with_logits(own_logits, own_01)
            else:
                o_loss = torch.zeros((), device=device)
            loss = p_loss + vlw * v_loss + ow_weight * o_loss

        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

        train_v_losses.append(float(v_loss.item()))
        train_p_losses.append(float(p_loss.item()))
        train_o_losses.append(float(o_loss.item()))

    post = held_out_metrics(net, buffer, held_out_idx, device)

    def _avg(xs, slc):
        if len(xs) < 5:
            return float("nan")
        return float(np.mean(xs[slc]))

    return {
        "name": name,
        "train_steps": train_steps,
        "vlw": vlw,
        "ow_weight": ow_weight,
        "lr": lr,
        "pre_v_mse": pre["v_loss_mse"],
        "post_v_mse": post["v_loss_mse"],
        "delta_v_mse": post["v_loss_mse"] - pre["v_loss_mse"],
        "pre_own_bce": pre["own_loss_bce"],
        "post_own_bce": post["own_loss_bce"],
        "delta_own_bce": post["own_loss_bce"] - pre["own_loss_bce"],
        "train_v_first5": _avg(train_v_losses, slice(0, 5)),
        "train_v_last5": _avg(train_v_losses, slice(-5, None)),
        "train_o_first5": _avg(train_o_losses, slice(0, 5)),
        "train_o_last5": _avg(train_o_losses, slice(-5, None)),
        "post_sat_frac": post["sat_frac"],
        "post_below_resign": post["below_resign_frac"],
        "post_v_mean": post["v_mean"],
        "post_v_std": post["v_std"],
    }


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
    own = buffer.ownership[:len(buffer)]
    print(f"[ab] value mean={float(val.mean()):+.3f}, "
          f"+1 frac={float(np.mean(val > 0)):.3f}")
    print(f"[ab] ownership +1/-1/0 = {float(np.mean(own>0)):.3f}/"
          f"{float(np.mean(own<0)):.3f}/{float(np.mean(own==0)):.3f}")
    print()

    # Recipes — Path A second pass: focus on steps=30 + lower vlw +
    # ownership weight. Hypothesis: with ownership driving trunk
    # learning, we can dial DOWN the value-loss weight to slow the
    # value head's overfit on noisy labels, and the trunk-via-
    # ownership keeps providing the spatial supervision the value
    # head needs to read off later. Goal: find post_v_mse < 1.0.
    # Keep two from the first pass as anchor points.
    recipes = [
        # name,                steps, vlw, ow,  lr
        ("A0-OW0-baseline",      100, 2.0, 0.0,  0.005),  # = run3 R1
        ("A1-OW1.5-30-anchor",    30, 2.0, 1.5,  0.005),  # best of pass1
        # Path A: low vlw + ownership at steps=30
        ("A2-30-vlw0.5-ow1.5",    30, 0.5, 1.5,  0.005),
        ("A3-30-vlw0.5-ow2.0",    30, 0.5, 2.0,  0.005),
        ("A4-30-vlw0.5-ow3.0",    30, 0.5, 3.0,  0.005),
        ("A5-30-vlw0.25-ow2.0",   30, 0.25, 2.0, 0.005),
        ("A6-30-vlw0-ow2.0",      30, 0.0, 2.0,  0.005),  # extreme: no value
        ("A7-50-vlw1-ow1.5",      50, 1.0, 1.5,  0.005),
        ("A8-50-vlw0.5-ow2.0",    50, 0.5, 2.0,  0.005),
        ("A9-30-vlw0.5-ow1.5-lo", 30, 0.5, 1.5,  0.001),  # low LR
    ]

    rows = []
    for spec in recipes:
        name, steps, vlw, ow, lr = spec
        t0 = time.time()
        row = train_recipe(name, steps, vlw, ow, lr, buffer, model_cfg, device)
        row["wall"] = time.time() - t0
        rows.append(row)
        print(f"[{name}] v: {row['pre_v_mse']:.4f}->{row['post_v_mse']:.4f} "
              f"(Δ{row['delta_v_mse']:+.4f}) | "
              f"own: {row['pre_own_bce']:.4f}->{row['post_own_bce']:.4f} "
              f"(Δ{row['delta_own_bce']:+.4f}) | "
              f"resign={row['post_below_resign']*100:.1f}% "
              f"sat={row['post_sat_frac']:.2f} "
              f"({row['wall']:.0f}s)")

    print("\n=== Summary (held-out v MSE; success = post < 1.00) ===")
    print(f"{'recipe':16} {'pre_v':>7} {'post_v':>7} {'Δv':>8} "
          f"{'tro_1st':>8} {'tro_lst':>8} {'sat':>5} {'resign%':>8}")
    for r in rows:
        winner = " ★" if r["post_v_mse"] < 1.0 else ""
        print(f"{r['name']:16} {r['pre_v_mse']:7.4f} "
              f"{r['post_v_mse']:7.4f} {r['delta_v_mse']:+8.4f} "
              f"{r['train_o_first5']:8.4f} {r['train_o_last5']:8.4f} "
              f"{r['post_sat_frac']:5.2f} {r['post_below_resign']*100:7.1f}%"
              f"{winner}")

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
