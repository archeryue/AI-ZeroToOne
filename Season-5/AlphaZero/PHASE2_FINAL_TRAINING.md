# Phase 2 Final Training — 13x13 Go AlphaZero (Run 5)

Living document for the 13x13 Phase 2 final run. Previous runs (1-4d)
are documented in `PHASE_TWO_TRAINING.md`.

---

## Architecture

```
Input (17, 13, 13)
  → Conv3x3(17→128) + LayerNorm + ReLU
  → 15 × ResidualBlock(Conv3x3+LN+ReLU → Conv3x3+LN + skip → ReLU)
  → Policy Head:    Conv1x1(128→2) + LN + ReLU + FC(338→170)
  → Score Head:     Conv1x1(128→1) + LN + ReLU + FC(169→32→1)
  → Ownership Head: Conv1x1(128→1, bias=True) → (B, 13, 13) logits
  → Value derived:  tanh(value_scale * score_pred + value_bias)
```

- **Parameters**: ~5.8M (15 blocks × 128 channels)
- **LayerNorm** everywhere (no BatchNorm). Eliminates train/eval drift.
- **Score head** (5.5k params) replaces value MLP (44k params). Predicts
  /N-normalized territory margin. Can't memorize cold data.
- **Value derived** from score: `tanh(1.0 * score + 0.0)`. Two frozen
  scalars (vlw=0), well-calibrated at init.
- **Ownership head**: KataGo-style auxiliary. Dense per-cell BCE
  regularizes trunk. Not used during MCTS inference.
- **No dropout, no model gating** — same as KataGo / Leela Zero.

---

## Training Config

```
Model:           15b × 128ch (~5.8M params)
Sims/move:       400
c_puct:          1.5
Parallel games:  128
Games/iter:      1024
Iterations:      60
Buffer:          1M positions (uint8 obs, augment-on-sample)
Batch:           1024
Train steps/it:  50
LR:              0.00125 → 0.0001 cosine
Optimizer:       SGD + momentum 0.9, weight_decay 1e-4
Grad clip:       max_norm=5.0
Dirichlet:       α=0.07, ε=0.25
Temperature:     τ=1.0 for first 40 moves, then τ=0.25
Max game moves:  250
Pass floor:      pass_min_move=60 (zeroed in both action + target)
Resign:          thresh=-0.95, consec=5, min_move=80,
                 disabled_frac=0.20, min_child_visits_frac=0.05
```

### Loss function

```
L = -π·log(p) + 1.0·(score_target - score_pred)² + 1.5·BCE(own) + 1e-4·||θ||²
```

- **Policy**: cross-entropy with MCTS visit distribution
- **Score**: MSE on /N-normalized territory margin (std≈1.5)
- **Ownership**: per-cell BCE-with-logits ({-1,0,+1} → {0,0.5,1})
- **Value loss weight = 0**: value is derived from score, not trained directly

---

## Go/no-go gates

**GREEN** (let it run):
- `score_loss` decreasing or flat across iters 0→5
- `eval_vs_random` ≥ 15% by iter 5, non-regressing
- `avg_moves/game` stays in 100-200 range
- Memory stable, no OOM

**RED** (kill, investigate):
- `eval_vs_random` stuck at 0% for 3 consecutive iters
- `avg_moves/game` < 80 (resign or pass collapse)
- Score predictions have no variance (value signal dead)
- Any crash / OOM

---

## Key decisions and rationale

**Why score head instead of value MLP?** The 44k-param value MLP
memorized cold ±1 labels in <1 epoch across runs 1-3 and the
standard-value-MLP restart (see failed attempts below). The 5.5k-param
score head predicts territory margin — denser signal than binary
win/loss, fewer params, can't memorize.

**Why LayerNorm?** BatchNorm's running stats caused train/eval drift
in Phase 1 (Problem 3) and were a candidate for Phase 2 strength
drift. LayerNorm has no running state. KataGo uses LayerNorm.

**Why ownership head?** 169× supervision density vs scalar value.
Regularizes the trunk with dense spatial labels from the start.
Training-only auxiliary (zero MCTS inference cost). Same as KataGo.

---

## Failed attempts (condensed)

### Attempt 1: Standard value MLP + LayerNorm (seed 42, batch 256)

Eval degraded 32→14→6% over 3 iters. Value MLP memorized despite
LayerNorm and ownership auxiliary.

### Attempt 2: Bigger batch (seed 100, batch 1024, lr 0.00125)

Eval 43→0% at iter 1. v_loss crossed cold floor (1.01 > 1.0).
Bigger batch did NOT prevent value head overfit.

### Attempt 3: Score head, unnormalized targets (seed 200)

Score head architecture correct — v_loss stuck at cold floor (can't
memorize). But score targets were raw territory margin (±50 range)
with score_loss_weight=0.01. Score head got near-zero gradient,
predicted ~0 for all positions. Derived value ≈ 0.016 everywhere.
MCTS blind. Eval 0%/0% at iters 0-1.

**Root cause**: implementation bug — target not normalized by /N.
Fixed: `target_score = ownership.sum() / N`, `score_loss_weight=1.0`,
`value_scale=1.0`. Verified by `_test_correctness.py` stage 4.

---

## Current run: score head with /N normalization (seed 300)

Score head with proper target normalization. Fresh start.

```bash
PYTHONPATH=engine PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
nohup python3 -u -m training.train \
    --board-size 13 --iterations 60 --num-workers 5 \
    --games-per-iter 1024 \
    --output-dir checkpoints/13x13_run5 \
    --seed 300 \
    > logs/13x13_run5.log 2>&1 &
```

### Progress log

| iter | total | pi | v | score | own | self-play time | avg moves | eval vs random | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 11.2340 | 5.1875 | 1.0623 | 5.0548 | 0.6611 | 1680.5s (28.0m) | 166 | **10.0%** | score_loss=5.05 (was 555 before fix). Value range [-0.50,+0.44]. Pass not argmax. |
| 1 | 9.9154 | 5.1526 | 1.1220 | 3.8926 | 0.5801 | 1175.3s (19.6m) | 123 | **60.0%** | FIRST iter-1 improvement in Phase 2! 10→60%. Score head working. |
| 2 | 9.5725 | 5.1278 | 1.1118 | 3.6156 | 0.5527 | 2332.1s (38.9m) | 213 | **13.3%** | Losses improving but eval regressed 60→13%. Value range widened. Investigating. |

_(Append new iters as they land.)_
