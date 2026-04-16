# Phase 2 Final Training — 13x13 Go AlphaZero (Run 5)

Living document for the 13x13 Phase 2 final run. Previous runs (1-4d)
are documented in `PHASE_TWO_TRAINING.md`. Pass/resign research in
`PASS_RESIGN_RESEARCH.md`.

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
Pass floor:      pass_min_move=120 (zeroed in both action + target)
Resign:          thresh=-0.95, consec=5, min_move=80,
                 disabled_frac=0.20, min_child_visits_frac=0.05
```

### Loss function

```
L = -π·log(p) + 1.0·score_MSE + 10.0·score_bias_reg + 1.5·BCE(own) + 1e-4·||θ||²
```

- **Policy**: cross-entropy with MCTS visit distribution
- **Score**: MSE on /N-normalized territory margin (std≈1.5)
- **Score bias reg**: `10.0 * (score_pred.mean() - target.mean())²`
  Anchors batch mean prediction to batch mean target. Prevents score
  head bias oscillation (+0.11→-0.44→+0.22) that caused eval swings.
- **Ownership**: per-cell BCE-with-logits ({-1,0,+1} → {0,0.5,1})
- **Value loss weight = 0**: value is derived from score, not trained directly

### Sampling

Decided-position downweighting (KataGo-style): positions with
|value| > 0.95 sampled at 10% rate. Prevents endgame outliers from
dominating gradient noise on the score head.

---

## Go/no-go gates

**GREEN** (let it run):
- `score_loss` decreasing or flat across iters 0→5
- `eval_vs_random` ≥ 15% by iter 5, non-regressing
- `avg_moves/game` stays in 120-200 range
- Memory stable, no OOM

**RED** (kill, investigate):
- `eval_vs_random` stuck at 0% for 3 consecutive iters
- `avg_moves/game` < 100 (pass collapse despite floor at 120)
- Score predictions have no variance (value signal dead)
- Any crash / OOM

---

## Key decisions and rationale

**Why score head instead of value MLP?** The 44k-param value MLP
memorized cold ±1 labels in <1 epoch across runs 1-3 and the
standard-value-MLP restart. The 5.5k-param score head predicts
territory margin — denser signal, fewer params, can't memorize.

**Why LayerNorm?** BatchNorm's running stats caused train/eval drift
in Phase 1 and Phase 2. LayerNorm has no running state. KataGo uses it.

**Why ownership head?** 169× supervision density vs scalar value.
Regularizes the trunk with dense spatial labels. Training-only
auxiliary (zero MCTS inference cost). Same as KataGo.

**Why pass_min_move=120?** Seed 300 showed 51% of games ending at
move 60-70 (right after the old floor of 60 lifted). 73% of iter 3
positions had >80 dame cells — garbage territory data. 120 ensures
the middlegame is fully played out.

**Why score bias regularization?** Seed 300 showed score head mean
oscillating +0.11→-0.44→+0.22 across iters while training losses
improved monotonically. This caused eval swings (10→60→13→17→33%).
The regularization anchors the mean without affecting spatial learning.

**Why downweight decided positions?** Endgame positions (3.8% of
buffer) have score targets 10× the median, creating high-variance
gradient noise on the score FC bias. KataGo downweights to 10%.

---

## Failed attempts (condensed)

### Attempt 1: Standard value MLP + LayerNorm (seed 42, batch 256)

Eval degraded 32→14→6%. Value MLP memorized despite ownership aux.

### Attempt 2: Bigger batch (seed 100, batch 1024, lr 0.00125)

Eval 43→0% at iter 1. Bigger batch did NOT prevent value overfit.

### Attempt 3: Score head, unnormalized targets (seed 200)

Score head predicted ~0 for all positions. Target was raw territory
margin (±50) with weight 0.01 — near-zero gradient. Eval 0%/0%.
**Fix:** normalize by /N, weight=1.0, value_scale=1.0.

### Attempt 4: Score head, normalized targets (seed 300)

First ever iter-1 improvement (10→60%). But two problems emerged:
1. **Score bias oscillation**: mean swung +0.11→-0.44→+0.22, causing
   eval swings 10→60→13→17→33%. Training losses not predictive.
2. **Pass-collapse**: 51% of games ended at move 60-70 after pass
   floor lifted. Buffer filled with poorly-resolved positions (73%
   with >80 dame cells at iter 3). Avg moves: 166→123→213→82→74.

**Fixes applied for current run:** pass_min=120, score bias reg=10.0,
decided-position downweighting at 10%.

---

## Current run (seed 400)

All fixes applied. Fresh start.

```bash
PYTHONPATH=engine PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
nohup python3 -u -m training.train \
    --board-size 13 --iterations 60 --num-workers 5 \
    --games-per-iter 1024 \
    --output-dir checkpoints/13x13_run5 \
    --seed 500 \
    > logs/13x13_run5.log 2>&1 &
```

Seed 400 (bias reg=10, no playout cap) ran 2 iters: eval 20→13%.
Bias reg crushed score std (0.028 vs 0.21). Dropped for seed 500.
Added playout cap (100/400 sims) for 2.3× speedup.

### Progress log

| iter | total | pi | v | score | own | self-play time | avg moves | eval vs random | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 11.1648 | 5.2075 | 1.0185 | 4.8643 | 0.7287 | 854.0s (14.2m) | 182 | **66.7%** | Best iter-0 in Phase 2! 2.3× speedup. Score std=0.113, pass=0.56%. |

_(Append new iters as they land.)_
