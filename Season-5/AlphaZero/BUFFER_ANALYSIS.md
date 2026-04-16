# Replay Buffer Analysis — Run 5 (Score Head, Seed 300)

Analysis at iter 3 self-play (buffer contains iters 0-2 complete + iter 3 partial).
Buffer snapshot: `checkpoints/13x13_run5/latest_buffer.npz`.

---

## 1. Buffer Overview

| Metric | Value |
|---|---|
| Size | 618,057 / 1,000,000 (61.8% full) |
| Obs | (618057, 17, 13, 13) uint8 |
| Policy | (618057, 170) float32 |
| Value | (618057,) float32 |
| Ownership | (618057, 13, 13) int8 |
| Sampling | Uniform random with 8-fold dihedral augmentation |

---

## 2. Per-Iter Breakdown

| Iter | Positions | Avg Moves | Value Mean | Score Mean | Score Std | Stone Mean |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 169,931 | 166 | +0.0002 | +0.0010 | 2.25 | 81.6 |
| 1 | 125,972 | 123 | +0.0002 | +0.0005 | 1.61 | 64.9 |
| 2 | 218,112 | 213 | -0.0006 | +0.0002 | 1.93 | 93.1 |
| 3 (partial) | 104,042 | ? | +0.0003 | +0.0003 | 0.60 | 42.8 |

**Observations:**
- Value and score means are near-zero across all iters — no systematic bias in the data.
- Avg moves vary widely (123 → 213) — iter 1 games were shorter, iter 2 games longer. This is driven by how the self-play net evaluates positions.
- Iter 1 has lower score std (1.61) and fewer stones (64.9) — shorter games produce less decisive territory outcomes.
- Iter 3 partial has very low score std (0.60) and low stones (42.8) — only early-game positions collected so far.

---

## 3. Value Target Distribution

| Metric | Value |
|---|---|
| Mean | -0.000083 (perfectly balanced) |
| +1 (black wins) | 309,003 (50.00%) |
| -1 (white wins) | 309,054 (50.00%) |
| Other | 0 |

Per-iter black win fraction: 50.01%, 50.01%, 49.97%, 50.01% — all balanced.

**Value by color-to-play:**
- Black to play (309,348 positions): value mean = **-0.3759**
- White to play (308,709 positions): value mean = **+0.3765**

This is correct and expected: when it's your turn, you haven't moved yet, so
positions where it's black's turn are on average slightly worse for black
(white just played and has the initiative). The symmetry (±0.376) confirms
the sign convention is consistent.

---

## 4. Score Target Distribution

Score target = `ownership.sum(axis=(1,2)) / N` where N=13.

| Metric | Value |
|---|---|
| Mean | +0.0005 (centered) |
| Std | 1.82 |
| Range | [-11.23, +11.23] |

Percentiles:

| 1st | 5th | 10th | 25th | 50th | 75th | 90th | 95th | 99th |
|---|---|---|---|---|---|---|---|---|
| -6.08 | -3.00 | -1.77 | -0.54 | 0.00 | +0.54 | +1.77 | +3.00 | +6.15 |

**Score by game outcome:**
- Black wins: mean = +0.93, std = 1.56
- White wins: mean = -0.93, std = 1.56

Symmetric and consistent. The ±0.93 means typical wins are by ~12 cells
(0.93 × 13), which is reasonable for cold self-play games.

**Score by game phase:**

| Phase | Stones | Count | Frac | Score Mean | Score Std | |Score|>2 |
|---|---|---:|---:|---:|---:|---:|
| Opening | [0, 20) | 96,236 | 15.6% | -0.005 | 1.41 | 10.1% |
| Early | [20, 40) | 87,859 | 14.2% | -0.023 | 1.45 | 10.4% |
| Mid-early | [40, 60) | 89,024 | 14.4% | -0.034 | 1.46 | 10.5% |
| Middle | [60, 80) | 70,505 | 11.4% | -0.067 | 1.68 | 14.1% |
| Mid-late | [80, 120) | 113,768 | 18.4% | -0.197 | 2.01 | 22.1% |
| Late | [120, 160) | 137,308 | 22.2% | -0.077 | 2.11 | 23.0% |
| Endgame | [160, 200) | 22,886 | 3.7% | **+1.782** | 2.14 | **41.4%** |
| Final | [200, 300) | 471 | 0.1% | **+5.743** | 1.45 | **100%** |

**Key finding:** Endgame/final positions (3.8% of buffer) have extreme
positive score targets (+1.78 to +5.74). These are heavily won positions
where the winner has filled most of the territory. If a training batch
oversamples these, it pulls the score head's bias positive. Conversely,
batches that undersample them see more neutral/negative positions.

This is the mechanism behind the score head mean oscillation: with 50 SGD
steps per iter (each batch 1024 from ~300k-600k buffer), the endgame
positions create high-variance gradient noise on the score head's bias.

---

## 5. Ownership Distribution

| Cell value | Count | Fraction |
|---|---:|---:|
| -1 (white) | 41,661,584 | 39.9% |
| 0 (dame) | 21,124,621 | 20.2% |
| +1 (black) | 41,665,428 | 39.9% |

Balanced between black and white. ~20% dame cells (unowned territory).
Dame count per position: mean = 34.2 cells (20.2% of 169).

---

## 6. Policy Target Distribution

**Pass action:**

| Metric | Value |
|---|---|
| Mean pass prob | 4.10% |
| pass = 0 (zeroed by floor) | 262,516 (42.5%) |
| pass > 0.5 | 16,711 (2.7%) |
| pass > 0.9 | 8,044 (1.3%) |

**Pass by game phase:**

| Phase | Stones | Pass Mean | pass > 0.5 |
|---|---|---:|---:|
| Opening | [0, 20) | 0.001% | 0 |
| Early | [20, 40) | 0.002% | 1 |
| Mid-early | [40, 60) | 1.78% | 1,327 |
| Middle | [60, 80) | 7.09% | 3,407 |
| Endgame | [160, 200) | 22.81% | 4,033 |

Pass floor is working: opening/early positions have near-zero pass.
Pass rises naturally in the middle/endgame as territory settles.
The mid-early [40,60) phase has 1.78% pass — these are positions just below
the pass_min_move=60 threshold where MCTS explores pass but the floor
zeroes it in the stored target.

**Policy concentration:**

| Metric | Value |
|---|---|
| Mean entropy | 1.89 (uniform = 5.14) |
| Top action > 50% | 47.7% of positions |
| Top action > 90% | 11.5% of positions |

Policy is fairly concentrated — about half of positions have one dominant
move. This is expected for MCTS-generated targets (MCTS focuses visits on
the best action).

---

## 7. Sampling Uniformity

50 batches of 1024 sampled from buffer:

| | Buffer | Sampled Mean | Sampled Std | Max |Deviation| |
|---|---:|---:|---:|---:|
| Value | -0.0001 | -0.0073 | 0.0281 | 0.0761 |
| Score | +0.0005 | -0.0126 | 0.0500 | 0.1186 |

Sampling is unbiased. The per-batch score noise (std=0.05, max=0.12) is
small relative to the score target std (1.82). **Sampling is not the source
of the score head bias oscillation** — the oscillation comes from the
SGD dynamics on the score head's FC bias parameter.

---

## 8. Diagnosis: Why Score Head Mean Oscillates

The data is clean:
- Buffer targets are perfectly balanced (value mean=0, score mean=0)
- Sampling is uniform and unbiased
- Per-iter segments all have near-zero means

The problem is in the score head's learning dynamics:
1. The score FC has one output bias that controls the mean prediction
2. Endgame positions (3.8% of buffer) have extreme score targets (mean +1.78)
3. Each SGD step sees a random batch; some batches oversample endgame
4. With only 50 steps per iter, the bias doesn't converge — it overshoots
5. Next iter, the bias overcorrects, creating oscillation

**Recommended fix:** Add a bias regularization term to the loss:
```python
score_bias_penalty = (score_pred.mean() - target_score.mean()).pow(2)
loss += 10.0 * score_bias_penalty
```

This anchors the per-batch mean prediction to the per-batch target mean,
preventing the bias from drifting while allowing spatial learning to continue.
