# Phase 2 Final Training — 13x13 Go AlphaZero (Run 5)

Living document for the 13x13 Phase 2 final run. Previous runs (1-4d)
are documented in `PHASE_TWO_TRAINING.md`. Pass/resign research in
`PASS_RESIGN_RESEARCH.md`. Playout cap plan in `PLAYOUT_CAP_PLAN.md`.

---

## Final Architecture

```
Input (17, 13, 13)
  → Conv3x3(17→128) + LayerNorm + ReLU
  → 15 × ResidualBlock(Conv3x3+LN+ReLU → Conv3x3+LN + skip → ReLU)
  → Policy Head:    Conv1x1(128→2) + LN + ReLU + FC(338→170)
  → Score Head:     Conv1x1(128→1) + LN + ReLU + FC(169→32→1)
  → Ownership Head: Conv1x1(128→1, bias=True) → (B, 13, 13) logits
  → Value derived:  tanh(1.0 * score_pred + 0.0)
```

- **5.8M parameters** (15 blocks × 128 channels)
- **LayerNorm** everywhere — no train/eval drift
- **Score head** (5.5k params) replaces value MLP (44k params) — can't memorize
- **Value derived** from score with frozen scalars (vlw=0)
- **Ownership head**: KataGo-style auxiliary, training-only

---

## Final Training Config (Seed 500)

```
Model:           15b × 128ch (~5.8M params)
Sims/move:       400 full / 100 reduced (playout cap, 25%/75%)
Parallel games:  128
Games/iter:      1024
Buffer:          1M positions (uint8, augment-on-sample)
Batch:           1024
Train steps/it:  50
LR:              0.00125 → 0.0001 cosine
Optimizer:       SGD + momentum 0.9, weight_decay 1e-4
Pass floor:      pass_min_move=120
Resign:          thresh=-0.95, consec=5, min_move=80
Sampling:        |value|>0.95 downweighted to 10% (KataGo-style)
```

### Loss

```
L = -π·log(p) + 1.0·score_MSE(/N-normalized) + 1.5·BCE(own) + 1e-4·||θ||²
```

---

## Journey: 5 Attempts, 1 Success

### Attempt 1 — Standard value MLP (seed 42)
Eval: 32→14→6%. Value MLP (44k params) memorized cold ±1 labels.

### Attempt 2 — Bigger batch (seed 100)
Eval: 43→0%. Bigger batch did NOT prevent memorization.

### Attempt 3 — Score head, unnormalized targets (seed 200)
Eval: 0%/0%. Score head correct architecture but targets were raw
±50 range with weight 0.01 — near-zero gradient. Score head predicted
~0 for all positions. **Fix:** normalize by /N, weight=1.0.

### Attempt 4 — Score head, normalized (seed 300)
Eval: 10→60→13→17→33%. **First ever iter-1 improvement!** But two
problems: (a) score bias oscillated ±0.44 causing eval swings,
(b) pass-collapse at move 60-70 (51% of games).

Buffer analysis showed 73% of iter 3-4 positions had >80 dame
cells — garbage territory data from short games.

**Fixes applied:** pass_min 60→120, KataGo-style decided-position
downweighting (|value|>0.95 at 10% sampling rate).

Also tried score bias regularization (weight 10.0) — crushed score
std from 0.21 to 0.028, killing spatial learning. Dropped.

### Attempt 5 — Final config (seed 500) ✓
All fixes + playout cap (100/400 sims, 2.3× speedup).

**Result: 70% vs random at iter 9 — best in Phase 2 history.**

---

## Seed 500 Progress Log

| iter | total | pi | v | score | own | time | avg moves | eval | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 11.16 | 5.208 | 1.019 | 4.864 | 0.729 | 14.2m | 182 | **66.7%** | Best iter-0 ever. Playout cap 2.3× speedup. |
| 1 | 10.60 | 5.169 | 1.078 | 4.584 | 0.567 | 14.3m | 180 | **53.3%** | First non-collapse iter 1 in Phase 2. |
| 2 | 10.22 | 5.149 | 1.154 | 4.252 | 0.549 | 14.5m | 183 | **60.0%** | Stable. |
| 3 | 9.67 | 5.133 | 1.134 | 3.723 | 0.542 | 14.5m | 181 | **16.7%** | Eval dipped (score bias flipped). |
| 4 | 8.99 | 5.125 | 1.152 | 3.051 | 0.540 | 10.4m | 136 | **16.7%** | Games shortened. |
| 5 | 8.72 | 5.118 | 1.183 | 2.810 | 0.528 | 10.6m | 131 | **3.3%** | Low point. |
| 6 | 8.00 | 5.103 | 1.163 | 2.103 | 0.526 | 10.8m | 137 | **16.7%** | Recovered. |
| 7 | 7.39 | 5.065 | 1.177 | 1.547 | 0.521 | 14.0m | 177 | **20.0%** | Games rebounded. |
| 8 | 6.79 | 5.013 | 1.227 | 1.004 | 0.513 | 14.8m | 189 | **66.7%** | Back to peak! Score <1.0. |
| 9 | 6.47 | 4.987 | 1.322 | 0.722 | 0.508 | 13.6m | 177 | **70.0%** | **NEW HIGH.** pi<5.0 first time. |

### Key metrics across 10 iters

- **Score loss:** 4.86 → 0.72 (monotonically improving)
- **Policy loss:** 5.21 → 4.99 (broke 5.0 at iter 9)
- **Ownership loss:** 0.73 → 0.51 (monotonically improving)
- **Avg game length:** 131-189, stable above pass floor (120)
- **Eval vs random:** oscillating 3-70%, peak 70% at iter 9
- **Total training time:** ~2.3 hours for 10 iters (playout cap)

---

## What Worked

1. **Score head replacing value MLP.** 5.5k params can't memorize
   cold ±1 labels. Predicts territory margin — denser signal.

2. **Score target /N normalization.** Raw targets (±50) gave zero
   gradient. Normalizing by board side length gives std≈1.5.

3. **pass_min_move=120.** Prevents pass-collapse. Games stay
   130-190 moves, generating quality territory data.

4. **Decided-position downweighting.** KataGo-style 10% sampling for
   |value|>0.95. Reduces endgame outlier gradient noise.

5. **Playout cap randomization.** 25% full (400 sims), 75% reduced
   (100 sims). 2.3× speedup with no quality loss — our targets come
   from game outcomes, not search estimates.

6. **LayerNorm.** No train/eval drift. Eval measures real strength.

7. **Comprehensive smoke test.** 7 stages catching score normalization,
   pass floor, gradient flow, engine staleness. Saved hours.

## What Didn't Work

1. **Standard value MLP** — memorized at every config (runs 1-3, attempt 1-2).
2. **Score bias regularization** — weight 10 crushed spatial learning.
3. **pass_min_move=60** — too low, games ended at move 60-70.

## Open Problems

1. **Eval oscillation.** Score head bias still swings mildly across
   iters (±0.1-0.2), causing eval to oscillate 3-70%. The spatial
   learning is real (score loss 4.86→0.72) but the bias creates
   MCTS confidence swings. May dampen with more iters as the buffer
   stabilizes.

2. **Training losses not fully predictive of strength.** Score and
   ownership losses improve monotonically, but eval doesn't track
   them directly. This is inherent to the derived-value architecture —
   the score head can get better at predicting territory while
   miscalibrating the win/loss boundary.

3. **Eval only vs random.** 70% vs random is a low bar. Need
   eval vs GnuGo or self-play Elo to measure real strength.

## Next Steps (if continuing)

1. **Keep running.** Score loss is still dropping (0.72 at iter 9).
   Policy is sharpening (4.99). The model hasn't plateaued.

2. **Eval vs GnuGo.** Measure real strength beyond random baseline.

3. **Consider small vlw.** A tiny value_loss_weight (0.01-0.05)
   might calibrate value_scale/value_bias without risk — only 2
   learnable scalars, can't memorize.

4. **Phase 3 (19x19).** If 13x13 reaches >90% vs random
   consistently, consider scaling up.
