# Phase 2 Final Training — 13x13 Go AlphaZero (Run 5)

Living document for the 13x13 Phase 2 final run. Previous runs (1-4d)
are documented in `PHASE_TWO_TRAINING.md`. This run starts fresh with
a cleaned-up architecture informed by all prior failures.

---

## What changed from run4

Three architectural decisions, all reverting toward the standard
AlphaGo Zero / KataGo recipe:

### 1. LayerNorm replaces BatchNorm everywhere

BatchNorm's running statistics caused recurring train/eval drift
throughout the project:
- Phase 1 Problem 3: BN specialization narrowed the replay buffer
  distribution, causing strength regression (iter 9 = 96% → iter 16
  = 81% vs random) even as training loss kept dropping.
- Phase 2 strength drift candidate: BN moving stats calibrated for
  self-play distribution mismatch eval-time distribution.

LayerNorm normalizes per-sample with no running stats. Train and eval
behavior are identical. Modern KataGo also uses LayerNorm.

### 2. Standard value MLP restored (replacing derived-value)

Run4's derived-value architecture (2 learnable scalars reading off
ownership predictions) was an attempt to prevent value-head
memorization. It solved the memorization problem but introduced new
ones:
- Pass-collapse: ownership head learned "territory is settled" from
  late-game data, leaked into early-game policy.
- Strength drift: vs-random dropped 72% → 42% over 4 iters while
  all training losses looked healthy.
- Training losses were not predictive of strength.

Restored the standard AlphaGo Zero value head:
`Conv1x1(ch→1) + LN + ReLU + FC(169→256) + ReLU + FC(256→1) + tanh`

The original memorization problem (runs 1-3) happened **without** the
ownership head. With ownership now providing dense trunk supervision,
the trunk learns meaningful spatial features that the value FC layers
can generalize from — the conditions that caused memorization no
longer apply.

### 3. Standard loss weights restored

| Knob | Run4 | Run5 (this run) |
|---|---|---|
| `value_loss_weight` | 0.0 | **1.0** |
| `train_steps_per_iter` | 30 | **100** |
| `ownership_loss_weight` | 2.0 | **1.5** |

Loss = `policy_loss + 1.0 * value_loss + 1.5 * ownership_loss`

Standard recipe. No tricks.

---

## Architecture

Standard AlphaGo Zero / KataGo structure. Nothing exotic.

```
Input (17, 13, 13)
  → Conv3x3(17→128) + LayerNorm + ReLU
  → 15 × ResidualBlock(Conv3x3+LN+ReLU → Conv3x3+LN + skip → ReLU)
  → Policy Head: Conv1x1(128→2) + LN + ReLU + FC(338→170)
  → Value Head:  Conv1x1(128→1) + LN + ReLU + FC(169→256) + ReLU + FC(256→1) + tanh
  → Ownership Head: Conv1x1(128→1, bias=True) → (B, 13, 13) logits
```

- **Parameters**: ~4.5M (15 blocks × 128 channels)
- **Ownership head**: KataGo-style auxiliary. Dense per-cell BCE
  supervision regularizes the trunk. NOT used during MCTS inference —
  only policy + value go to the search. Same as KataGo.
- **No dropout**: Open-source Go AIs (KataGo, Leela Zero, AlphaGo
  Zero) don't use dropout. With ownership providing dense trunk
  supervision, the value head has sufficient gradient signal to
  generalize without explicit regularization.
- **No model gating**: Always use latest weights for self-play, same
  as KataGo and Leela Zero. The replay buffer (1M positions, ~3 iters
  of history) provides natural stability against single-iter
  regressions. Per-iter eval + checkpoints allow manual rollback if
  needed.

---

## Training Config

```
Model:           15 blocks × 128 channels (~4.5M params)
Sims/move:       400
c_puct:          1.5
Parallel games:  128
Games/iter:      2048
Iterations:      60
Buffer:          1M positions (uint8 obs, augment-on-sample)
Batch:           256
Train steps/it:  100
LR:              0.005 → 0.0001 cosine over 6000 steps
Optimizer:       SGD + momentum 0.9, weight_decay 1e-4
Grad clip:       max_norm=5.0
Dirichlet:       α=0.07, ε=0.25
Temperature:     τ=1.0 for first 40 moves, then τ=0.25
Max game moves:  250
Pass floor:      pass_min_move=60
Resign:          thresh=-0.95, consec=5, min_move=80,
                 disabled_frac=0.20, min_child_visits_frac=0.05
Checkpoint:      every iter
Eval:            every iter (--eval-in-loop or post-hoc)
```

### Loss function

```
L = -π·log(p) + (z - v)² + 1.5 · BCE(own_logits, own_target) + 1e-4·||θ||²
```

- **Policy**: cross-entropy with MCTS visit distribution
- **Value**: MSE with game outcome ±1
- **Ownership**: per-cell BCE-with-logits. Target mapped from
  {-1, 0, +1} → {0, 0.5, 1}. Dame cells (0) → 0.5 where BCE is
  minimized at logit=0 (uncertain).

### Resign strategy (strict, unchanged from run4)

Resign is deliberately conservative:
- **`resign_threshold=-0.95`**: value head must be 95% confident of
  losing before resign is even considered.
- **`resign_consecutive=5`**: must see 5 losing moves in a row. One
  single move above -0.95 resets the counter to 0.
- **`resign_min_move=80`**: no resign before move 80 (~45% of a
  typical 180-move game).
- **`resign_disabled_frac=0.20`**: 20% of games never resign,
  ensuring losing positions stay in the buffer.
- **Credible-child cross-check**: if any child node with ≥5% of root
  visits has Q > -0.95, resign is blocked (recovery line exists).

### Pass floor (kept from run4)

`pass_min_move=60`: below move 60, the pass action is zeroed from both
the stored policy target and the sampled action. Prevents the policy
from learning "pass is valid in the opening" from MCTS visit noise.
After move 60, the net is free to pass if territory is genuinely
settled.

---

## Key design decisions and rationale

### Why no dropout

AlphaGo Zero, KataGo, Leela Zero — none use dropout. They rely on
data volume for regularization. Our cold start has limited data, but
the ownership head changes the equation: 169 dense per-cell labels
per position (vs 1 scalar value label per game) give the trunk
sufficient supervision to learn generalizable features from the start.
The value FC layers then fit on meaningful trunk features, not noise.

### Why no model gating

AlphaGo Zero used gating (new model must beat old 55% to be adopted).
KataGo, Leela Zero, and ELF OpenGo all dropped it — monotone
improvement emerges naturally with sufficient buffer diversity.

For us, gating would add ~5-10 hours of eval compute across 60 iters
(50-game head-to-head per iter at 100 sims/move). Our defenses against
regression are:
- 1M replay buffer (~3 iters of history) — one bad iter can't flush
  good data
- Per-iter checkpoints — manual rollback available
- Per-iter eval — regression detected immediately
- Conservative resign — bad value head can't collapse game lengths

### Why LayerNorm over BatchNorm

BatchNorm has running statistics (moving mean/var) that specialize to
the training distribution. During eval against a different opponent
(random, GnuGo, or a different checkpoint), the input distribution
differs from self-play, and the stale running stats produce wrong
normalizations. This was the root cause of Phase 1 Problem 3 and a
candidate for Phase 2 strength drift.

LayerNorm normalizes each sample independently using that sample's own
statistics. No running state, no train/eval divergence. Modern KataGo
also uses LayerNorm.

### Why keep ownership head

Ownership is a training-only auxiliary — computed during forward pass
but discarded during MCTS inference. Same as KataGo. Benefits:
- 169× supervision density vs scalar value loss alone
- Regularizes the trunk with dense spatial labels
- Helps the trunk learn territory concepts from the start
- Zero inference cost (Conv1x1 output is dropped)

---

## Comparison with open-source Go AIs

| | **Ours** | **AlphaGo Zero** | **KataGo** | **Leela Zero** |
|---|---|---|---|---|
| Blocks × Ch | 15×128 | 20×256 | 10-20×128-256 | 20-40×128-256 |
| Norm | LayerNorm | BatchNorm | LayerNorm | BN→LN |
| Value head | Standard MLP | Standard MLP | Standard MLP | Standard MLP |
| Ownership | Yes (aux) | No | Yes (aux) | No |
| Dropout | No | No | No | No |
| Model gating | No | Yes | No | No |
| MCTS sims | 400 | 400 | 400-800 | 400-1600 |
| c_puct | 1.5 | 1.0 | 1.0-1.5 | 1.0-1.5 |
| Optimizer | SGD 0.9 | SGD 0.9 | SGD 0.9 | SGD 0.9 |
| Buffer | 1M | 500K-1M | 1M+ | 50M+ |
| Training | Iterative | Continuous | Continuous | Continuous |

Our setup is standard except for iterative (vs continuous) training
and slightly higher c_puct (1.5 vs 1.0). Both are acceptable for a
budget single-GPU run.

---

## Environment

- **Host:** TBD (RunPod RTX 4090 24 GB)
- **Python 3.11**, **torch 2.4.1+cu124**
- **Rate:** ~$0.44-0.60/hr
- **Estimated cost:** 60 iters × ~35-55 min/iter ≈ 35-55 hours ≈
  $15-33

## Launch checklist

```bash
cd Season-5/AlphaZero/engine && python setup.py build_ext --inplace && cd ..
PYTHONPATH=engine python training/_test_correctness.py
PYTHONPATH=engine python -m training.train --board-size 13 --smoke-test \
    --output-dir checkpoints/13x13_smoke

# Production launch:
PYTHONPATH=engine PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
nohup python3 -u -m training.train \
    --board-size 13 --iterations 60 --num-workers 5 \
    --output-dir checkpoints/13x13_run5 \
    --seed 42 \
    > logs/13x13_run5.log 2>&1 &
```

## Go/no-go gates

**GREEN** (let it run):
- `value_loss` flat-or-falling across iters 0→5
- `eval_vs_random` ≥ 15% by iter 5, non-regressing
- `avg_moves/game` stays in 100-200 range
- Memory stable, no OOM

**RED** (kill, investigate):
- `value_loss` rising 2+ iters in a row
- `eval_vs_random` stuck near 0% for 3 consecutive iters past iter 2
- `avg_moves/game` < 80 (resign or pass collapse)
- Any crash / OOM

---

## Progress log

_(Append new iters as they land.)_

| iter | total | pi | v | own | self-play time | avg moves | games/s | eval vs random | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| | | | | | | | | | |
