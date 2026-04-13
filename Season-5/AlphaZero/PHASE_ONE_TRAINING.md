# Phase 1 Training — 9x9 Go AlphaZero

Living document for the 9x9 Phase 1 run. The pipeline has hit three
distinct problems in sequence, each rooted and fixed before the next
was reachable:

1. **Speed** — self-play appeared hung; torch thread contention and
   pinned-memory misuse
2. **Crash** — reliable SIGSEGV inside `MCTS::select_leaf` a few
   iterations into training
3. **Regression** — training loss kept dropping but tournament
   strength degraded; BatchNorm specializing to a narrow self-play
   distribution

The current run is **`9x9_run2`**, started 2026-04-13 from random
weights with all three classes of fix applied.

---

## Environment

- **Host:** RunPod RTX 4090 (24 GB VRAM), AMD EPYC 7763
- **Container:** cgroup CPU quota 27.2 cores, memory 125 GB
- **Python 3.11**, **torch 2.4.1+cu124** with inductor / Triton
- **Rate:** ~\$0.60/hr

## Code layout

```
Season-5/AlphaZero/
├── engine/                        # C++ Go + MCTS + SelfPlayWorker (pybind11)
│   ├── go.{h,cpp}                 # Board + Game rules
│   ├── mcts.h                     # MCTSTree<N>, select_leaf, expand, backup
│   ├── worker.h                   # SelfPlayWorker<N>, resign logic
│   ├── bindings.cpp               # pybind11 glue
│   └── go_engine.*.so             # built extension (inplace)
├── model/
│   ├── network.py                 # AlphaZeroNet (ResNet policy+value)
│   └── config.py                  # ModelConfig / TrainingConfig / CONFIGS
├── training/
│   ├── train.py                   # main loop, CLI, checkpoint + buffer save
│   ├── parallel_self_play.py      # orchestrator + worker threads + barriers
│   ├── trainer.py                 # SGD trainer with grad clip + NaN guard
│   ├── replay_buffer.py           # circular buffer, 8-fold aug, save/load
│   ├── eval_matchup.py            # paired head-to-head evaluator
│   ├── eval_vs_random.py          # standalone vs-random eval
│   └── _test_correctness.py       # end-to-end smoke test
├── logs/9x9_run2.log
└── checkpoints/9x9_run2/
```

---

## Problem 1 — Speed (pre-training session)

### Symptom

Launching `python -m training.train --board-size 9 --iterations 73`
looked hung. After ~6 min of wall time iter 0 still hadn't printed,
but the process was clearly alive:

- `%CPU` reported **2620%** (~26 cores worth)
- Cumulative CPU `TIME` was **3h 30m** after 6 min wall (~35×
  parallelism)
- **199 threads** in the process
- GPU only ~33% utilized

### Root cause 1a — torch CPU thread-pool contention

`torch.get_num_threads()` returned **128**. Every orchestrator CPU
tensor op fanned out across a 128-wide intra-op pool and fought the
5 C++ self-play threads on a 27-core quota.

Fix — set thread limits *before* importing torch, in
`training/train.py`:

```python
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision("high")
```

Thread count dropped **199 → 9**. Smoke-test iter time **9–11 s →
1.2–2.9 s** (~7× faster).

### Root cause 1b — pinned memory misused on the D2H path

The original code was:

```python
# WRONG — defeats pinned memory
self.policy_pinned[:total_nn].copy_(policies.cpu(), non_blocking=False)
```

`.cpu()` is a blocking transfer into a fresh non-pinned tensor; the
pre-allocated pinned buffer contributed nothing.

Fix:

```python
self.policy_pinned.copy_(policies, non_blocking=True)
self.value_pinned.copy_(values, non_blocking=True)
torch.cuda.synchronize()
```

### Other optimizations landed

1. **`torch.compile(mode="max-autotune")`** with **fixed-shape
   forward** (always pass the full `total_max_nn` buffer so
   torch.compile locks onto a single kernel shape — no recompile
   spikes).
2. **Zero-copy pinned buffers** — numpy view over the pinned tensor
   so C++ workers write leaf observations straight into pinned
   memory, eliminating a staging `np.concatenate` + copy.
3. **`network.py`: `.view()` → `.flatten(1)`** in the policy/value
   heads (layout-agnostic).
4. **TF32 matmul** via `set_float32_matmul_precision("high")`.

### What was rejected

- **Channels-last / NHWC** — +1.8 ms GPU+xfer on our model; transpose
  cost outweighs kernel win.
- **Cutting `num_simulations` / `num_parallel_games`** — no MCTS
  quality sacrifice.
- **Pipelining worker-select with GPU forward** — adds a 1-tick lag
  between selection and tree update, subtle MCTS quality tradeoff.

### Result

Iter 0 of Phase 1 (first run): 2049 games, 157K positions, 644 s
self-play time, **3.2 games/s**. PLAN.md had assumed 0.5 ms/forward;
real number is 15.2 ms, so PLAN's \$1.1 / 2.5 h per run was never
reachable. Real cost is **~\$8 / ~13.5 h per run**.

---

## Problem 2 — Crash (SIGSEGV hunt)

### Symptom

After iter 0 printed successfully, the process began segfaulting
reliably inside the C++ self-play worker. Crash signature always:

```
Fatal Python error: Segmentation fault

[5 threads, all at]
  parallel_self_play.py line 145  (worker.tick_select(obs_slice))
```

Crash landed at different wall times depending on starting state —
fresh runs survived ~28 min, resumes from a stronger checkpoint
crashed in ~3–5 min. A stronger checkpoint produces more confident
MCTS, which grows deeper trees sooner, which hit the real bug faster.

### Wrong leads chased (do not repeat)

- Inductor compile-subprocess pool
- Zero-copy pinned-memory-backed numpy views
- Numpy view slicing into a shared `obs_buffer`
- `torch.compile` itself (disabling it bought longer survival but
  still crashed — pressure reduction, not a fix)
- `trainer.load_checkpoint` corruption

### Root cause: `mcts.h::select_leaf` out-of-bounds write

```cpp
static constexpr int MAX_PATH_DEPTH = 64;   // <-- too small

int select_leaf(int* out_path) {
    int len = 0;
    int current = root_idx;
    out_path[len++] = current;
    while (nodes[current].is_expanded() && !nodes[current].is_terminal) {
        int child = select_child(current);
        if (child < 0) break;
        nodes[child].virtual_loss++;
        current = child;
        out_path[len++] = current;   // NO BOUNDS CHECK
    }
    return len;
}
```

With MCTS tree reuse keeping the surviving subtree across moves, the
effective depth grows throughout a single game. At 400 sims/move and
`virtual_loss_batch=8`, PUCT chases confident lines deep, and depth
easily exceeds 64. At that point `out_path[64++]` scribbles node
indices into the next `LeafInfo`'s `path_len`, `leaf_idx`, ... and
then into the following `LeafInfo`'s `path[]`. The next
`select_leaves` iteration dereferences the corrupted path and
crashes.

### Fix

```cpp
// MCTS tree depth is usually ~10-30, but with tree reuse across many
// moves the effective depth can grow unboundedly. 256 is a safe bound
// for any realistic Go position.
static constexpr int MAX_PATH_DEPTH = 256;

int select_leaf(int* out_path) {
    int len = 0;
    int current = root_idx;
    out_path[len++] = current;
    while (nodes[current].is_expanded() && !nodes[current].is_terminal) {
        if (len >= MAX_PATH_DEPTH) break;   // hard cap — prevents OOB
        int child = select_child(current);
        if (child < 0) break;
        nodes[child].virtual_loss++;
        current = child;
        out_path[len++] = current;
    }
    return len;
}
```

Memory cost is ~625 KB/worker — negligible vs ~8 GB RSS.

### Auxiliary change: per-worker owned numpy arrays

While hunting the bug, the shared `obs_buffer` with per-worker slice
views was replaced with fully owned per-worker numpy arrays in
`parallel_self_play.py`. This did NOT fix the segv (the bug was in
C++) but it's cleaner and removes a class of shared-pointer concerns.
Kept.

### Post-fix validation

Restart from the previously-reliable sub-3-minute crash path ran
iters 4–12 without crashing. The crash was unambiguously fixed.

---

## Problem 3 — Regression (BatchNorm drift)

### Symptom

Run 1 ran cleanly from iter 4 through iter 19 post-crash-fix.
Training loss kept dropping (4.89 → 2.56, policy 3.99 → 1.89), but
**tournament strength regressed**:

| iter | vs random |
|---|---:|
| 9 | 96% |
| 16 | 81% |
| 19 | 85% |

Head-to-head vs iter 9 at 400 sims looked tied (iter 19 = 0.520),
which masked the regression — search at 400 sims compensates for
policy-head drift. The vs-random metric at 100 sims is the cleaner
signal.

### Weight trajectory (the mechanical proof)

```
iter   policy_bn.max   input_conv.max
   9       31.19           0.160
  10       30.60           0.163
  11       32.15           0.168
  12       44.34           0.170   ← drift begins
  15       43.73           0.224
  16       51.35           0.334
  17       61.68           0.419
  18       65.05           0.477
  19       73.57           0.549
```

`policy_bn` grew **2.4×**, `input_conv` grew **3.4×**, both
monotonically from iter 12. All other layers stayed stable. This is
specific to BN + the input conv, not a global optimization problem.

### Root cause

**Distribution collapse inside the replay buffer plus BatchNorm
specializing to that narrow distribution.** Contributing factors:

1. **Resign logic too aggressive** (`thresh=-0.95`, 3 consec, no
   move floor, 10% disabled) — cut middle/endgame positions out of
   the buffer once the model got confident
2. **Replay buffer lost on every restart** — each resume refilled
   from the current (already-narrow) model
3. **Exploration phase too short** — `temperature_moves=15` over
   ~85-move games meant only ~18% of each game was sampled
   proportionally; the remaining ~82% was near-argmax
4. **`temp_low=0.1`** collapsed training policy targets to near
   one-hot after move 15 — no runner-up signal for the model to
   learn relative move strength
5. **No anchor against any known-good distribution** — once the
   buffer narrowed, training had no recovery force

An iter-13 NaN crash (one SGD gradient spike sent weights to ∞)
surfaced during this period. Fixed with
`clip_grad_norm_(max_norm=5.0)` + `isfinite(loss)` skip-step guard.
Kept in run 2.

### Fixes landed for run 2

Run 1 artifacts (checkpoints, logs) lived on the prior host and are
not present on the current machine. Decision: start run 2 fresh from
random weights with all of the following applied together.

| # | Fix | Files |
|---|---|---|
| 1 | **Resign v2** — loosen `-0.95 → -0.90`, raise `consecutive 3 → 5`, add `resign_min_move=20` floor, add credible-child cross-check (don't resign if any child with ≥5% of root visits has `Q > threshold`), double `disabled_frac 0.10 → 0.20` | `engine/worker.h`, `engine/bindings.cpp`, `model/config.py`, `training/parallel_self_play.py::_make_sp_config` |
| 2 | **Wider exploration** — `temperature_moves 15 → 30`, `temperature_low 0.1 → 0.25` (9x9 preset) | `model/config.py` |
| 3 | **Replay buffer persistence** — `save_to`/`load_from` with atomic temp+rename; save after each iter, reload on resume | `training/replay_buffer.py`, `training/train.py` |
| 4 | **Anchor-mixing code path (optional)** — `train_step` / `train_epoch` accept `anchor` + `anchor_frac`; `--anchor-buffer` / `--anchor-frac` CLI flags. Unused on a fresh run, kept as a recovery lever. | `training/trainer.py`, `training/train.py` |
| 5 | **Eval cadence** — `eval_interval 10 → 5` for 9x9 | `model/config.py` |
| 6 | **Keep all checkpoints** — removed `KEEP_LAST=5` retention. Each ckpt ~36 MB × 73 iters ≈ 2.6 GB, trivial. Prior run lost iter 14 to retention and we couldn't reconstruct the weight-drift trajectory post-hoc. | `training/train.py` |

Grad clipping and NaN guard from run 1 are kept as-is.

---

## Run 2 — live training

### Verification before launch

1. `training/_test_correctness.py` — self-play, buffer sanity,
   5-epoch loss descent, MCTS-vs-random eval. Passed.
2. Targeted test: resign v2 fields propagated through
   `_make_sp_config`; `ReplayBuffer.save_to`/`load_from` exact
   round-trip; `train_step` with and without anchor.
3. `train.py --smoke-test` over 3 iters — all checkpoints saved,
   `latest_buffer.npz` written, resume restores the buffer.

### Config

```
Model:           10 blocks × 128 channels (3.009M params)
Sims/move:       400
Parallel games:  256 (5 workers × ~52)
Games/iter:      2048
Iterations:      73
Buffer:          500K, persistent (latest_buffer.npz)
Batch:           256
Train steps/it:  100
LR:              0.01 → 0.0001 cosine over 7300 steps
Optimizer:       SGD + momentum 0.9, weight_decay 1e-4
Dirichlet:       α=0.11, ε=0.25
Temperature:     τ=1.0 for first 30 moves, then τ=0.25
Max game moves:  150
Resign v2:       thresh=-0.90, consec=5, min_move=20,
                 disabled_frac=0.20, min_child_visits_frac=0.05
Checkpoint:      every iter, no pruning
Eval:            every 5 iters (100 games vs random)
```

### Run metadata

- **Run label:** `9x9_run2`
- **Checkpoint dir:** `checkpoints/9x9_run2/`
- **Log file:** `logs/9x9_run2.log`
- **Started:** 2026-04-13
- **Launch pid:** 3374
- **Command:**
  ```bash
  PYTHONPATH="$PWD:$PWD/engine/build/lib.linux-x86_64-cpython-311:$PYTHONPATH" \
  PYTHONFAULTHANDLER=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  nohup python3 -u -m training.train \
      --board-size 9 --iterations 73 --num-workers 5 \
      --output-dir checkpoints/9x9_run2 \
      > logs/9x9_run2.log 2>&1 &
  ```

### Progress log

_(Append new iters as they land. Flag any Fatal/Segmentation/Traceback
lines immediately and root-cause before restarting.)_

| iter | total | pi | v | self-play time | avg moves | games/s | note |
|---|---:|---:|---:|---:|---:|---:|---|

### Red flags (stop and investigate)

- `policy_bn.max` > **40** at any iter → BN specialization recurring.
  First try `--anchor-buffer` from the most recent good checkpoint,
  or halve `lr_init` (0.01 → 0.005).
- `input_conv.max` drifts above **0.25** monotonically across ≥3 iters.
- `vs_random` drops below **90%** on 2 consecutive evals.
- Training loss drops faster than 0.1 per iter sustained — likely
  overfitting to a narrow distribution, not real improvement. Run 1's
  iter 12 jump to loss 3.99 was exactly this failure mode.

### Green lights (keep going)

- `vs_random` in 92–97% range
- `policy_bn.max` in 30–40 range
- `input_conv.max` in 0.15–0.25 range
- Training loss drops gently with oscillation
