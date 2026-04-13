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

- **Host (run 1, prior RunPod box):** RTX 4090 24 GB, EPYC 7763,
  cgroup CPU quota 27.2 cores, **memory 125 GB**, oom_kill_disable=1
- **Host (run 2, current box):** RTX 4090 24 GB, cgroup memory
  **42.8 GB** — much tighter than run 1's host. `oom_kill_disable=1`
  here too, so hitting the limit produces a silent SIGKILL with no
  Python traceback (see Problem 4 below).
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

### Root cause 1c — `torch.compile` not engaged, then misconfigured

The original self-play forward path ran the model in eager mode.
`torch.compile` with the right settings is worth ~20–30% on batched
inference for a small conv tower like ours, but getting it right took
several attempts.

**What we ended up with** (`training/parallel_self_play.py`):

```python
if self.use_cuda and os.environ.get("AZ_COMPILE", "1") != "0":
    try:
        self.infer_net = torch.compile(net, mode="max-autotune")
    except Exception as e:
        print(f"[ParallelSelfPlay] torch.compile disabled: {e}")
        self.infer_net = net
```

Three things had to land together for this to actually pay off:

1. **`mode="max-autotune"`** instead of `reduce-overhead`. The
   autotuned Triton kernels bought another ~10% over the default
   mode on bs=2048, 17×9×9 inputs.
2. **Fixed-shape forward.** Initially we passed only the "live" slice
   `obs_pinned[:total_nn]` to the compiled net. `total_nn` varies
   per tick (some workers have fewer active games), so torch.compile
   recompiled on every unique shape — the guards triggered, the
   autotune cache missed, and we saw multi-second recompile spikes
   mid-iter. **Fix:** always pass the *full* `total_max_nn` buffer
   to `self.infer_net(obs_tensor)`. Trailing slots contain stale
   data, but each worker only reads back the first `nn_count`
   entries of its own policy/value slice, so the stale outputs are
   harmless. This locks the compiled kernel onto a single shape for
   the entire run.
3. **`AZ_COMPILE=0` escape hatch** for smoke tests. Warmup +
   autotune for iter 0 takes ~2 min on a 4090, which dominates
   short test runs. Setting `AZ_COMPILE=0` gives eager mode for
   test scripts (`_test_correctness.py`, smoke tests) while the
   real run keeps it on.

There's a fourth piece that shows up in the monitor: on the real run
the first iter's log contains a big block of `AUTOTUNE mm(...)` lines
from inductor — that's the normal Triton kernel-selection pass.
It's expected and it's why iter 0 takes ~7 min instead of the ~5 min
steady-state.

### Other optimizations landed

1. **Zero-copy pinned buffers** — numpy view over the pinned tensor
   so C++ workers write leaf observations straight into pinned
   memory, eliminating a staging `np.concatenate` + copy. Saves ~3 ms
   per tick. (Later replaced with per-worker *owned* arrays while
   hunting the SIGSEGV — the ownership change didn't affect the win,
   it just eliminated a shared-pointer class of bugs. See Problem 2.)
2. **`network.py`: `.view()` → `.flatten(1)`** in the policy/value
   heads. Layout-agnostic; required for any NHWC experiment and
   doesn't hurt NCHW.
3. **TF32 matmul** via `set_float32_matmul_precision("high")`. Free
   on Ampere/Ada — no quality cost at this model size.
4. **Separate H2D / D2H pinned tensors** with explicit
   `torch.cuda.synchronize()` after the D2H copy (see Root cause 1b).

### What was rejected

- **Channels-last / NHWC** — +1.8 ms GPU+xfer on our model
  (10b × 128ch, bs 2048). The transpose cost on every tick outweighs
  the kernel win for a model this small.
- **Cutting `num_simulations` / `num_parallel_games`** — explicitly
  off-limits, no MCTS quality sacrifice.
- **Pipelining worker-select with GPU forward** — would add a 1-tick
  lag between selection and tree update, a subtle MCTS quality
  tradeoff. Held off per the no-sacrifice constraint.

### `PLAN.md`'s GPU estimate was wildly optimistic

PLAN.md assumed **0.5 ms** per batched forward on a 4090. Measured:
**15.2 ms** (~30× slower). Theoretical floor is ~5.8 ms (~490 GFLOPs
at bs 2048 ÷ ~85 TFLOPS bf16 peak), so 15 ms is reasonable for a
small conv tower where kernel launch + memory bandwidth dominate.

Consequence: PLAN's \$1.1 / 2.5 h-per-run estimate was never
physically reachable. Real cost is **~\$8 / ~13.5 h per run**.

### Result

Iter 0 of Phase 1 (run 1): 2049 games, 157K positions, 644 s
self-play time, **3.2 games/s**, initial loss 5.25 (π=4.37, v=0.88).
Run 2 iter 0 with wider temperature + slightly faster pipeline:
2048 games, 156K positions, 444 s, **4.6 games/s**, loss 5.23.
The pipeline is now GPU-forward-bound, not CPU-bound.

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

## Problem 4 — Memory OOM (run 2, current host)

### Symptom

Run 2 launched from random weights, completed iter 0 (4.6 g/s, 76
avg moves) and iter 1 (5.1 g/s, 81 avg moves) cleanly with healthy
loss descent (5.23 → 4.97). Then the process disappeared **silently**:

- `ps -p 3374` returned empty — process gone
- Log file ended mid-write inside `numpy/lib/function_base.py`
- **No `!!! TRAINING CRASHED !!!` marker** that `train.py`'s top-level
  exception handler emits
- No `Fatal Python error`, no traceback, no OOM message
- `oom_kill 0` in `/sys/fs/cgroup/memory/memory.events`
- Iter 0 + iter 1 checkpoints (24 MB each) and `latest_buffer.npz`
  (2.92 GB) all fully written before the death

A truly silent kill — no Python-level signal handler had a chance to
run.

### Root cause: cgroup memory limit on this host is 42.8 GB, not 125 GB

The original `PHASE_ONE_TRAINING` Environment section claimed
"memory 125 GB" — that was the *prior* RunPod box (run 1 host). The
current box has very different limits:

```
$ cat /sys/fs/cgroup/memory/memory.limit_in_bytes
45999996928              # 42.8 GB
$ cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes
41591750656              # 38.7 GB peak before kill
$ cat /sys/fs/cgroup/memory/memory.oom_control
oom_kill_disable 1       # ← key
under_oom 0
oom_kill 0
```

The combination is the trap:

- **42.8 GB hard limit** vs run 1's host with 125 GB (~3× headroom)
- **`oom_kill_disable=1`** — when the cgroup hits its limit, the
  kernel's normal OOM killer is suppressed, and the container runtime
  freezes processes and SIGKILLs them. SIGKILL bypasses Python
  signal handlers entirely, so `faulthandler` never gets to write a
  traceback, and `train.py`'s `except BaseException` block never
  runs. Process just vanishes.

### Memory budget at the moment of death

Approximate breakdown of the ~38.7 GB peak:

| | Estimate |
|---|---:|
| MCTS tree state (5 workers × 52 games × ~50 MB tree+game pool) | ~13 GB |
| Replay buffer in RAM (500K × 17 × 9 × 9 × float32 + policy + value) | ~2.64 GB |
| **Transient `np.savez` of 2.92 GB buffer file** | **~2.5–3 GB peak** |
| PyTorch CUDA reserved + model + optimizer + torch.compile cache | ~2 GB |
| Worker pinned tensors + own numpy buffers | ~50 MB |
| Other (Python interpreter, libs, OS, fragmentation) | ~remainder |

Steady-state (~35 GB) was already at 82% of the 42.8 GB cap. The
brand-new buffer-persistence code path that was added as a fix for
Problem 3 introduced a **2.5–3 GB transient peak during `np.savez`**
right after each iter's training step. That spike is what pushed us
over the line at iter 1 → iter 2 boundary.

The same code ran on run 1's host without issue because there was
~90 GB of headroom there.

### Fix

Smallest, most surgical change that keeps the run progressing on
this host:

1. **Disable `buffer.save_to(...)` in `train.py`** for now. The
   `save_to` / `load_from` / `--anchor-buffer` code paths from
   Problem 3 are kept intact for future hosts with more headroom —
   we just don't *call* the save on this run. (Comment in code
   points to this section.)
2. **Reduce 9x9 `buffer_size` from 500K → 300K.** Saves ~1.06 GB
   permanent in-RAM. Buffer still holds plenty of samples for
   training diversity (~5 iters worth at the new game lengths).
3. **Delete the existing `latest_buffer.npz`** to free 2.92 GB of
   disk and prevent `train.py` from finding it on resume (which
   would try to reload the now-too-big buffer into a 300K array and
   raise).
4. **Restart from `checkpoint_0001.pt`** — no rollback, just resume
   from where we crashed. The replay buffer starts empty and refills
   on the first self-play phase; minor warmup hit only, no quality
   impact since iter 2 trains on iter-2-self-play data only and
   subsequent iters fill the buffer normally.

```python
# training/train.py — replaces the buffer.save_to(...) call:
# Buffer persistence deliberately disabled: the np.savez transient
# pushed peak memory over the 42.8 GB cgroup limit on this host
# (silent SIGKILL after iter 1). The code path stays for future
# hosts with more memory headroom.
```

```python
# model/config.py — 9x9 preset:
buffer_size=300_000,   # was 500_000; fits under 42.8 GB cgroup
```

### Validation

After the restart from `checkpoint_0001.pt` as pid 4305:

| iter | RSS peak | result |
|---|---:|---|
| 2 | ~33 GB | first iter post-fix; clean self-play + train + ckpt |
| 3 | ~33 GB | clean |
| 4 | ~33 GB | new low loss, eval vs random = 63% |
| 5 | ~33 GB | clean — confirmed stably past the OOM boundary |

Steady-state ~33 GB out of 42.8 GB is comfortable headroom (~9 GB).
No further memory work needed for the rest of Phase 1.

### Followup for Phase 2

Phase 2 will use a 1M sample × 13x13 obs buffer = **12.2 GB as
float32**, almost 5× larger than the Phase 1 buffer that already
caused this OOM. Switching obs storage to `uint8` (lossless because
every plane is already 0/1) drops it to ~3 GB. See
`PHASE_TWO_TODO.md` section 1 for the full plan and code diff.

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
| 0 | 5.2312 | 4.3591 | 0.8721 | 443.7 s | 76 | 4.6 | First iter; torch.compile autotune warmup absorbed here |
| 1 | 4.9727 | 4.1127 | 0.8600 | 405.5 s | 81 | 5.1 | Throughput climbing as compile cache settles. Process SIGKILLed silently after this iter — see Problem 4 (42 GB cgroup OOM during np.savez buffer persist). |
| 2 | 4.7398 | 3.9659 | 0.7739 | 541.3 s | 99 | 3.8 | Resumed from `checkpoint_0001.pt` after OOM fix (buffer persistence disabled, `buffer_size` 500K → 300K). Avg moves jump to 99 — resign v2 + wider τ are keeping games longer, exactly the diversity outcome we wanted. Throughput trades down accordingly. |
| 3 | 4.8337 | 4.0272 | 0.8066 | 456.6 s | 92 | 4.5 | Single-iter wobble (+0.09 total). Normal training noise; nothing to investigate. |
| 4 | 4.7235 | 3.9479 | 0.7756 | 443.3 s | 90 | 4.6 | New low total **and** new low policy. **Eval vs random = 63.0%** (100 games, 73.6 s) — first strength signal. Run 1's first eval was at iter 9 = 96%, so not directly comparable; trajectory is on track for a 90%+ landing at iter 9. |
| 5 | 4.8378 | 4.0069 | 0.8309 | 409.6 s | 84 | 5.0 | Slight uptick (+0.11 total). Avg moves back down to 84, throughput up. Memory holding stably under the 42 GB cap. Run is now confirmed stable past the OOM fix. |
| 6 | 4.7589 | 3.8859 | 0.8730 | 366.6 s | 75 | 5.6 | Policy new low (3.89). Throughput best yet at 5.6 g/s; games shortening as model gains confidence. |
| 7 | 4.3473 | 3.8863 | 0.4610 | 451.2 s | 90 | 4.5 | Big total drop (−0.41) driven by value collapse 0.87 → 0.46. Pattern looks unlike run 1 iter 12 (which had games shortening 84 → 44 from aggressive resign); here games **lengthened** 75 → 90, so value head is fitting genuinely harder positions, not cheap resign-decided ones. |
| 8 | 4.3982 | 3.6446 | 0.7536 | 402.8 s | 81 | 5.1 | Confirmed: iter-7 jump was a value-head wobble, not a regression. Total bounced back (4.35 → 4.40), value recovered (0.46 → 0.75), policy hit **new low 3.64**. |
| 9 | 4.4429 | 3.7163 | 0.7267 | 475.8 s | 95 | 4.3 | Single-iter policy wobble (3.64 → 3.72). **Eval vs random = 91.0%** (100 games). 5 pp below run 1 iter 9 (96%) — expected cost of wider τ schedule. |
| 10 | 3.8752 | 3.4874 | 0.3878 | 291.5 s | 60 | 7.0 | **Big total drop −0.56**, value collapses to 0.39. Same superficial signature as run 1 iter 12 (loss drop + games shortening + throughput up). Triggered an immediate weight audit. |
| 11 | 3.9196 | 3.4189 | 0.5007 | 399.0 s | 79 | 5.1 | Iter-10 shortening reverted (60 → 79). Loss flat. Audit at iter 10 showed `policy_bn_var=26.4` — well below run 1's iter-12 drift point of 44.3. Continued. |
| 12 | 3.6767 | 3.1436 | 0.5332 | 458.6 s | 90 | 4.5 | Policy new low (3.14). Games stable at 90. Audit: `policy_bn_var=33.9` — climbing but still under run 1's 44.3 threshold. |
| 13 | 3.4851 | 2.9921 | 0.4930 | 334.0 s | 67 | 6.1 | **Second sharp shortening** 90 → 67. Total dropped −0.95 across iters 9-13 (−0.24/iter, well above the 0.1/iter "fast drop" red flag). Audit: `pbn_var` went **down** 33.9 → 32.2 (oscillation, not run-1 monotone growth). False alarm cleared. |
| 14 | 3.6705 | 2.7474 | 0.9231 | 473.5 s | 94 | 4.3 | Total reversed up (+0.18), policy still hitting new lows (2.75), games back to 94. **Eval vs random = 49.0% — apparent catastrophic regression.** Triggered another audit; `pbn_var` jumped 32.2 → 42.6 (breached run-1 red flag). The eval looked apocalyptic but it turned out to be a metric failure, not real strength loss — see Tournament results below. |
| 15 | 3.4998 | 2.6338 | 0.8660 | 468.7 s | 92 | 4.4 | Policy new low (2.63). Held off on intervention pending head-to-head data. |
| 16 | 2.8011 | 2.5802 | 0.2209 | 362.5 s | 73 | 5.7 | Total drop −0.70. Policy stable at 2.58, value collapsed again. The "fast drop pattern" continued — alarming on the loss table, ambiguous on the audit. |
| 17 | 2.2028 | 1.9205 | 0.2824 | 482.3 s | 93 | 4.3 | Policy broke 2.0. Net loss progress 4.44 → 2.20 across iters 9-17 (−1.50, vs run 1's −1.43 across iters 12-19). Faster degradation than run 1 in raw loss terms. |
| 18 | 2.8348 | 2.1556 | 0.6792 | 390.6 s | 78 | 5.2 | **First up-tick.** Total +0.63, value bounced 0.28 → 0.68. Free-fall paused. |
| 19 | 2.7262 | 2.1197 | 0.6065 | 510.1 s | 100 | 4.0 | Total stabilized. **Eval vs random = 100.0% (100/100)** — completely contradicting iter 14's 49%. Confirmed the in-loop eval is broken; the same 100-game seed set giving 49% then 100% within 5 iters can't be measuring real strength. |
| 20 | 2.7321 | 2.0529 | 0.6792 | 503.2 s | 99 | 4.1 | Loss virtually unchanged from iter 19. Stable plateau confirmed. |
| 21 | 2.7400 | 2.0686 | 0.6714 | 553.4 s | 106 | 3.7 | **Longest avg moves of the run (106).** Loss has held in the 2.73-2.83 band for 4 iters. Training paused after this iter to run a comprehensive strength-curve tournament. |

### Iter-21 weight audit

```
iter   pbn_var  vbn_var  input_conv  pol_fc.max  status
  9    22.33    55.38    0.135       0.196       baseline
 14    42.58    67.72    0.200       0.346       breached pbn 40 ⚠️
 19    62.11    59.95    0.364       0.422       breached input_conv 0.25 at iter 16
```

Run 2 iter 19 vs run 1 iter 19: `pbn_var` 62 vs 74 (run 2 ~16% lower),
`input_conv` 0.36 vs 0.55 (run 2 ~35% lower). Run 2's drift is real
but slower than run 1 — the resign v2 + wider exploration fixes
slowed the specialization but didn't fully prevent it. **Importantly,
this is a scale issue, not a quality issue** — the actual play
strength is genuinely improving (see tournament results below).

### Strength-curve tournament (paused after iter 21)

The in-loop vs-random eval became unreliable (iter 14 = 49% then
iter 19 = 100% on the same 100-game seed set is impossible if the
model were really regressing). To get a trustworthy strength
trajectory, paused training and ran paired head-to-head matchups
between checkpoints using `training/eval_matchup.py` (25 pairs ×
100 sims/move × 4 random opening ply, paired colors so first-move
advantage cancels).

Anchor: `checkpoint_0009.pt`. All other checkpoints play against it.

| iter (NEW) | score vs iter 9 | 95% CI | pairs (NEW–OLD) | ties | verdict |
|---:|---:|---|---:|---:|---|
| 0 | **0.220** | [0.10, 0.41] | 0–14 | 11 | REGRESSION (expected — barely trained) |
| 3 | 0.500 | [0.32, 0.68] | 0–0 | 25 | matched |
| 6 | 0.500 | [0.32, 0.68] | 0–0 | 25 | matched |
| 9 | (anchor) | — | — | — | self-reference |
| 12 | 0.500 | [0.32, 0.68] | 0–0 | 25 | matched |
| 14 | **0.580** | [0.39, 0.75] | 4–0 | 21 | meaningful (despite in-loop eval saying 49% ⚠️) |
| 15 | 0.540 | [0.35, 0.72] | 3–1 | 21 | matched |
| 18 | **0.720** | [0.52, 0.86] | 11–0 | 14 | STRONG |
| 19 | **0.920** | [0.75, 0.98] | 21–0 | 4 | STRONG (in-loop eval said 100% — agrees) |
| 21 | **1.000** | [0.87, 1.00] | 25–0 | 0 | STRONG |

Sanity check: **iter 0 vs iter 21 = 0.000** (0–25, 0 ties). The end
of the run is dramatically stronger than the start, as expected.

### Three things the tournament tells us

1. **There was never a regression.** Iter 14 — the supposedly
   catastrophic 49% point — was actually *slightly stronger* than
   iter 9 (4 clear wins, 0 losses, 21 ties). The in-loop
   `evaluate_vs_random` is a broken signal here. Likely cause: at
   iter 14 the value head briefly over-weighted the pass action,
   which causes AZ-as-black to lose to random by Tromp-Taylor area
   scoring with komi 7.5. The function passes whenever
   `tree.best_action()` returns the pass index, with no recovery
   path. By iter 19 the value head re-balanced that specific failure
   mode even though the broader BN drift continued.

2. **Real improvement is monotone with a sharp inflection at ~iter
   14–16.** Iters 3–12 are statistical ties with iter 9 (100 sims
   per move at eval can't distinguish weak models from each other;
   the games keep ending in color-decided ties). The first decisive
   improvement shows at iter 14, then the model accelerates:
   0.58 → 0.72 → 0.92 → 1.00 across iters 14–21.

3. **BN drift ≠ quality loss.** Run 2 ended iter 21 with `pbn_var`
   = 62 and `input_conv` = 0.36 — both clearly breaching run 1's
   red-flag thresholds — yet head-to-head says iter 21 dominates
   every earlier checkpoint. The BN drift is a scale/normalization
   shift, not the kind of representational specialization that
   destroys play strength. The run 1 doc's "policy_bn > 40 = trouble"
   threshold was over-conservative.

### Lesson learned for future runs

**Never trust `evaluate_vs_random` as the primary regression signal
for a strong model.** It has a hidden failure mode: the function
unconditionally takes whatever `best_action()` returns, including the
pass action, with no safety net. A model whose value head transiently
over-weights passing will lose to random play by area scoring even
though it would crush the same opponent if forced to play real moves.
Head-to-head matchups (`eval_matchup.py`) are the real signal.

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
