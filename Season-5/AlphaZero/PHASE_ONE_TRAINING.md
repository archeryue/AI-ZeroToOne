# Phase 1 Training — 9x9 Go AlphaZero

Living document for the 9x9 Go Phase 1 run. Merges the original
optimization session notes and the SIGSEGV crash handover, then
continues as the ongoing training journal. Append new progress at the
bottom; keep the historical sections above immutable so we always
have the full context for future debugging.

---

## Environment

- **Host:** RunPod RTX 4090 (24 GB VRAM), AMD EPYC 7763
- **Container:** cgroup CPU quota **27.2 cores** (not the "6 vCPU" plan
  label; read from `/sys/fs/cgroup/cpu/cpu.cfs_quota_us`), memory
  limit 125 GB, `oom_kill_disable=1`
- **Python:** 3.11, **torch:** 2.x with inductor / Triton
- **GPU driver:** NVIDIA 580.126.20, CUDA 13.0
- **OS / FS:** Linux 6.8.0-107, cgroup v1 for `cpu,cpuacct` + `memory`
- **Blocked capabilities:** no root, `ptrace_scope=1`, `dmesg`
  unreadable — no live debugger, core dumps effectively blocked
- **Actual RunPod rate:** \$0.60/hr (not the \$0.44/hr in `PLAN.md`)

## Training config (9x9)

- Model: **10 residual blocks × 128 channels**, ~3.0 M parameters
- 256 parallel games × 5 self-play worker threads
- 400 MCTS sims/move, virtual-loss batch 8
- 2048 games / iter, 73 iters total
- Replay buffer 500K positions, 8-fold symmetry augmentation
- Checkpoint every iter (overridden for 9x9); retain last 5

## Code layout

```
Season-5/AlphaZero/
├── engine/                        # C++ Go + MCTS + SelfPlayWorker (pybind11)
│   ├── go.{h,cpp}                 # Board + Game rules
│   ├── mcts.{h,cpp}               # MCTSTree<N>, select_leaf, expand, backup
│   ├── worker.{h,cpp}             # SelfPlayWorker<N>::tick_select / tick_process
│   ├── bindings.cpp               # pybind11 glue
│   └── build/lib.linux-.../go_engine.*.so
├── model/
│   ├── network.py                 # AlphaZeroNet (ResNet policy+value)
│   └── config.py                  # ModelConfig / TrainingConfig / CONFIGS
├── training/
│   ├── train.py                   # entry point, main() loop
│   ├── parallel_self_play.py      # orchestrator + worker threads + barriers
│   ├── trainer.py                 # SGD trainer, checkpoints
│   ├── replay_buffer.py           # circular buffer, 8-fold symmetry augment
│   ├── _bench_selfplay.py         # GPU-forward + tick micro-bench
│   ├── _bench_batch.py            # batch-size scan
│   ├── _test_correctness.py       # end-to-end smoke test
│   └── _test_segv_repro.py        # multi-threaded worker stress repro
├── logs/9x9_run1.log              # current training stdout/stderr
└── checkpoints/9x9_run1/          # checkpoint_NNNN.pt, training_log.jsonl
```

---

## Part 1 — Self-play optimization (pre-training session)

The first session took the self-play pipeline from "appears hung" to
a working Phase 1 run. Captured here because the same mistakes are
very easy to re-introduce.

### Starting symptom

Launching `python -m training.train --board-size 9 --iterations 73`
looked hung: after ~6 minutes of wall time, iter 0 still hadn't
printed. But the process was clearly alive —

- `%CPU` reported **2620%** (~26 cores' worth)
- Cumulative CPU `TIME` was **3h 30m** after 6 min wall (~35× parallelism)
- GPU ~33% utilization
- **199 threads** in the process

### Root cause 1 — torch CPU thread-pool contention

`torch.get_num_threads()` returned **128**. Every orchestrator CPU
tensor op fanned out across a 128-wide intra-op pool and fought the
5 C++ self-play threads on a 27.2-core quota.

**Fix — set thread limits *before* importing torch** in
`training/train.py`:

```python
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_float32_matmul_precision("high")
```

Thread count: **199 → 9**. Smoke-test iter: **9–11 s → 1.2–2.9 s**
(~7× faster).

### Root cause 2 — pinned memory misused on the D2H path

The old D2H code was:

```python
# WRONG — defeats pinned memory
self.policy_pinned[:total_nn].copy_(policies.cpu(), non_blocking=False)
```

`.cpu()` is a *blocking* transfer into a *fresh non-pinned* tensor.
The pre-allocated pinned buffer contributed nothing.

**Fix:**

```python
self.policy_pinned.copy_(policies, non_blocking=True)
self.value_pinned.copy_(values, non_blocking=True)
torch.cuda.synchronize()
```

### Optimizations applied (ordered by impact)

1. **`torch.compile(mode="max-autotune")`** — Triton autotuned
   kernels, ~10% over `reduce-overhead`.
   - Gotcha: `dynamic=True` recompiles per unique shape. Fix was to
     always pass the **full fixed-size** pinned buffer to forward,
     even when `total_nn < total_max_nn` — trailing slots carry stale
     data but their outputs are harmless (each worker reads only its
     own offset range).
2. **Zero-copy pinned buffers** — numpy view over the pinned tensor
   so C++ workers write leaf observations straight into pinned
   memory, eliminating the staging `np.concatenate` + copy.
   `obs_pinned.to(device, non_blocking=True)` is a single DMA from
   there. Saved ~3 ms per tick. *(Note: reverted later for a
   different reason — see Part 2 — and since put back in a modified
   form with per-worker owned buffers.)*
3. **Fixed-shape GPU forward.** Always forward at `total_max_nn`,
   never at varying `total_nn`. Locks torch.compile onto one shape
   and eliminates recompile spikes.
4. **`network.py`: `.view()` → `.flatten(1)`** in the policy/value
   heads. Layout-agnostic; required for any NHWC experiment.
5. **TF32 matmul:** `torch.set_float32_matmul_precision("high")` —
   free on 4090.

### Things tried and rejected

- **Channels-last / NHWC.** +1.8 ms GPU+xfer on our model (10b × 128ch,
  bs 2048). Transpose cost on every tick outweighs the kernel win.
- **Cutting `num_simulations` / `num_parallel_games`.** Explicitly
  off-limits — no MCTS quality sacrifice.
- **Pipelining (worker-select overlapped with GPU forward).** Would
  add a 1-tick lag between selection and tree update — a subtle MCTS
  quality tradeoff. Held off per the no-sacrifice constraint.

### `PLAN.md`'s GPU estimate was wildly optimistic

PLAN assumed **0.5 ms** per batched forward. Measured on 4090:
**15.2 ms** (~30× slower). Theoretical floor is ~5.8 ms
(~490 GFLOPs at bs 2048 ÷ ~85 TFLOPS bf16 peak), so 15 ms is
reasonable for a small conv tower where kernel launch + memory
bandwidth dominate.

Consequence: PLAN's \$1.1 / 2.5 h-per-run estimate was never
physically reachable. Real cost is **~\$8 / ~13.5 h per run**.

### Post-optimization throughput (iter 0 of Phase 1, first run)

| Metric | Value |
|---|---:|
| Games completed | 2049 |
| Positions generated | 157,642 |
| Avg moves / game | 77 |
| Self-play time | 644.6 s |
| Train time (100 steps) | 1.8 s |
| **Games / s** | **3.2** |
| Initial loss | 5.25 (policy 4.37, value 0.88) |

---

## Part 2 — The SIGSEGV hunt

### Symptom

After iter 0 printed successfully, the process began segfaulting
reliably inside the C++ self-play worker. The crash signature was
always:

```
Fatal Python error: Segmentation fault

[5 threads, all at]
  parallel_self_play.py line 145  (worker.tick_select(obs_slice))
```

plus main thread parked at `select_barrier.wait()`. The crash landed
at different wall times depending on the starting state:

| # | Checkpoint | `AZ_COMPILE` | pinned | Survived | Iters done | Crash |
|---|---|---|---|---|---|---|
| 1 | fresh | on | zero-copy | ~28 min | 0,1,2 | after iter 2 |
| 2 | fresh | on | zero-copy | ~28 min | 0,1 | after iter 1 |
| 3 | ckpt_0001 | on | zero-copy | ~5 min | 0 | during iter 0 |
| 4 | ckpt_0001 | **off** | zero-copy | ~35 min | 2,3 | at iter 4 start |
| 5 | ckpt_0003 | on | plain np | ~3 min | 0 | at iter 4 start |
| 6 | ckpt_0003 | on | plain np | ~3 min | 0 | at iter 4 start |

### Wrong leads chased (do not repeat)

1. **Inductor compile-subprocess pool.** The `subproc_pool._read_thread`
   frame showed up in some crash dumps but only 2 of 4. Setting
   `TORCHINDUCTOR_COMPILE_THREADS=1` (disables the 32-child pool)
   did NOT stop the segv — a fresh run with the env var crashed in
   ~1 minute from ckpt_0003.
2. **Zero-copy pinned-memory-backed numpy views.** Earlier hypothesis;
   reverted to plain `np.zeros` for worker buffers; segv persisted.
3. **Numpy view slicing into a shared `obs_buffer`.** Refactored to
   per-worker OWNED numpy arrays so workers never touch a sliced
   view. Segv reproduced on the first tick_select anyway — proving
   the bug is not in the Python↔pybind11 buffer handoff.
4. **`torch.compile`**. Disabling it bought longer survival but
   still crashed at iter 4 start (run #4). Not the cause, just
   reduced pressure.
5. **`trainer.load_checkpoint` corruption.** Seemed to correlate —
   resume crashed faster than fresh — but the real mechanism is
   that a stronger checkpoint produces more confident MCTS, which
   grows deeper trees sooner, which hits the real bug faster.

### Root cause: `engine/mcts.h::select_leaf` out-of-bounds write

```cpp
static constexpr int MAX_PATH_DEPTH = 64;

struct LeafInfo {
    int path[MAX_PATH_DEPTH];    // <-- fixed 64 slots
    int path_len;
    int leaf_idx;
    bool needs_nn;
    float terminal_value;
};

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

With MCTS tree reuse via `advance()` keeping the surviving subtree
from prior moves, the effective depth grows throughout a single
game. At 400 sims/move and `virtual_loss_batch=8`, PUCT chases
"confident" lines deep, and depth easily exceeds 64. At that point
`out_path[64++]` scribbles node indices into the next `LeafInfo`'s
`path_len`, `leaf_idx`, ... and then into the following `LeafInfo`'s
`path[]`. The next `select_leaves` iteration dereferences the
corrupted `info.path[info.path_len - 1]` and crashes inside
`fill_observations` or `select_child`.

**Why the dump shows "5 workers at line 145":** faulthandler catches
SIGSEGV on the offending thread and walks *all* Python frames. The
four non-crashing workers are also inside pybind11 C++ (GIL
released) in their own `tick_select`, so every worker reports the
same Python frame.

### Fix

`engine/mcts.h`:

```cpp
// MCTS tree depth is usually ~10-30, but with tree reuse across many
// moves the effective depth can grow unboundedly. 256 is a safe bound
// for any realistic Go position; select_leaf also hard-caps against
// this limit to prevent out-of-bounds writes.
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

Memory cost: `16 × 4 × (256 − 64) = 12 KB per GameSlot.leaves[]`; with
52 games/worker, ~625 KB/worker. Negligible vs. ~8 GB process RSS.

**Rebuild (setuptools doesn't always notice header-only changes):**

```bash
cd engine
rm -rf build/temp.linux-x86_64-cpython-311 build/lib.linux-x86_64-cpython-311 go_engine.*.so
python3 setup.py build_ext --inplace
```

### Auxiliary code change: per-worker owned numpy arrays

While hunting the bug, the shared `obs_buffer`/`policy_buffer`/
`value_buffer` (with per-worker slice views) was replaced with fully
owned per-worker arrays in `training/parallel_self_play.py`:

```python
self.worker_obs    = [np.zeros((mnn, 17, N, N), dtype=np.float32) for mnn in ...]
self.worker_policy = [np.zeros((mnn, N*N+1),    dtype=np.float32) for mnn in ...]
self.worker_value  = [np.zeros(mnn,             dtype=np.float32) for mnn in ...]
```

Each worker passes its own array directly to
`worker.tick_select(obs_buf)` — no slicing of a shared backing
array. `run_games` stages each worker's obs into the correct offset
of `self.obs_pinned` before the GPU forward and splits the D2H
policy/value outputs back into the per-worker arrays afterward.

This change did NOT fix the segv (the bug was in C++), but it's
cleaner and removes a class of shared-pointer concerns. Kept.

### Validation

Restart from `checkpoints/9x9_run1/checkpoint_0003.pt` — the
previously-reliable sub-3-minute crash path:

```bash
cd /root/AI-ZeroToOne/Season-5/AlphaZero
export PYTHONPATH="$PWD:$PWD/engine/build/lib.linux-x86_64-cpython-311:$PYTHONPATH" \
       PYTHONFAULTHANDLER=1 \
       TORCHINDUCTOR_COMPILE_THREADS=1
nohup python3 -u -m training.train \
    --board-size 9 --iterations 73 --num-workers 5 \
    --checkpoint checkpoints/9x9_run1/checkpoint_0003.pt \
    --output-dir checkpoints/9x9_run1 \
    > logs/9x9_run1.log 2>&1 &
```

`TORCHINDUCTOR_COMPILE_THREADS=1` is defensive — the 32-child compile
pool is unnecessary here and is exactly the kind of thing we don't
want to co-exist with a live C++ worker pool. It is **not**
load-bearing for the fix.

### If it ever regresses

1. Grep the new log for `Fatal Python error` and compare the top
   frame. If it's still at `tick_select`, audit other fixed-size
   buffers in the MCTS / worker hot path — but none of the remaining
   ones (`MCTSTree::expand::masked[ACTIONS]`, `apply_dirichlet_noise::noise[ACTIONS]`,
   `SelfPlayWorker::tick_select` obs bound) are unsafe.
2. If the top frame moved elsewhere, treat it as a fresh bug — the
   MCTS depth bug is unambiguously fixed.
3. The old counter-hypotheses (inductor subproc, pinned memory,
   ckpt corruption) are **all disproved**; do not chase them again.

---

## Files touched (cumulative)

| File | Change |
|---|---|
| `engine/mcts.h` | `select_leaf` bounds check; `MAX_PATH_DEPTH 64 → 256` |
| `training/parallel_self_play.py` | Per-worker owned numpy arrays; `torch.compile(max-autotune)`; fixed-shape GPU forward; pinned D2H fix; `np.concatenate` removed |
| `training/train.py` | Thread limits before torch import; TF32; import-safe interop threads; faulthandler (signal + periodic); hoisted `ParallelSelfPlay` outside the training loop; checkpoint retention |
| `model/network.py` | `.view()` → `.flatten(1)` in policy/value heads |
| `model/config.py` | `checkpoint_interval=1` for 9x9 |
| `training/_bench_selfplay.py` | Micro-benchmark (new) |
| `training/_bench_batch.py` | Batch-size scan (new) |
| `training/_test_correctness.py` | End-to-end smoke test (new) |
| `training/_test_segv_repro.py` | Multi-threaded stress reproducer (new) |

---

## Part 3 — Live training run

### Run metadata

- **Run label:** `9x9_run1`
- **Checkpoint dir:** `checkpoints/9x9_run1/`
- **Log file:** `logs/9x9_run1.log`
- **Command:** `python -m training.train --board-size 9 --iterations 73 --num-workers 5 --checkpoint checkpoints/9x9_run1/checkpoint_0003.pt --output-dir checkpoints/9x9_run1`
- **Env:** `TORCHINDUCTOR_COMPILE_THREADS=1`, `PYTHONFAULTHANDLER=1`
- **Restart pid (post-fix):** 22329
- **Started (post-fix):** 2026-04-12, shortly after the MCTS bounds
  patch + clean rebuild landed
- **Resumed from iteration:** 3 (iter 4 is the first new iter)

### Baseline loss (from the pre-crash `.degraded` log)

| iter | total | policy | value |
|---|---:|---:|---:|
| 0 | 5.25 | — | — |
| 1 | 5.02 | — | — |
| 2 | 4.78 | 3.94 | 0.84 |
| 3 | 4.75 | 3.92 | 0.83 |

Policy baseline: `log(82) ≈ 4.41` — anything below that beats
uniform random.

### Progress log (post-fix run)

| iter | total | pi | v | self-play time | avg moves | games/s | note |
|---|---:|---:|---:|---:|---:|---:|---|
| 4 | 4.8910 | 3.9888 | 0.9023 | 768.0 s | 99 | 2.7 | First post-fix iter; bump over iter 3 attributable to resume noise |
| 5 | 4.7319 | 3.8885 | 0.8434 | 758.0 s | 111 | 2.7 | Below pre-crash iter 3 (4.75) — recovered |
| 6 | 4.5885 | 3.8375 | 0.7509 | 496.3 s | 73 | 4.1 | Resign threshold firing more often → shorter games → ~1.5× faster iters |
| 7 | 4.6555 | 3.8798 | 0.7758 | 580.4 s | 90 | 3.5 | Small uptick (+0.07); games back up to 90 moves. Still within noise band; value loss stable. |
| 8 | 4.6980 | 3.8862 | 0.8118 | 633.2 s | 98 | 3.2 | Slow two-iter drift upward (4.59→4.66→4.70). Still under pre-crash iter 3 (4.75). Games lengthening; watch through iters 9–10. |
| 9 | 4.7394 | 3.8799 | 0.8595 | 633.7 s | 97 | 3.2 | Fourth uptick in a row. **Eval: 96/100 vs random (197.8 s).** Loss drift is the value head fitting harder mid-game positions from a stronger replay distribution — NOT a regression. Win rate says the model is fine. |
| 10 | 4.6207 | 3.8152 | 0.8056 | 633.7 s | 94 | 3.2 | Drift broken. Policy loss hit a new low (3.82, below iter 6's 3.84); value loss recovered 0.05. Iter 7–9 uptick confirmed noise. |
| 11 | 4.4532 | 3.6879 | 0.7653 | 550.5 s | 84 | 3.7 | New best across all iters — total down 0.17, policy breaks 3.70 barrier. Games shortening, throughput climbing. Clean improvement trajectory. |
| 12 | 3.9908 | 3.4124 | 0.5784 | 266.1 s | 44 | 7.7 | **Phase transition.** Total loss breaks 4.0 (−0.46 in one iter), clearly below `log(82)=4.41`. Value loss 0.77→0.58. Avg moves halved (84→44): resign threshold firing on majority of games. Throughput doubles to 7.7 g/s; projected remaining runtime drops from ~10 h to ~4.6 h. |
| 13 (pre-fix) | **NaN** | **NaN** | **NaN** | 851.0 s | 103 | 2.4 | **Weight explosion.** Games jumped 44→103 moves (opposite direction from iter 12 — resign no longer firing), throughput collapsed, then the training step produced NaN loss. Weight norm of `checkpoint_0013.pt` = NaN. iter 9–12 weights remained clean (max\|w\|=68–78). Root cause: no `clip_grad_norm_` in `trainer.train_step` — a single gradient spike sent the weights to ∞. Fix: add `clip_grad_norm_(max_norm=5.0)` + `torch.isfinite(loss)` skip guard + apply-count-based epoch averaging. **Rolled back to `checkpoint_0012.pt` and restarted as pid 23196.** |
| 13 (post-fix) | 4.4282 | 3.6363 | 0.7918 | 809.4 s | 105 | 2.5 | Clean run with grad clipping. Loss ~iter 11's level — confirms iter 12's 3.99 was an artifact of the pre-collapse state, not real improvement. Self-play characteristics (105 moves, 2.5 g/s) match pre-fix iter 13, so iter 12's model really does produce longer less-decisive games. Zero skipped steps → clipping alone was sufficient, NaN guard never fired. |
| 14 | 4.1543 | 3.3239 | 0.8304 | 504.0 s | 60 | 4.1 | Policy loss new all-time low (3.32), below iter 12's suspect 3.41. Games shortening (105→60), throughput recovering (2.5→4.1 g/s). Value loss (0.83) is honestly higher than iter 12's artifactual 0.58 — this is the real trajectory. Zero NaN, zero skipped steps. |
| 15 | 3.8360 | 3.1324 | 0.7036 | 797.8 s | 96 | 2.6 | Total loss 3.84 — honestly beats iter 12's artifactual 3.99. Policy 3.13 new low; value 0.70 recovering. Games back to 96 moves (less aggressive resign). Clean improvement trajectory. |
| 16 | 3.2084 | 2.7687 | 0.4396 | 546.8 s | 65 | 3.7 | **Big real-weights jump.** −0.63 total in one iter. Policy 2.77 now clearly below `log(82)=4.41` baseline by a massive margin; value 0.44 strong. Games shortening, throughput up. This is the phase transition iter 12 was faking, now on clean weights. |
| 17 | 3.0118 | 2.2271 | 0.7847 | 768.9 s | 91 | 2.7 | Policy keeps crushing (2.23 new low). Value bounced back up to 0.78. ⚠️ Iter 15 eval vs iter 9 = 0.380 / 0.400 (seeds 0, 42) — local dip. |
| 18 | 2.3848 | 2.0499 | 0.3349 | 471.5 s | 59 | 4.4 | Another big loss drop (−0.63). Policy 2.05 new low. **Iter 16 eval vs iter 9 = 0.600 (STRONG).** Iter 15 dip was a local per-iter wobble, not catastrophic forgetting — iter 16 already recovered and iter 12 → iter 16 trajectory is genuinely improving (0.640 → 0.600, both STRONG). Keep training as-is. |
| 19 | 2.5626 | 1.8929 | 0.6698 | 682.1 s | 85 | 3.0 | Policy 1.89 new low. Value bounced (0.33→0.67), total slightly up. Same policy-monotonic / value-oscillating pattern. Matchup vs iter 9 queued for when `checkpoint_0019.pt` lands. |

_(Append new iters as they land. Flag any Fatal/Segmentation/Traceback
lines immediately and root-cause before restarting.)_

### Monitoring in place

- **Monitor `bklbbqd08`:** `tail -F logs/9x9_run1.log` filtered for
  `^Iter` / Fatal / Segmentation / Traceback / Error / Exception.
  Fires a notification on every iter completion and on any crash
  signal.
- **Cron `5b46ba02`:** every 15 min at `:07 :22 :37 :52`. Prints
  process uptime + RSS + GPU util, last 3 `Iter` lines, and flags
  any fatal signals. Session-scoped, auto-expires after 7 days.

### Expected cost / wall time

At the current ~3 games/s average (iters 4–5) dropping toward
~4 games/s as resign fires more often (iter 6), the remaining
67 iterations should land somewhere in the **9–12 h wall time**
band, for a total run cost around **\$5–7**.
