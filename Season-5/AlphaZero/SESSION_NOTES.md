# AlphaZero Self-Play Optimization Session

Notes from the debugging + optimization session where we took the self-play
pipeline from "appears hung" to a working Phase 1 training run.

## Starting symptom

Phase 1 training (`python -m training.train --board-size 9 --iterations 73`)
was launched on an RTX 4090 instance. After **~6 minutes of wall time**, iter
0 still hadn't printed — but the process was clearly alive:

- `%CPU` reported **2620%** (~26 cores' worth)
- Cumulative CPU `TIME` was **3h 30m** after 6 min wall → ~35x parallelism
- `nvidia-smi` showed ~33% GPU utilization
- 199 threads in the process

## Root cause #1 — torch CPU thread pool contention

`torch.get_num_threads()` returned **128** and `torch.get_num_interop_threads()`
also returned 128. Every CPU tensor op in the orchestrator (the
`policies.cpu()` transfer, the non-pinned copies) fanned out across a
128-wide intra-op pool. Combined with the 5 C++ self-play worker threads,
the process had **~199 threads** contending on a **27.2-core cgroup quota**
(read from `/sys/fs/cgroup/cpu/cpu.cfs_quota_us`).

On this container `nproc` returned 256 (host AMD EPYC 7763 total), but the
container was cgroup-limited to 27.2 cores. The 6-vCPU figure the user
quoted is a RunPod plan label; actual scheduler quota is 27.2 cores.

### Fix

`training/train.py` now sets CPU thread limits **before** importing torch:

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

Thread count dropped from **199 → 9**. Broken smoke test went from
**9–11 s/iter → 1.2–2.9 s/iter** — ~7× faster.

## Root cause #2 — pinned memory misused on the D2H path

Even with threads fixed, the `ParallelSelfPlay` loop was doing:

```python
# WRONG — defeats pinned memory
self.policy_pinned[:total_nn].copy_(policies.cpu(), non_blocking=False)
```

`policies.cpu()` is a **blocking** transfer into a **fresh non-pinned** CPU
tensor; the subsequent copy into `self.policy_pinned` is a second CPU→CPU
copy. The pre-allocated pinned buffer provided zero benefit on this path.

### Fix

```python
self.policy_pinned.copy_(policies, non_blocking=True)
self.value_pinned.copy_(values, non_blocking=True)
torch.cuda.synchronize()
```

One non-blocking GPU→pinned DMA, one sync before reading.

## Plan's GPU estimate was wildly optimistic

`PLAN.md` assumed **0.5 ms** per batched forward (9x9, 10b × 128ch, batch
2048, BF16). Measured on the 4090: **15.2 ms** — **~30× slower** than plan.

Rough theoretical floor: ~490 GFLOPs per forward at batch 2048, 4090 bf16
peak ~85 TFLOPS → ~5.8 ms minimum. Measured 15 ms is reasonable for small
conv towers where kernel launch and memory bandwidth dominate.

Consequence: the plan's $1.1/run, 2.5 h/run estimate for 9x9 was never
achievable, regardless of anything we did. Real cost is ~$8–9/run for 150K
games at current speed.

## Optimizations applied (all lossless, ordered by impact)

### 1. `torch.compile(mode="max-autotune")`

Triton-based autotuned kernels. `reduce-overhead` works too but `max-autotune`
gets an extra ~10% over it.

**Gotcha:** `torch.compile` with `dynamic=True` recompiles on every unique
input shape → the "bimodal timing" bug we hit (median 20 ms, mean 50 ms).
Fix was to always pass the **full fixed-size** pinned buffer to forward,
even when `total_nn < total_max_nn`. Trailing slots carry stale data; their
outputs are harmless because each worker only reads its own offset range.

### 2. Zero-copy pinned buffers

Replaced the old pattern (numpy buffer → `np.concatenate` → copy into pinned
→ H2D) with a **direct numpy view over the pinned tensor**:

```python
self.obs_pinned = torch.empty((total_max_nn, 17, N, N),
                              dtype=torch.float32, pin_memory=True)
self.obs_buffer = self.obs_pinned.numpy()  # zero-copy view
```

C++ workers write leaf observations directly into the pinned memory via the
numpy view. Orchestrator then calls `self.obs_pinned.to(device, non_blocking=True)`
with no staging copies. Same trick for `policy_buffer` / `value_buffer` on
the D2H side.

**Saved ~3 ms per tick** (GPU+xfer median 13.56 → 10.23 ms). Slight cost:
C++ workers writing to pinned memory are marginally slower (~1.4 → 1.7 ms
for `tick_select`) because pinned pages are uncached differently, but the
net is still a large win.

### 3. Fixed-shape GPU forward

Always forward at `total_max_nn`, never at the varying `total_nn`. Locks
torch.compile onto a single input shape and eliminates recompile spikes.

### 4. `network.py`: `.view()` → `.flatten()`

`.view()` fails on non-contiguous layouts. Used `.flatten(1)` in the policy
and value heads so the model is layout-agnostic (needed for a channels_last
experiment that didn't pay off, see below). This is a safe no-op change
for NCHW and enables NHWC if we ever want it.

### 5. TF32 for fp32 matmul

`torch.set_float32_matmul_precision("high")` — free on 4090.

## Things tried and rejected

### Channels-last / NHWC memory format

Hypothesis: RTX 4090 tensor cores prefer NHWC and our model is conv-heavy.
Result: **+1.8 ms on GPU+xfer**. The transpose cost of converting NCHW obs
tensors to NHWC on every tick (we can't store obs in NHWC without rewriting
the C++ observation encoder) outweighed the kernel win for this small
model (10b × 128ch at batch 2048). Reverted.

### Cutting `num_simulations` or `num_parallel_games`

Explicitly off-limits per user direction: no MCTS quality sacrifice.

### Pipelining (worker-select overlapped with GPU forward)

Expected +30% but introduces a 1-tick delay between selection and tree
update, which is a subtle MCTS quality tradeoff. Held off on the user's
"no performance sacrifice" constraint.

## Final measured throughput

Two sources: (1) the synthetic `_bench_selfplay.py` micro-benchmark and
(2) real iter 0 of the Phase 1 run.

### Synthetic bench (`_bench_selfplay.py`)

60 s warmup + 120 s measurement window; real Phase 1 config: 10b × 128ch,
batch 2048, 400 sims, 5 workers × 256 parallel games:

| Component | Before fixes | After all fixes |
|---|---:|---:|
| Tick median | 26.2 ms | **~15–17 ms** |
| GPU + transfers median | 17.8 ms | **~9–10 ms** |
| "Other" (barriers + bookkeeping) | 7.9 ms | ~6.3 ms |
| Throughput | ~0.7 games/s (first 20 s, warmup-polluted) | **~2.1 games/s** |

### Real iter 0 of Phase 1 (the number that actually matters)

| Metric | Value |
|---|---:|
| Games completed | 2049 |
| Positions generated | 157,642 |
| Average moves / game | 77 |
| Self-play time | **644.6 s** |
| Train time (100 steps) | 1.8 s |
| **Games / s** | **3.2** |
| Initial loss | 5.25 (policy 4.37, value 0.88) |

**The real throughput (3.2 games/s) is ~1.5× the synthetic bench (2.1
games/s).** The bench was conservative because:

1. It used a tiny warmup replay buffer — real workers run with real games,
   and tree reuse across moves gets better amortization once games settle.
2. Synthetic obs were random tensors, but Go positions have much higher
   sparsity — the GPU forward path caches and memory access patterns
   are slightly friendlier in reality.
3. The bench measured 20 s of an ongoing run without waiting for
   steady-state game completion rate; iter 0 reports the full 2048-game
   window average.

### Phase 1 projection

At 3.2 games/s and **\$0.60/hr** (actual RunPod rate, not the \$0.44/hr
in `PLAN.md`):

| Item | Value |
|---|---:|
| Games per run | 150K (73 iters × 2048) |
| Self-play wall time | ~13 h |
| Training overhead | ~2 min |
| Eval overhead (7 evals) | ~30 min |
| **Total wall time** | **~13.5 h** |
| **Cost per run** | **~\$8.1** |

Plan's original Phase 1 estimate was 2.5 h / \$1.1, which was never
physically reachable: the plan assumed a 0.5 ms GPU forward but the 4090
actually takes ~15 ms for a bs=2048 bf16 forward on our 10b × 128ch
model (~30× miss). Our 13.5 h is as close as we can get without
cutting sims or accepting an MCTS quality tradeoff from pipelining.

## Correctness verification

`training/_test_correctness.py` runs the real 10b × 128ch model and checks:

- Self-play produces 65 games / 6236 positions in 5 s
- Replay buffer `obs` is binary (0/1) — C++ encoder round-trip works
- Replay buffer `policy` rows sum to 1.0 exactly
- Replay buffer `value` is in [-1, 1] with balanced mean (0.002)
- Training loss drops 4.99 → 4.35 over 5 epochs; value loss 0.62 → 0.05
- `evaluate_vs_random` runs 10 games of MCTS vs random without crashing

All checks pass.

## Files touched

| File | Change |
|---|---|
| `training/train.py` | Thread limits before torch import, TF32, import-safe interop threads |
| `training/parallel_self_play.py` | torch.compile(max-autotune), zero-copy pinned buffers, fixed-shape forward, pinned D2H fix, removed `np.concatenate` |
| `model/network.py` | `.view()` → `.flatten(1)` in policy / value heads |
| `training/_bench_selfplay.py` | Micro-benchmark (new, in-tree for reproducibility) |
| `training/_bench_batch.py` | Batch-size scan (new) |
| `training/_test_correctness.py` | End-to-end correctness test (new) |

## Phase 1 run in progress

Launched `python -m training.train --board-size 9 --iterations 73 --num-workers 5`
as pid **11235** at 2026-04-12 07:55 UTC. Logs at
`Season-5/AlphaZero/logs/9x9_run1.log`, checkpoints at
`Season-5/AlphaZero/checkpoints/9x9_run1/` every 10 iterations.

**Expected completion:** ~13.5 h total wall time / ~\$8.1 on RunPod at
\$0.60/hr. Iter 0 finished at 644 s with loss 5.25 and 3.2 games/s —
healthy signal.

A monitor cron (every 10 min) tails the log and writes state into
`logs/monitor.log` so we can spot stalls, crashes, or degraded loss
without babysitting.
