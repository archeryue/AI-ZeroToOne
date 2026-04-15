# AlphaZero training — workload profile and GPU selection

Notes on how AlphaZero Zero-style training actually loads a GPU, why
most of it is *not* what you'd expect from a deep learning workload,
and how to choose hardware cost-effectively. Numbers are grounded in
this project's measurements on 13×13 / 15b×128ch training with
`num_parallel_games=256`, `virtual_loss_batch=8`,
`num_simulations=400`.

## TL;DR

- **Zero training is memory-bandwidth and orchestration bound, not
  compute bound.** On an RTX 4090, the actual workload runs at
  **~1.5 %** of the card's theoretical BF16 peak.
- **For this workload, the 4090 is the cost-effective sweet spot.**
  Higher-end cards (A100, H100) cost 6–10× more but deliver <2×
  speedup because neither can saturate its tensor cores on a 4.5M-
  parameter ResNet at batch 2048.
- **Real speedups live in software, not hardware**: shrinking the
  net, multi-GPU self-play parallelism, CUDA Graphs to eliminate
  per-tick sync, or async inference pipelines.
- **The exception**: if you train at a much larger scale (≥15 blocks
  × 256 channels on 19×19) the tensor-core utilization improves and
  A100/H100 start to pay off. Our 13×13 / 15b×128ch workload is not
  that exception.

---

## What AlphaZero training actually looks like

A Zero training iter has three very different phases by compute
profile:

| phase | wall time (our iter 0) | what's running | bottleneck |
|---|---|---|---|
| **self-play** | ~50 min (~98 %) | C++ MCTS tree search on CPU threads + GPU batch inference per "tick" | GPU forward passes + barrier sync + Python overhead |
| **training** | ~2 s (<0.1 %) | SGD on replay buffer, 30 steps × 256 batch | Pure GPU compute on the same net |
| **eval vs random** | ~2 min (~4 %) | 50 games × 100 sims/move vs a random opponent | GPU forward passes, much smaller batch per move |

Unlike supervised learning where training is the expensive part,
**~98 % of the wall time goes into self-play inference**, not
backprop. This inverts your intuition about what matters for hardware
selection: you want a card that's fast at **many small forward
passes**, not one that's fast at big batch SGD steps.

### What a "tick" is, and why it's the unit that matters

`parallel_self_play.py` runs a barrier-synchronized loop across N
worker threads and a single orchestrator:

```
tick = {
    1. workers select leaves from their MCTS trees (C++, GIL released)
    2. barrier — all workers wait for the slowest
    3. orchestrator batches all selected leaves' observations into one
       GPU forward pass, gets (policy, value, ownership) logits back
    4. barrier — workers wait for the GPU result
    5. workers process NN results, expand nodes, back up values,
       maybe complete moves + start new games (C++, GIL released)
}
```

Each tick advances every active game by 8 simulations (one
`virtual_loss_batch` worth). A 150-move 13×13 game at 400 sims
therefore needs ~7,500 ticks per game × 256 parallel games, amortized
to ~6,000 ticks per iter of 2048 games. At 25 ticks/sec that's ~40 min
of pure tick work.

**The tick is where all of Zero training's latency lives.** Every
single other design decision — batch size, parallelism, model shape,
number of workers — flows from "how do I make ticks fast."

## Why Zero training is a bad match for modern GPUs

Three independent reasons the workload runs well under 5 % of a
modern accelerator's theoretical peak:

### 1. The model is small

A 15 blocks × 128 channels ResNet at 13×13 has:
- ~**4.5 M parameters**
- ~**50 GFLOPs per forward pass per sample**

At batch 2048, a single forward pass is ~100 TFLOPs of arithmetic.
The RTX 4090's theoretical BF16 peak is ~165 TFLOPs/sec. So one
forward pass "should" take 0.6 seconds of compute.

**Measured per-tick time: 40 ms.** That means each tick completes
~15× faster than a naive "just count FLOPs" model would predict…
which sounds great, until you realize it means we're running at
**1.5 % of peak**. The tensor cores are idle most of the tick.

Why? Small ResNets are **memory-bandwidth bound**, not compute bound.
Each `Conv+BN+ReLU` layer does very little arithmetic per byte of
memory traffic (the feature maps are 256 × 128 × 13 × 13 =
~5.5 M floats per layer = 11 MB in fp16). Reading those feature
maps through the L2 cache dominates wall time; the tensor cores
finish their multiply-accumulates in microseconds and then sit idle
waiting for the next load.

This is a fundamental property of the workload. You can't fix it
with a better GPU — the next-generation tensor core will just finish
even faster and sit idle longer.

### 2. The batch is small relative to modern GPUs

2048 samples per forward pass sounds like a lot, but for modern
hardware it's tiny. An H100's optimal operating point for a
15b×128ch ResNet is closer to 8,192–16,384 samples per batch. At
2048, you don't fill the SM schedulers and you don't hide latency.

Can we raise the batch? Not really — it's bounded by
`num_parallel_games × virtual_loss_batch = 256 × 8 = 2048`. Raising
`num_parallel_games` linearly grows the MCTS tree memory (which was
our Problem 1 in Phase 2), raising `virtual_loss_batch` degrades
MCTS quality by doing more speculative leaf selection. Both hit
hard limits before they saturate an A100, let alone an H100.

### 3. Python/PyTorch overhead per tick is non-negligible

Each tick:
- `torch.from_numpy` for the pinned obs tensor
- `.to(device, non_blocking=True)` H2D copy
- One forward pass
- `torch.softmax` on the policy logits
- D2H copy for (policy, value)
- `torch.cuda.synchronize()` (this is the big one)
- Barrier sync in Python

The Python-side overhead per tick is ~2–5 ms even on a warm process.
On a 40 ms tick, that's 5–12 % of the budget spent on framework
dispatch. On a faster GPU, this fraction grows as the compute shrinks
— an H100 might do the forward in 15 ms but still pays the same
~5 ms Python overhead, so the speedup caps at 40/(15+5) = 2×.

CUDA Graphs + an async inference pipeline would eliminate the
`cuda.synchronize()` per tick and flatten the overhead. The prior
session estimated this at "~1 day of careful rewrite" for a ~20 %
speedup. Real engineering cost, real payoff, but not the kind of
thing that changes the GPU selection question.

## Measurements from this project

Everything above is generic. Here are the concrete numbers we
measured on this project's training runs (RTX 4090, 13×13,
15b × 128ch, 256 parallel games, 400 sims):

| metric | value | notes |
|---|---:|---|
| tick rate (steady state) | 23–25 /s | after torch.compile autotune (first ~1–2 min are slower) |
| per-tick wall time | ~40 ms | 1 / 25 |
| per-forward-pass samples | 2048 | 256 games × vl_batch 8 |
| per-forward-pass FLOPs | ~100 TF | rough estimate |
| implied throughput | ~2.5 TF/s | 100 TF / 0.04 s |
| 4090 theoretical BF16 peak | ~165 TF/s | spec sheet |
| **% of peak** | **~1.5 %** | the bottom line |
| GPU utilization reported by nvidia-smi | 70–85 % (mid-iter) | nvidia-smi counts "any kernel running" not "tensor cores saturated" |
| GPU utilization (iter 0 tail) | 40–50 % | barrier sync waits for straggler games |
| VRAM in use | ~1 GB | 24 GB card is vastly oversized; any 8 GB+ card works |
| Host RSS (peak) | 30–35 GB | MCTS trees + replay buffer + CUDA reserved |

The **70–85 % nvidia-smi utilization number is misleading**. It
reports whether *any* CUDA kernel is running during the sampling
interval, not whether the tensor cores are saturated. A kernel that
reads 11 MB from HBM and does 30 MFLOPs of arithmetic keeps the SM
busy (so nvidia-smi says 100 %) while using 0.1 % of the tensor
throughput. The `implied throughput / peak` ratio above is the honest
metric.

## GPU comparison for this specific workload

The table below shows what to expect *for 15b×128ch / 13×13 Zero
training at batch 2048*, which is the workload we've been tuning.
These are **extrapolations** from 4090 measurements plus public spec
sheets, not benchmarks — but the relative ordering is reliable
because all these cards hit the same 1–3 % peak utilization regime
on this model size.

| GPU | BF16 peak | HBM bandwidth | ~iter 0 time | ~cloud $/hr | train 20 iters cost | notes |
|---|---:|---:|---:|---:|---:|---|
| RTX 3090 (24 GB) | 71 TF | 936 GB/s | ~80 min | $0.2–0.3 | **$5–9** | best $/iter if you already own one |
| **RTX 4090 (24 GB)** | **165 TF** | **1 TB/s** | **~60 min** | **$0.3–0.5** | **$6–10** | **sweet spot: memory bandwidth matters more than TF peak** |
| RTX 5090 (32 GB, 2025) | ~210 TF | 1.8 TB/s | ~45 min (est.) | $0.6–0.9 | $9–14 | if available; Blackwell, not yet cloud-common |
| L40S (48 GB) | 90 TF | 864 GB/s | ~70 min | $1.0–1.5 | $23–35 | datacenter-rated 4090 equivalent; more headroom, not more speed |
| A100 40 GB | 312 TF | 1.6 TB/s | ~45 min | $1.0–1.5 | $15–22 | expensive per min, ~1.3× 4090 wall-clock |
| A100 80 GB | 312 TF | 2 TB/s | ~40 min | $1.5–2.5 | $20–35 | same compute, more HBM — doesn't help at 1 GB working set |
| H100 80 GB | 990 TF | 3 TB/s | ~30 min | $2.5–4.0 | $25–40 | 2× faster than 4090, 8× more expensive |
| H200 141 GB | 990 TF | 4.8 TB/s | ~28 min | $3.5–5.0 | $35–55 | marginal over H100 for this workload |

**The pattern**: cost scales much faster than speedup because no card
past the 4090 can saturate its tensor cores on a small ResNet at
batch 2048. You pay H100-class prices for A100-class wall-clock.

### What the table does NOT say

- This is for **this specific workload**. If you scale up to 19×19
  with 20b×256ch (KataGo-class), the tensor core utilization climbs
  to maybe 10–15 % of peak and the A100/H100 advantage widens — an
  H100 might actually do 3× a 4090 at that scale.
- It says nothing about **inference for a trained model**, where
  latency per single-position evaluation matters more than batch
  throughput.
- **Used markets change the ranking**. A used 3090 at $700 has
  better $/iter than a new 4090, and the compute gap is small enough
  that it's the pragmatic pick if availability is constrained.

## What actually makes training faster (not hardware)

Ordered by effort vs payoff, from a cold start with our current
code:

### 1. Shrink the net 15b → 10b  ★ cheapest win

**One config line** in `model/config.py` (`num_blocks=15 → 10`).
Expected impact: **~33 % faster per iter** (40 ms tick → ~27 ms).
Memory drops proportionally. Quality trade-off: the Phase 1 9×9
preset is 10b×128ch and converged fine; whether 13×13 needs 15b is
unverified — the jump was a guess, not a measurement. If you're
cost-constrained, try 10b first and bump only if convergence is
demonstrably anemic.

### 2. Multi-GPU self-play  ★ best scaling

Self-play is embarrassingly parallel across workers. A 2×4090 box
running two independent `SelfPlayWorker` groups doubles tick
throughput with minimal orchestration overhead (the only shared
state is the network weights, which sync once per iter). Cost: ~1 day
of code to split workers across devices and aggregate harvests,
plus a second card (~$1,600 used or cloud-equivalent). Result:
**2× iter time reduction**, effectively, for the hardware cost of
one 4090.

This is strictly a better use of $1,600 than upgrading the first
4090 to an A100.

### 3. CUDA Graphs + async inference  ★ real engineering, real payoff

The per-tick `torch.cuda.synchronize()` in `parallel_self_play.py`
stalls the GPU between ticks. Capturing the forward pass + softmax +
D2H copies into a CUDA Graph and pipelining them with the next
tick's worker barrier would unlock the last ~20 % of GPU utilization
that the nvidia-smi 84 % number hints at. ~1 day of careful rewrite.
Not infrastructure-level, just tight C++-ish Python.

### 4. Drop `num_simulations` 400 → 300

~25 % faster per iter. Off-spec from the AGZ 13×13 published value of
400, but 300 is still reasonable. The quality cost is real
(fewer per-move rollouts → worse policy targets) but unquantified on
13×13 at this net size. Acceptable for a "good enough" run, not
recommended for chasing maximum strength.

### 5. Drop `num_parallel_games` modestly

Inverse of the Phase 2 Problem 1 analysis — fewer parallel games
means shorter iter 0 tail (fewer straggler games holding the barrier)
but lower per-tick throughput. Only helps if your bottleneck is the
tail, not the steady state. For us, steady state dominates, so this
doesn't help.

## Decision tree — which card for your situation

```
Are you training at KataGo scale (19×19, 20b+ × 256ch+)?
│
├── YES → H100 / A100 80GB pays off. Tensor cores stay fed.
│         Budget ~$4/hr cloud × 100+ hours = $400+ per run.
│
└── NO (13×13 or smaller, <20b net, <256ch):
    │
    ├── Do you already own a 3090 or 4090?
    │   │
    │   ├── YES → Use it. Don't upgrade. The cost/perf of any
    │   │         newer card is worse.
    │   │
    │   └── NO → Continue...
    │
    ├── Do you need this trained in <12 hours?
    │   │
    │   ├── YES → 2× 4090 with multi-GPU self-play (~1 day of
    │   │         code work first). ~30 min/iter × 10-20 iters.
    │   │
    │   └── NO → Continue...
    │
    ├── Is the training a one-off experiment?
    │   │
    │   ├── YES → Rent a cloud 4090 ($0.3–0.5 /hr).
    │   │         20 iters × 1h ≈ $6–10 total.
    │   │
    │   └── NO, recurring → Buy a used 3090 ($600–900).
    │                       Breakeven vs cloud is ~100 hours.
    │
    └── Default: RTX 4090.
```

## Why I didn't suggest renting an A100/H100 for this project

Because for our specific workload the A100 is ~1.3× faster than the
4090 at ~3–5× the cost per hour. That's strictly worse cost/perf.
H100 is ~2× faster at ~8× the cost per hour. Also strictly worse.

The only case where it would make sense is if **wall-clock matters
more than dollars** — e.g., "I need results before a meeting
tomorrow" — but even then the right move is multi-GPU 4090 rather
than upgrading to a single fancier card, because parallelism scales
better than per-card speedup at this scale.

## Summary

- **The 4090 is the right choice for 13×13 / 15b×128ch Zero training.**
  No other card has materially better cost-per-iter on this workload.
- **Cost/perf winner if you can tolerate 33 % slower iters**: used 3090.
- **Cost/perf winner if you want faster iters without paying H100 prices**:
  a second 4090 + multi-GPU self-play code.
- **What doesn't work**: upgrading the GPU. At batch 2048 with a 4.5 M
  parameter ResNet, every card past the 4090 is bottlenecked on memory
  bandwidth and Python overhead, not FLOPs.
- **What does work**: shrinking the net, parallelizing across GPUs,
  CUDA Graphs — all software changes on the same 4090.

Phase 2's 60 min/iter × ~20 iters = ~20 hours wall time is not a
hardware problem. It's the inherent cost of doing Zero training on a
small board with a mid-size network and a consumer GPU. The only way
to meaningfully beat that cost is to change the algorithm (continuous
async self-play, score-based auxiliary heads, smaller net) — not to
buy a bigger card.
