# Phase 2 Training — 13x13 Go AlphaZero

Living document for the 13x13 Phase 2 run. Phase 1 (9x9) wrapped with
the iter 39 checkpoint as the Elo peak of `9x9_run2`; Phase 2 moves to
a bigger board, a bigger net, and more sims per move.

This document is organized into three sections mirroring how the work
actually unfolded:

1. **Training preparation** — the uint8 buffer memory optimization
   plus the other code and config work landed before any 13x13 training
   started.
2. **Fixing OOM problems** — the MCTS tree memory leak, cgroup
   page-cache accounting, and the memory-budget audit that followed.
3. **Fixing the regression issue** — the value-head instability that
   broke every early attempt, including the run1/run2/run3 wrong
   diagnoses, the offline-A/B-driven recipe selection, and the final
   KataGo-style ownership head + derived value architecture that
   actually addressed the root cause.

---

## Environment

- **Host:** RTX 4090 24 GB.
- **cgroup v1 memory limit:** **~87.5 GiB** (re-measured at run3 —
  earlier runs assumed 42 GB then 62 GB based on stale notes; the real
  hard cap on this instance is 93,999,996,928 bytes). Run1/2/3 peaks
  landed around 40 GiB, leaving ~47 GiB headroom.
- **Host MemTotal:** 755 GiB (the container is tiny relative to the
  host).
- **Python 3.11**, **torch 2.4.1+cu124** with inductor / Triton.
- `oom_kill_disable=1` in this container, so hitting the cgroup cap
  would freeze rather than SIGKILL.

## Config at a glance (`CONFIGS[13]` in `model/config.py`)

Values are the current state after all prep + fixes. Anywhere a number
changed during Phase 2, the final value is shown and the change is
documented in the relevant section below.

| knob | value | notes |
|---|---:|---|
| `num_blocks × channels` | 15 × 128 | vs 10 × 128 on 9x9 |
| `num_simulations` | **400** | AlphaGo Zero 13x13 default; cut from 600 for wall-time |
| `dirichlet_alpha` | 0.07 | ≈ 10 / avg_legal_moves (~150 early) |
| `num_parallel_games` | **256** | restored after Problem 1 follow-up |
| `num_games_per_iter` | 2048 | same as 9x9 |
| `buffer_size` | 1,000,000 | vs 500k on 9x9 |
| `max_game_moves` | 250 | vs 150 on 9x9 |
| `temperature_moves` | 40 | ≈ 1/3 of avg 120-move game |
| `temperature_low` | 0.25 | preserves runner-up signal |
| `lr_init` | 0.005 | halved vs 0.01 default; 9x9 run 2 fix |
| `resign_min_move` | **40** | raised 20→40 after Problem 3 |
| `train_steps_per_iter` | **30** | down from 100 — see regression section |
| `value_loss_weight` | **0.0** | run4 derived-value fix, see regression section |
| `ownership_loss_weight` | **2.0** | new run4 knob, KataGo-style |
| `checkpoint_interval` | 1 | per-iter for post-hoc Elo audits |
| `eval_interval` | 1 | every iter, for run3/run4 iter-by-iter visibility |

---

# Section 1 — Training preparation

Work done before any 13x13 training started. Ordered by the order done,
not by importance. Every item was validated — test/run output is either
quoted or committed.

## 1. uint8 replay buffer ✅

**Motivation.** The 13x13 1M-sample buffer as float32 is **12.2 GB**
(vs 1.6 GB for the 9x9 500k buffer). On the 42 GB cgroup host that the
notes at the time assumed, that did not fit alongside ~30 GB of MCTS
trees + torch.compile cache + model state — the savez transient alone
would push RSS over the edge. Every obs plane cell is exactly 0 or 1
(verified in `training/_test_correctness.py`), so storing them as
`uint8` is a **lossless 4× memory cut**.

| | float32 | uint8 |
|---|---:|---:|
| 13x13 obs per sample | 11,492 B | 2,873 B |
| 1M-sample buffer (obs only) | 11.49 GB | **2.87 GB** |
| Total buffer (obs + policy + value) | 12.17 GB | **3.56 GB** |

**Code changes** (~3 lines of substance, in commit `59e0c84`):

`training/replay_buffer.py`:

```python
self.obs = np.zeros((capacity, input_planes, N, N), dtype=np.uint8)
# ...
self.obs[idx] = obs.astype(np.uint8, copy=False)   # cast on push
```

`training/trainer.py::train_step`:

```python
# Cast back to float AFTER the H2D copy. The uint8 bytes cross PCIe
# instead of float32 — free 4× bandwidth win on top of the memory cut.
obs = torch.from_numpy(obs_np).to(self.device).float()
```

The C++ self-play side keeps writing float32 observations; the cast
happens at the Python boundary inside `replay_buffer.push`.

**Why the GPU cast is free.** `Conv2d` reads `float32` regardless.
With float32 storage, the GPU copies 4 B/cell across PCIe and reads
directly. With uint8 storage, the GPU copies 1 B/cell (4× faster) and
`.float()` materializes a `float32` tensor — modern CUDA fuses the
dtype conversion into the load, so it's not new work.

**Validation.** `training/_test_correctness.py` has stage `[1c]`:
dtype check, float32→uint8→float32 round-trip identity, save_to /
load_from round-trip identity, 8-way symmetry correctness match
against `augment_8fold` reference, pass-action invariance, and legacy
float32 save compat (`load_from` casts on load). Full test passes:

```
[1c] uint8 obs round-trip + save/load
  float32→uint8→float32 round-trip exact ✓
  save_to / load_from round-trip exact ✓ (7086 samples)
...
ALL CORRECTNESS CHECKS PASSED
```

**Out of scope for Phase 2 (maybe Phase 3).** *Option B — bitpack obs*
would cut each binary cell to 1 bit instead of 1 byte, shrinking the
buffer another 8× to 0.36 GB. Needs ~30 lines for `np.packbits` index
bookkeeping plus a ~10 µs per-sample unpack. Revisit if Phase 3 (19x19,
6.4 GB even as uint8) actually hits memory pressure.

## 2. Replay buffer persistence re-enabled ✅

Persistence was **disabled** in `train.py` during 9x9 run 2 because
`np.savez` on the float32 buffer materialized a transient that pushed
peak RSS to exactly the 42.83 GiB cap (see `PHASE_ONE_TRAINING.md`
Problem 4). With uint8 storage the 13x13 1M buffer is 3.6 GB and the
savez transient is also ~4× smaller, so `train.py` now calls
`buffer.save_to(buffer_path)` every iteration again. The comment in
`train.py` points at this section for the memory rationale.

Upside beyond just not losing the buffer on crash: crash recovery is
now a one-liner — resume with `--checkpoint
checkpoints/13x13_runN/checkpoint_XXXX.pt` and the buffer reloads from
`latest_buffer.npz` automatically.

## 3. 13x13-specific exploration & resign tuning ✅

These aren't bugs, they're choices that need to be revisited whenever
the board size changes.

- **Dirichlet α = 0.07.** Formula α ≈ 10 / avg_legal_moves. 13x13 has
  ~150 legal moves early game → 0.067.
- **Temperature schedule: `temperature_moves=40`, `temperature_low=0.25`.**
  9x9 run 2's `(30, 0.25)` was tuned for ~85-move games; 13x13 averages
  ~120 moves. 40 keeps the exploration window at ~1/3 of the avg game
  like the tuned 9x9 setup. `temperature_low=0.25` (vs default 0.1)
  preserves runner-up move signal in training targets so the net sees
  a smoother policy, not collapsed one-hot.
- **Resign v2 thresholds initially unchanged from 9x9 run 2.**
  `resign_min_move=20` was 17 % of a 120-move 13x13 game (vs 13 % of
  an 85-move 9x9 game). This was **later raised to 40** under Problem 3
  after the iter-1 early-resign data bias hit.
- **`max_game_moves=250`** per PLAN.md vs 150 on 9x9.

## 4. Lower `lr_init` + per-iter checkpoints + tight eval cadence ✅

Three `TrainingConfig` tweaks on the 13x13 preset, all cheap insurance:

- **`lr_init=0.005`** (halved from the default 0.01). 9x9 run 1 used
  0.01 and exhibited the iter-4→19 BN-drift regression; run 2 at 0.005
  was stable. On a bigger net (15b vs 10b) and a bigger game, halving
  LR is basically free.
- **`checkpoint_interval=1`.** Phase 1 proved you cannot reconstruct
  weight-drift trajectories from sparse snapshots. 60 iters × ~36 MB
  ≈ 2.2 GB of checkpoints — trivial.
- **`eval_interval=5`** (later tightened to `1` for run2/3/4 to get
  iter-by-iter strength visibility while the regression work was
  happening).

## 5. Anchor-buffer mixing — kept optional

The `--anchor-buffer` / `--anchor-frac` CLI path stays in. Phase 2
starts **without** an anchor (fresh Zero training). If 13x13 hits the
same BN-drift pattern as 9x9 run 1, generate an anchor from the last
known-good 13x13 checkpoint and restart with
`--anchor-buffer path/to/anchor_buffer.npz --anchor-frac 0.2`, exactly
like the 9x9 run 2 recovery recipe.

## 6. Fresh-device handoff checklist

```bash
cd Season-5/AlphaZero/engine && python setup.py build_ext --inplace && cd ..
PYTHONPATH=engine python training/_test_correctness.py   # → ALL CORRECTNESS CHECKS PASSED
PYTHONPATH=engine python training/_test_tree_cap.py      # → PASS: tree cap honored
PYTHONPATH=engine python -m training.train --board-size 13 --smoke-test --output-dir checkpoints/13x13_smoke
# then: nohup PYTHONPATH=engine python -m training.train --board-size 13 --iterations 60 ...
```

Diagnostic hooks already wired: `faulthandler` 5-min thread dumps,
`kill -USR1 <pid>` on-demand stacks, grad-clip + NaN-guard skip
(watch `skipped_total`), per-iter `latest_buffer.npz` persistence.

---

# Section 2 — Fixing OOM problems

## Problem 1 — Dryrun OOM (MCTS tree memory unbounded)

Date: **2026-04-14**. Status: **fixed**.

### Symptom

```
python -m training.train --board-size 13 --output-dir checkpoints/13x13_dryrun
```

… ran with the dryrun config (256 parallel games, 600 sims, 15b×128ch,
1M buffer) and hit a cgroup SIGKILL **before iter 0 finished**. The
tail of `logs/13x13_dryrun.log` was a `faulthandler`
`dump_traceback_later(300)` stack dump showing all workers stuck in
`_worker_thread` at `tick_process` — no traceback, no Python error, no
OOM message from Python. Docker crashed.

### Root cause: `MCTSTree::advance()` never frees orphaned subtrees

(The `faulthandler` stack dumps in the log initially looked like a
deadlock in `tick_process` / `tick_select`, but that's the 5-min
periodic dump firing while self-play is just slow — not a hang.)

Re-read of `engine/mcts.h::advance`:

```cpp
void advance(int action) {
    // ... find new_root ...
    if (new_root >= 0) {
        // ... ensure new root has game state ...
        root_idx = new_root;
    } else {
        // Action not in tree — create fresh root
        // (NOTE: still appends to nodes, never clears)
        root_idx = alloc_node();
        // ...
    }
    root_noise_applied = false;
}
```

`advance()` re-roots by changing `root_idx`, but **the `nodes` vector
and `game_pool` deque are never compacted**. Orphaned subtrees from
every prior move of the game sit around until the tree is destructed
at game end. Across a full 13x13 game, the retention is:

| knob | 9x9 run 2 | 13x13 dryrun |
|---|---:|---:|
| moves × sims per game | 85 × 400 ≈ 34k | 250 × 600 ≈ 150k |
| avg children / expansion | ~50 | ~100 |
| nodes at end of game | ~1.7M | ~15M |
| `sizeof(MCTSNode)` | 32 B | 32 B |
| per-tree `nodes` bytes | ~54 MB | ~480 MB |
| per-tree `game_pool` bytes (Game ≈ 4.3 KB on 13x13) | ~80 MB | ~640 MB |
| `num_parallel_games` | 256 | 256 |
| **total across all trees** | **~35 GB** | **~280 GB** |

9x9 run 2 already sat right at the 42 GB cap (that's why Problem 4 in
`PHASE_ONE_TRAINING.md` and commit `1017ea7` even exist). 13x13 dryrun
is ~8× worse and had zero chance of fitting.

The 9x9 smoke test (8 parallel games, 16 sims, 2 workers, 3 iters)
masked this entirely — tree growth is per-game-length, and the smoke
test games never got long enough for it to matter.

### Fix

Three changes, all small:

1. **`engine/worker.h` — per-game tree node cap.** New constant and a
   check in `complete_move()` after `advance(action)`:

   ```cpp
   static constexpr int MAX_TREE_NODES = 200000;  // later raised to 1M
   // ...
   s.game.make_move(action);
   s.tree->advance(action);
   if (s.tree->num_nodes() > MAX_TREE_NODES) {
       s.tree->reset(s.game);   // rebuild from current game state
   }
   ```

   `reset(game)` wipes `nodes` + `game_pool` and seeds a fresh tree
   rooted at `s.game`, so the next move starts with no inherited
   visits but still gets its full `num_sims` budget.

2. **`model/config.py` 13x13 preset — `num_parallel_games=128`**
   (down from the 256 default). Defense in depth: halves tree memory,
   pinned buffers, and worker numpy arrays all at once. GPU batch is
   still 128 × `vl_batch(8)` = 1024 which saturates a 4090 for a
   15b×128ch net, so throughput is essentially unchanged.

3. **`engine/mcts.h` — revert speculative `nodes.reserve(131072)`
   back to `16384`.** A preemptive bump during an earlier audit; with
   the cap in place it's just a startup optimization, and the larger
   value wasted ~1 GB of baseline RSS sitting behind 256
   allocated-but-unused vectors.

Also added two diagnostic helpers:

- **`engine/worker.h::max_tree_nodes()`** — returns the max
  `num_nodes()` across all slots.
- **`MAX_TREE_NODES`** exposed as a Python-visible static property on
  each bound worker class, so tests can read it without hardcoding.

### Validation

`training/_test_tree_cap.py` drives a real `SelfPlayWorker13` at
production `num_sims=600` with fake NN outputs for 30 s and asserts
`max_tree_nodes` never exceeds the cap. Peak observed on 4 parallel
games: 289,803 nodes (below the 320k bound), RSS growth 0.33 GB over
30 s. Passes.

**Why not proper compaction?** Proper compaction (DFS-relabel the
reachable subtree into fresh storage, preserving inherited visits) is
~80 lines of fragile pointer relocation across `nodes`, `game_pool`,
and every `children_start` index — a bug there silently corrupts
search. Reset-on-threshold is ~5 lines and obviously correct, only
losing inherited visits every 3–5 moves on 13x13. Revisit only if
convergence is bottlenecked on reset frequency.

### Follow-up tuning: raise cap 200k → 1M, restore 256 parallel games

Same day, after the first fix was in place. The initial 200k cap was
sized against a conservative 36 GB target, which made the reset fire
**every ~3 moves** — ~33 % of moves started cold, translating to
roughly **~13 % slower convergence**. With 35 GB of real headroom
confirmed available on the host, there's no reason to leave that on
the floor.

Per-tree memory scales as `~75 bytes/node` (nodes array + proportional
`game_pool`). Trading cap sizes against parallelism:

| cap | per-tree | 256 trees | reset every | cold-move frac | convergence hit |
|---:|---:|---:|---:|---:|---:|
| 200k | ~16 MB | ~4 GB | ~3 moves | ~33 % | ~13 % |
| **1,000k** | **~75 MB** | **~19 GB** | **~14 moves** | **~7 %** | **~3 %** |
| 2,000k | ~150 MB | ~38 GB (over) | ~28 moves | ~3.5 % | ~1.5 % |

1M is the sweet spot under 35 GB. At the same time `num_parallel_games`
was restored 128 → 256; the tree cap makes this safe now (peak MCTS
state ~19 GB, not the ~280 GB of the uncapped version). The bigger
GPU batch (256 × `vl_batch(8)` = 2048) amortizes per-tick CPU+sync
overhead and should give an additional ~20–40 % speedup per iter from
better GPU saturation.

**Updated memory budget at peak (during savez transient, 256 parallel
games, 1M cap):**

| component | GB |
|---|---:|
| MCTS state (256 trees × ~75 MB) | 19.0 |
| Replay buffer (uint8) | 3.6 |
| savez transient | 3.6 |
| Model + optimizer + torch.compile | 1.0 |
| CUDA reserved + Python/C++ overhead | 3.0 |
| **Total** | **30.2** |

**Files touched:**

- `engine/worker.h`: `MAX_TREE_NODES` 200000 → **1000000**.
- `model/config.py` 13x13 preset: `num_parallel_games` 128 → **256**.
- Engine rebuilt via `setup.py build_ext --inplace`.

### Speed tuning: drop `num_simulations` 600 → 400

At 600 sims, 256 parallel, 15b×128ch the measured per-tick time was
~347 ms (matching the 3.15× scale from Phase 1's 110 ms on 9x9 10b).
That put iter time at ~52 min and total at ~58 hours for 60 iters.

Dropping to 400 sims — the AlphaGo Zero 13x13 published value, not a
shortcut — cuts iter time ~33 % to ~35 min and total to ~35 hours.

### Dryrun v4 — ground truth

1-iter dryrun with the final tuned config (256 parallel, 400 sims,
1M tree cap): **iter 0 in 41.6 min**, peak cgroup memory **31.0 GB**
(under the 35 GB ceiling with ~4 GB headroom), train loss 5.97
(π=5.15, v=0.82), **150 avg moves/game** (prediction was 120; 19%
miss explains the 41 vs 35 min gap). No NaN, no grad skips, buffer
1094 MB on disk. Pipeline correct end-to-end.

---

## Problem 2 — `memory.usage_in_bytes` double-counts page cache

Date: **2026-04-14**. Status: **fixed by raising the ceiling**.

### What happened

Run1 attempt 1 completed iter 0 cleanly, then iter 0→1 boundary saw
cgroup `memory.usage_in_bytes` climb past v4's 31 GB steady state and
accelerate:

```
check         cgroup   peak     note
T+42 min      33.87 GB 33.87    iter 0 just completed, iter 1 ~2 min in
T+48 min      34.01 GB 34.25    +0.14 GB over 2 min
T+50 min      34.52 GB 34.55    +0.51 GB over 2 min (accelerating)
abort window  -        35.02    crossed the line between checks
```

run1 was aborted at T+50:22 before iter 1's end-of-iter savez
transient could fire — projected peak would have been 36–37 GB.

### Post-kill diagnosis via `memory.stat`

```
cache:         2.57 GB   ← reclaimable page cache
rss:           0.45 GB   ← actual anon memory
inactive_file: 2.38 GB   ← almost all of it is the 1 GB latest_buffer.npz
                           plus compiled inductor artifacts
```

Only **~0.45 GB was real anon memory** on an idle container; **~2.57
GB was page cache** sitting in `inactive_file` — exactly what you'd
expect from a ~1 GB `latest_buffer.npz` (written at iter 0 end and
still hot in the cache) plus the inductor compile artifacts written
during training.

**The 35 GB limit was monitored against `memory.usage_in_bytes`, which
sums anon + cache.** Cgroup v1's usage metric doesn't distinguish
reclaimable cache from unreclaimable rss. The kernel would have
reclaimed the cache under real memory pressure before OOM, so ~3–5 GB
of "memory pressure" on the dashboard was phantom reclaimable cache.
**The real anon-memory peak was probably around 30–31 GB**, comfortably
under any real hardware limit.

### Resolution: raise operational ceiling, then re-measure

A real cgroup audit during run1 recovery showed
`memory.limit_in_bytes = 61.99 GB` with `oom_kill_disable=1` — the
35 GB target was a self-imposed soft limit, not a real constraint.
Decision: **raise operational ceiling 35 GB → 48 GB** (still 14 GB
below the hard cap). Zero code changes, zero AI quality cost.

A further re-measurement during run3 setup showed the current
instance's cgroup v1 limit is actually **~87.5 GiB** (93,999,996,928
bytes). Production peaks on run1/2 were ~40 GiB, leaving ~47 GiB
headroom. The OOM problem is fully retired; everything from run3
onward assumes the 87.5 GiB ceiling.

### Other options that were considered but not taken

- `posix_fadvise(POSIX_FADV_DONTNEED)` after `buffer.save_to` to drop
  page cache (~5 lines) — unnecessary after the ceiling fix
- Monitor `memory.stat: rss` instead of `memory.usage_in_bytes`
- Lower `MAX_TREE_NODES` 1M → 500k (AI quality hit ~3 % → ~7 %)
- Lower `num_parallel_games` 256 → 192

---

# Section 3 — Fixing the regression issue

The second-hardest part of Phase 2 — and the reason run1, run2, and
run3 all had to be aborted before completing. The story here is
**four wrong diagnoses before the right one**. This section is
organized chronologically so each wrong hypothesis and its
falsification are visible.

## Problem 3 — early-resign data bias (run1 iter 1)

Date: **2026-04-14**. Status: **fixed** (`resign_min_move: 20 → 40`),
but this was only the surface-level symptom of a deeper problem.

### Symptom

Iter 1 completed in **12.6 min** vs iter 0's **40.5 min** (3.2× faster)
and showed a counter-intuitive training signal:

```
Iter    1 | Self-play: 2052 games, 84841 pos (41 avg moves), 2.7 games/s, buf=379657, 753.0s
         | Train: loss=5.9476 (pi=5.0447, v=0.9029), lr=0.004997, 2.5s
```

| metric | iter 0 | iter 1 | Δ |
|---|---:|---:|---:|
| self-play time | 2426 s | 753 s | ×3.2 faster |
| avg moves / game | 144 | **41** | **×3.5 shorter** |
| games / sec | 0.84 | 2.70 | — |
| total loss | 5.974 | 5.948 | −0.026 |
| policy loss | 5.145 | 5.045 | −0.100 ✓ |
| **value loss** | **0.828** | **0.903** | **+0.075 ⚠️** |

Value loss going UP while policy loss goes down was the exact
fingerprint of the iter-4→19 regression on 9x9 run 1 — the trainer
was fitting on positions whose value labels got flipped by early
resign.

### Root cause + fix

After 100 training steps on cold self-play data, the value head
moved just enough for some positions to cross the `-0.90` resign
threshold around move ~25 (just above `resign_min_move=20`). The
credible-child cross-check (child with Q > −0.9 **and** ≥ 5% of root
visits) didn't fire often enough to block it because on a cold-ish
policy visits were spread too thin for any single child to hit 5%.
Games cut off at move ~25–40 → buffer filled with early-game
positions → value labels came from the resign decision itself →
doom loop.

**Fix:** `resign_min_move: 20 → 40` on the 13x13 preset. 27 % of a
typical 150-move 13x13 game, matching the 23 % (20/85) ratio that
worked on 9x9 run 2. Config knob only.

Attempt 3 resumed from `checkpoint_0000.pt` (iter 1's regressed
checkpoint deleted, buffer trimmed back to iter 0's 294k positions,
log truncated to iter 0 only).

### Problem 3 was only a band-aid

Attempt 3 resumed from `checkpoint_0000.pt` under the fix above. Iters
0-4 all completed; memory behavior was healthy. **But the first
in-training `evaluate_vs_random` at iter 4 returned 5 %**, and a
post-kill per-checkpoint audit showed strength was **oscillating
wildly** across every iter. The resign-floor fix stopped the
immediate feedback loop but the underlying instability was still
there — which became Problem 4 below.

---

## Problem 4 (wrong diagnosis v1) — "value-head cannibalization"

Status: **initially shipped as `value_loss_weight=2.0`, later refuted**.

### Symptom

Post-kill checkpoint audit
(`training/_eval_checkpoints.py`, 100 games × 100 sims/move vs random
per checkpoint):

| iter | win rate | total | policy | **value** | avg moves/game |
|---:|---:|---:|---:|---:|---:|
| 0 | **20.0 %** | 5.974 | 5.145 | **0.828** | 144 |
| 1 | **2.0 %** ⚠️ | 5.956 | 5.045 | **0.911** | 41 |
| 2 | **18.0 %** | 5.876 | 4.980 | **0.896** | 94 |
| 3 | **0.0 %** ⚠️ | 5.811 | 4.891 | **0.921** | 154 |
| 4 | **8.0 %** | 5.745 | 4.820 | **0.925** | 155 |

Three signals, all pointing the same direction:

1. **Strength oscillates wildly, not monotonically.** Every iter is a
   crapshoot between "barely viable" and "completely broken".
2. **Policy loss drops monotonically** (5.145 → 4.820, −6.3 %), but
   **value loss rises monotonically** (0.828 → 0.925, **+11.7 %**).
3. **Win rate tracks value loss directly**.

### Hypothesis at the time + fix

`loss = policy_loss + value_loss` with equal weight meant policy
(~5 cross-entropy) dominated value (~0.9 MSE) ~80:20 in the gradient.
Theory: the value head was "under-trained," wandering on noisy
targets. Standard "policy cannibalizes value head" framing.

Fix: new `value_loss_weight` config knob, **2.0 for 13x13**, and
`eval_interval=1` for iter-by-iter visibility.

### Run 2 results — fix is partial, not sufficient

Launched 2026-04-14 ~15:02 UTC, aborted after iter 1 completed.
Memory behavior was perfect. **But iter 1 showed the same value-head
drift pattern as run1**, just at smaller magnitude:

| metric | iter 0 | iter 1 | Δ | run1 Δ |
|---|---:|---:|---:|---:|
| avg moves/game | 153 | 86 | −67 | −103 (144→41) |
| policy_loss | 5.1346 | 5.0516 | −0.083 ✓ | −0.100 |
| **value_loss** | **0.8723** | **0.9207** | **+0.048 ⚠️** | **+0.083 ⚠️** |
| eval win rate | 1.0 % | 0.0 % | −1 pp | −18 pp |

**`value_loss_weight=2.0` dampened the drift by ~40 %** (+0.048 vs
run1's +0.083) but **did not reverse the direction**. The
cannibalization hypothesis was **partially validated** — a 2× weight
helped, but not enough.

---

## Run 3 — offline-A/B-driven recipe selection

Picked up the next session with a stricter approach: don't pick a
lever blind, **measure every candidate against a real cold buffer
first** and only commit a recipe with controlled-experiment evidence.

### Step 1 — code audit (no bug found)

Audited every path that touches the value target end-to-end: sign
convention in `play_one_game` and C++ `finish_game`, observation
encoding in `Game::to_observation`, replay buffer 8-fold augmentation,
train/eval mode plumbing through `torch.compile`. Verdict: no hidden
bug. The failure is in the optimization dynamics.

### Step 2 — offline A/B harness

`training/_phase2_offline_ab.py`, two phases:

1. **`gen-buffer`** — seeded RNGs, real 13×13 net at random init,
   256 cold games at production settings → saved cold weights +
   28,253 positions, balanced labels.
2. **`run-ab`** — each recipe reloads the **same** cold weights and
   trains against the **same** buffer. Records held-out value MSE
   (5k-sample slice never seen in training), `below_resign_frac`
   (fraction of held-out `v < −0.9`, proxy for triggering the iter-1
   resign loop), value-head saturation rate, first-5 vs last-5
   training loss trajectory.

The offline test is single-iter only — it cannot capture iter-over-
iter buffer evolution — but it IS faithful in per-position exposure
(~8.5% buffer coverage per iter).

### Step 3 — recipes tested and results

Tested 14 recipes (10 MSE variants + 4 BCE/WDL-style). Held-out MSE
baseline (cold weights, no training) is **1.0035** — the "predict
zero against ±1 labels" floor. **Δv > 0 means the recipe made things
worse on unseen positions.**

```
recipe         pre_v  post_v       Δv   tr1st   trlst   sat  resign%
R1-current    1.0035  1.6342  +0.6307  0.9368  0.0476  0.25    21.0%   ← prod
R2-steps30    1.0035  1.1122  +0.1087  0.9368  0.2693  0.00     0.0%   ← BEST
R3-vlw1       1.0035  1.5408  +0.5373  0.9466  0.0533  0.25    19.3%
R4-huber      1.0035  1.6588  +0.6553  0.4709  0.0269  0.35    15.6%
R5-lowLR      1.0035  1.3616  +0.3580  0.9809  0.1313  0.00     2.8%
R6-vlw0.5     1.0035  1.6086  +0.6051  0.9619  0.0701  0.24    10.1%
R7-bundled    1.0035  1.2919  +0.2884  0.4864  0.2879  0.00     0.0%
R8-vlw4       1.0035  1.6614  +0.6578  0.9332  0.0623  0.28    17.8%
R9-warmup     1.0035  1.5810  +0.5775  0.9614  0.0656  0.31    15.9%
R10-valOnly   1.0035  1.7150  +0.7115  0.9319  0.0436  0.29    25.8%
B1-bce100     1.0035  1.9566  +0.9530  0.6814  0.0616  0.78    33.1%
B2-bce30      1.0035  1.5054  +0.5018  0.6814  0.4290  0.00     0.2%
B3-bce-low    1.0035  1.5809  +0.5773  0.6888  0.3491  0.06    17.6%
B4-bce-vlw1   1.0035  1.7603  +0.7568  0.6855  0.0910  0.51    40.1%
```

### Step 4 — what the data says

**Every recipe has Δv > 0.** Three distinct fingerprints:

1. **Catastrophic overfit** (R1/R3/R4/R8/R9/R10): training loss
   collapses to <0.1 (95% drop), held-out jumps to ~1.6, saturation
   25–35%. **R10 (value-only, zero policy gradient) is in this
   group** — the WORST recipe — which kills the original
   "policy is cannibalizing the value head" framing entirely. The
   value head fails even when nothing else touches the trunk.
2. **Under-train** (R2, R7): training loss only drops to ~0.27,
   held-out +0.11/+0.29, saturation 0%, `below_resign_frac` 0%. The
   value head barely moves.
3. **WDL/BCE is WORSE than MSE** (B1–B4): at 100 steps BCE shows
   saturation 78 % (vs MSE 25 %) and held-out 1.96 (vs 1.63). MSE
   gradient caps at 2 once `|v − target| = 1`; BCE-with-logits keeps
   pushing toward extremes. On noisy ±1 labels, BCE memorizes harder.
   **WDL escalation is ruled out.**

### Step 5 — failure mode reframed

The "value-head cannibalization" story from Problem 4 is **refuted**:
R10 (value-only) is the worst, not the best. Huber and warmup don't
help. What does help (weakly) is reducing *exposure* (fewer steps).

The real failure mode is **classic overfit to noisy labels on a small
cold buffer**. Cold-game outcomes are essentially random (50.1% black
wins). The 4.5M-param network memorizes 28k random labels in <1
epoch and produces confident wrong predictions on unseen positions.

### Step 6 — Run 3 recipe and launch

Picked **R2** as the production candidate with the smallest possible
diff: `train_steps_per_iter: 100 → 30` on the 13x13 preset, plus a
`set_all_seeds()` call in `train.py` for run-to-run reproducibility
(run1 vs run2 had a 19pp eval gap at iter 0 partly because RNGs were
unseeded).

### Run 3 result — R2 falsified at iter 1

Aborted at iter 1 (~70 min wall time after launch). Iter 0 looked
normal but iter 1 immediately tripped two abort gates:

| metric | iter 0 | iter 1 | Δ | gate |
|---|---:|---:|---:|---|
| avg moves/game | 146 | **56** | −90 | ❌ < 80 |
| v_loss | 0.8821 | **0.9653** | **+0.083** | ❌ rising |
| pi_loss | 5.1909 | 5.0764 | −0.115 | ✓ |
| eval vs random | 15 % | **5 %** | −10 pp | ❌ |
| self-play wall | 2742 s | 814 s | −1928 s | confirms collapse |

The iter-0 → iter-1 v_loss drift is **+0.083, identical in magnitude
to run1's** +0.083 on the same pair of iters. **R2 decisively
falsified:** iter-over-iter bootstrapping with under-training does
NOT escape cold-start at this scale.

Run3 checkpoints + buffer preserved for post-mortem in
`checkpoints/13x13_run3/`.

### What the run3 work definitively ruled out

- Loss-formulation instability (WDL/BCE is worse than MSE)
- Policy-trunk cannibalization (R10 value-only is the worst)
- Huber loss (R4 matches R1)
- LR alone (R5 reduces overfit but doesn't fix it)
- Warmup (R9 is no better than R6)
- `train_steps_per_iter` reduction alone (R2 fails live)

---

## Run 4 — KataGo-style ownership head + derived value

Status: **launched 2026-04-15**, iter 0 in progress at time of writing.
This is the architectural fix that addresses the actual root cause.

### Diagnosis that led here

The run3 A/B pinned the memorization to the **value head MLP** (FC
169→256→1, ~44k params sitting on top of 28k cold training positions
— ~1.6× overparameterized, trivial to memorize). Outcome-only
supervision (one scalar per 150-move game) is too sparse to train a
4.5M-param net at this scale.

### First attempt: ownership head as auxiliary on top of existing value MLP

Implemented the KataGo-style ownership head as a pure addition:

- **C++** `Board::compute_ownership()` — Tromp-Taylor flood-fill
  writing per-cell int8 labels (+1 BLACK / −1 WHITE / 0 dame).
- **Worker** `finish_game` — computes ownership once at game end and
  copies it into each MoveRecord with a current-player sign flip.
- **Buffer** — new `ownership` tensor, 8-fold sym-aware augmentation,
  save/load round-trip.
- **Network** — `ownership_conv = Conv1×1(ch→1)` → `(B, N, N)` logits;
  trained with BCE-with-logits against `(own+1)/2`.
- **Trainer** — new `ownership_loss_weight` knob; total loss =
  `policy_loss + vlw·value_loss + ow_weight·ownership_loss`.

**First-pass offline A/B (10 recipes varying `ow_weight`, `vlw`,
steps, LR):**

```
recipe          pre_v  post_v       Δv  tro_1st  tro_lst   sat  resign%
OW0-baseline   1.0035  1.6342  +0.6307   0.0000   0.0000  0.25    21.0%
OW0.5-100      1.0035  1.5538  +0.5502   1.0132   0.5625  0.30    25.2%
OW1.0-100      1.0035  1.5819  +0.5784   0.9687   0.5425  0.29    20.0%
OW1.5-100      1.0035  1.6103  +0.6068   0.9613   0.5310  0.29    25.3%
OW1.5-30       1.0035  1.1990  +0.1955   0.9613   0.5781  0.00     0.0%
OW1.5-200      1.0035  1.4980  +0.4945   0.9619   0.5345  0.29    24.6%
OW1.5-vlw1     1.0035  1.5260  +0.5225   0.9613   0.5310  0.28    11.9%
OW1.5-lowLR    1.0035  1.3300  +0.3264   0.9614   0.5920  0.00     2.3%
OW2.0-100      1.0035  1.5791  +0.5755   0.9616   0.5248  0.32    17.0%
OW3.0-100      1.0035  1.6021  +0.5986   0.9632   0.5155  0.31    15.8%
```

- **Ownership head IS learning** — training ownership loss drops
  0.96 → 0.54 in every recipe, held-out drops 0.69 → 0.55.
- **But every recipe still has post_v_mse > 1.0.** Ownership trains
  the trunk fine; it doesn't stop the value MLP from memorizing
  independently.

A second pass focused on `steps=30 + low vlw + ownership` showed the
same pattern. The only recipe that ever stayed at the cold floor was
**A6 (`vlw=0`, ow=2.0)** — because with `vlw=0` the value MLP never
updated at all. Not a fix, but a definitive pin: the memorization
is in the value MLP, not the trunk or the ownership head.

### Second attempt: derive value from ownership (no independent value MLP)

The decisive architectural change. **Replace the value MLP entirely**
with a deterministic readout of the ownership head:

```python
# Old value head (~44k params, memorized 28k cold labels in <1 epoch):
value = tanh(fc2(relu(fc1(bn(conv(trunk))))))

# New value head (2 learnable scalars, can't memorize anything):
own_logits = ownership_conv(trunk)              # (B, N, N)
own_probs  = sigmoid(own_logits)                # (B, N, N)
own_signed = 2 * own_probs - 1                  # (B, N, N) in [-1, 1]
margin     = own_signed.sum((1, 2))             # (B,) expected net cells
value      = tanh(value_scale * margin + value_bias)
# value_scale, value_bias are nn.Parameter(), init (0.02, 0.0)
```

**Properties of the new architecture:**

- The "value head" has exactly **two learnable scalars** (`value_scale`,
  `value_bias`). Cannot memorize 28k labels by any mechanism.
- Value is **mathematically bound** to ownership. The network can only
  produce a value prediction that's consistent with its ownership
  prediction, and ownership has 169 dense per-cell real-territory
  labels per position.
- Value loss gradients still flow back into the ownership head and
  the trunk (via sigmoid, sum, tanh), so value supervision is not
  lost — it's rerouted through a structure that can't overfit.
- This is the KataGo "value-from-score" idea in its simplest form.
  KataGo has both a direct value head AND a score-derived one and
  combines them; we have only the score-derived one.

### Offline A/B on the derived-value architecture

Reran the same 10 recipes from the second pass against the new
architecture:

```
recipe               post_v       Δv   sat   resign%
A0-OW0-baseline      1.4878  +0.4660  0.61    11.7%   ← ow=0, vlw=2
A1-OW1.5-30-anchor   1.9776  +0.9558  0.52    43.1%   ← ow=1.5, vlw=2 (WORST)
A2-30-vlw0.5-ow1.5   1.4642  +0.4424  0.14    25.1%
A3-30-vlw0.5-ow2.0   1.7370  +0.7152  0.60    47.2%
A5-30-vlw0.25-ow2.0  1.5452  +0.5234  0.29     0.0%
A6-30-vlw0-ow2.0     0.9631  −0.0586  0.00     0.0%  ★ below floor
A7-50-vlw1-ow1.5     1.8052  +0.7835  0.71    34.5%
A8-50-vlw0.5-ow2.0   1.8570  +0.8352  0.77    42.8%
A9-30-vlw0.5-lowLR   1.3698  +0.3480  0.01     6.1%
```

**A6 broke the floor** (post_v_mse = **0.9631** < 1.02, **Δv = −0.059**,
**resign% = 0%**, **saturation = 0%**). This was the **first recipe
across run1/2/3/4 offline A/Bs to ever produce negative Δv on
held-out data**.

**Critical subtlety**: the pattern in this table is the opposite of
the pattern with the old MLP architecture. Now **any nonzero value
loss weight HURTS**: A1 (`vlw=2`, ow=1.5) has the worst result (Δv =
+0.96) because with the derived value, value-loss gradients flow back
through sigmoid/sum into the ownership head — directly **hijacking it
into making per-cell predictions that sum to noisy game outcomes**,
which conflicts with the ownership loss's "predict real territory"
supervision. `vlw=0` removes the conflict entirely.

With `vlw=0`:
- The ownership loss alone trains the ownership head and the trunk
  on dense per-cell real-game labels.
- `value_scale=0.02` and `value_bias=0` never update.
- The derived value is a **frozen deterministic readout** of a
  well-trained ownership head.
- For 13x13, `scale=0.02` is a reasonable calibration: `tanh(0.02 ×
  ±50)` ≈ ±0.76, and ±100 approaches saturation — fine dynamic range.

### Run 4 recipe

Three config changes from the run3 state:

```python
# model/config.py — 13x13 preset
value_loss_weight=0.0,          # was 2.0 — removes the hijack of ownership head
train_steps_per_iter=30,        # was 100 — prevents per-iter overfit
ownership_loss_weight=2.0,      # new; KataGo-range supervision weight
```

Plus the derived-value architecture in `model/network.py`: the old
`value_conv`/`value_bn`/`value_fc1`/`value_fc2` modules are **removed**
entirely; replaced by `value_scale` and `value_bias` as
`nn.Parameter()`.

### Smoke test (tiny 2b×32ch, 3 iters)

```
Iter 0 | loss=6.5736 (pi=5.2012, v=1.2104, own=0.6862) | Eval 75%
Iter 1 | loss=6.3388 (pi=5.1813, v=0.9308, own=0.5788) | Eval 75%
Iter 2 | loss=6.2620 (pi=5.1567, v=1.0404, own=0.5526) | Eval 65%
```

Arithmetic `5.2012 + 2·0.6862 = 6.5736` ✓ (`vlw=0`, `ow=2` both
active). Ownership loss drops monotonically past the 0.693 entropy
floor. Derived value reported (not in total): 1.21 → 0.93 → 1.04.

### Hard go/no-go gate at iter 5 (same as run3)

**GREEN** (let the full 60 iters run unattended ~38h):
- `value_loss` flat-or-falling across iters 0→5 (the reported
  derived-value MSE, even though it's not in the total loss)
- `eval_vs_random` non-regressing AND ≥ ~10–15 % by iter 5
- `avg_moves/game` stays in the 100–160 range (no resign collapse)
- cgroup memory stable, well under 60 GB

**RED** (kill, re-evaluate):
- `value_loss` rising 2 iters in a row
- `eval_vs_random` stuck near 0 % for 3 consecutive iters past iter 2
- `avg_moves/game` < 80
- any crash / OOM

### Run 4 — first attempt died silently mid-eval

Launched at 04:56 UTC. Iter 0 self-play ran cleanly for ~52 min
(2050 games, 355,569 positions, 173 avg moves — longer than run3
because the derived value is flat at cold init so resign never
fires). `buffer.save_to()` successfully wrote `latest_buffer.npz`
(1.32 GB). Then the process **vanished silently around 05:51** with:

- no Python traceback
- no `TRAINING CRASHED` from the `BaseException` handler
- no core dump
- no cgroup OOM (`memory.failcnt=0`, `oom_kill=0`)
- the iter 0 summary line was never flushed to the log (Python
  block-buffers `nohup`-redirected stdout by default)

Most likely cause: **external kill on a shared host** (host load
average was 6.21 at the time — other processes contending for
memory, and we were at ~30 GB RSS). No cgroup mechanism prevents
the kernel OOM killer targeting a high-RSS container when the
global system is under pressure. The `oom_kill_disable=1` flag only
applies to the cgroup's own OOM path.

Unable to prove it was an external kill vs a subtle bug with the
available logs, so I made the training loop **more resilient and
observable** before relaunching. Three train.py changes (commit
`ac7751d`):

1. **`PYTHONUNBUFFERED=1` launch env + `flush=True` on every
   iter-loop print.** Guarantees output hits disk immediately; a
   crash can't bury the iter summary in a stdout buffer.
2. **Reorder loop: checkpoint BEFORE eval.** Run4 died during eval
   AFTER training completed, losing the trained weights for no
   reason. Save first so a crash mid-eval still preserves the
   checkpoint for post-hoc audit / resume.
3. **Drop eval `num_games` 100 → 50.** Halves eval wall time
   (~5 min → ~2.5 min) to shrink the window where an external kill
   can hit. Statistical cost tiny: binomial σ at p=0.15 is ±5pp on
   50 games vs ±3.6pp on 100.

None of these touch what the network learns; they only harden the
training loop against the specific failure mode run4 hit.

### Run 4b — relaunched, iter 0 broke every prior baseline

Launched at 06:13 UTC. Iter 0 completed cleanly at **~54 min**
total. **This is the first iter 0 across run1/2/3/4b where all
four signals land above (or at) the best previously-seen value:**

```
Iter 0 | Self-play: 2050 games, 355569 pos (173 avg moves), 3115.8s
       | Train: loss=6.3737 (pi=5.1689, v=0.9942, own=0.6024), 0.9s
       | Checkpoint saved: checkpoints/13x13_run4b/checkpoint_0000.pt
       | Eval vs random: 60.0% (50 games, 129.7s)
       | Total: 3254.1s
```

| | run1 | run2 | run3 | **run4b** |
|---|---:|---:|---:|---:|
| iter 0 eval vs random | 20 % | 1 % | 15 % | **60 %** |
| avg moves/game | 144 | 153 | 146 | 173 |
| self-play time | 40.5 min | 42.5 min | 45.7 min | 52 min |
| value loss (iter 0) | 0.828 | 0.872 | 0.882 | 0.9942 (derived) |
| ownership loss | — | — | — | **0.6024** |

**The 60 % eval number is 3× the best previous iter-0 baseline.** 50
games has a binomial 95 % CI of ±14 pp, so the true win rate could be
as low as ~46 % — still higher than any prior run's iter 0.

Arithmetic check on the train line: `5.1689 + 0·0.9942 + 2.0·0.6024
= 6.3737` ✓ (confirms `vlw=0`, `ow=2.0` active). The `v_loss=0.994`
sitting at the cold floor confirms the 2-learnable-scalar derived
head is NOT memorizing — exactly what the architecture was designed
to prevent. The real work is happening in `own_loss=0.6024`, which
is **already below the 0.693 BCE entropy floor** after just 30 SGD
steps: the ownership head has learned meaningful per-cell features
on real cold-game data in a single iter.

This is the first iter 0 across Phase 2 that clears the "does the
architecture actually work on production data" bar. Iter 1 (in
progress at time of writing) is the decisive test of whether
iter-over-iter bootstrapping compounds — i.e., whether `avg_moves`
stays ≥ 80 and `eval` stays above the cold baseline instead of
collapsing like it did in run1/2/3 iter 1.

### Things still untested (post-run4b iter 0)

- **Score-distribution head** (KataGo's secondary innovation; only
  worth adding on top of ownership if run4 plateaus early)
- **9×9 → 13×13 weight transfer / supervised pretraining** (~1 day of
  architecture-transfer code, worth doing only if run4 fails outright)
- **Learnable value_scale/value_bias**: currently `vlw=0` means they
  never update; could try training them with a tiny weight (e.g.
  `vlw=0.05`) once ownership is well-trained, for calibration

---

# Section 4 — Run 4c/4d: resilience, pass-collapse, and the silent strength regression

Three separate problems landed in one session, each surfacing only once the previous one was solved.

## Problem A — silent-kill survivability (run4c)

**Symptom.** Run4b iter 2+ self-play kept dying silently after ~5 min (no Python traceback, no OOM, no core). Diagnosed as probable external SIGKILL on a shared container (load avg 5–6, we were a ~30 GB RSS top-tenant).

**Fixes landed as `run4c`.** Three changes that together survived 5+ iters:

1. **`num_parallel_games` 256 → 128** (`model/config.py`). Halves MCTS tree state (~19 GB → ~10 GB peak). Per `HARDWARE_NOTES.md` the 4090 already saturates at 128 parallel, so GPU throughput is effectively unchanged; iter wall-time stays similar. This alone moved us out of the OOM-kill top-candidate range.
2. **Intra-iter buffer persistence** (`parallel_self_play.py::run_games` accepts a `save_callback` + `save_interval_s`; default 120 s). `train.py` passes `buffer.save_to(buffer_path)`. If the process dies mid-iter, the already-harvested self-play positions survive — on restart the buffer reloads with the partial harvest and the next attempt picks up where the previous one died.
3. **Shell watchdog** (`run_resilient.sh`). Relaunches `train.py --checkpoint <latest>` on non-zero exit, with MIN_RUNTIME_S crash-loop guard (60 s) and MAX_RETRIES cap. Forwards SIGINT/SIGTERM as user stops (rc 130/143).

**Result.** Attempt 1 of run4c cleared 4 iters cleanly before hitting an **actual** segfault (rc=139, SIGSEGV + core dumped) at ~75 min of continuous runtime during iter 3 self-play. The heartbeat dump caught corrupt memory (`threading.py line 9651904`, a nonsense line number) — the corruption was already there when the dump fired, not caused by it. Watchdog auto-restarted; attempt 2 resumed from `checkpoint_0002.pt` with the partial iter 3 buffer preserved and iter 3 completed on the rerun. Training made forward progress despite the crash. Core dumps go through an apport pipe on this host, so no local file was available for post-mortem symbol analysis.

**Open question.** Root cause of the segfault not identified. Candidates: MCTS tree-reset use-after-free (`reset()` called tens of thousands of times per iter with max-cap trees), torch inductor compile-worker subprocess drift, `faulthandler.dump_traceback_later` interrupting C++ in a non-signal-safe state, or long-run CUDA/pinned allocator drift. Mitigation is the watchdog; diagnosis deferred.

## Problem B — pass-collapse (v1 fix)

**Symptom.** Run4c iter 2 self-play produced **69 avg moves/game** (vs run4b iter 0/1 at 173/182). Games were ending via consecutive passes before move 80, despite the resign floor being at 80. The handover's earlier attribution to "resign firing early" was wrong — `resign_min_move=80` is a lower bound on *resignation*, not on *game length*. A 13×13 game can naturally terminate via `consecutive_passes >= 2` (`go.h:470`) at any move.

**Root cause.** The iter 0/1 training buffer contained ~2× pass moves per game (the final two moves of every pass-pass ending), which is ~1–2 % of all recorded positions. Combined with the ownership head learning "territory is settled" after ~60 SGD steps, the policy overgeneralized "pass when territory looks settled" to early-game positions. Once both players at a self-play game landed on pass in the same window, the game ended short. Iter-2 self-play collapsed to ~69 moves/game; the policy was training on the collapsed distribution.

**Fix (v1 — partial).** New `pass_min_move` config knob in `SelfPlayConfig` (and `TrainingConfig`, default 0 for 9×9, set to **60** on the 13×13 preset). In `worker.h::complete_move`, sample the played action from a local copy of `rec.policy` with `sample_policy[ACTIONS-1] = 0` when `move_num < pass_min_move`. Policy target in `rec.policy` left untouched (this is the part that was wrong — see Problem C).

**Result.** Iter 2 rerun jumped to **130 avg moves/game**, iter 3 to 190, iter 4 to 208 (*above* run4b iter 0/1). Games generating longer by the iter — apparent positive feedback loop, architecture seemingly working.

## Problem C — silent strength regression (the real bug)

**Symptom.** With iter 2–5 completed, we ran `training/_eval_checkpoints.py`-style per-iter eval-vs-random (50 games, 100 sims/move, fresh subprocess each to avoid the multi-load hang the handover documented). Results:

| iter | avg_moves | vs-random | pi_loss | own_loss |
|---:|---:|---:|---:|---:|
| 0 | 173 | **72.0 %** | 5.1689 | 0.6024 |
| 1 | 182 | **3.3 %** ⚠️ | 5.0762 | 0.5163 |
| 2 | 130 | **10.0 %** | 5.0392 | 0.5109 |
| 3 | 190 | **22.0 %** | 4.9278 | 0.4884 |
| 4 | 208 | **32.0 %** | 4.8866 | 0.4771 |
| 5 | 227 | **0.0 %** ⚠️ | 4.8399 | 0.4633 |

Iter 1 dropped 72 % → 3.3 % after only 30 more SGD steps on iter-1 self-play data. Iter 2–4 climbed back ~10 pp/iter but never reached iter 0's level. Iter 5 **completely collapsed** to 0/50 wins vs random.

**Critical discovery.** Training losses (`pi`, `own`, total) dropped **monotonically across every iter**, including iter 1 and iter 5. The self-play `avg_moves` also climbed monotonically 130 → 190 → 208 → 227. None of the in-loop signals caught the regression. `run4b` never measured iter 1's eval because in-loop eval was disabled after iter 0 — we were flying blind on real strength.

**Root cause.** Forward-pass diagnostic on iter 5 `checkpoint_0005.pt` against an empty 13×13 board:

```
value_scale=0.0200 (healthy, unchanged)
policy argmax: 169 (PASS)
policy top5: [169, 99, 33, 54, 70]
logits range [-0.29, 2.94] with pass=2.94 peak
```

Running the same diagnostic across iters 0–5 on an empty board:

| iter | pass_logit | pass_prob | pass_is_argmax |
|---:|---:|---:|:---:|
| 0 | +0.61 | 1.09 % | yes |
| 1 | +1.11 | 1.77 % | yes |
| 2 | +1.12 | 1.80 % | yes |
| 3 | +1.34 | 2.23 % | yes |
| 4 | +1.80 | 3.50 % | yes |
| 5 | +2.94 | **10.27 %** | yes |

Pass is argmax on an empty board for **every** checkpoint, and the pass prior grows monotonically. At iter 5 the prior crossed a threshold where MCTS 100-sim rollouts started committing visits to pass — at move 1 on an empty board, where the derived value sees "territory is undetermined" and every child Q ≈ 0, MCTS visit allocation is driven by the policy prior and pass wins the argmax. In `evaluate_vs_random` (which has no pass-gate, unlike self-play), the net played pass → opponent played random → net passed again → two consecutive passes → `end_game()` scored an empty board → komi goes to white → 0/50 losses. Self-play never exhibited this because the C++ pass-gate still blocked the played action below move 60; the *policy prior* was broken but the *played move* was not.

**Why the v1 fix missed it.** v1 gated pass only in the *sampled action*, not in the *stored policy target*. MCTS still explored pass during search, visits still accumulated on the pass slot, and `s.tree->get_policy()` wrote those visits into `rec.policy` which got pushed to the replay buffer unchanged. Over many training iters the policy head slowly learned "pass has non-trivial probability in the opening" from those targets. Eventually the learned prior crossed the MCTS-visit-allocation threshold.

**Fix (v2).** Two-line change in `worker.h::complete_move`: zero `rec.policy[ACTIONS-1]` **before** `push_back`, and renormalize the non-pass probabilities to sum to 1, whenever `move_num < pass_min_move`. The sampling code was simplified to sample directly from `rec.policy` since pass is already zeroed there for gated positions. The net will now see a stored target with strictly zero pass probability in early-game positions and can't learn to prefer pass in the opening.

## Run4d — rollback and relaunch

Decided to restart from `checkpoint_0000.pt` with a fresh buffer:

- **Keep** iter 0's weights (72 % vs random, the only validated-strong checkpoint across the whole phase) — pass logit +0.61 is small enough for the fixed training to drive down over a few iters.
- **Discard** `checkpoint_0001..5.pt` and the iter 0-5 replay buffer (all contained polluted targets and/or polluted priors). Backed up to `checkpoints/13x13_run4b_contaminated_iter1-5/` for post-mortem.
- **Relaunch** via `run_resilient.sh` with `--games-per-iter 1024` (half of the 2048 baseline — per-iter wall-time drops from ~55 min to ~30 min, each crash costs proportionally less). Pass-floor v2 fix is in the engine the training loads.

Success criterion at iter 1: **eval ≥ 60 % vs random** (keeps or improves iter 0's baseline). If iter 1 collapses again, the bug is deeper than pass-in-target and we need to reconsider the derived-value architecture.

## Takeaways for the next session

1. **Eval-vs-random per iter is mandatory for this architecture.** The previous "disable in-loop eval because it hangs" decision saved wall time but hid a catastrophic silent regression for 5 iters. Run the post-hoc eval loop (`run_eval_loop.sh`) in a separate process — one fresh subprocess per checkpoint avoids the accumulation hang.
2. **Training losses are not predictive of strength for this architecture.** `pi_loss` and `own_loss` dropped monotonically while vs-random went 72 % → 3.3 % → ... → 0 %. Do not use loss trajectories as a training health check on the derived-value net.
3. **`num_games_per_iter` directly trades iter wall-time for per-iter data volume.** Cutting 2048 → 1024 halved iter time and roughly halved the crash-recovery cost without visible quality impact so far (training samples from the full 1M buffer regardless).
4. **Intra-iter buffer persistence + shell watchdog + shrunken MCTS footprint survived the run4b silent-kill regime.** Even when a real SIGSEGV hit, the watchdog resumed from the last checkpoint with partial iter data preserved.
5. **Pass gating must cover the stored target, not just the sampled action.** Any auxiliary constraint that blocks an action in self-play must also remove it from the policy training target for the same positions, otherwise the net learns the forbidden action is "valid" via the MCTS visit distribution even when it can't actually be played.
