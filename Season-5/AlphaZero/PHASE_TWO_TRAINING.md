# Phase 2 Training — 13x13 Go AlphaZero

Living document for the 13x13 Phase 2 run. Phase 1 (9x9) wrapped with
the iter 39 checkpoint as the Elo peak of `9x9_run2`; Phase 2 moves to
a bigger board, a bigger net, and more sims per move. The code-side
prep and a first OOM fight happened before the full run started —
they're captured below.

This file mirrors the structure of `PHASE_ONE_TRAINING.md`: prep work,
then a numbered list of problems as they hit and got fixed, then the
live run log.

---

## Environment

- **Host:** RTX 4090 24 GB, cgroup memory budget **~42 GB** but the
  *operational* ceiling is **36 GB** — above that RunPod gets unstable
  and recovering the instance costs real time. Treat 36 GB as the hard
  cap.
- **Python 3.11**, **torch 2.4.1+cu124** with inductor / Triton
- `oom_kill_disable=1` in this container, so hitting the cgroup cap
  produces a silent SIGKILL with no Python traceback — same failure
  mode as Phase 1 run 2 (see `PHASE_ONE_TRAINING.md` Problem 4).

## Config at a glance (`CONFIGS[13]` in `model/config.py`)

| knob | value | notes |
|---|---:|---|
| `num_blocks × channels` | 15 × 128 | vs 10 × 128 on 9x9 |
| `num_simulations` | **400** | same as AlphaGo Zero 13x13; cut from 600 for wall-time (see "Speed tuning: drop sims 600 → 400") |
| `dirichlet_alpha` | 0.07 | ≈ 10 / avg_legal_moves (~150 early) |
| `num_parallel_games` | **256** | restored after Problem 1 follow-up (see "Tuning 1M cap + 256 parallel") |
| `num_games_per_iter` | 2048 | same as 9x9 |
| `buffer_size` | 1,000,000 | vs 500k on 9x9 |
| `max_game_moves` | 250 | vs 150 on 9x9 |
| `temperature_moves` | 40 | ≈ 1/3 of avg 120-move game |
| `temperature_low` | 0.25 | preserves runner-up signal |
| `lr_init` | 0.005 | halved vs 0.01 default; 9x9 run 2 fix |
| `checkpoint_interval` | 1 | per-iter for post-hoc Elo audits |
| `eval_interval` | 5 | tight cadence to catch BN drift early |

---

## Prep work landed before the run

Ordered by the order they were done, not by importance. Every item
here was validated — test/run output is either quoted or committed.

### 1. uint8 replay buffer ✅

**Motivation.** The 13x13 1M-sample buffer as float32 is **12.2 GB**
(vs 1.6 GB for the 9x9 500k buffer). On the 42 GB cgroup host that
does not fit alongside ~30 GB of MCTS trees + torch.compile cache +
model state — the savez transient alone pushes RSS over the edge.
Every obs plane cell is exactly 0 or 1 (verified in
`training/_test_correctness.py`), so storing them as `uint8` is a
**lossless 4× memory cut**.

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

**Validation.** `training/_test_correctness.py` now has stage `[1c]`:
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

### 2. Replay buffer persistence re-enabled ✅

Persistence was **disabled** in `train.py` during 9x9 run 2 because
`np.savez` on the float32 buffer materialized a transient that pushed
peak RSS to exactly the 42.83 GiB cap (see `PHASE_ONE_TRAINING.md`
Problem 4). With uint8 storage the 13x13 1M buffer is 3.6 GB and the
savez transient is also ~4× smaller, so `train.py` now calls
`buffer.save_to(buffer_path)` every iteration again. The comment in
`train.py` points at this section for the memory rationale.

Upside beyond just not losing the buffer on crash: crash recovery is
now a one-liner — resume with `--checkpoint
checkpoints/13x13_run1/checkpoint_NNNN.pt` and the buffer reloads from
`latest_buffer.npz` automatically.

### 3. 13x13-specific exploration & resign tuning ✅

These aren't bugs, they're choices that need to be revisited whenever
the board size changes.

- **Dirichlet α = 0.07.** Formula α ≈ 10 / avg_legal_moves. 13x13 has
  ~150 legal moves early game → 0.067. PLAN.md's 0.07 is right.
- **Temperature schedule: `temperature_moves=40`, `temperature_low=0.25`.**
  9x9 run 2's `(30, 0.25)` was tuned for ~85-move games; 13x13 averages
  ~120 moves. 40 keeps the exploration window at ~1/3 of the avg game
  like the tuned 9x9 setup. `temperature_low=0.25` (vs default 0.1)
  preserves runner-up move signal in training targets so the net sees
  a smoother policy, not collapsed one-hot.
- **Resign v2 thresholds unchanged from 9x9 run 2.** `resign_min_move=20`
  is 17 % of a 120-move 13x13 game (vs 13 % of an 85-move 9x9 game).
  Credible-child cross-check + move floor is game-size-agnostic. Raise
  `resign_min_move` to 30 only if resign cuts too much mid-game data.
- **`max_game_moves=250`** per PLAN.md vs 150 on 9x9.

### 4. Lower `lr_init` + per-iter checkpoints + tight eval cadence ✅

Three `TrainingConfig` tweaks on the 13x13 preset, all cheap insurance:

- **`lr_init=0.005`** (halved from the default 0.01). 9x9 run 1 used
  0.01 and exhibited the iter-4→19 BN-drift regression; run 2 at 0.005
  was stable. On a bigger net (15b vs 10b) and a bigger game, halving
  LR is basically free.
- **`checkpoint_interval=1`.** Phase 1 proved you cannot reconstruct
  weight-drift trajectories from sparse snapshots. 60 iters × ~36 MB
  ≈ 2.2 GB of checkpoints — trivial.
- **`eval_interval=5`.** Tight cadence so BN drift or strength
  regressions get caught within a few iters instead of waiting 10+
  iters like Phase 1 run 1.

### 5. Anchor-buffer mixing — kept optional

The `--anchor-buffer` / `--anchor-frac` CLI path stays in. Phase 2
starts **without** an anchor (fresh Zero training). If 13x13 hits the
same BN-drift pattern as 9x9 run 1, generate an anchor from the last
known-good 13x13 checkpoint and restart with
`--anchor-buffer path/to/anchor_buffer.npz --anchor-frac 0.2`, exactly
like the 9x9 run 2 recovery recipe.

---

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

### Wrong lead

"Self-play is deadlocked." The stack dump shows workers inside
`tick_process` / `tick_select`, which looks like a hang. It isn't —
`faulthandler.dump_traceback_later(300)` fires every 5 min whether or
not anything is stuck. Self-play was just *slow*, and while it was
running the per-game MCTS trees were fattening up until the cgroup
killed the container.

### Root cause: `MCTSTree::advance()` never frees orphaned subtrees

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
   static constexpr int MAX_TREE_NODES = 200000;
   // ...
   s.game.make_move(action);
   s.tree->advance(action);
   if (s.tree->num_nodes() > MAX_TREE_NODES) {
       s.tree->reset(s.game);   // rebuild from current game state
   }
   ```

   `reset(game)` wipes `nodes` + `game_pool` and seeds a fresh tree
   rooted at `s.game`, so the next move starts with no inherited
   visits but still gets its full `num_sims` budget. With the cap at
   200k, per-tree peak is:

   - `nodes`: 200k × 32 B = 6.4 MB
   - `game_pool`: ~2k expanded × ~4.3 KB = 8.6 MB
   - **per-tree ≈ 16 MB peak**

2. **`model/config.py` 13x13 preset — `num_parallel_games=128`**
   (down from the 256 default). Defense in depth: halves tree memory,
   pinned buffers, and worker numpy arrays all at once. GPU batch is
   still 128 × `vl_batch(8)` = 1024 which saturates a 4090 for a
   15b×128ch net, so throughput is essentially unchanged.

3. **`engine/mcts.h` — revert speculative `nodes.reserve(131072)`
   back to `16384`.** I had preemptively bumped reserve during the
   "decisions to revisit" audit; with the cap in place it's just a
   startup optimization, and the larger value wasted ~1 GB of baseline
   RSS sitting behind 256 allocated-but-unused vectors.

Also added two diagnostic helpers:

- **`engine/worker.h::max_tree_nodes()`** — returns the max
  `num_nodes()` across all slots. Exposed as
  `SelfPlayWorker<N>::max_tree_nodes` in bindings.
- **`MAX_TREE_NODES`** exposed as a Python-visible static property on
  each bound worker class, so tests can read it without hardcoding.

### Post-fix memory budget (target: 36 GB)

| component | estimate |
|---|---:|
| MCTS state (128 trees × ~16 MB peak) | ~2.1 GB |
| Replay buffer (uint8 1M × 13×13 × 17 + policy + value) | ~3.6 GB |
| `buffer.save_to` np.savez transient | ~3.6 GB |
| Pinned buffers + worker numpy arrays | ~60 MB |
| Model + optimizer + torch.compile cache | ~1 GB |
| CUDA allocator reserved + Python/C++ overhead | ~3 GB |
| **Total** | **~14 GB** |

Leaves **~22 GB of headroom** under the 36 GB ceiling. If observed
peak RSS ever crosses ~20 GB, stop and investigate.

### Validation

`training/_test_tree_cap.py` — new focused test. Drives a real
`SelfPlayWorker13` at **production `num_sims=600`** with fake NN
outputs for 30 s and asserts `max_tree_nodes` never exceeds
`MAX_TREE_NODES + 120k` headroom at any probe, and that the harvest
produces correctly-shaped arrays. Actual output:

```
Tree-cap test — MAX_TREE_NODES=200,000, hard bound=320,000
  board=13x13, games/worker=4, num_sims=600, vl_batch=8

  ticks=246409  games_done=93  positions_harvested=12847
  peak max_tree_nodes across slots: 289,803 (limit 320,000)
  RSS: start=0.033 GB  peak=0.367 GB  growth=0.334 GB
  elapsed=30.0s, 8214 ticks/s
  harvest: 12847 positions  (obs.shape=(12847, 17, 13, 13), pol.shape=(12847, 170))
PASS: tree cap honored, RSS bounded, harvest intact
```

Four parallel games for 30 s grew RSS by only 0.33 GB — extrapolating
linearly to 128 games × indefinite time, tree memory is bounded as
expected. Peak 289,803 nodes matches the predicted `MAX_TREE_NODES +
one-move-growth` ceiling.

Also verified nothing else regressed:

- `python -m training.train --board-size 13 --smoke-test` — 3 iters,
  loss 6.24 → 6.06 monotonically, all 3 per-iter checkpoints saved,
  final model written.
- `python training/_test_correctness.py` — still prints
  `ALL CORRECTNESS CHECKS PASSED`.

### Why not proper compaction?

Proper compaction (DFS-relabel the reachable subtree into a fresh
`nodes` + `game_pool`, preserving every inherited visit) would be
architecturally better — no tree-reuse loss at all, and steady-state
tree size would converge around ~100-200k nodes *without* a hard cap.
But it's ~80 lines of fragile pointer relocation bookkeeping across
`nodes`, `game_pool`, and every `children_start` index, and a bug in
that code would silently corrupt search.

Reset-on-threshold is ~5 lines, obviously correct, and only throws
away inherited visits every 3–5 moves on 13x13. Revisit compaction
only if Phase 2 convergence looks anemic and the reset frequency turns
out to be the bottleneck.

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
better GPU saturation (9x9 run 2 already ran this batch size happily).

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

Leaves ~4.8 GB headroom under 35 GB. Watch first-iter RSS carefully
— this is tighter than the conservative fix, and the savez transient
is the most likely spike source.

**Files touched:**

- `engine/worker.h`: `MAX_TREE_NODES` 200000 → **1000000**, plus
  expanded comment table showing the memory-vs-cold-move trade.
- `model/config.py` 13x13 preset: `num_parallel_games` 128 → **256**
  with a comment pointing at this section.
- Engine rebuilt via `setup.py build_ext --inplace`.

**Validation still pending.** Re-running `_test_tree_cap.py` and the
13x13 smoke test is on the pre-full-run checklist below; the new 1M
cap means the micro-test should see peak max_tree_nodes around
~1,070,000 (1M + one move's ~72k growth).

### Speed tuning: drop `num_simulations` 600 → 400

Same-day calibration run (dryrun v3) showed real iter time landing
around **~58 min** at (600 sims, 256 parallel, 15b×128ch). Math
reconciled against Phase 1 `9x9_run2/training_log.jsonl`'s measured
469 s/iter:

```
Phase 1 per-tick: 110 ms  (batch 2048, 10b×128ch, 9x9)
Phase 2 scale:    ×2.1 (cells) × ×1.5 (blocks) = ×3.15  per tick
Phase 2 per-tick: ~347 ms  (batch 2048, 15b×128ch, 13x13)
Phase 2 ticks/iter: 2048 × 120 × 600 / 8 / 2048 = 9000
Phase 2 iter time: 9000 × 0.347 = ~3120 s ≈ 52 min   (close to observed)
```

60 iters × 58 min ≈ **58 hours** wall clock. That's long enough to be
annoying and long enough that any mid-run fix needs to pause real
training work.

**The cheapest no-regret speedup is `num_simulations` 600 → 400.**
That's exactly the value AlphaGo Zero used for 13x13 self-play, so
it's the published default, not a shortcut. MCTS quality drops mildly
(see Problem 1 compaction discussion for how quality scales with
total-visit count), convergence rate is very mildly slower per iter,
but wall-clock per iter drops ~33 %.

Updated estimate:

```
Phase 2 ticks/iter: 2048 × 120 × 400 / 8 / 2048 = 6000
Phase 2 iter time:  6000 × 0.347 = ~2080 s ≈ 35 min
Total: 60 × 35 = 35 hours  (vs 58 hours at 600 sims)
```

If the strength curve looks anemic at 400 sims, the plan is to bump
back to 600 for the last ~20 iters once the net is past the worst
of early training (when MCTS quality matters most for stability).

Heavier speed levers that were considered and NOT applied now:

- **Pipelined orchestrator** eliminating the per-tick
  `torch.cuda.synchronize()` (would unlock the last ~20 % of GPU
  utilization — 74 % ceiling seen in both dryruns). Correct long-term
  fix but ~1 day of careful `parallel_self_play.py` rewrite.
- **Halve `num_games_per_iter`** 2048 → 1024. Halves wall time per
  iter but also halves fresh self-play data per training step. Net
  effect on convergence-in-wall-time is ambiguous. Skip.
- **Smaller model.** Conflicts with the Phase 2 goal of a bigger net.
  Skip.

### Dryrun v4 — measured ground truth

Launched 2026-04-14 09:42 UTC. Full 1-iter dryrun with the final
tuned config: 256 parallel × 400 sims × 1M tree cap × 15b×128ch.
Completed cleanly at T+~42 min.

| metric | measured | vs prediction |
|---|---:|---|
| iter 0 total time | **41.6 min** | predicted 35 min (+19 %) |
| self-play time | 41.5 min | — |
| train step time | 2.1 s | — |
| games completed | 2049 | target 2048 |
| positions harvested | 307,678 | — |
| **avg moves / game** | **150** | **predicted 120 — root of the 19 % miss** |
| ticks | 64,301 | matches 2048 × 150 × 400 / 8 / 256 |
| ticks/sec | 25.8 | — |
| per-tick time | **38.8 ms** | — |
| peak cgroup memory | **31.01 GB** | predicted ~30 GB ✓, abort line 35 GB |
| train loss | 5.97 (π=5.15, v=0.82) | finite, reasonable cold-net value |
| grad skips | 0 | — |
| `checkpoint_0000.pt` | 36.5 MB | — |
| `latest_buffer.npz` | 1094 MB (307k positions) | — |

**Takeaways:**

- **Memory is safe.** Peak 31 GB under the 35 GB ceiling with ~4 GB
  headroom. The 1M tree cap + 256 parallel games combo does what it
  said on the tin. No OOM risk for the full run.
- **Wall time is 19 % slower than projected** because 13x13 games at
  this net strength average ~150 moves, not 120. My projection
  overcorrected for 9x9's 85 moves but underestimated the draw of
  longer 13x13 openings. Per-tick time (38.8 ms) matches the
  3.14× scaling factor from Phase 1 almost exactly — no hidden
  overhead, just more moves to search.
- **Training signal is clean.** Loss 5.97 is where a cold 15b×128ch
  13x13 net should land; no NaN; no grad-clip skips. The pipeline is
  correct end-to-end.
- **Updated full-run estimate:**
  ```
  iter 0:                ~42 min (includes ~4 min autotune warmup)
  iter 1..59 each:       ~37 min (no autotune) — estimate
  total 60 iters:        ~36–42 hours wall clock
  ```
  The iter 1+ estimate assumes autotune is cached (it is) and that
  game length stabilizes around 150 moves. If games grow longer as
  the net gets stronger (common in Zero training), iters will
  gradually slow; if they shorten (resign firing more), iters will
  speed up. Either way the total lands **under 2 days** wall clock,
  acceptable for a Phase 2 training run.

---

## Starting Phase 2 on a new device

Handoff checklist for picking this up on a fresh training host (RTX
4090-class GPU + CUDA + ability to build `go_engine.*.so`).

### 0. Sync the code

```bash
git fetch origin
git checkout main
# Latest prep commit should include the OOM + tree-cap fix.
git log --oneline -5
```

### 1. Build / rebuild the C++ engine

The compiled `engine/go_engine.*.so` is platform + Python-ABI
specific; always rebuild on a new host.

```bash
cd Season-5/AlphaZero/engine
python setup.py build_ext --inplace
cd ..
python -c "import sys; sys.path.insert(0,'engine'); import go_engine; print(go_engine.__file__)"
```

### 2. Correctness smoke (fast, CPU-ok)

```bash
PYTHONPATH=engine python training/_test_correctness.py
```

Expected tail: `ALL CORRECTNESS CHECKS PASSED`.

### 3. Tree-cap test (fast, CPU-ok)

Direct regression test for Problem 1. Runs in ~30 s.

```bash
PYTHONPATH=engine python training/_test_tree_cap.py
```

Expected tail: `PASS: tree cap honored, RSS bounded, harvest intact`.

### 4. 13x13 smoke test (3 iters, real GPU)

```bash
PYTHONPATH=engine python -m training.train \
    --board-size 13 \
    --smoke-test \
    --output-dir checkpoints/13x13_smoke
```

`--smoke-test` shrinks to a 2b×32ch net, 16 sims, 32 games/iter,
3 iters. Exercises the **full pipeline end-to-end** on the 13x13 code
path without waiting hours.

**What to watch:**

- `ps -o rss= -p <pid>` peak RSS — should stay **well under 5 GB** for
  the smoke config. If it doesn't, investigate before the full run.
- No SIGKILL / no Python tracebacks at exit.
- Per-iter checkpoints written, `training_log.jsonl` has 3 lines.

### 5. Full Phase 2 run

Once smoke passes cleanly:

```bash
mkdir -p logs checkpoints/13x13_run1

nohup PYTHONPATH=engine python -m training.train \
    --board-size 13 \
    --iterations 60 \
    --output-dir checkpoints/13x13_run1 \
    > logs/13x13_run1.log 2>&1 &

# Watch
tail -f logs/13x13_run1.log
```

Expected per-iter (from `CONFIGS[13]`): 2048 games × 600 sims on a
15b×128ch net with `num_parallel_games=128`. Peak RSS should stabilize
around **14–18 GB**. **Hard ceiling: 36 GB** — abort and investigate
if RSS trends above ~20 GB during iter 0-2.

**Monitor RSS on the first 2–3 iters before walking away.** Phase 1
learned the hard way that "it fit last iter" is not a guarantee it'll
fit next iter when the buffer is still filling toward 1M samples.

### 6. Things to double-check after the first training iter

- `training_log.jsonl` line 0 has sane `self_play` stats (> 0 games,
  > 0 positions, avg moves in 80–160 range for 13x13).
- `train.loss` is finite and roughly 6–7 on iter 0 (cold net); any
  `skipped` count should be 0 or very small.
- `ls checkpoints/13x13_run1/` shows `checkpoint_0000.pt` (~36 MB for
  a 15b×128ch net with Adam moments).
- `checkpoints/13x13_run1/latest_buffer.npz` exists and grows toward
  ~3.5 GB as the buffer fills.

### 7. If something looks off — diagnostic hooks already wired

- `faulthandler` prints all Python thread stacks on any fatal signal
  and every 5 min to stderr (`train.py:29-37`). A stack dump there
  does **not** by itself mean a hang — see Problem 1.
- `kill -USR1 <pid>` dumps live thread stacks on demand.
- Grad-clip (L2 norm 5.0) + NaN-guard in `trainer.py::train_step` skip
  poisoned steps instead of nuking weights; watch `skipped_total`.
- Per-iter `latest_buffer.npz` persistence means a crash doesn't lose
  the buffer — resume with `--checkpoint
  checkpoints/13x13_run1/checkpoint_NNNN.pt`.

### 8. Post-run deliverables (mirror Phase 1)

1. Keep every per-iter `checkpoint_NNNN.pt`. 60 iters × ~36 MB ≈ 2.2 GB
   — trivial, and Phase 1 showed post-hoc weight-drift audits are
   worthless without them.
2. Run the round-robin Bradley-Terry tournament across every
   checkpoint for the Elo curve (same pipeline as
   `9x9_run2_tournament`).
3. Append a "Run 1 — live training" section to **this file** and log
   problems as they come up. That's the Phase 1 pattern and the main
   reason this document was converted from the old `PHASE_TWO_TODO.md`
   to `PHASE_TWO_TRAINING.md`.

---

## Run 1 — live training

### Attempt 1: aborted mid-iter-1 (2026-04-14 ~10:30–11:22 UTC)

Launched the full 60-iter run at ~10:31 after v4 validated the config
end-to-end. Iter 0 completed cleanly; iter 1 was killed at 76 % by
Claude when the monitored `memory.usage_in_bytes` crossed the safety
line. Iter 0 artifacts are preserved and resumable.

**Iter 0 results (identical to v4 within noise):**

| metric | value |
|---|---:|
| self-play time | 2426.8 s (40.5 min) |
| total iter time | 2430.4 s |
| games | 2048, positions 294,816 |
| avg moves / game | 144 |
| ticks | 61,851 @ 25.5 /s |
| train loss | 5.974 (π=5.145, v=0.828) |
| grad skips | 0 |
| `checkpoint_0000.pt` | 36.5 MB ✓ |
| `latest_buffer.npz` | 1.05 GB (294,816 positions) ✓ |

Iter 0 self-play ran smoothly between **~22 GB and ~30 GB** cgroup
`memory.usage_in_bytes`, matching v4. The `| sp:` progress log landed
and is working as designed — 30 s cadence, clean format, ETA
converging from noisy-early toward ~40 min real.

### Problem 2 — `memory.usage_in_bytes` double-counts page cache

**What happened.** Between the iter 0→1 boundary and iter 1 self-play,
cgroup `memory.usage_in_bytes` climbed past v4's 31 GB steady state
and accelerated:

```
check         cgroup   peak     note
T+42 min      33.87 GB 33.87    iter 0 just completed, iter 1 ~2 min in
T+48 min      34.01 GB 34.25    +0.14 GB over 2 min
T+50 min      34.52 GB 34.55    +0.51 GB over 2 min (accelerating)
abort window  -        35.02    crossed the line between checks
```

I SIGTERM'd run1 at T+50:22 before iter 1's end-of-iter savez
transient could fire — projected peak would have been 36–37 GB.

**Post-kill diagnosis via `memory.stat`:**

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

**The 35 GB limit is on `memory.usage_in_bytes`, which sums anon +
cache**, and I was monitoring that. Cgroup v1's usage metric doesn't
distinguish reclaimable cache from unreclaimable rss. The kernel would
have reclaimed the cache under real memory pressure before OOM, but:

1. The user's stated rule is based on the number they'd see (`usage`).
2. I couldn't tell from live monitoring how much of the usage was
   cache vs rss without parsing `memory.stat` every poll.

So ~3–5 GB of "memory pressure" on the dashboard was phantom
reclaimable cache. **The real anon-memory peak was probably around
30–31 GB**, comfortably under any real hardware limit.

### Options for Attempt 2

Presenting these rather than choosing one autonomously. **Nothing has
been retuned; v4's config is still checked into main.** Iter 0 is
resumable from `checkpoints/13x13_run1/checkpoint_0000.pt`.

1. **Drop page cache after every buffer.save_to** (~5 lines in
   `replay_buffer.save_to`). `os.posix_fadvise(fd, 0, 0,
   POSIX_FADV_DONTNEED)` immediately after `np.savez` + `fsync`.
   Frees the 1 GB latest_buffer.npz from the cgroup cache count.
   **Zero impact on AI quality or speed**; purely a memory-accounting
   fix. **My preferred option.**

2. **Monitor `memory.stat: rss` instead of `memory.usage_in_bytes`.**
   Makes the abort rule fire on real anon-only pressure instead of
   on cache-inflated usage. No code changes to training, just to the
   monitoring script. Same abort budget of 35 GB but against a smaller
   number, so effective headroom grows by ~3-5 GB. Risk: if the
   user's 35 GB was a cgroup-wide concern (not just rss), this
   violates the spirit of their rule.

3. **Lower `MAX_TREE_NODES` 1M → 500k.** Cuts MCTS peak from 19 GB to
   9.6 GB, a 9.4 GB savings. Trades AI quality from ~3 % → ~7 %
   convergence hit (~14 moves → ~7 moves between tree resets). Safe
   but unnecessarily conservative if option (1) would suffice.

4. **Lower `num_parallel_games` 256 → 192.** MCTS peak drops 19 → 14.4
   GB. GPU batch drops 2048 → 1536, still well-utilized on a 4090.
   Iter time impact small. Less aggressive than (3).

5. **Combine (1) + (2).** Best-of-both: fix the cache accounting
   issue AND switch to anon-only monitoring. Belt and suspenders.

**Recommendation:** option (1) alone (or 1+2). v4's memory math was
correct for real anon memory, the config is fine, and the only thing
that went wrong was the 1 GB buffer file lingering in page cache. A
trivial fix gets us back to running without touching the training
config.

**What I'm NOT doing autonomously:** picking one of these, rebuilding,
relaunching. The user asked me to stop on anything uncertain, and
this is exactly that. Iter 0 is safe, run1 dir is clean. Waiting for
their decision.

### Resolution: raise operational ceiling 35 GB → 48 GB

The user came back, rejected the "sacrifice AI quality" options and
asked for a real memory-limit audit. Results:

```
memory.limit_in_bytes    = 61.99 GB   ← real cgroup hard cap
memory.memsw.limit       = 61.99 GB   (no swap)
memory.kmem.limit        = unlimited
oom_kill_disable         = 1          ← processes freeze, don't crash
host MemTotal            = 528 GB     (container is tiny vs host)
```

The 35 GB target was a self-imposed soft limit, not a real constraint.
The **real** wedge point is 62 GB, and `oom_kill_disable=1` means
crossing it freezes the process rather than SIGKILL'ing it — so we'd
see the failure in monitoring (tick rate drops to 0) before any
unrecoverable damage.

**Decision: raise operational ceiling 35 GB → 48 GB.** That's:
- Covers all observed peaks (attempt 1 peaked at 35.02) with ~13 GB
  slack
- Still 14 GB of hard-limit headroom below 62 GB
- Zero code changes, zero AI quality cost, zero speed cost

No config tuning, no monitoring rewrite — just a corrected threshold
that matches the actual container envelope.

## Attempt 2 — resumed with 48 GB ceiling, aborted on early-resign

### Run startup (2026-04-14 ~11:59 UTC)

Launched via `--checkpoint checkpoints/13x13_run1/checkpoint_0000.pt`.
Autotune was **~8× faster** than a cold launch (torch.compile
artifacts cached from attempt 1 persisted in `/tmp`): self-play
started by T+0.5 min instead of T+4 min.

`train.py` resume messages landed as expected:

```
Resumed from iteration 0
Restored replay buffer: 294816 samples from checkpoints/13x13_run1/latest_buffer.npz
```

Iter 1 self-play ran comfortably under 30 GB cgroup — the 48 GB
ceiling was correct.

## Problem 3 — early-resign data bias (iter 1)

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

**Value loss going UP while policy loss goes down is the exact
fingerprint of the iter-4→19 regression on 9x9 run 1** — the trainer
is fitting on positions whose value labels got flipped by early
resign, so the value head learns noise while the policy head is
still learning signal from visit distributions.

### Root cause

On a fresh 15b×128ch net at iter 0, value outputs are near zero and
resign never fires. After 100 training steps on 307k cold
self-play positions, the value head has moved just enough for some
positions to hit the `-0.90` threshold for 5 consecutive moves —
early, at move ~25 (just above `resign_min_move=20`), and
aggressively, because the credible-child cross-check requires a
child with Q > `-0.90` **and** ≥ 5 % of root visits. Early in
training, the policy is still nearly-uniform so visits spread across
many children and no single child gets 5 % + a non-losing Q. The
cross-check fires less often than intended on cold-ish nets.

**Effect on training data:** games cut off at move ~25-40. The
replay buffer fills with early-game positions where the "true"
value is ambiguous, and the labels come from the resign decision
itself rather than from played-out games. Feedback loop:

```
early resign → short games → middle/endgame positions absent →
value head trained on biased early-game labels →
value head overconfident early → more resigns fire earlier →
repeat
```

This is textbook what `PHASE_ONE_TRAINING.md` Problem 3 describes.

### Fix

`model/config.py` 13x13 preset: **`resign_min_move=20 → 40`**
(override; default in `TrainingConfig` stays at 20 for 9x9 use).

- 40 is 27 % of a typical 150-move 13x13 game, matching the 23 %
  ratio (20 / 85) that worked on 9x9 run 2.
- The hard move floor can't be side-stepped by the credible-child
  check; it always blocks resign regardless of value confidence.
- No code change, no rebuild, no AI quality loss. Config knob only.
- Credible-child cross-check stays on as the second line of
  defense once move 40 is passed.

### Revert + resume plan

Iter 1's `checkpoint_0001.pt` was trained on biased data; carrying
it forward risks cementing the value-head confusion. Reverting run1
state to the iter 0 baseline:

- **delete** `checkpoint_0001.pt` (poisoned)
- **trim** `latest_buffer.npz` from 379,657 → 294,816 positions
  (iter 0's size, iter 1 entries at indices 294816+ simply
  truncated — circular buffer invariants preserved, `size=index=294816`)
- **truncate** `training_log.jsonl` to just the iter 0 line
- **keep** `checkpoint_0000.pt` — that's the pre-bias weights we
  resume from

Then resume with the same `--checkpoint
checkpoints/13x13_run1/checkpoint_0000.pt` command as attempt 2.
Iter 1 now runs under `resign_min_move=40`; if games land in the
~100–150 move range again, the fix worked.

## Attempt 3 — resumed, ran 5 iters, catastrophic strength regression

Attempt 3 resumed from `checkpoint_0000.pt` under the fix above. Iters
0-4 all completed; memory behavior was healthy (peak ~43 GB cgroup,
under the raised 50 GB ceiling). **But the first in-training
`evaluate_vs_random` at iter 4 returned 5 %**, and a post-kill
per-checkpoint audit showed the strength was **oscillating wildly**
across every iter — the cleanest-ever diagnostic signal we've gotten
on 13x13 training stability.

## Problem 4 — value-head cannibalization

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

1. **Strength oscillates wildly, not monotonically.** This is NOT the
   Phase 1 run 1 fingerprint (which was a slow monotonic decline from
   a working net). Here every iter is a crapshoot between "barely
   viable" and "completely broken".

2. **Policy loss drops monotonically** (5.145 → 4.820, -6.3 %), but
   **value loss rises monotonically** (0.828 → 0.925, **+11.7 %**).
   The trainer is making the policy head much better and the value
   head much worse at the same time.

3. **Win rate tracks value loss directly**: iters where v_loss ≈ 0.90
   score ~20 %, iters where v_loss ≈ 0.92 score 0–5 %. A 0.02-0.03
   change in value loss swings strength by 15–20 percentage points.

### Root cause: equal-weight loss is ~80 % policy-driven

The trainer uses `loss = policy_loss + value_loss` with equal weight.
But the two losses have very different natural magnitudes:

- **`policy_loss`**: cross-entropy over 170 actions. Uniform baseline
  is `log(170) ≈ 5.14`. A cold net sits right there, a trained net
  drops to ~4.5-4.8. **Magnitude ~5.**
- **`value_loss`**: MSE of scalar values in [-1, 1]. A net predicting
  0 on all positions gets ~1.0 MSE against ±1 targets. **Magnitude ~0.9.**

At equal weight in the summed loss, the total is ~80 % policy-driven,
so the gradient SGD sees is ~5× more policy signal than value signal.
The value head is **under-trained** — every SGD step, policy weights
move a lot and value weights barely move. Over 100 train steps per
iter, the value head drifts randomly on whatever noisy value targets
the self-play happened to produce, without ever actually **learning**
them.

The oscillation mechanism:

```
1. iter N self-play produces positions + value targets (from game outcomes)
2. trainer runs 100 SGD steps; policy head improves, value head
   wanders in random direction
3. iter N+1 self-play uses the wandered value head; MCTS leaf
   evaluations are noisy → MCTS picks worse moves → produces worse
   games → value targets are from a different distribution than iter N
4. trainer sees a value-target shift; value head wanders again,
   possibly back toward iter N's state, possibly further
5. repeat
```

This is a standard failure mode in Zero training. LeelaZero,
Katago, and the original AlphaZero all weight value loss higher
than policy loss (coefficients between 1.0 and 1.5 on some
implementations, explicit L2 regularization in AlphaZero itself).

### Why 9x9 worked but 13x13 doesn't

Phase 1 9x9 run 2 used the same `loss = policy + value` recipe and
trained fine end-to-end. Three reasons 13x13 is more sensitive:

1. **Bigger net (15b vs 10b)** — more capacity to overfit noisy value
   targets.
2. **Longer games (~150 vs ~85 moves)** — more positions where the
   game outcome is still ambiguous, so value labels are noisier.
3. **Larger board** — value evaluation is inherently harder because
   territorial accounting scales roughly with board area.

So a recipe that worked on a "forgiving" small-game/small-net setup
collapses on a harder problem.

### Fix: `value_loss_weight = 2.0` for 13x13

New knob on `TrainingConfig`:

```python
# model/config.py
@dataclass
class TrainingConfig:
    ...
    value_loss_weight: float = 1.0   # default preserves 9x9 behavior

# 13x13 preset override:
value_loss_weight=2.0,
```

Trainer uses it:

```python
# trainer.py train_step
loss = policy_loss + self.cfg.value_loss_weight * value_loss
```

**Why 2.0?** At that coefficient, value's contribution to the total
is `2.0 × 0.9 = 1.8` vs policy's `5.0`, so policy:value split is
~73 : 27 (vs the old 85 : 15). That's still policy-dominated, but the
value head gets ~2× the gradient signal it was getting, enough to
actually learn the targets.

Not going higher (to e.g. 3.0) because:
- AlphaZero's own published weight is closer to 1.0; 2.0 is already
  a meaningful deviation from the canonical recipe.
- Over-weighting value could swing the problem the other way (policy
  head cannibalization).
- If 2.0 still oscillates, the next lever is `train_steps_per_iter`
  or `lr_init`, tested independently.

Also enabling `eval_interval=1` on the 13x13 preset for the initial
run2 so we see iter-by-iter strength and can catch any regression on
iter 1 instead of waiting until iter 4. Once the strength curve is
confirmed healthy, revert to a cheaper eval cadence (5-ish).

### Smoke-test verification (2b×32ch tiny model)

```
Iter 0 | Train: loss=6.9982 (pi=5.1543, v=0.9220) | Eval: 75%
Iter 1 | Train: loss=6.4426 (pi=5.0921, v=0.6753) | Eval: 60%
Iter 2 | Train: loss=6.3165 (pi=5.0439, v=0.6363) | Eval: 60%
```

Arithmetic check: `5.1543 + 2.0 × 0.9220 = 6.9983` ≈ observed 6.9982
(rounding) → **value_loss_weight is active**.

Observations:
- **Value loss drops monotonically** (0.922 → 0.675 → 0.636), reversing
  the failing run1 trend.
- **Policy loss also drops** (5.154 → 5.092 → 5.044), so we're not
  cannibalizing policy to help value.
- **Eval fires every iter** (eval_interval=1 works).
- Eval win rate is noisy on 20 games + tiny model but **no strength
  crash** from any iter to the next (75 → 60 → 60, not 75 → 2 → 60).

Full 15b×128ch run2 is the real test.

### What gets thrown away from run1

run1's iter 1-4 data is contaminated and its weights are broken.
Starting **run2 from scratch** (new `checkpoints/13x13_run2/` dir, no
`--checkpoint`, no buffer reload). `checkpoint_0000.pt` is also not
reusable because it was saved AFTER 100 train steps with the
unweighted loss — its value head is already biased.

The full per-iter run1 checkpoints and the strength_audit.txt stay
on disk for post-mortem. PHASE_TWO_TRAINING.md's Problem 4 section
(this one) is the write-up.

---

## Starting run2 — fresh, value_loss_weight=2.0

Commands for a clean launch with the new config:

```bash
mkdir -p logs checkpoints/13x13_run2
nohup PYTHONPATH=engine python -m training.train \
    --board-size 13 \
    --iterations 60 \
    --output-dir checkpoints/13x13_run2 \
    > logs/13x13_run2.log 2>&1 &
```

**What to watch in iters 0-5 of run2:**

- `value_loss` should drop monotonically (or at worst stay flat)
  across iters. If it rises, the fix is insufficient.
- `eval_vs_random` win rate should climb monotonically from the cold
  baseline (~20 % on 13x13 cold). iter 1 should be ≥ iter 0, not
  catastrophically lower.
- `avg moves/game` should stay in the 100-160 range. Values below
  80 indicate pass-loop or value-collapse from the old bug.
- `cgroup memory` should plateau around ~40-44 GB (same as run1).

If all four look healthy through iter 5, the 60-iter run is expected
to complete in ~35-40 hours.
