# Phase 2 — Things to do before starting

Followups from Phase 1 work that should land before kicking off the
13x13 self-play run. Not blocking Phase 1 itself.

**Status (2026-04-14):** code-side prep items 1–3 are landed and
committed on `claude/review-alphazero-project-2xT2d` (commit
`59e0c84`). Items 4–5 run on the training host — see
[Starting Phase 2 on a new device](#starting-phase-2-on-a-new-device)
at the bottom for the exact commands.

---

## 1. Replay buffer: store obs as `uint8` (Option A)

**Why blocking for Phase 2:** Phase 2 buffer at the planned 1M × 13x13
samples is **12.2 GB as float32** (vs 1.6 GB for the 9x9 Phase 1
buffer). On the current 42.8 GB cgroup host, that does not fit
alongside ~30 GB of MCTS trees + torch.compile cache + model state.

**The fix:** every obs plane value is exactly 0 or 1 (verified in
`_test_correctness.py`). Storing them as `uint8` instead of `float32`
is a lossless 4× memory cut.

| | float32 | uint8 |
|---|---:|---:|
| 13x13 obs per sample | 11,492 B | 2,873 B |
| Phase 2 buffer (1M samples, obs only) | 11.49 GB | **2.87 GB** |
| Total buffer (obs + policy + value) | 12.17 GB | **3.56 GB** |

H2D PCIe bandwidth for the obs portion of every training step also
drops 4× as a free side-effect.

### Code changes (~3 lines of substance)

`training/replay_buffer.py`:

```python
# was:
self.obs = np.zeros((capacity, input_planes, N, N), dtype=np.float32)

# becomes:
self.obs = np.zeros((capacity, input_planes, N, N), dtype=np.uint8)
```

```python
# in _store(), was:
self.obs[idx] = obs

# becomes:
self.obs[idx] = obs.astype(np.uint8)   # cast on push, obs is small
```

`training/trainer.py::train_step`:

```python
# was:
obs = torch.from_numpy(obs_np).to(self.device)

# becomes:
obs = torch.from_numpy(obs_np).to(self.device).float()
```

That's it. The C++ self-play side keeps writing float32; the cast
happens at the Python boundary.

### Why the GPU cast is free

`Conv2d` reads `float32` regardless. With float32 storage, the GPU
copies 4 bytes/cell across PCIe and reads them directly. With uint8
storage, the GPU copies 1 byte/cell across PCIe (4× faster) and
materializes a `float32` tensor with `.float()` — modern CUDA fuses
the dtype conversion into the load, so it's not new work. We win
both memory and bandwidth.

### Risks (all very low)

1. `_test_correctness.py` asserts `set(obs_vals) ⊆ {0.0, 1.0}` on the
   raw buffer — needs to become `⊆ {0, 1}` after the dtype change.
   One-character fix.
2. `np.rot90` / `np.flip` for 8-fold augmentation work transparently
   on `uint8`. No change.
3. Pinned-memory transfer works the same on `uint8`. The materialized
   float32 GPU tensor is one extra ~3 MB allocation per training step
   — negligible vs 24 GB VRAM.

### Verification before merging

```python
# 1. Round-trip identity
obs_f32_orig = ...                     # float32 obs from self-play
buffer.push(obs_f32_orig, ...)
obs_back = torch.from_numpy(buffer.obs[0]).float().numpy()
assert np.array_equal(obs_f32_orig, obs_back)

# 2. Loss reproducibility — train one seeded step on identical batches
# under float32 vs uint8 buffers, losses must match to ~1e-6

# 3. Memory check — process RSS after pushing 300K samples should drop
# by ~1.2 GB on 9x9 and ~8.6 GB on 13x13
```

### Out of scope (defer to Phase 3 if it's needed)

**Option B — bitpack obs:** stores each binary cell as 1 bit instead
of 1 byte. Cuts the Phase 2 buffer further to **0.36 GB** (32×).
Adds ~30 lines for `np.packbits` index bookkeeping plus a ~10 µs
per-sample unpack cost. Worth it for Phase 3 (1M × 19x19 obs is
25.7 GB as float32, 6.4 GB as uint8 — still tight on 42 GB) but not
needed for Phase 2.

---

## 2. Decisions to revisit for 13x13

These are not bugs, just choices that need a fresh look because
13x13 is a different game from 9x9.

- **Dirichlet α.** PLAN.md uses 0.07 for 13x13 (vs 0.11 for 9x9).
  AlphaGo Zero formula is α ≈ 10 / avg_legal_moves. 13x13 has ~150
  legal moves early-game → α ≈ 0.067. PLAN.md's 0.07 is right.
- **Temperature schedule.** Run 2's 9x9 schedule (`τ=1.0` for 30
  moves, then `τ=0.25`) was tuned for ~85-move 9x9 games. Phase 2's
  13x13 averages ~120 moves. Probable update: `temperature_moves=40`
  (still ~1/3 of avg game) and keep `temp_low=0.25`.
- **Resign v2 thresholds.** Resign v2 was tuned against the 9x9
  failure mode. The same thresholds should be safe for 13x13 (the
  credible-child cross-check + move floor is game-size-agnostic), but
  the `resign_min_move=20` floor that was 13% of an 85-move 9x9 game
  is only 17% of a 120-move 13x13 game. Probably leave it; raise to
  30 if early resign cuts too much mid-game data again.
- **`max_game_moves=250`** per PLAN.md for 13x13 vs 150 for 9x9.
  Already in `model/config.py` 13x13 preset.

## 3. Buffer persistence — re-enabled ✅

Buffer persistence was disabled in `train.py` during Phase 1 because
the `np.savez` transient pushed the 9x9 run over the 42.8 GB host's
memory limit. With `uint8` storage the 13x13 1M obs buffer is 2.87 GB
instead of 11.49 GB, and the savez transient also shrank ~4×.
`train.py` now calls `buffer.save_to(buffer_path)` every iteration
again; watch peak RSS on the first iter or two to confirm the
transient fits comfortably under the cgroup cap on the new host.

## 4. Anchor-buffer mixing — keep optional

The `--anchor-buffer` / `--anchor-frac` code path stays in. Phase 2
should start without an anchor (fresh Zero training). If 13x13 hits
the same BN drift pattern as 9x9 run 1, generate an anchor from the
last known-good 13x13 checkpoint and restart with it.

## 5. C++ engine size sanity check

`engine/mcts.h` has `nodes.reserve(65536)` — fine for 9x9 but might
be undersized for 13x13 trees with more sims and longer games. Audit
once Phase 1 finishes and bump to 131072 if `MCTSTree::num_nodes` is
hitting the reservation limit late in 13x13 self-play.

---

## Order of operations when Phase 1 finishes

1. ✅ **Land Option A (uint8 obs).** `training/replay_buffer.py` stores
   obs as `uint8`, push/load_from cast explicitly, sample() preserves
   dtype. `training/trainer.py` casts to float after H2D. Verified
   via standalone test: dtype check, round-trip identity, 8-way
   symmetry correctness (match `augment_8fold` reference), pass
   invariant, save_to/load_from exact round-trip, legacy float32
   save compat, ±4σ symmetry uniformity.
2. ✅ **Re-enable `buffer.save_to(...)` in `train.py`.** Comment
   updated to point at this TODO section and the uint8 savings
   (12.2 GB → 3.6 GB on the 13x13 1M buffer, savez transient ~4×
   smaller).
3. ✅ **Update 13x13 preset in `model/config.py`** per section 2:
   `temperature_moves=40`, `temperature_low=0.25`.
4. ⏭ **Run a 3-iter smoke test on the 13x13 config** once the full
   CUDA + go_engine env is available. Watch peak RSS — should stay
   well under 35 GB with uint8 buffer.
5. ⏭ **Launch full Phase 2.**

---

## Starting Phase 2 on a new device

Handoff checklist for picking this up on a fresh training host (RTX
4090-class GPU + CUDA + built `go_engine.*.so`).

### 0. Sync the code

```bash
git fetch origin claude/review-alphazero-project-2xT2d
git checkout claude/review-alphazero-project-2xT2d
# Sanity: HEAD should include commit 59e0c84
#   "Phase 2 prep: uint8 obs buffer + re-enable persistence + 13x13 temp schedule"
git log --oneline -1
```

### 1. Build / rebuild the C++ engine

The compiled `engine/go_engine.*.so` is platform + Python-ABI
specific; always rebuild on a new host.

```bash
cd Season-5/AlphaZero/engine
python setup.py build_ext --inplace
cd ..
python -c "import sys; sys.path.insert(0,'.'); import go_engine; print(go_engine.__file__)"
```

### 2. Correctness smoke (fast, CPU-ok)

Run the end-to-end correctness test first. Stage `[1c]` is the new
uint8 round-trip + save/load check added for Phase 2. All stages
should pass:

```bash
# From Season-5/AlphaZero/ :
python training/_test_correctness.py
```

Expected tail output: `ALL CORRECTNESS CHECKS PASSED`.

### 3. 13x13 smoke test (3 iters, real GPU)

```bash
python -m training.train \
    --board-size 13 \
    --smoke-test \
    --output-dir checkpoints/13x13_smoke
```

`--smoke-test` shrinks model to 2b×32ch, 16 sims, 32 games/iter,
3 iters. Goal is to exercise the **full pipeline end-to-end** on
the 13x13 code path (board size, buffer shapes, augmentation,
save/load) without waiting hours.

**What to watch:**

- `ps -o rss= -p <pid>` or equivalent peak RSS — should stay **well
  under 20 GB** for the smoke config. If it doesn't, investigate
  before the full run.
- `buffer.save_to` runs each iter → expect `latest_buffer.npz` to
  appear in the output dir and grow roughly as samples accumulate.
- No SIGKILL / no Python tracebacks at exit.
- `evaluate_vs_random` at iter 2/3 completes without crashing (win
  rate will be ~50% — we only trained 3 iters of a tiny model).

### 4. Full Phase 2 run

Once the smoke test passes cleanly:

```bash
# Logs + checkpoints to their own dir
mkdir -p logs checkpoints/13x13_run1

nohup python -m training.train \
    --board-size 13 \
    --iterations 60 \
    --output-dir checkpoints/13x13_run1 \
    > logs/13x13_run1.log 2>&1 &

# Watch
tail -f logs/13x13_run1.log
```

Expected per-iter (from `CONFIGS[13]`): 2048 games × 600 sims on a
15b×128ch net. Peak RSS should stabilize around **25–30 GB**
(1M-sample uint8 obs buffer = 3.6 GB, MCTS trees + compile cache
dominate the rest). **Hard ceiling is the 42.8 GB cgroup** —
`oom_kill_disable=1` means going over produces a silent SIGKILL
with no traceback, so watch RSS on the first 2–3 iters before
walking away.

**Phase 2 starts without anchor-buffer mixing.** If the
iter-4→19 BN-drift pattern from 9x9 run 1 reappears (tournament
strength stalls or regresses while training loss keeps dropping),
restart from the last known-good checkpoint with `--anchor-buffer
path/to/anchor_buffer.npz --anchor-frac 0.2`, same as 9x9 run 2's
recovery.

### 5. Things to double-check after the first training iter

- `training_log.jsonl` line 0 contains sane `self_play` stats (>0
  games, >0 positions, avg moves in 80–160 range for 13x13).
- `train.loss` is finite and roughly 6–7 on iter 0 (cold net);
  `train.skipped` is 0 or very small.
- `ls checkpoints/13x13_run1/` shows `checkpoint_0000.pt` (36 MB at
  15b×128ch ≈ 7 MB params × 4 B + optimizer state).
- `checkpoints/13x13_run1/latest_buffer.npz` exists and is ~3.5 GB
  after the buffer fills.

### 6. If something looks off — diagnostic hooks already wired

- `faulthandler` prints all Python thread stacks on any fatal signal
  and every 5 min to stderr (`train.py:28-37`). Tail the log for a
  traceback when RSS looks suspicious.
- `kill -USR1 <pid>` dumps live thread stacks on demand.
- Grad-clip (L2 norm 5.0) + NaN-guard in `trainer.py::train_step`
  skip poisoned steps instead of nuking the weights; watch the
  `skipped_total` counter in the per-iter train log.
- Per-iter `latest_buffer.npz` persistence means a crash doesn't
  lose the buffer — just resume with `--checkpoint
  checkpoints/13x13_run1/checkpoint_NNNN.pt`.

### 7. Post-run deliverables (mirror Phase 1)

When the run finishes or you stop it:

1. Keep all per-iter `checkpoint_NNNN.pt` files — they're cheap
   (~36 MB each × 60 iters ≈ 2.2 GB) and Phase 1 showed that
   post-hoc weight-drift audits are worthless without them.
2. Run the round-robin Bradley-Terry tournament across every
   checkpoint to get the Elo curve (same pipeline as
   `9x9_run2_tournament`).
3. Open a `PHASE_TWO_TRAINING.md` diary file mirroring
   `PHASE_ONE_TRAINING.md` and log problems as they come up.
