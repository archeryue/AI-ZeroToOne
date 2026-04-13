# Phase 2 — Things to do before starting

Followups from Phase 1 work that should land before kicking off the
13x13 self-play run. Not blocking Phase 1 itself.

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

## 3. Buffer persistence — re-evaluate

Buffer persistence is currently disabled in `train.py` because the
`np.savez` transient pushed the 9x9 run over the 42.8 GB host's
memory limit. With `uint8` storage the Phase 2 obs buffer is 2.87 GB
instead of 11.49 GB, so the savez transient is also ~4× smaller and
should fit. Re-enable for Phase 2 once Option A lands.

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

1. Land Option A in a single PR. Verify with the three-step test
   above + a full `_test_correctness.py` smoke run.
2. Re-enable `buffer.save_to(...)` in `train.py`.
3. Update 13x13 preset in `model/config.py` per section 2 above.
4. Run a 3-iter smoke test on the 13x13 config to validate memory
   peaks under 35 GB.
5. Launch full Phase 2.
