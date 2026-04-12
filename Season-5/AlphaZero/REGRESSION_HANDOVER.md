# Regression Handover — 9x9 Phase 1, post iter 19

Written at pause. Pick up here tomorrow.

## TL;DR

Training ran iters 4–19 post-fix on top of `preserved_iter0012.pt` (the
last pre-NaN weights). Training loss kept dropping (4.89 → 2.56, policy
loss 3.99 → 1.89), but **tournament strength regressed** measured two
ways:

- `iter 9 vs random = 96%`, `iter 16 vs random = 81%`, `iter 19 vs
  random = 85%` — **~10–15 percentage points worse**
- Weight audit shows `policy_bn` grew **2.4×** (31 → 74) and
  `input_conv` grew **3.4×** (0.16 → 0.55), both monotonically across
  iters 9 → 19. Classic BatchNorm specialization to a narrow
  distribution.
- Head-to-head vs iter 9 at 400 sims is roughly *matched* (iter 19 =
  0.520), which **masks** the regression — search at 400 sims
  compensates for the policy-head drift; at 100 sims (vs-random) it
  doesn't, and the regression shows through.

Training is **paused**. pid 23196 killed cleanly. Checkpoints 9–19
preserved as `checkpoints/9x9_run1/preserved_iter00NN.pt`.

Root cause: **distribution collapse inside the replay buffer plus
BatchNorm specializing to that narrow distribution**, compounded by:

1. Replay buffer is lost on every restart (`train.py` creates a fresh
   empty buffer on resume, doesn't save/load it)
2. Resign logic triggers too aggressively, cutting off middle-game and
   endgame positions from the buffer (10% disable-resign rate and no
   move-count floor means aggressive early resign during drift)
3. No gradient clipping was the root of the earlier NaN (fixed); but
   grad clipping alone doesn't prevent the slow monotonic drift
4. No anchor against known-good distributions — once the buffer
   narrows the training has no recovery force

## Plan for tomorrow

Implement three coupled fixes together (**resign v2**, **buffer
persistence**, **anchor buffer**), rebuild the C++ extension, roll back
to `preserved_iter0011.pt`, and restart. All three fixes are necessary;
none is sufficient alone.

Rollback target is iter 11 because the weight audit shows `policy_bn`
jumps from 32 (iter 11) → 44 (iter 12), i.e. iter 12 is where the
drift begins.

---

## Evidence

### vs-random trajectory (100 sims/move)

| iter | vs random | source |
|---|---:|---|
| 9 | 96.0% | in-loop eval (pre-NaN run) |
| 16 | 81.0% | standalone via `eval_vs_random.py` |
| 19 | 85.0% | in-loop eval (post-fix run) |

### Head-to-head matchups vs `eval_opponent_iter0009.pt` (50 pairs, 400 sims, 8 random ply)

| iter | both-wins | tied | zero-wins | score | CI | verdict |
|---|---:|---:|---:|---:|---|---|
| 12 | 16 | 32 | 2 | 0.640 | [0.501, 0.759] | STRONG |
| 15 (seed 0) | 1 | 36 | 13 | 0.380 | [0.259, 0.518] | regression |
| 15 (seed 42) | 1 | 38 | 11 | 0.400 | [0.276, 0.538] | regression |
| 16 | 15 | 30 | 5 | 0.600 | [0.462, 0.724] | STRONG |
| 19 | 13 | 26 | 11 | 0.520 | [0.385, 0.652] | tied |

The head-to-head numbers are **noisy** and **misleading** at 400 sims —
search compensates for policy drift. The vs-random numbers are the
cleaner signal.

### Weight trajectory (the mechanical proof)

```
iter      max|w|       ||W||       rms    input_conv  policy_bn  value_bn
   9     77.6359      134.99    0.0777      0.160       31.19    77.64
  10     76.1878      133.51    0.0769      0.163       30.60    76.19
  11     70.7556      133.32    0.0768      0.168       32.15    70.76
  12     67.8962      145.74    0.0839      0.170       44.34    67.90   ← drift begins
  15     81.3519      147.45    0.0849      0.224       43.73    81.35
  16     67.2174      146.92    0.0846      0.334       51.35    67.22
  17     85.4022      160.72    0.0926      0.419       61.68    85.40
  18     70.5563      163.28    0.0940      0.477       65.05    70.56
  19     79.7501      168.00    0.0968      0.549       73.57    79.75
```

- `policy_bn` goes 31.2 → 73.6 (**2.4× growth**) monotonically
- `input_conv` goes 0.16 → 0.55 (**3.4× growth**) monotonically
- All other layers (residual tower, value head, FC layers) stay roughly
  stable. This is specific to BN and the input conv, not a global
  optimization problem.

Interpretation: the residual tower's activations on self-play positions
have narrow variance; `policy_bn`'s learned affine scale grows to
amplify the signal; this amplification is specialized to the in-
distribution statistics and **breaks on out-of-distribution inputs**
(random play, older checkpoint's play).

### Training loss trajectory (for context)

```
iter  total    pi      v       avg moves   games/s
  4   4.891   3.989   0.902       99         2.7
  5   4.732   3.889   0.843      111         2.7
  6   4.589   3.838   0.751       73         4.1
  7   4.656   3.880   0.776       90         3.5
  8   4.698   3.886   0.812       98         3.2
  9   4.739   3.880   0.860       97         3.2
 10   4.621   3.815   0.806       94         3.2
 11   4.453   3.688   0.765       84         3.7
 12   3.991   3.412   0.578       44         7.7   ← resign firing heavily; loss jump suspicious in hindsight
 13   4.428   3.636   0.792      105         2.5   (post-fix re-train from iter 12)
 14   4.154   3.324   0.830       60         4.1
 15   3.836   3.132   0.704       96         2.6
 16   3.208   2.769   0.440       65         3.7
 17   3.012   2.227   0.785       91         2.7
 18   2.385   2.050   0.335       59         4.4
 19   2.563   1.893   0.670       85         3.0
```

Policy loss drops monotonically from 3.99 → 1.89 across 16 iters. Total
loss drops from 4.89 → 2.56. On paper this is great training. In
reality it's the policy head memorizing its own self-play visit
patterns while losing general Go skill.

---

## Current state on disk

### Preserved checkpoints (retention-proof, never pruned)

```
checkpoints/9x9_run1/
  eval_opponent_iter0009.pt      ← the 96%-vs-random verified baseline
  preserved_iter0010.pt
  preserved_iter0011.pt          ← rollback target
  preserved_iter0012.pt          ← verified iter-12 (drift start)
  preserved_iter0015.pt
  preserved_iter0016.pt
  preserved_iter0017.pt
  preserved_iter0018.pt
  preserved_iter0019.pt
```

(iter 13 was the NaN-dead one, deleted; iter 14 was pruned by retention
before I thought to preserve it, not recoverable.)

### Scripts ready to use

- `training/eval_matchup.py` — paired head-to-head eval between two
  checkpoints. Flags: `--new`, `--old`, `--pairs`, `--sims`,
  `--random-moves`, `--seed`. Default 50 pairs × 400 sims × 8 random
  opening ply is the "v5" config that actually differentiates strength.
- `training/eval_vs_random.py` — standalone vs-random eval. Used for
  iter 16's standalone number (81%).

### Killed processes

- **pid 23196** training process — killed via `kill -TERM` then
  `kill -KILL`. Completely gone.
- Monitors and cron `89026d6f` still running and watching for events
  that will not come. Safe to leave; they'll expire or can be cleared
  in the morning session.

### Tools running against a live directory

Nothing is actively modifying `checkpoints/9x9_run1/` or
`logs/9x9_run1.log`. Safe to edit anything.

---

## The three coupled fixes

### Fix 1 — Resign logic v2

**Current logic** (`engine/worker.h` line ~138):

- Single signal: `root_value()` < `-0.95`
- `resign_consecutive=3`
- No move-count floor
- `resign_disabled_frac=0.10`

**New logic:**

```cpp
// In SelfPlayConfig (engine/worker.h):
float resign_threshold             = -0.90f;  // was -0.95, loosened
int   resign_consecutive           = 5;        // was 3, raised
int   resign_min_move              = 20;       // NEW: hard floor
float resign_disabled_frac         = 0.20f;    // was 0.10, doubled
float resign_min_child_visits_frac = 0.05f;   // NEW: filter NN hallucinations
```

```cpp
// Replace the resign block in complete_move():
if (!s.disable_resign && s.move_num >= cfg_.resign_min_move) {
    float root_v = s.tree->root_value();
    bool losing = (root_v < cfg_.resign_threshold);

    // Cross-check: no credibly-visited child has a non-losing Q-value.
    // A child is "credibly visited" if it has ≥5% of root's total
    // visits (filters NN-hallucinated low-visit high-Q children).
    if (losing) {
        const auto& root = s.tree->nodes[s.tree->root_idx];
        int total_visits = root.visit_count;
        int min_visits = std::max(
            1, (int)(total_visits * cfg_.resign_min_child_visits_frac));
        for (int i = 0; i < root.num_children; ++i) {
            const auto& child = s.tree->nodes[root.children_start + i];
            if (child.visit_count >= min_visits &&
                child.q_value() > cfg_.resign_threshold) {
                losing = false;  // there's a credible recovery line
                break;
            }
        }
    }

    if (losing)
        s.consecutive_low++;
    else
        s.consecutive_low = 0;

    if (s.consecutive_low >= cfg_.resign_consecutive) {
        s.game.resign(s.game.current_turn);
        finish_game(idx);
        return;
    }
}
```

**Wiring:**

- `engine/worker.h` — add the 2 new fields to `SelfPlayConfig`, implement logic
- `engine/bindings.cpp` — expose the 2 new `def_readwrite`s
- `model/config.py` — add matching fields to `TrainingConfig` with the new defaults
- `training/parallel_self_play.py::_make_sp_config` — propagate the new fields

### Fix 2 — Replay buffer persistence

**Add save/load methods to `training/replay_buffer.py`:**

```python
def save_to(self, path: str):
    np.savez(
        path,
        obs=self.obs[:self.size],
        policy=self.policy[:self.size],
        value=self.value[:self.size],
        index=np.int64(self.index),
        size=np.int64(self.size),
    )

def load_from(self, path: str):
    data = np.load(path)
    n = int(data["size"])
    self.obs[:n]    = data["obs"]
    self.policy[:n] = data["policy"]
    self.value[:n]  = data["value"]
    self.index = int(data["index"])
    self.size = n
```

**Integrate into `training/train.py`:**

- After `buffer = ReplayBuffer(...)`, if resuming from a checkpoint:
  ```python
  buf_path = os.path.join(args.output_dir, "latest_buffer.npz")
  if os.path.exists(buf_path):
      buffer.load_from(buf_path)
      print(f"  Restored replay buffer: {len(buffer)} samples from {buf_path}")
  ```
- At the end of every iter (after buffer.push but before checkpoint
  save is fine):
  ```python
  buffer.save_to(os.path.join(args.output_dir, "latest_buffer.npz"))
  ```

**Disk cost:** ~3 GB when full (500K × (17*81 + 82 + 1) × 4 bytes).
Saves are ~5 s on NVMe. Overwrite single file per iter — no
retention/versioning; we only need the latest.

### Fix 3 — Anchor buffer from iter 9

**Goal:** permanently mix iter 9's distribution into every training
batch so BN/conv layers can't specialize to whatever the current model
is playing.

**Step 1 — generate the anchor buffer once** (standalone script, new
file `training/gen_anchor_buffer.py`):

Load `preserved_iter0009.pt`, run 2048 self-play games via
`ParallelSelfPlay`, dump the raw replay buffer contents to
`checkpoints/9x9_run1/anchor_buffer_iter9.npz`. Takes ~13 minutes
(same as a normal iter). Do this ONCE, never regenerate.

Pseudocode:

```python
net = load_net("preserved_iter0009.pt")
buf = ReplayBuffer(capacity=500_000, ...)
sp  = ParallelSelfPlay(net, ...)
sp.run_games(2048, buf)
buf.save_to("anchor_buffer_iter9.npz")
```

Expected content: ~160K positions × 8 augmentation = ~1.3M → but
capped at 500K by the buffer. Plenty.

**Step 2 — mix anchor samples into every train_step:**

In `training/trainer.py::train_step`:

```python
def train_step(self, buffer, anchor=None, anchor_frac=0.2):
    n_anchor = int(self.cfg.batch_size * anchor_frac) if anchor else 0
    n_main   = self.cfg.batch_size - n_anchor

    obs_main, pol_main, val_main = buffer.sample(n_main)
    if n_anchor:
        obs_anc, pol_anc, val_anc = anchor.sample(n_anchor)
        obs_np    = np.concatenate([obs_main,    obs_anc], axis=0)
        policy_np = np.concatenate([pol_main,    pol_anc], axis=0)
        value_np  = np.concatenate([val_main,    val_anc], axis=0)
    else:
        obs_np, policy_np, value_np = obs_main, pol_main, val_main
    # ... rest of the train_step as before
```

- `anchor_frac=0.2` — 20% of each batch from iter 9's distribution
- Anchor is a normal `ReplayBuffer` loaded from `anchor_buffer_iter9.npz`
- Never modified; samples only

**Wire into `train.py`:**

```python
parser.add_argument("--anchor-buffer", type=str, default=None)

anchor = None
if args.anchor_buffer:
    anchor = ReplayBuffer(500_000, model_cfg.board_size)
    anchor.load_from(args.anchor_buffer)
    print(f"  Loaded anchor buffer: {len(anchor)} samples")

# Pass to trainer.train_epoch → train_step
tr_stats = trainer.train_epoch(buffer, args.iterations, anchor=anchor)
```

`trainer.train_epoch` forwards `anchor` into each `train_step` call.

### Fix 0 — Gradient clipping is already in place

(Sanity check, do not re-add.) `trainer.train_step` already has:

```python
torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
```

plus the `torch.isfinite(loss)` skip-step guard. Don't remove these.

---

## Step-by-step implementation order

All of these are in `/root/AI-ZeroToOne/Season-5/AlphaZero/`.

1. **Resign logic code changes** (no rebuild yet — do code first, all
   at once, then rebuild)
   - `engine/worker.h` — add 2 new fields to `SelfPlayConfig`,
     implement new resign block in `complete_move()`
   - `engine/bindings.cpp` — expose `resign_min_move` and
     `resign_min_child_visits_frac` via `def_readwrite`
   - `model/config.py` — add matching fields to `TrainingConfig` and
     the 9x9 `CONFIGS` override; update values for the raised
     `resign_disabled_frac` / `resign_consecutive` / loosened
     `resign_threshold`
   - `training/parallel_self_play.py::_make_sp_config` — propagate new
     fields

2. **Replay buffer persistence**
   - `training/replay_buffer.py` — add `save_to()` / `load_from()`
   - `training/train.py` — load on resume, save after each iter

3. **Anchor buffer mixing**
   - `training/trainer.py::train_step` — accept optional anchor buffer,
     mix `anchor_frac=0.2` of batch
   - `training/trainer.py::train_epoch` — forward anchor to each step
   - `training/train.py` — `--anchor-buffer` CLI flag, load buffer at
     start, forward to trainer

4. **Rebuild C++ extension**
   ```bash
   cd engine
   rm -rf build/temp.linux-x86_64-cpython-311 build/lib.linux-x86_64-cpython-311 go_engine.*.so
   python3 setup.py build_ext --inplace
   ```

5. **Write + run anchor buffer generator**
   - `training/gen_anchor_buffer.py` — new file, see step above
   - Run once:
     ```bash
     export PYTHONPATH="$PWD:$PWD/engine/build/lib.linux-x86_64-cpython-311:$PYTHONPATH"
     python3 -u -m training.gen_anchor_buffer \
         --checkpoint checkpoints/9x9_run1/preserved_iter0009.pt \
         --output checkpoints/9x9_run1/anchor_buffer_iter9.npz \
         --board-size 9
     ```
   - ~13 minutes
   - Verify the file exists, ~1.5 GB or so

6. **Rollback training target**
   - Do NOT delete `preserved_iter0011.pt`; keep it (it's small).
   - The actual restart uses it as `--checkpoint`.

7. **Clean up live state before restart**
   - Archive current log: `mv logs/9x9_run1.log logs/9x9_run1.log.pre_regression_fix`
   - Delete `latest_buffer.npz` if it exists in `checkpoints/9x9_run1/`
     (shouldn't, since the current run didn't have buffer persistence,
     but double-check)

8. **Restart command**

   ```bash
   cd /root/AI-ZeroToOne/Season-5/AlphaZero
   export PYTHONPATH="$PWD:$PWD/engine/build/lib.linux-x86_64-cpython-311:$PYTHONPATH" \
          PYTHONFAULTHANDLER=1 \
          TORCHINDUCTOR_COMPILE_THREADS=1
   nohup python3 -u -m training.train \
       --board-size 9 --iterations 73 --num-workers 5 \
       --checkpoint checkpoints/9x9_run1/preserved_iter0011.pt \
       --anchor-buffer checkpoints/9x9_run1/anchor_buffer_iter9.npz \
       --output-dir checkpoints/9x9_run1 \
       > logs/9x9_run1.log 2>&1 &
   ```

9. **Re-arm monitoring**
   - Kill any stale monitors from the previous session
   - Arm a persistent monitor on `logs/9x9_run1.log` for
     `^Iter |Fatal|nan|Segmentation|skipped`
   - Reset the 15-min cron to track the new pid

---

## Verification plan

After restarting from iter 11 with all three fixes, we need to
confirm the regression is fixed before celebrating. The symptom we're
hunting for is "weight drift that correlates with strength loss".

### Every 3–4 iters, check:

1. **Eval vs random using `eval_vs_random.py`** on the latest
   preserved checkpoint. Target ≥ 94% (iter 9's 96% minus a 2pp margin
   for 100-game binomial noise). If it drops below 90% we have a
   regression.

2. **Eval vs iter 9 using `eval_matchup.py`** (400 sims, 8 random ply,
   50 pairs). Target aggregate score ≥ 0.55 with the upper CI > 0.60.

3. **Weight audit using the one-liner from earlier:**
   ```python
   # Look for: policy_bn.max, input_conv.max should be close to iter 9
   # (~31, ~0.16). If they drift above 40 / 0.25, something is wrong.
   ```

### Red flags that mean "stop and retry with bigger hammer":

- `policy_bn.max` exceeds **40** at any iter → specialization is
  starting to recur. Stop, reduce `lr_init` from 0.01 → 0.005, consider
  raising `anchor_frac` to 0.30.
- `vs_random` drops below **90%** for 2 consecutive preserved-iter
  checks → real regression, stop and investigate.
- `training loss` starts dropping faster than 0.1 per iter → suspicious
  overfitting. Stop.

### Green lights that mean "keep going":

- `vs_random` stays in 92–97% across iters
- `policy_bn.max` stays in 30–40 range across iters
- `input_conv.max` stays in 0.15–0.25 range across iters
- Training loss drops gently and non-monotonically (lots of tiny
  oscillation = normal)

## Open questions (think about tomorrow)

1. **Should `anchor_frac` be 0.1, 0.2, or 0.3?** I picked 0.2 as a
   reasonable default. 0.3 would more aggressively anchor but dilute
   the current self-play signal. If we see BN drift return at 0.2, try
   0.3.
2. **Should we lower `lr_init` from 0.01 to 0.005?** Not in the
   initial plan but it's an option if anchor mixing alone doesn't stop
   the drift.
3. **Gating as a last resort.** If the anchor buffer + resign v2 +
   buffer persistence all together can't stop the regression, the next
   step is full AlphaGo-Zero-style gating: after each train step, run
   a paired eval vs the previous best and only accept the new weights
   if they win ≥55%. This is what the original paper does. Complex to
   implement but guarantees monotonic improvement.
4. **Should the anchor buffer be regenerated periodically?** E.g.,
   after iter 30 when the model is presumably much stronger, regenerate
   the anchor from iter 30 itself. That would update the "what Go
   positions are worth being good at" target. For now, just use iter
   9's anchor — we can reconsider after we see the fix working.
5. **Does it make sense to anchor against iter 11 instead of iter 9?**
   Iter 11 is the last pre-drift but we only verified iter 9 against
   random. For cleanliness of baseline I picked iter 9. Could generate
   both anchors and pick whichever gives better results.

## What NOT to do tomorrow

- **Don't try to "fix" training by adjusting hyperparameters alone**
  (lr, weight_decay, batch size). The root cause is distribution
  narrowing — fix the data, not the optimization.
- **Don't skip the anchor buffer** because "it's 13 minutes to
  generate". Without it, buffer persistence only preserves a narrow
  recent-iter distribution; it doesn't prevent further narrowing.
- **Don't re-remove the gradient clipping.** It was load-bearing for
  the iter 13 NaN crash; the drift here is a different failure mode.
- **Don't chase head-to-head matchup results above 0.55** as a
  success metric. At 400 sims, head-to-head masks policy-head drift.
  The vs-random metric is the source of truth for policy quality.
- **Don't delete any `preserved_iter*.pt` file** until the fix is
  confirmed and we have new preserved versions from the new run.

## Files that exist and should not be touched

- `checkpoints/9x9_run1/eval_opponent_iter0009.pt`
- `checkpoints/9x9_run1/preserved_iter00{10,11,12,15,16,17,18,19}.pt`
- `logs/9x9_run1.log` (soon to be archived)
- `training_log.jsonl` inside `checkpoints/9x9_run1/`

## Related code locations (quick reference)

```
engine/worker.h                          # GameSlot, complete_move, resign logic
engine/bindings.cpp                      # SelfPlayConfig bindings
engine/mcts.h                            # MCTSTree, select_leaf (MAX_PATH_DEPTH=256)
training/replay_buffer.py                # ReplayBuffer with 8-fold aug
training/trainer.py                      # SGD + grad clip + NaN guard
training/parallel_self_play.py           # ParallelSelfPlay orchestrator
training/train.py                        # main training loop, eval_vs_random
training/eval_matchup.py                 # paired head-to-head evaluator (v5)
training/eval_vs_random.py               # standalone vs-random eval
model/config.py                          # ModelConfig, TrainingConfig, CONFIGS
model/network.py                         # AlphaZeroNet (residual tower + heads)
```
