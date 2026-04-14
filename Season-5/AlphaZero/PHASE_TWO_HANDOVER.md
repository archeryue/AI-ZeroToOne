# Phase 2 — Handover for next session

**Status as of 2026-04-14 ~16:20 UTC:** nothing running, run1 and run2
both aborted after strength regressions, fix hypothesis partially
validated but insufficient. Need to try a different lever tomorrow.

Read `PHASE_TWO_TRAINING.md` for the full narrative and all four
Problems documented so far. This handover is the **short version**
with everything you need to resume.

---

## TL;DR

Phase 2 prep (uint8 buffer, persistence, tuning, dryrun v4) is done
and works. The **actual full training run won't converge** — the
value head destabilizes during the very first training step on cold
self-play data, and subsequent iters oscillate between "near-cold"
and "worse-than-cold" instead of improving. Two attempts made it as
far as iter 4 (run1) and iter 1 (run2) before we aborted. The
`value_loss_weight=2.0` fix committed on `main` dampens the drift
but does not cure it.

**Do NOT just re-run training and hope.** Try one of the hypotheses
at the bottom first.

---

## What exists on disk

### run1 (failed at iter 5 start, 5 % vs random at iter 4)

```
checkpoints/13x13_run1/
├── checkpoint_0000.pt .. checkpoint_0004.pt   # 5 iters, all preserved
├── latest_buffer.npz                          # iter 4 buffer, trimmed back to iter 0 size in attempt 3
├── training_log.jsonl                         # 5 iter entries
└── strength_audit.txt                         # per-ckpt vs-random results from audit
```

### run2 (failed at iter 2 start, 0 % vs random at iter 1)

```
checkpoints/13x13_run2/
├── checkpoint_0000.pt                         # after iter 0 training
├── checkpoint_0001.pt                         # after iter 1 training (regressed)
├── latest_buffer.npz                          # 1.74 GB, iter 1 end
└── training_log.jsonl                         # 2 iter entries
```

### Logs

```
logs/
├── 13x13_dryrun.log                           # v2, OOM'd
├── 13x13_dryrun_v4.log                        # v4, succeeded (ground truth iter)
├── 13x13_run1.log                             # run1 attempt 1 (the 35 GB abort)
├── 13x13_run1_a2.log                          # run1 attempt 2 (resume)
├── 13x13_run1_a3.log                          # run1 attempt 3 (ran to iter 5 start)
├── 13x13_run2.log                             # run2
├── 13x13_smoke_postfix.log                    # smoke test
├── 13x13_smoke_progress.log                   # smoke test with progress log
├── 13x13_smoke_vw.log                         # smoke test with value_loss_weight=2
└── ckpt_audit.log                             # per-checkpoint strength audit from run1
```

### Critical git commits on `main` (read these first)

```
09ce5e4 Phase 2 Problem 4: value-head cannibalization — weight value_loss 2x
4d50196 Phase 2: raise resign_min_move 20 → 40 for 13x13 + raise ceiling 35 → 48 GB
8bd5633 Phase 2 run 1 attempt 1: aborted mid-iter-1 — page cache accounting
ace6dd9 Phase 2 dryrun v4 measured results
d6f7f56 parallel_self_play: intra-iter progress log
058091d Phase 2: num_simulations 600 → 400 for wall-time
91193d0 Phase 2 tuning: 1M tree cap + restore 256 parallel games
cc1a69c Phase 2 dryrun OOM fix: cap MCTS tree nodes per game
```

All are on `origin/main`. `PHASE_TWO_TRAINING.md` contains the full
diagnostic narrative for each.

---

## The data we have (both runs combined)

| run | iter | avg_mv | pi_loss | v_loss | eval wr | notes |
|---|---:|---:|---:|---:|---:|---|
| run1 | 0 | 144 | 5.1454 | **0.8284** | 20 % | baseline |
| run1 | 1 | 41 | 5.0451 | **0.9113** ⚠️ | 2 % | **catastrophic** |
| run1 | 2 | 94 | 4.9799 | **0.8957** | 18 % | recovered |
| run1 | 3 | 154 | 4.8908 | **0.9206** ⚠️ | 0 % | crashed again |
| run1 | 4 | 155 | 4.8200 | **0.9248** ⚠️ | 8 % | mild recovery |
| run2 | 0 | 153 | 5.1346 | **0.8723** | 1 % | lower baseline (why?) |
| run2 | 1 | 86 | 5.0516 | **0.9207** ⚠️ | 0 % | same drift, smaller Δ |

**Two consistent patterns across both runs:**

1. **Policy loss monotonically drops.** Policy head trains fine.
2. **Value loss oscillates upward.** Zig-zag around a slowly-rising
   trend. `value_loss_weight=2.0` dampened the rise (+0.048 vs
   +0.083 between iter 0 and iter 1) but did not reverse it.

**One weird data point we don't understand:**

- run1 iter 0 eval was 20 %; run2 iter 0 eval was 1 % under
  otherwise-identical conditions.
  - Both use random init (no `torch.manual_seed`) → weights differ.
  - Both use unseeded `np.random` in the random-player evaluator →
    random's moves differ.
  - run1 used `value_loss_weight=1`; run2 used `2.0` → the 100-step
    iter-0 training produced meaningfully different checkpoints.
  - Part of the 19 pp gap is eval variance; part of it is real.
  - **Don't read too much into a single iter-0 eval.** Set a seed
    next time.

---

## What's committed and in effect in `main`

`model/config.py`, 13x13 preset (as of `09ce5e4`):

```python
CONFIGS[13] = (
    ModelConfig(board_size=13, num_blocks=15, channels=128),
    TrainingConfig(
        num_simulations=400,             # down from 600 (wall-time)
        dirichlet_alpha=0.07,
        num_games_per_iter=2048,
        num_parallel_games=256,
        buffer_size=1_000_000,
        max_game_moves=250,
        temperature_moves=40,
        temperature_low=0.25,
        lr_init=0.005,
        resign_min_move=40,              # up from 20 (Problem 3)
        value_loss_weight=2.0,           # up from 1.0 (Problem 4, PARTIAL FIX)
        eval_interval=1,                 # for run2 visibility
        checkpoint_interval=1,
    ),
)
```

`engine/worker.h`:

```cpp
static constexpr int MAX_TREE_NODES = 1000000;
```

`training/parallel_self_play.py::run_games` has the intra-iter
progress-log patch (every 30 s).

`training/_eval_checkpoints.py` is the per-checkpoint strength
audit script. Rerunnable:

```bash
PYTHONPATH=engine python training/_eval_checkpoints.py
```

Edit the `ckpt_dir` variable to point at whichever run's
checkpoints you want to audit.

---

## What to try next (ranked by expected impact, lowest risk first)

**Before launching ANYTHING, set a torch manual seed** so results
are reproducible. Add to `train.py`:

```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

Then pick one lever at a time and test it for 2-3 iters:

### 1. ⭐ Cut `train_steps_per_iter` 100 → 20 (recommended first)

**Hypothesis:** 100 SGD steps per iter is too many for a
15b×128ch net fitting to noisy cold-self-play value targets. The
value head drift happens inside the first few tens of steps; the
rest amplifies it. 20 steps may be enough to make a small
consistent improvement without drift.

**Change:** add `train_steps_per_iter=20` to the 13x13 preset.
Run 2-3 iters and check if `v_loss` stops rising.

**Cost:** ~same wall time per iter (training is <3 s of each
40-min iter), but total learning over 60 iters drops 5× so you
may need more iters total.

### 2. Value-loss weight WARMUP

**Hypothesis:** 2.0 is wrong for early iters (when the value
targets are noisy) but right for later iters (when they're
informative). Start with a lower value.

**Change:** replace the scalar `value_loss_weight` with a schedule:

```python
# in trainer.train_epoch, after the LR schedule:
vw_warmup_iters = 5
vw_start, vw_end = 0.25, 2.0
progress = min(self.total_steps / (vw_warmup_iters * self.cfg.train_steps_per_iter), 1.0)
self.cfg.value_loss_weight = vw_start + progress * (vw_end - vw_start)
```

Equivalent to "almost no value training for the first 5 iters,
ramp up after".

### 3. Lower `lr_init` 0.005 → 0.001

**Hypothesis:** 5× smaller steps in iter 0's training reduce
per-step weight movement, which limits how far the value head
can drift from its cold-init position in one iter.

**Change:** `lr_init=0.001` in the 13x13 preset.

**Cost:** 5× slower learning across the whole run. Would need
more iters.

### 4. Huber loss for value head

**Hypothesis:** MSE penalizes extreme outlier value targets
quadratically. On cold self-play the game outcome is ±1 and
predictions are near 0, so MSE gradients are large and
destabilizing. Huber (L1 for large errors, L2 for small) reduces
sensitivity to those outliers.

**Change:** in `trainer.train_step`, replace

```python
value_loss = F.mse_loss(value, target_value)
```

with

```python
value_loss = F.huber_loss(value, target_value, delta=0.5)
```

### 5. Batch normalization audit / switch to GroupNorm

**Hypothesis:** BN running stats drift when the distribution of
training batches shifts between iters. On 13x13 with heterogeneous
cold-self-play data, the running mean/var might be updating in
ways that hurt inference-time predictions.

**Change:** audit `model/network.py` for BN layers, consider
replacing with `nn.GroupNorm` (no running stats) or setting
`track_running_stats=False`. More invasive, save for last.

### 6. Give up on Zero training from scratch on 13x13 with this recipe

The fallback plan is **supervised pretraining** from the 9x9
`9x9_run2/model_final.pt` weights, transferred to a 13x13 board
(with the appropriate tower expansion). The 9x9 net has a trained
value head; transferring its weights would avoid the cold-start
instability entirely.

This is a ~1-day project (architecture transfer code + validation)
and only worth doing if the simpler fixes all fail.

---

## Concrete resume command once a fix is picked

Fresh run, new output dir, no resume:

```bash
cd /root/AI-ZeroToOne/Season-5/AlphaZero

# confirm engine is built and imports
PYTHONPATH=engine python -c "import go_engine; print(go_engine.__file__)"

# run the correctness smoke test (should pass, ~30s)
PYTHONPATH=engine python training/_test_correctness.py

# run the tree-cap test (should pass, ~30s)
PYTHONPATH=engine python training/_test_tree_cap.py

# 13x13 smoke test with progress log (should pass, ~1 min)
rm -rf checkpoints/13x13_smoke_next
PYTHONPATH=engine python -m training.train \
    --board-size 13 --smoke-test \
    --output-dir checkpoints/13x13_smoke_next

# full run (use a new dir name per attempt — 13x13_run3, _run4, ...)
mkdir -p checkpoints/13x13_run3 logs
rm -f /tmp/run3.pid
PYTHONPATH=engine nohup python -m training.train \
    --board-size 13 --iterations 60 \
    --output-dir checkpoints/13x13_run3 \
    > logs/13x13_run3.log 2>&1 &
echo $!
# grab the actual python PID (not the bash wrapper) from ps --forest
ps --forest -e -o pid,ppid,cmd | grep training.train
```

Use the checkpoint audit script after each run to get the
per-iter strength curve:

```bash
# edit _eval_checkpoints.py line 44: ckpt_dir = "checkpoints/13x13_run3"
PYTHONPATH=engine python training/_eval_checkpoints.py
```

---

## Monitoring cheat-sheet for the next session

- **Memory:** `cgroup memory.usage_in_bytes` (at
  `/sys/fs/cgroup/memory/`), **abort at 50 GB**. Use
  `memory.stat: rss` as the real-anon number if the total looks
  inflated by cache.
- **Real hard limit:** 62 GB cgroup, `oom_kill_disable=1` (process
  freezes, doesn't crash).
- **GPU:** `nvidia-smi`. Expect 70-85 % util, ~1 GB VRAM.
- **Progress:** tail `logs/13x13_runN.log` for `| sp:` lines
  (every 30 s) and `Iter N |` summaries.
- **training_log.jsonl:** per-iter metrics in structured form —
  the authoritative source. Use the python one-liner in this doc
  to parse.
- **Faulthandler 5-min stack dumps** in the log are NOT errors,
  they're periodic health checks. Main thread in `cuda.synchronize`
  inside `run_games` is normal self-play.

### Key decision rules

- **v_loss rising iter-over-iter** = the old Problem 4 drift is
  still there. The fix being tested isn't enough. Abort and try
  the next lever.
- **eval vs random stuck at 0-5 % for >3 iters** = net isn't
  learning or is actively deteriorating. Abort.
- **avg_moves/game < 80** = value head is probably triggering
  early resigns / pass loops. Abort.
- **cgroup > 48 GB** = tighten monitoring, abort at 50 GB.
- **total loss dropping + policy loss dropping + value loss
  dropping + eval rising** = fix is working, let it cook.

---

## Things that are NOT the problem (don't re-investigate)

These were ruled out conclusively this session:

- ❌ uint8 buffer leak — audited, zero changes needed, actually
  SAVES memory vs float32
- ❌ MCTS tree memory leak — bounded by `MAX_TREE_NODES=1M`,
  stable at steady state
- ❌ Page cache inflating memory numbers — it does, but it's
  reclaimable and doesn't matter for anon rss
- ❌ Early-resign data bias — `resign_min_move=40` fixed this,
  games are normal length from iter 0 onwards (before training
  breaks them)
- ❌ OOM at 35 GB — the real cgroup limit is 62 GB, operational
  budget raised to 50 GB
- ❌ Python/C++ worker memory leaks — memory behavior is stable
  across iters, it just LOOKS like growth during spawn/warmup

---

## Environment quick-check (for a fresh shell)

```bash
cd /root/AI-ZeroToOne/Season-5/AlphaZero
git status                      # should be clean or show only .claude/
git log --oneline -10            # should show 09ce5e4 as HEAD
ls -la engine/go_engine.*.so    # should exist
PYTHONPATH=engine python -c "import go_engine; print(go_engine.SelfPlayWorker13.MAX_TREE_NODES)"  # should print 1000000
python -c "from model.config import CONFIGS; _, tc = CONFIGS[13]; print(tc.value_loss_weight, tc.resign_min_move)"  # should print 2.0 40
nvidia-smi | head -15           # GPU should be idle / free
python3 -c "print(open('/sys/fs/cgroup/memory/memory.usage_in_bytes').read())"  # should show baseline ~1-3 GB
```

If any of these fail, investigate before launching.

---

## One thing to do FIRST tomorrow

Pick ONE lever from "What to try next" above. **Don't change
multiple things at once** — we need to know which lever moved the
needle. My recommendation: start with lever #1
(`train_steps_per_iter=20`) because it's the simplest to implement
(one number change), has a clear hypothesis behind it, and is
maximally conservative (does less training per iter, can't make
things worse than doing zero training).

Run 3 iters, look at the strength curve from `_eval_checkpoints.py`,
decide. 3 iters ≈ 2 hours wall time. If v_loss stops rising and
eval stays ≥ cold baseline, let it run longer. If not, try the
next lever.

**Don't commit to a 60-iter run until you have 3 consecutive iters
with v_loss flat-or-dropping AND eval non-regressing.** That's the
go/no-go gate for a full production run.

---

Good luck tomorrow.
