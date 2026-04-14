# run_artifacts

Small, git-tracked copies of the authoritative numerical data from
Phase 2 runs. The full `checkpoints/` and `logs/` directories are
excluded by `.gitignore` (checkpoint files are 36 MB each, buffer
files are 1-3 GB, logs are large text) — this folder preserves the
text-sized metadata that doesn't fit in git normally.

## Files

- **`dryrun_v4_training_log.jsonl`** — the successful 1-iter dryrun
  that validated the memory fix + config before launching run1.
  Single iter, ~450 bytes. Ground-truth iter time: 2496 s = 41.6 min.
- **`run1_training_log.jsonl`** — run1 (resign_min_move=20 → 40 fix,
  `value_loss_weight=1.0` original). 5 iters. Strength oscillated
  20 % → 2 % → 18 % → 0 % → 8 % vs random.
- **`run1_strength_audit.txt`** — post-run per-checkpoint eval from
  `training/_eval_checkpoints.py`. Source of the "run1 was
  oscillating, not monotonically failing" diagnosis.
- **`run2_training_log.jsonl`** — run2 (`value_loss_weight=2.0`
  partial fix). 2 iters before abort. Same direction failure,
  smaller magnitude: +0.048 v_loss rise vs run1's +0.083.

## How to read

Each line in a `.jsonl` file is one iter's full record:

```python
import json
for line in open('run1_training_log.jsonl'):
    r = json.loads(line)
    print(r['iteration'], r['train'], r.get('eval'))
```

Key fields per iter:
- `self_play.games` / `self_play.positions` → compute `avg_moves = positions / games`
- `self_play.time` → iter self-play wall time (seconds)
- `train.policy_loss`, `train.value_loss`, `train.loss` (**total is weighted** in run2; raw pi and v are the apples-to-apples numbers)
- `train.skipped_total` → cumulative grad-skip count (should be 0 or tiny)
- `eval.vs_random_winrate` → per-iter strength (present when `eval_interval` fires)

## Why these files matter for tomorrow

The `PHASE_TWO_TRAINING.md` and `PHASE_TWO_HANDOVER.md` docs contain
the numbers in markdown tables, but those are **transcribed
summaries** — they lose float64 precision and the full context of
each iter. If tomorrow we want to plot a loss curve or compute a
precise ratio, the raw `.jsonl` here is the source of truth.

If the training sandbox disk is wiped (as the user noted, "we will
lose everything other than git"), these 4 files are what survive.
The actual network weight checkpoints do not survive, but they're
also not reusable — we've established that any run1/run2 checkpoint
past iter 0 is contaminated, and iter 0 itself is regeneratable from
a fresh cold-init run.
