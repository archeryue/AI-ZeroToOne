# Phase 2 — Handover

**Status as of 2026-04-15 ~08:35 UTC:** nothing running. Run4b made
it through iter 0 + iter 1 cleanly with a **genuinely healthy
training trajectory**, then died mid-iter-2 twice in a row on a
mystery silent kill. The model architecture + recipe is validated;
the loop wrapper needs hardening for the shared-host environment
before any more training can complete.

Read `PHASE_TWO_TRAINING.md` for the full 3-section narrative
(Preparation / OOM fixes / Regression fix). This handover is the
short version + what to do next.

---

## TL;DR

- **Architecture works**: KataGo-style ownership head + derived
  value (2 learnable scalars, no value MLP).
  Iter 0 eval vs random = **60 %**, iter 1 training losses all
  dropped (own 0.602 → 0.516, v 0.994 → 0.932, pi 5.169 → 5.076),
  iter 1 self-play avg = **182 moves/game** (vs 173 at iter 0 — no
  collapse, cleanest iter-1 behavior across run1/2/3/4b).
- **Loop is fragile**: the training process keeps dying silently
  with no Python traceback, no OOM, no core dump. Run4 iter 0 eval,
  run4b iter 1 eval, run4b resume iter 2 self-play (twice) —
  different phases, same no-diagnostic-signal death pattern. Host
  load avg is 5–6 → shared-host kills are the leading hypothesis
  but I cannot prove it.
- **Preserved on disk**: `checkpoint_0000.pt`, `checkpoint_0001.pt`,
  `latest_buffer.npz` (728k positions, verified intact, 173 avg
  moves in iter 0 slice, 182 in iter 1 slice). **Nothing is
  corrupted.** The next AI should resume from `checkpoint_0001.pt`.
- **Most promising next step**: make the training loop resilient to
  external kills — most likely by running self-play in a child
  subprocess with checkpoint-based checkpoint/resume on every iter
  boundary. That way a kill during any iter loses at most one iter
  of work, and the parent can restart the child automatically.

---

## What's on disk right now

### `checkpoints/13x13_run4b/`

```
checkpoint_0000.pt       36 MB   iter 0 trained weights (30 SGD steps, 60 % eval)
checkpoint_0001.pt       36 MB   iter 1 trained weights (60 SGD steps, losses all dropped)
latest_buffer.npz       2.7 GB   iter 0 + iter 1 buffer, 728,158 positions
training_log.jsonl       512 B   iter 0 entry only (iter 1 died before log write)
```

The buffer was **verified intact** by loading it and inspecting:

```
size: 728158, index: 728158
value: mean=+0.000 std=1.000 +1 frac=0.500
ownership: +1/-1/0 frac = 0.452/0.452/0.096
iter 0 slice (first 355569 / 2050 games) = 173.4 moves/game  ✓
iter 1 slice (last  372589 / 2050 games) = 181.8 moves/game  ✓
```

These numbers exactly match what the live logs printed, so the
buffer on disk is the genuine iter-0 + iter-1 self-play data, not
corrupted.

### Logs

```
logs/13x13_run4.log           run4 (iter 0), died mid-eval
logs/13x13_run4b.log          run4b (iters 0+1), died iter-1 mid-eval — has the 60 % result
logs/13x13_run4b_resume.log   resume attempt 1 (iter 2 self-play), short games, I aborted mid-iter
logs/13x13_run4b_resume2.log  resume attempt 2 (iter 2 self-play), died at ~5 min into self-play
```

### Critical commits on `main`

```
64347bc Phase 2 run4b: gate in-loop eval behind --eval-in-loop flag (default off)
71fb68a Phase 2 run4b: document the mystery crash + iter 0 breakthrough
ac7751d Phase 2 run4b: train.py resilience + HARDWARE_NOTES.md
5ac11a0 Phase 2: restructure PHASE_TWO_TRAINING into 3 sections, trim trivial content
844bb3e Phase 2 run4: KataGo-style ownership head + value derived from ownership
```

All on `origin/main`. Read `PHASE_TWO_TRAINING.md` for the 3-section
narrative (prep / OOM fixes / regression fix) and
`HARDWARE_NOTES.md` for the GPU-selection analysis.

---

## What's validated (you can trust these)

| claim | evidence |
|---|---|
| Ownership head implementation is correct | `engine/go.h::compute_ownership()` passes multi-color flood-fill tests in `_test_correctness.py` and `_test_tree_cap.py` |
| Derived-value architecture trains | Smoke test shows own loss 0.69 → 0.65 → 0.62 monotonic, arithmetic `pi + 0·v + 2·own = total` verified |
| `vlw=0, ow=2.0, steps=30` is the right recipe | Run4 offline A/B: A6 recipe achieved held-out v_mse = 0.9631 (below the 1.02 cold floor), ONLY recipe across run1/2/3/4 offline A/Bs with Δv < 0 |
| Iter 0 / iter 1 training is healthy | Losses all dropped iter 0→1 (pi −0.09, v −0.06, own −0.09), game length went 173 → 182 (no resign collapse), eval 60 % vs random |
| Replay buffer is intact | Loaded and inspected directly — 728,158 positions matching the exact iter 0 + iter 1 stats from the live log |
| Memory is fine | cgroup v1 cap on this host is **87.5 GiB** (not 62, re-measured), run4b peaked at ~43 GB — ~45 GB headroom |

---

## What's broken (the next problem)

**The training process keeps dying silently.** No traceback, no
OOM counter, no core dump. Three incidents so far:

| incident | when | phase | main thread frame when faulthandler last fired |
|---|---|---|---|
| run4 iter 0 | 2026-04-15 05:51 | **eval** | `batchnorm.forward` |
| run4b iter 1 | 2026-04-15 07:58 | **eval** | `conv.forward` |
| run4b resume iter 2 (2nd attempt) | 2026-04-15 08:30 | **self-play** | truncated mid-dump |

The run4 and run4b eval crashes were worked around by gating
in-loop eval behind `--eval-in-loop` (default off). But the resume
attempt ALSO died, this time during self-play, at ~5 min in — which
means the in-loop eval gating didn't fix the root cause, it just
moved the crash to a different phase.

**Everything I know that points at external kill:**

- cgroup `memory.failcnt = 0`, `oom_kill = 0`, `oom_kill_disable = 1`
- memory usage at death was ~25–35 GiB, well under the 87.5 GiB cap
- host load average at time of death: **5–6** (`uptime` shows
  `load average: 4.87, 5.58, 6.11`) → significant contention from
  other tenants on this shared container
- **No Python exception path fired** — the `except BaseException`
  handler in `train.py::main` would have printed a traceback. It
  didn't. That's only possible with an external SIGKILL.
- No core dump files in `/tmp/` or `/root/`
- `dmesg` is not readable in this container, `/var/log/syslog` not
  present — can't verify externally

**Everything I verified that is NOT the cause:**

- Not a network/architecture bug (isolated eval repro via
  `training/_debug_eval_crash.py` ran all 50 games cleanly on
  `checkpoint_0001.pt` with flat GPU memory 43 MB alloc / 68 MB
  reserved; no leak, no deadlock, no crash)
- Not a memory leak (resume process RSS was 19–35 GB, stable; never
  hit the 87.5 cap)
- Not `torch.compile` — survived 60 minutes in earlier runs
- Not the buffer (verified)
- Not `compute_ownership` (tests pass)
- Not the derived-value head (smoke + A/B + live iters 0+1 all work)

**Leading hypothesis**: **host-level OOM killer or container runtime
SIGKILL on a shared machine.** When global memory pressure rises,
the kernel OOM killer picks the largest-RSS process — our training
process (~30 GB) is always in the top-3 candidates. `oom_kill_disable`
only applies to the cgroup's own OOM path, not to the host kernel
hitting the container.

**Second hypothesis**: some GPU driver / CUDA state issue that
manifests only after ~5 min of sustained self-play. Less likely
given the cross-phase consistency.

---

## What to try next (ranked by what I think is most worth doing first)

### 1. ⭐ Subprocess-isolated training loop (recommended)

The problem is that **one silent kill takes out ALL training
progress, including the iter we were mid-way through**. Fix: run
each iter's self-play in a subprocess, checkpoint between iters
from the parent.

Concrete design:

- `train.py` outer loop (parent) — loads net + buffer, saves
  checkpoint/buffer, forks a subprocess for each iter's self-play
  phase, reads back the new positions from a pipe or temp file,
  trains in-process, saves checkpoint/buffer.
- `self_play_subproc.py` (child) — loads the latest checkpoint +
  network weights, instantiates `ParallelSelfPlay`, runs 2048
  games, writes the harvested (obs, policy, value, ownership)
  tuples to a temp file, exits.
- If the child dies silently, the parent retries. Parent only
  loses the child's in-progress games, not training state.

Cost: ~100–150 lines of code + testing. Biggest payoff if the host
keeps killing us.

### 2. Smaller per-iter footprint

If the host kills us because we're the biggest tenant, shrink
ourselves:

- **Net**: 15b × 128ch → 10b × 128ch (one config line). Cuts
  forward-pass compute ~33 %, RSS roughly −3 GB, iter wall time
  −33 %. Needs re-validation of recipe (offline A/B + smoke + one
  live iter).
- **`num_parallel_games`**: 256 → 128. Cuts MCTS tree memory
  ~10 GB. Halves batch size, roughly same tick rate.
- **`num_simulations`**: 400 → 300. ~25 % faster per iter. Off-spec
  but viable.

Any of these alone drops peak RSS by 5–15 GB. Combined, could get
us under 20 GB RSS where we're no longer a top-3 OOM candidate on
this host.

### 3. Move to a less contended host

This is not a code problem; it's an environment problem. Running
the same job on a dedicated 4090 (not a shared container with load
avg 6) would almost certainly work with the existing code.

- RunPod / Vast.ai / Lambda single-tenant 4090 pod: ~$0.30–0.50/hr
  cloud. See `HARDWARE_NOTES.md` for the cost analysis.
- 20 iters × ~50 min = ~16 hours × $0.40/hr ≈ **$6.50** to complete
  a "good enough" training run.
- **This is the cheapest, fastest path to a trained model if the
  host contention is the root cause.**

### 4. If none of the above — the resign spiral is real

In the first resume attempt (before I aborted it), iter 2 self-play
was producing **~50 avg moves/game** because the iter-1-trained
ownership head is confident enough that derived value crossed −0.9
at move 40+ often enough to trigger resigns. I raised
`resign_threshold: −0.90 → −0.95` and `resign_min_move: 40 → 80`
(commit in current working tree but NOT yet pushed). That run died
before I could confirm it helped.

**If you see iter 2 (or any later iter) producing short games
(~50 avg moves) after a clean resume**, that's the resign spiral,
not a new bug. Options:

- Raise `resign_min_move` further (80 → 120)
- Raise `resign_threshold` further (−0.95 → −0.98)
- Disable resign entirely in training: set
  `resign_disabled_frac: 0.20 → 1.0` on the 13×13 preset

---

## Concrete resume command (current code, no changes)

```bash
cd /root/AI-ZeroToOne/Season-5/AlphaZero

# 1. Sanity
PYTHONPATH=engine python -c "import go_engine; print(go_engine.__file__)"
PYTHONPATH=engine python -c "from model.config import CONFIGS; _, t = CONFIGS[13]; print('resign_threshold', t.resign_threshold, 'resign_min_move', t.resign_min_move, 'ow_w', t.ownership_loss_weight, 'vlw', t.value_loss_weight, 'steps', t.train_steps_per_iter)"
# Expected: -0.95 80 2.0 0.0 30  (run4b resume2 config, in working tree, NOT yet pushed)

# 2. Build engine if needed
cd engine && python setup.py build_ext --inplace && cd ..

# 3. Smoke
PYTHONUNBUFFERED=1 PYTHONPATH=engine python -m training.train --board-size 13 --smoke-test --output-dir checkpoints/13x13_smoke_check

# 4. Resume from iter 1 checkpoint
PYTHONUNBUFFERED=1 PYTHONPATH=engine nohup python -m training.train \
    --board-size 13 --iterations 60 \
    --output-dir checkpoints/13x13_run4b \
    --checkpoint checkpoints/13x13_run4b/checkpoint_0001.pt \
    > logs/13x13_run4b_resumeN.log 2>&1 &

# 5. Monitor
tail -f logs/13x13_run4b_resumeN.log
```

**Note:** the working tree has uncommitted changes to
`model/config.py` (resign_threshold -0.95, resign_min_move 80) from
my last retune attempt. Either commit them or revert with
`git checkout model/config.py` — your call.

---

## Things that are NOT the problem (don't re-investigate)

- ❌ **Network architecture bug** — smoke test passes, offline A/B
  A6 achieved Δv < 0 (first ever), live iters 0+1 learn healthily
- ❌ **Replay buffer corruption** — verified intact with exact iter
  0/1 stats
- ❌ **uint8 obs storage** — audited in run3, zero issues
- ❌ **MCTS tree leak** — bounded by MAX_TREE_NODES=1M
- ❌ **cgroup OOM** — failcnt=0, oom_kill=0 verified
- ❌ **`compute_ownership` sign convention** — verified by multi-
  color test in debug, value head calibration is correct
- ❌ **torch.compile / compile worker deadlock** — ruled out by
  isolated eval repro
- ❌ **"Value-head cannibalization" hypothesis from the original
  handover** — this was refuted by run3's A/B (see
  PHASE_TWO_TRAINING.md section 3, "value-head cannibalization
  (wrong diagnosis v1)")
- ❌ **Early-resign data bias in iter 1** — fixed by raising
  resign_min_move, but this was a band-aid; the REAL problem at
  that time was the value MLP overfit that run4 fixed

---

## Diagnostic tools already wired

```bash
# Test ownership / Tromp-Taylor computation
PYTHONPATH=engine python training/_test_correctness.py

# Test MCTS tree cap + ownership plumbing through harvest
PYTHONPATH=engine python training/_test_tree_cap.py

# Reproduce the eval crash in isolation (used to rule out network
# bug, runs fine — confirms eval is not the intrinsic problem)
NUM_GAMES=50 PYTHONPATH=engine python training/_debug_eval_crash.py

# Run4 offline A/B harness (single-iter recipes vs cold buffer)
PYTHONPATH=engine python training/_phase2_run4_offline_ab.py gen-buffer
PYTHONPATH=engine python training/_phase2_run4_offline_ab.py run-ab

# Post-hoc strength audit of saved checkpoints (edit ckpt_dir inside)
PYTHONPATH=engine python training/_eval_checkpoints.py
```

`faulthandler` is wired in `train.py:29-37` — dumps all Python
thread stacks on any fatal signal and every 5 min to stderr. The
5-min dumps look scary but are normal heartbeats.

---

## Key state snapshot

```
Date:              2026-04-15 ~08:35 UTC
Host:              RTX 4090, cgroup limit 87.5 GiB, shared (load avg 5-6)
Torch:             2.4.1+cu124
Python:            3.11.10
CUDA driver:       570.195.03
Nothing running.   Last process (PID 11612) died at ~08:30.

Working tree:      model/config.py has uncommitted edits
                   (resign_threshold=-0.95, resign_min_move=80)
origin/main head:  64347bc  (eval-in-loop flag commit)

Best checkpoint:   checkpoints/13x13_run4b/checkpoint_0001.pt
                   - iter 1 trained weights (60 SGD steps total)
                   - healthy losses: pi=5.076, v=0.932, own=0.516
                   - last known strength: iter 0 eval showed 60 %
                     vs random
Buffer:            checkpoints/13x13_run4b/latest_buffer.npz
                   728,158 positions, verified intact
```

---

## One thing to do FIRST

**Read `PHASE_TWO_TRAINING.md` end-to-end** before making any
changes. It's 980 lines but organized into three sections — you
can skim sections 1 and 2 and read section 3 in detail. The run4
narrative (ownership head, offline A/Bs, derived value architecture)
is where all the non-obvious decisions live and the ones that
matter for understanding why things are the way they are.

Then decide: subprocess-isolated loop (option 1), smaller footprint
(option 2), different host (option 3), or a tighter retune (option
4). In my honest opinion, **option 3 is the cheapest and highest-
probability path to a completed training run** — this host's
contention is an environmental problem that no amount of code
cleverness can fix reliably.

Good luck.
