# Distributed Training Plan — Go AlphaZero, Community Edition

A KataGo/Leela-Zero-style volunteer-driven training system for our Go
AlphaZero. Goal: break out of the single-4090 / 10K-game regime by letting
anyone contribute their GPU, so that over ~1 year we can reach meaningful
19×19 strength.

This document is the *design* plan. Implementation is deferred — we want to
agree on the shape first.

---

## Mission

Turn our single-box AlphaZero project into a community-trained system:

- **Anyone with a GPU can contribute** self-play games (KataGo/Leela model).
- **No dedicated GPU runs 24/7.** The coordinator lives on a small GCP VM;
  the trainer is *also a task* picked up by a whitelisted client (initially
  just the project owner's local 5060 Ti 16GB).
- **Target**: 19×19 AlphaZero, reached over ~12 months of community compute.

The motivating asymmetry (from `KATAGO_PROJECT.md:333`):

> KataGo has ~93M self-play games from 1,382 volunteers. We have ~10K games
> from 1 GPU. That is a 9,300× data deficit — and it is the single reason
> our auxiliary heads overfit and their value MLP does not. This is a
> data-scale problem, not an algorithm problem.

Self-play is embarrassingly parallel (`HARDWARE_NOTES.md:228`). Volunteers
are the only scaling lever with compound returns for 19×19.

---

## System Architecture

Three loosely-coupled services, coordinated via HTTPS + object storage:

```
┌───────────────────────────────┐
│  GCP (small, always-on)       │      e2-small VM + GCS bucket
│  ────────────────────────     │
│  • Coordinator API (FastAPI)  │
│  • Postgres (jobs, stats)     │
│  • Static site / leaderboard  │
│  • GCS: weights + buffer      │
└──────────────┬────────────────┘
               │  HTTPS
               │
   ┌───────────┴──────────────────────────────────┐
   ▼                    ▼                         ▼
┌────────┐        ┌────────────┐          ┌──────────────┐
│ Owner  │        │ Volunteer  │   ...    │ Spot GPU     │
│ 5060Ti │        │ 3090/4090  │          │ (fallback)   │
│        │        │            │          │              │
│ SP +   │        │ Self-play  │          │ Self-play or │
│ Train  │        │ only       │          │ Training     │
└────────┘        └────────────┘          └──────────────┘
  whitelisted       anyone                 coordinator-
  for training      with auth              managed
```

Three task types flow through the coordinator:

| task type      | who picks it up       | frequency         | output                   |
|----------------|-----------------------|-------------------|--------------------------|
| `selfplay_job` | any authed client     | continuous, N/hr  | training rows (.npz)     |
| `rating_job`   | any authed client     | per candidate net | win/loss vs incumbent    |
| `training_job` | whitelisted trainers  | when buffer fills | new candidate network    |

---

## Why Training Is Also a Task (And Why That Is Not "Distributed SGD")

An earlier draft argued against distributing training. That argument was
specifically against **distributing within one SGD step** (gradient
all-reduce over WAN). What we are doing here is different: we distribute
**which machine runs each self-contained training iter**.

```
REJECTED — distributed SGD:
     ┌─────┐      ┌─────┐      ┌─────┐
     │ GPU │ ←──► │ GPU │ ←──► │ GPU │   gradient all-reduce, ~90 MB/step
     └─────┘      └─────┘      └─────┘   WAN-hostile, non-deterministic,
        all running simultaneously        stale-gradient hell

ADOPTED — training-job-as-a-task:
     ┌────────────────────────────────────┐
     │  Trainer client (one at a time)    │
     │  1. pull current weights (~90 MB)  │
     │  2. pull optimizer state (~90 MB)  │
     │  3. pull replay buffer sample      │
     │  4. run 100 SGD steps locally      │
     │  5. upload new weights + optim     │
     └────────────────────────────────────┘
       runs on 5060Ti, or friend's box,
       or GCP spot GPU as fallback
```

No gradient traffic between machines. No stragglers. Iter K fully completes
on one machine before iter K+1 starts on (possibly) another. Output is
gated by match games before it replaces the incumbent network.

This is the pattern KataGo implicitly uses — their trainer happens to
always be the same box, but nothing in the design requires that.

---

## Why This Works on a 5060 Ti 16GB

From `HARDWARE_NOTES.md:159`, the 15b×128ch net used ~1 GB VRAM during
training. For the worst case (19×19, 20b×256ch, 23M params, BF16):

| component                              | VRAM   |
|----------------------------------------|--------|
| model weights (BF16)                   | ~45 MB |
| gradients (BF16)                       | ~45 MB |
| SGD momentum state                     | ~90 MB |
| batch 256 activations (BF16)           | ~2–3 GB |
| PyTorch / CUDA reserved                | ~1–2 GB |
| **total**                              | **~4–5 GB** |

16 GB leaves ample room. Self-play on this same card (MCTS trees + 256
parallel games) is strictly heavier; if the card handles self-play it
handles training.

---

## Components

### 1. Coordinator (GCP, always-on)

Stack: FastAPI + Postgres + GCS. Runs on `e2-small` (~$7/mo).
Stateless; all durable state in Postgres + object storage.

Endpoints (minimal v1):

```
GET  /api/v1/network/current           → {net_id, url, sha256, sig}
GET  /api/v1/network/<id>              → binary (cached via Cloudflare)

POST /api/v1/job/request                → client claims a job
  body  : {client_id, capabilities: [selfplay, training?], ...}
  reply : {job_id, type, config, net_id, seed, ...}

POST /api/v1/job/heartbeat              → keep lease alive
POST /api/v1/job/submit                 → upload artifact + stats
POST /api/v1/job/abandon                → give up a job

GET  /api/v1/leaderboard                → volunteer stats
GET  /api/v1/status                     → pipeline health page
```

Job lifecycle: `queued → leased → submitted → validated → (accepted | rejected)`
with 30-minute lease timeouts.

### 2. Client (runs anywhere)

Shipped as a pinned Docker image (Linux+CUDA first; Metal/ROCm later if
there is demand). Single CLI:

```bash
docker run --gpus all ghcr.io/<proj>/az-client:v1.2.3 \
    --token $AZ_TOKEN --capabilities selfplay,training
```

Internally, self-play reuses `ParallelSelfPlay` mostly as-is. The only
refactor required on our side is Phase 0 below.

### 3. Trainer

Logically the same as our current `training/train.py`, but the
self-play step is replaced by "pull the last K iters of rows from
GCS". Trainer is invoked by the client when a `training_job` is claimed:

```python
def run_training_job(job):
    weights    = download(job.current_net_url)
    optim      = download(job.optim_state_url)
    rows       = sample_from_gcs(job.buffer_window, n=job.batch * job.steps)
    new_w, new_o, stats = train_k_steps(weights, optim, rows, job.steps)
    upload(new_w, new_o, stats)
```

### 4. Gating / Rating

A new network does NOT become "current" on upload. Instead:

1. Trainer output is staged as `candidate_net_{K}`.
2. Coordinator queues ~400 `rating_job`s: `candidate_K` vs `current`.
3. Volunteers (any) play the match games with low sims (~100).
4. If candidate wins ≥55%, it gets promoted to `current`. Otherwise discard.

Gating catches both (a) bad trainer iters (bug / bad LR spike) and
(b) malicious trainer uploads. It is the single most important piece of
the whole system.

---

## Anti-Cheat / Validation

Threat model: self-play volunteers are untrusted; trainers are whitelisted.

Defenses, stacked:

| threat                         | defense                                             |
|--------------------------------|-----------------------------------------------------|
| wrong engine / tuned sims      | coordinator dictates all of `SelfPlayConfig`        |
| fabricated game data           | spot-replay: redo ~0.5% of jobs, compare move dist  |
| uploads under wrong network    | clients verify ed25519 signature on net file        |
| rating-game tampering          | quorum: average across many volunteers before promo |
| malicious trainer upload       | gating (55% match threshold) + whitelist only       |
| rate abuse / botnet            | per-token rate limits, reputation score             |

Gradient-level attacks are simply out of scope because gradients never
cross the wire — only completed networks do, and those are gated.

---

## Phased Rollout

| phase | scope                                                                          | volunteers | est. time |
|-------|--------------------------------------------------------------------------------|------------|-----------|
|   0   | Refactor `ParallelSelfPlay` into `run_selfplay_job(net, cfg) → .npz`. No net.  | 0 (local)  | 1 wk      |
|   1   | Minimal coordinator + single-volunteer end-to-end loop on 9×9. Localhost only. | 1 (self)   | 2 wk      |
|   2   | Gating, signed nets, auth tokens, leaderboard. Still 9×9.                      | 5–10 friends | 2 wk    |
|   3   | Docker packaging, onboarding doc, open to public at 13×13.                     | 20–100     | 4 wk      |
|   4   | 19×19 launch with pretrained seed net. Add global pooling; ownership head already exists. | 100+ | ongoing  |

Phases 0–2 (~5 weeks) prove the full pipeline end-to-end on the board size
we have the most data on. Phases 3–4 are operational — run it for a year,
see how far the volunteer pool carries us.

---

## GCP Cost (Steady State)

| resource                                            | monthly   |
|-----------------------------------------------------|-----------|
| e2-small VM (coordinator + static site)             | ~$7       |
| Cloud SQL (or Postgres on the VM itself)            | $0–10     |
| GCS storage (50 GB: buffer + nets + game records)   | ~$1       |
| Egress (~50 MB net × ~100 pulls/day)                | ~$1       |
| Cloudflare in front of weights (free tier)          | $0        |
| **Total, steady**                                   | **~$10–20** |
| Spot GPU fallback if trainer offline >24h (opt-in)  | +$3–10    |

No always-on GPU on GCP. The owner's 5060 Ti is the trainer. A spot-GPU
fallback on GCP can be wired in later so the pipeline does not stall if
the owner is offline for a week.

---

## Scheduling Rules

Coordinator invariants to keep things sane:

- **At most one active `training_job`** at a time. 30-minute lease;
  expired leases re-queue.
- **Training job enqueued when**: buffer has ≥ N new rows since last
  accepted iter (e.g., 50K new positions).
- **Iter monotonicity**: each training job tagged `iter=K`. Uploads must
  match the tag; out-of-order uploads rejected.
- **Windowed sampling**: trainer samples from the last M networks' worth
  of rows (not only the latest), matching KataGo's windowed recipe.
  Prevents policy collapse when one promoted net is weaker than its
  data suggests.
- **Candidate retention**: keep the last K candidate nets even after
  rejection — useful for post-hoc Bradley-Terry tournaments.

---

## What We Keep from Today's Code

Most of `Season-5/AlphaZero/` survives unchanged:

| component                                   | reuse?     | notes                                          |
|---------------------------------------------|------------|------------------------------------------------|
| `engine/` (C++ Go + MCTS + bindings)        | as-is      | already portable                               |
| `model/network.py`                          | as-is      | needs only a weight serialization helper       |
| `training/parallel_self_play.py`            | refactor   | extract `run_selfplay_job(...)` entry point    |
| `training/trainer.py`                       | refactor   | extract `run_training_step_batch(...)` entry   |
| `training/replay_buffer.py`                 | keep local | trainer clones a slice from GCS per iter       |
| `training/train.py` orchestration           | replace    | becomes "poll coordinator for jobs"            |
| `model/config.py`                           | keep       | coordinator references named configs           |

The Phase 0 refactor is deliberately minimal: tease the self-play and
training loops out of `train.py` and into pure functions that take a
config + weights and return artifacts. No networking yet. That alone
de-risks ~60% of the design.

---

## Open Questions

1. **How public?** Friends-only vs. OGS-announcement-level public.
   Changes auth model and abuse posture significantly. Recommend
   friends-only through Phase 2.
2. **Project license.** Currently none. Pick MIT (KataGo's choice) before
   Phase 3 public release.
3. **Client distribution.** Docker-only (v1), or pip + PyInstaller too?
   Docker-only is dramatically simpler and matches KataGo's trend.
4. **Board-size bootstrap.** Start the public rollout at 13×13 (existing
   infra) or 9×9 (cheapest smoke-test)? Recommend 9×9 through Phase 2,
   13×13 at Phase 3, 19×19 at Phase 4.
5. **Trainer whitelist policy.** Solo (owner only) vs. invited trusted
   contributors. Solo through Phase 3; expand only if training becomes a
   scheduling bottleneck (it almost certainly will not for 12+ months).
6. **Pretrained seed for 19×19.** Re-use the pretraining recipe from
   `PLAN.md:211-214` (featurecat 21.1M-game dataset, ~2–4 hours local
   pretraining) before launch. Ships with the first public 19×19 net.

---

## Relation to Prior Planning Docs

- `PLAN.md` — original single-box technical plan. Self-play + training on
  one rented 4090. This doc supersedes its Phase-4 "if we had more
  compute" discussion with a concrete volunteer model.
- `KATAGO_PROJECT.md` — reference material; the distributed model here
  is explicitly modeled on KataGo's.
- `HARDWARE_NOTES.md` — cost-per-iter analysis for the single-box case.
  Still accurate for the trainer client. Self-play throughput scales
  linearly with volunteer count, which this doc assumes without
  re-deriving.

---

## TL;DR

- Coordinator: small always-on GCP VM (~$10–20/mo). No GPU.
- Self-play: any volunteer's GPU, via HTTPS job queue.
- Training: also a job — picked up by a whitelisted client (initially the
  owner's 5060 Ti). Full iter runs on one box; no distributed SGD.
- Gating via match games before any candidate net is promoted.
- Phase 0 (local refactor) → Phase 4 (public 19×19 launch) over ~12 months.
- The hard problem was always self-play data scale. This plan fixes it.
