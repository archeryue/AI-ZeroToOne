# KataGo — The Open-Source Go AI That Changed Everything

A deep dive into the most efficient and influential open-source Go AI
project. Written as reference material for our AlphaZero project.

---

## The Creator

**David Jian Wu** — a software engineer at Jane Street Capital (a
quantitative trading firm). Harvard College B.Sc. (2011), thesis on
AI for the game of Arimaa.

Before KataGo, Wu created **bot_Sharp**, an Arimaa-playing program
that defeated three top human players to win the Arimaa AI Challenge
in 2015. Sharp used alpha-beta search enhanced with machine learning
— the same "combine classical search with learned evaluation" instinct
that would later drive KataGo.

Wu started KataGo in late 2017 as a personal project at Jane Street.
It evolved from experimentation into a genuine research effort.
Jane Street provided computational resources for the initial training
runs.

**First release:** February 27, 2019, alongside the paper.

---

## The Paper

**"Accelerating Self-Play Learning in Go"** (arXiv:1902.10565,
February 2019)

### The headline result

**50× more efficient than prior methods.** KataGo surpassed ELF
OpenGo's final strength after only 19 days on ~27 V100 GPUs
(~1.4 GPU-years). For context:

| System | Compute | Time | Cost (est.) |
|---|---|---|---|
| AlphaZero (DeepMind) | Thousands of TPUs | ~40 days | ~$3.5M |
| ELF OpenGo (Facebook) | Thousands of GPUs | ~2 weeks | ~$1M+ |
| Leela Zero (community) | Thousands of volunteer GPUs | ~2 years | Free (donated) |
| **KataGo** | **27 V100 GPUs** | **19 days** | **<$100K** |

A second run with bugfixes reached ELF's strength in **just 3 days**
on 28 V100s — careful engineering dramatically accelerating learning.

### The four categories of innovations

**1. General techniques (applicable beyond Go):**

- **Playout cap randomization**: 75% of moves use fewer simulations
  (fast), 25% use full simulations (high quality, used as policy
  target). Resolves the tension between policy targets (dense — every
  move) and value targets (sparse — one outcome per game). Total
  training speedup: ~1.4×.

- **Policy target pruning**: Reduce policy mass on moves that MCTS
  proved are bad. Prevents the network from wasting capacity learning
  to predict moves the search already rejected.

- **Global pooling layers**: Add global average pooling branches
  inside the residual tower. Lets the network condition on global
  board context (essential for ko fights, whole-board strategic
  patterns). Without this, the conv-only trunk can only see local
  features.

**2. Domain-specific improvements (Go-specific):**

- **Ownership auxiliary head**: Per-cell prediction of final territory
  control. 169 dense labels per position (on 13×13) vs 1 scalar value
  label per game. Provides massive auxiliary supervision that
  regularizes the trunk. Training speedup: ~1.6×.

- **Score prediction head**: Predicts final score margin, not just
  binary win/loss. Enables reasonable play in handicap games without
  special training. Score supervision is denser than binary outcome.

- **Hand-crafted input features** (18 binary planes):
  - Liberty counts (1, 2, 3 liberties)
  - Ladder detection (stones ladderable 0/1/2 turns ago, ladder
    capture moves)
  - Pass-alive regions
  - Ko/superko/suicide illegal moves
  - Komi parity
  - Board location markers

  These features give the network a head start — it doesn't have to
  discover ladder patterns or liberty counting from raw stone history.

### Combined acceleration factor: ~9.1×

Each innovation contributes multiplicatively. Together they explain
why KataGo can match systems that used 100-1000× more compute.

---

## Architecture Details

### Network structure

KataGo maintains AlphaGo Zero's residual tower but adds multiple
output heads and global pooling:

```
Input (22+ planes, including hand-crafted features)
  → Conv3x3 + Norm + ReLU
  → N × ResBlock (with global pooling branches)
  → Policy head:     move probabilities
  → Value head:      win probability (independent MLP)
  → Score head:      expected final score margin
  → Ownership head:  per-cell territory prediction
  → (optional: opponent move prediction, score variance)
```

### Key differences from AlphaGo Zero

| Aspect | AlphaGo Zero | KataGo |
|---|---|---|
| Input | 17 planes (stone history only) | 22+ planes (history + features) |
| Output heads | 2 (policy + value) | 5-6 (policy + value + score + ownership + more) |
| Normalization | BatchNorm | LayerNorm (modern versions) |
| Pooling | None | Global pooling in trunk |
| Value target | Binary ±1 | Binary ±1 + score margin |
| Aux supervision | None | Ownership (169 labels/position) |
| Training | Iterative batch | Continuous |

### Network sizes

KataGo trains multiple network sizes:

| Network | Blocks × Channels | Elo (approx.) |
|---|---|---|
| 6b × 96ch | Small | ~2,900 (6D) |
| 10b × 128ch | Medium | ~3,500 |
| 20b × 256ch | Large | ~4,500 |
| 40b × 256ch | XL (strongest) | ~5,274 (18D, professional level) |

---

## Distributed Training Model

### How it works

KataGo launched **community-distributed training** on November 28,
2020. The system is elegant in its simplicity:

1. **Volunteers** download the KataGo client and run self-play games
   on their own GPUs
2. **Game data** is uploaded to `katagotraining.org`
3. **Central server** aggregates data and trains improved networks
4. **New networks** are rating-tested against previous versions
5. **Cycle repeats** — volunteers automatically download the latest
   network

### Why distributed works for self-play

Self-play is **embarrassingly parallel** — each GPU plays games
independently with no synchronization needed. The only shared state
is the network weights, which update once per training batch (not per
game). Volunteers need minimal bandwidth: upload game records (small
text), download new weights periodically (~50 MB).

### Scale

| Metric | Total (as of 2025-2026) |
|---|---|
| Volunteer contributors | **1,382** |
| Training data rows | **4.6 billion** |
| Self-play games | **93.16 million** |
| Rating games | **1.99 million** |

For context: our Phase 2 has generated ~10,000 games total. KataGo's
community has generated **93 million**. That's a 9,300× data
advantage — which is exactly why their 44k-param value MLP doesn't
overfit and ours does.

---

## Current Strength

### Elo ratings

| Engine | Elo (approx.) | Level |
|---|---|---|
| Random play | ~0 | — |
| Beginner human | ~500 | 30 kyu |
| Strong amateur | ~2,000 | 1 dan |
| Professional 1p | ~3,000 | 1 dan pro |
| Lee Sedol (AlphaGo match) | ~3,500 | 9 dan pro |
| **KataGo 40b×256ch** | **~5,274** | **18D (superhuman)** |
| Top human pros | ~5,200-5,400 | Chinese 18D |

KataGo at full strength is **comparable to the world's strongest
human professionals** — and on unlimited hardware would surpass them.

### Scalable strength

Depending on hardware and time budget, KataGo operates at different
levels:

- **Limited resources**: Amateur 4 kyu
- **Consumer GPU, few days**: High amateur dan
- **Consumer GPU, months**: Superhuman
- **Multi-GPU cluster**: World-class professional level

---

## Impact on the Go Community

### How KataGo changed Go study

Before KataGo, AI analysis required expensive proprietary systems.
Now anyone with a consumer GPU can run state-of-the-art analysis
locally. This democratization has fundamentally changed how players
study:

- **Ownership heatmaps**: Visualize territorial influence, not just
  move quality. Players can see "who controls what" at a glance.
- **Score estimation**: Understand game urgency and decision quality
  in terms of actual points, not just "ahead/behind."
- **Move-by-move analysis**: Every move gets a winrate delta, showing
  exactly where a game was won or lost.
- **Handicap-friendly**: The score head enables meaningful analysis
  of high-handicap games — historically difficult for AI.

### Professional adoption

- **South Korean national team** uses KataGo for training
- Default analysis engine on **OGS** (Online Go Server, tens of
  thousands of users)
- Powers **AI Sensei** (free web-based game review)
- Used by professionals worldwide for game preparation

### Analysis ecosystem built on KataGo

| Tool | Description |
|---|---|
| **KaTrain** | Teaching tool with variable-difficulty bot play and analysis |
| **Lizzie / LizGoban** | Real-time search visualization, ownership heatmaps |
| **Sabaki** | Modern game editor with KataGo integration |
| **Ogatak** | KataGo-specific analysis GUI |
| **AI Sensei** | Web-based analysis platform |

### The kata-analyze GTP extension

KataGo extended the standard Go Text Protocol with `kata-analyze`,
which returns:
- Move probabilities (policy)
- Win probability
- Ownership heatmap (per-cell, -1 to +1)
- Expected score
- Score variance (uncertainty)

GUIs render these as real-time visualizations during analysis. This
is what makes KataGo uniquely useful for study — you don't just see
"this move is good/bad," you see *why* in terms of territory and
score.

---

## Interesting Facts

### The adversarial attack (2022)

Researchers at MIT demonstrated that a specifically-trained adversarial
policy could beat KataGo **>97% of the time** by exploiting blind
spots in its evaluation. The attack used a cyclic group strategy
that KataGo's training data never contained. This highlighted an
important lesson: superhuman game-playing AI can still have
exploitable vulnerabilities. KataGo has since been partially patched
against known attacks.

### The hardest Go problem in the world

David Wu used KataGo to analyze a 1713 Go problem from the
**Igo Hatsuyoron** (a classical problem collection compiled by the
Inoue house head). Jane Street published a blog post about this —
using modern AI to probe a 300-year-old puzzle. The intersection of
AI and Go history.

### Multi-board support

Unlike most Go AIs, KataGo supports **any board size** (9×9, 13×13,
19×19, rectangular boards) and **multiple rulesets** (Japanese,
Chinese, Tromp-Taylor, AGA, New Zealand, and more). This versatility
makes it the Swiss Army knife of Go engines.

### One person, world-class impact

KataGo remains primarily the work of one engineer (David Wu), with
community support for distributed training. The core algorithms,
C++ engine, Python training pipeline, and GTP interface are largely
his work. It's a remarkable demonstration of what one skilled
individual can accomplish with open-source collaboration.

### Commercial sponsorship

In March 2026, **ZhiziGo** company sponsored the creation of a new
KataGo model variant (`kata1-zhizi-b28c512nbt-muonfd2`), showing
that commercial entities recognize KataGo's value to the professional
Go community.

---

## Open Source

- **License**: MIT (free use/modification)
- **Repository**: https://github.com/lightvector/KataGo
- **Language**: C++ (engine) + Python (training)
- **Distributed training server**: https://katagotraining.org
- **Paper**: https://arxiv.org/abs/1902.10565

---

## What We Borrowed from KataGo

For our AlphaZero project, we adopted several KataGo innovations:

| Innovation | KataGo | Ours |
|---|---|---|
| Ownership head | Yes | Yes (BCE per cell) |
| Score head | Yes | Yes (MSE on territory margin) |
| Value from score | Combined with independent value | Derived (no independent MLP) |
| LayerNorm | Yes (modern versions) | Yes |
| Hand-crafted features | Yes (18 planes) | No (pure 17-plane stone history) |
| Global pooling | Yes | No |
| Playout cap | Yes | No |
| Distributed training | Yes (1,382 volunteers) | No (1 GPU) |

The key difference: KataGo has **93 million** self-play games from
1,382 volunteers. We have **~10,000 games** from 1 GPU. Their value
MLP works because 93M games provides enough data diversity. Ours
memorizes because ~10K games doesn't. Our score-head-derived-value
architecture is a direct response to this data-scale constraint.

---

## Lessons for Our Project

1. **Engineering matters more than hardware.** KataGo's 9.1×
   acceleration came from software innovations, not bigger GPUs.
   The same V100s that others used inefficiently, Wu used 50× better.

2. **Auxiliary heads are not optional at small scale.** On large
   data, a simple value head works. On small data, ownership and
   score supervision are essential to prevent the value head from
   memorizing noise. This is the central lesson of our Phase 2.

3. **Hand-crafted features help.** KataGo's ladder/liberty input
   planes give the network a head start. Our pure-history approach
   is principled ("Zero" philosophy) but forces the network to
   discover these patterns from scratch — expensive in data.

4. **Distributed self-play is the scaling lever.** If we wanted to
   match KataGo's data scale, we'd need volunteers running self-play
   — not a bigger GPU. This is fundamentally a data problem.

5. **One person can build world-class AI.** David Wu proved that a
   single skilled engineer with good taste in algorithm design can
   compete with corporate labs. That's inspiring, even when the
   engineering is hard.
