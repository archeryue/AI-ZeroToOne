# AlphaZero for Go — Technical Plan

## Mission

Build AlphaZero from scratch for the game of Go.

- **9x9 and 13x13**: Pure self-play from scratch — the true "Zero" approach
- **19x19**: Proper architecture with human pretraining — build it right, train it exploratory

**Budget: total < $100.** Target ~$44, leaving ~$56 headroom for extra experiments.

Season 5 finale — from RL basics all the way to AlphaZero on Go.

---

## Why The Cost Is Affordable: Massive Parallelism

The naive cost estimate (800 sims × millions of moves = billions of NN calls = $$$) is wrong because it ignores **batched parallelism across games**.

### The GPU Batching Trick

A single GPU forward pass takes ~0.3ms whether batch=1 or batch=256. So:

```
❌ Sequential thinking:
   1 game: 800 sims/move × 220 moves = 176,000 forward passes
   80K games × 176K passes = 14 billion NN calls 😱

✅ Parallel thinking:
   256 games run simultaneously
   Each "tick": collect 1 leaf per game → batch=256 → ONE forward pass (0.3ms)
   800 ticks per move × 220 moves = 176,000 ticks total
   Wall time: 176,000 × 0.3ms = 53 sec for ALL 256 games
   Throughput: 256 games / 53 sec ≈ 17,000 games/hr
```

With **virtual loss** (collect 8 leaves per game per tick):
```
   Batch = 256 games × 8 VL = 2,048 per forward pass (~0.5ms)
   Ticks per move: 800/8 = 100
   Ticks per game: 100 × 220 = 22,000
   Wall time: 22,000 × 0.5ms = 11 sec for ALL 256 games (GPU only)
```

The **real bottleneck is CPU** — MCTS tree traversal, game state copying, capture checking. This is why the **C++ engine is critical**. With C++ MCTS + engine, CPU cost per tick ≈ 2-10ms (depending on board size and thread count).

### Hardware: RunPod RTX 4090 24GB

**~$0.44/hr — 8-16 vCPUs, 32-64GB RAM, 24GB VRAM.**

Our largest model (23M) needs <10GB VRAM total (inference + training), so the
4090's 24GB is plenty. BF16 tensor core throughput (~330 TFLOPS) is more than
enough for our model sizes. **CPU is the actual bottleneck** — see Cost Estimates.

RAM constraint: keep replay buffer at 500K-1M positions (~7-15GB) to fit
comfortably in 32-64GB system RAM.

### Process Architecture (8 vCPUs)

```
RunPod RTX 4090 24GB — 8+ vCPUs, 32-64GB RAM

Process 0: Orchestrator (Python)                         [1 vCPU]
  ├── GPU inference server (torch BF16, batched)
  ├── Training loop (SGD, interleaved with self-play)
  └── Logging, checkpointing, eval

Processes 1-5: Self-Play Workers (C++ via pybind11)      [5 vCPUs]
  ├── Worker 1: manages 52 parallel games
  ├── Worker 2: manages 52 parallel games
  ├── Worker 3: manages 52 parallel games
  ├── Worker 4: manages 52 parallel games
  └── Worker 5: manages 48 parallel games
  Total: 256 parallel games

Reserved: [2 vCPUs] for OS, Python GC, I/O, etc.

Communication:
  Workers ──shared memory queue──→ GPU server ──shared memory──→ Workers
  Workers ──shared memory──→ Replay Buffer ──→ Training process
```

### Synchronization Flow (1 tick)

```
   Workers (C++)                  GPU Server (Python)
   ─────────────                  ───────────────────
1. Each worker selects 8 leaves
   per game (virtual loss)
   → 5 workers × ~50 games × 8
   = 2,048 leaf observations

2. Write obs to shared memory ──→ 3. Read all 2,048 obs
                                  4. torch forward pass
                                     batch=2048, BF16 (~0.5ms)
5. Read policy+value results  ←── 6. Write results to shared memory

7. Each worker: expand nodes,
   backup values, remove VL

8. Repeat until 400-800 sims done for current move
```

**Each worker is single-threaded C++** — no GIL issues, no Python overhead in the hot loop. Workers only call Python for GPU inference via shared memory.

---

## Cost Estimates (RTX 4090, Based on Actual Benchmarks)

> Benchmarks from Apple M1, -O3, after completing Steps 1-6 (C++ engine + MCTS).
> RunPod x86 CPUs (AMD EPYC / Intel Xeon) expected to be within ±20% of M1 single-core.

### Per-Sim CPU Cost (MCTS Benchmark)

Full sim cost includes tree traversal, Game<N> state copy (with 8-position
history), deque-based state storage, node allocation, and backup propagation.

| | 9x9 | 13x13 (est.) | 19x19 |
|---|---|---|---|
| Per-sim CPU cost | **3.2μs** | **~5.5μs** | **10.8μs** |

Source: MCTS benchmark, batch=8 virtual loss, DESIGN.md Step 6 results.
13x13 interpolated from board area (169 cells vs 81/361).

### Per Tick (256 games × 8 VL = 2,048 leaves)

| | 9x9 (5M) | 13x13 (7M) | 19x19 (23M) |
|---|---|---|---|
| CPU work (single thread) | 6.6ms | ~11.3ms | 22.1ms |
| CPU work (5 workers) | **1.3ms** | **~2.3ms** | **4.4ms** |
| GPU forward pass (BF16) | 0.5ms | 0.6ms | **~2.0ms** |
| Sync + overhead | ~0.2ms | ~0.2ms | ~0.2ms |
| **Effective per tick** | **~2.0ms** | **~3.1ms** | **~6.6ms** |

**CPU is the bottleneck for ALL configurations.** GPU has idle headroom,
which is why larger models (7M, 23M) don't hurt throughput.

### Throughput Estimates

| | 9x9 (400 sims) | 13x13 (600 sims) | 19x19 (800 sims) |
|---|---|---|---|
| Ticks/game | 3,000 | 9,000 | 22,000 |
| Time/256 games (2x overhead) | 12s | 56s | 290s |
| **Throughput** | **77K games/hr** | **16K games/hr** | **3.2K games/hr** |

### Per-Phase Cost (RTX 4090 at $0.44/hr)

**Phase 1 — 9x9 pure self-play (150K games/run)**:

| Component | Estimate |
|-----------|----------|
| Self-play: 150K / 77K games/hr | ~2.0h |
| Training | ~0.5h |
| **Total per run** | **~2.5h** |
| **Cost per run** | **~$1.1** |

**Phase 2 — 13x13 pure self-play (200K games/run)**:

| Component | Estimate |
|-----------|----------|
| Self-play: 200K / 16K games/hr | ~12.5h |
| Training | ~2h |
| Resign savings (-15%) | ~-2h |
| **Total per run** | **~12.5h** |
| **Cost per run** | **~$5.5** |

13x13 gets more games than 9x9 — bigger board, more complex patterns to learn from scratch.

**Phase 3 — 19x19 pretrained + self-play (30K games/run)**:

| Component | Estimate |
|-----------|----------|
| Self-play: 30K / 3.2K games/hr | ~9.4h |
| Training | ~1h |
| Resign savings (-25%) | ~-2.5h |
| **Total per run** | **~8h** |
| **Cost per run** | **~$3.5** |

### Total Budget

| Phase | Cost/run | Runs | Total |
|-------|----------|------|-------|
| 9x9 pure self-play (5M, 150K games) | ~$1.1 | 8 | **~$9** |
| 13x13 pure self-play (7M, 200K games) | ~$5.5 | 5 | **~$28** |
| 19x19 pretrained (23M, 30K games) | ~$3.5 | 2 | **~$7** |
| **Grand Total** | | **15** | **~$44** |

**~$44 total, ~$56 headroom.** 13x13 gets the lion's share of the budget — it's
the main training target with the most complex pure self-play task. Plenty of
headroom to add more games or runs if results look promising.

### Possible Optimizations

The CPU hot path can be optimized during Step 4 (SelfPlayWorker):
- **Lazy history copy**: only copy history when encoding observations, not on every clone
- **Contiguous arena for game states**: replace `deque<Game<N>>` with a flat arena
- **Increase VL batch from 8 to 16**: amortize per-tick sync overhead

A 2x reduction in per-sim cost would bring the total to ~$22.

---

## Lessons from ChessRL Candidate 4 (6 Failed Versions)

| Failure | Root Cause | Go Mitigation |
|---------|-----------|---------------|
| **Draw deadlock** — 99.5% draws, no signal | Chinese Chess draws; sparse reward | Go always has a winner (komi). No draw problem. |
| **Chicken-and-egg** — bad policy → bad MCTS → worse policy | Too few sims, too small model, no pretraining | 400-800 sims; 200K+ games for 9x9/13x13; pretraining for 19x19 |
| **Self-play overfitting** — memorizes one trick | Buffer too small (20K) | Large buffer (500K-1M), 8x symmetry augmentation |
| **Policy degradation** — MCTS targets ≤ pretrained policy | Weak value head + low sims = noise | High sim count + many games = strong targets |

---

## Training Pipeline

### Phase 1: 9x9 Pure Self-Play (1x RTX 4090, ~$1.1/run)

**No pretraining — learns Go from scratch.** The model starts with random weights
and discovers captures, eyes, territory, and strategy entirely through self-play.
This is the true AlphaZero/"Zero" approach.

**Config**:

| Parameter | Value |
|-----------|-------|
| Model | 10 blocks × 128ch (5M) |
| MCTS sims/move | 400 |
| Virtual loss batch | 8 |
| Parallel games | 256 (5 workers × ~52) |
| Games per run | 150K |
| Positions/game (×8 symmetry) | 60 × 8 = 480 |
| Total positions per run | **72M** |
| Replay buffer | 500K positions |
| Train batch size | 256 |
| Train steps per iter | 100 |
| LR | 0.01 → 0.001 (cosine) |
| Optimizer | SGD + momentum 0.9 |
| L2 | 1e-4 |
| Temperature | 1.0 first 15 moves, → 0.1 |
| Dirichlet alpha/epsilon | 0.11 / 0.25 |
| Komi | 7.5 |

**Targets**: >95% vs Random, >80% vs pure MCTS, beat GnuGo lv10.

### Phase 2: 13x13 Pure Self-Play (1x RTX 4090, ~$5.6/run)

**Also pure self-play from scratch.** 13x13 is the sweet spot — big enough for
real strategy (joseki, influence, territory balance, middle-game fighting), small
enough to train well. It's a real competitive format on many online Go servers.

| Parameter | Value |
|-----------|-------|
| Model | 15 blocks × 128ch (7M) |
| MCTS sims/move | 600 |
| Virtual loss batch | 8 |
| Parallel games | 256 |
| Games per run | 200K |
| Positions/game (×8 symmetry) | 120 × 8 = 960 |
| Total positions per run | **192M** |
| Replay buffer | 1M positions |
| Dirichlet alpha/epsilon | 0.07 / 0.25 |
| Komi | 7.5 |

**Targets**: >99% vs Random, beat GnuGo lv5+, solid mid-game strategy.

### Phase 3: 19x19 Pretrained + Self-Play (1x RTX 4090, ~$3.5/run)

**Human pretraining, then self-play refinement.** We build the proper AlphaGo Zero
architecture (20b×256ch, 800 sims) and full training pipeline. The goal is a
correct, production-quality implementation — not a fully converged model.

**Pretraining** (local GPU, free):

| Source | Games |
|--------|-------|
| featurecat/go-dataset | 21.1M |
| KGS archives | 100K+ |
| Japanese pro archives | 88K+ |

- Policy: cross-entropy on human moves (~50-55% top-1 accuracy)
- Value: MSE on game outcome (+1/-1)
- Local GPU (RTX 5060 Ti), ~2-4 hours

**Self-play refinement** (cloud):

| Parameter | Value |
|-----------|-------|
| Model | 20 blocks × 256ch (23M) |
| MCTS sims/move | 800 |
| Virtual loss batch | 8 |
| Parallel games | 256 |
| Games per run | 30K |
| Positions/game (×8 symmetry) | 220 × 8 = 1,760 |
| Total positions per run | **53M** |
| Replay buffer | 1M positions |
| Dirichlet alpha/epsilon | 0.03 / 0.25 |
| Resign threshold | v < -0.95 for 3+ moves |

**Targets**: Measurable improvement over pretraining baseline, demonstrate the
full AlphaZero loop works at 19x19 scale.

---

## Technical Design

### 1. Go Engine (C++ with pybind11)

```
engine/
├── go_game.h / go_game.cpp    # Board, rules, capture, ko, scoring
├── go_game_test.cpp           # Unit tests
├── bindings.cpp               # pybind11
└── setup.py
```

**Board**: `uint8_t board[N*N]`, 0=empty, 1=black, 2=white.

**Rules**:
- Stone placement + flood-fill capture (liberty counting)
- Simple ko (last capture position)
- Superko (Zobrist hash set)
- Two-pass → game end, Tromp-Taylor area scoring
- Komi 7.5

**Zobrist hashing**: random 64-bit per (position, color), XOR incremental update.

**Performance target**: >100K random playouts/sec per core on 9x9.

**Key**: Engine must support fast `clone()` for MCTS — each leaf expansion needs a state copy. Use stack-allocated board arrays, not heap.

### 2. C++ MCTS (Critical for Performance)

**This is the biggest difference from ChessRL** — MCTS in C++, not Python.

```
engine/
├── mcts.h / mcts.cpp         # MCTS tree, PUCT, virtual loss, batching
```

Python MCTS was the bottleneck in ChessRL. Object creation, GC, tree traversal in Python is 50-100x slower than C++. Moving MCTS to C++ makes the CPU cost negligible.

**C++ MCTS responsibilities**:
- Tree node allocation (arena allocator — no per-node malloc)
- PUCT selection with virtual loss
- Leaf collection for batch NN evaluation
- Tree backup after NN results return
- Tree reuse between moves (promote subtree)

**Python only handles**:
- GPU inference (torch batched forward pass)
- Training loop (loss, optimizer, checkpointing)
- Orchestration (start/stop workers, logging)

**Interface**:
```cpp
// C++ side: collect leaves that need NN evaluation
struct LeafRequest {
    float obs[17][N][N];   // observation planes
    int game_id;
    int node_id;
};

// Python calls:
std::vector<LeafRequest> collect_leaves(int num_leaves);  // C++ selects leaves
void process_results(std::vector<PolicyValue> results);    // C++ updates trees
```

### 3. Neural Network

**Input encoding** (17 planes, following AlphaGo Zero):

| Plane | Description | Count |
|-------|-------------|-------|
| Current player stones | Binary | 1 |
| Opponent stones | Binary | 1 |
| History (last 7 states) | Per-step stone planes | 14 |
| Color to play | All 1s or 0s | 1 |
| **Total** | | **17** |

**Model sizes are chosen per board size.** Since CPU is the bottleneck for all
configs (not GPU), larger models don't hurt throughput — size for representational
capacity, not inference speed.

**9x9 model — 10 blocks × 128ch ≈ 5M params**:
```
Input: (batch, 17, 9, 9)
  → Conv2d(17→128, k=3, pad=1) + BN + ReLU
  → 10 Residual Blocks (128 channels)
  → Policy Head: Conv1x1(128→2) + BN + ReLU + FC(162→82)  [81 + pass]
  → Value Head:  Conv1x1(128→1) + BN + ReLU + FC(81→256) + ReLU + FC(256→1) + tanh
```
Well-tested architecture. 200K games × 480 positions = 96M positions — 19x data-to-param ratio.

**13x13 model — 15 blocks × 128ch ≈ 7M params**:
```
Input: (batch, 17, 13, 13)
  → Conv2d(17→128, k=3, pad=1) + BN + ReLU
  → 15 Residual Blocks (128 channels)
  → Policy Head: Conv1x1(128→2) + BN + ReLU + FC(338→170)  [169 + pass]
  → Value Head:  Conv1x1(128→1) + BN + ReLU + FC(169→256) + ReLU + FC(256→1) + tanh
```
More depth than 9x9 (15 vs 10 blocks) for deeper tactical reading. Same channel
width — 128 is adequate for 13x13 spatial patterns. 100K games × 960 = 96M positions.

**19x19 model — 20 blocks × 256ch ≈ 23M params** (AlphaGo Zero architecture):
```
Input: (batch, 17, 19, 19)
  → Conv2d(17→256, k=3, pad=1) + BN + ReLU
  → 20 Residual Blocks (256 channels)
  → Policy Head: Conv1x1(256→2) + BN + ReLU + FC(722→362)  [361 + pass]
  → Value Head:  Conv1x1(256→1) + BN + ReLU + FC(361→256) + ReLU + FC(256→1) + tanh
```
The standard architecture from the AlphaGo Zero paper (not the 40-block final
version, but the one that already beat all previous AlphaGo versions within 3 days).
Human pretraining + exploratory self-play. Built right, not trained to convergence.

### 4. Data Augmentation (8-fold Symmetry)

Go boards are square → 4 rotations × 2 reflections = 8 equivalent positions.

For each self-play position `(obs, policy, value)`:
```python
def augment_8fold(obs, policy_map, value):
    # obs: (17, N, N), policy_map: (N, N) + pass
    augmented = []
    for rotation in [0, 90, 180, 270]:
        for flip in [False, True]:
            obs_t = transform(obs, rotation, flip)
            policy_t = transform(policy_map, rotation, flip)
            augmented.append((obs_t, policy_t, value))
    return augmented  # 8 samples from 1 position
```

150K games × 60 moves × 8 = **72M training positions** (9x9 pure self-play).
200K games × 120 moves × 8 = **192M training positions** (13x13 pure self-play).
30K games × 220 moves × 8 = **53M training positions** (19x19 pretrained + self-play).

### 5. Replay Buffer

- Circular buffer, 500K-1M positions (sized for 32-64GB RAM on RTX 4090 instances)
- Uniformly sampled for training
- Stored in shared memory (multiprocessing.SharedMemory) for zero-copy between self-play workers and trainer
- Old positions naturally evicted as buffer wraps

### 6. Training

- **Policy loss**: cross-entropy between MCTS visit distribution π and network policy p
- **Value loss**: MSE between game outcome z and network value v
- **L2 regularization**: 1e-4 on all weights
- **Total**: `L = -π·log(p) + (z-v)² + c·||θ||²`
- **Optimizer**: SGD + momentum 0.9 (following AlphaGo Zero)
- **LR schedule**: cosine decay from 0.01 to 0.0001
- **BF16 training** via torch.cuda.amp

### 7. Additional Optimizations

**Resign threshold**:
- If value head outputs < -0.95 for 3+ consecutive own moves, resign
- Saves ~25-30% game time in later training
- Disabled for 10% of games (to still see losing positions)

**Tree reuse**:
- After playing a move, promote the chosen child's subtree as new root
- Saves ~30-50% of MCTS work per move

**torch.compile**:
- Compile the inference model for ~20-30% speedup on batched forward passes

**KataGo's playout cap randomization** (optional):
- 75% of moves: 400 sims (fast)
- 25% of moves: 800 sims (high quality, used as policy target)
- Reduces average sims while maintaining target quality

---

## Evaluation

**Baselines** (in order of strength):
1. **Random** — trivial
2. **Pure MCTS** (no NN, random rollouts, 1000 sims)
3. **GnuGo** (level 1-10) — classical engine, level 10 ≈ 5-8 kyu
4. **Previous checkpoint** — Elo tracking

**Per-phase targets**:

| | 9x9 | 13x13 | 19x19 |
|---|---|---|---|
| vs Random | >99% | >99% | >95% |
| vs GnuGo lv5 | >80% | >60% | — |
| vs GnuGo lv10 | >50% | — | — |
| vs pretraining baseline | — | — | measurable gain |

**GTP protocol**: implement Go Text Protocol so our engine can play vs GnuGo, on OGS, or vs humans.

**Elo tracking**: 100+ games between checkpoints, plot Elo vs training time.

---

## Project Structure

```
AlphaZero/
├── PLAN.md                    # This file
├── engine/                    # C++ Go engine + MCTS + pybind11
│   ├── go_game.h/cpp          # Board, rules, capture, ko, scoring
│   ├── mcts.h/cpp             # C++ MCTS (PUCT, virtual loss, batching)
│   ├── go_game_test.cpp       # Unit tests for engine
│   ├── mcts_test.cpp          # Unit tests for MCTS
│   ├── bindings.cpp           # pybind11 Python bindings
│   └── setup.py               # Build script
├── model/
│   ├── network.py             # ResNet dual-head (policy + value)
│   └── config.py              # Model configs (9x9/5M, 13x13/7M, 19x19/23M)
├── training/
│   ├── pretrain.py            # Supervised pretraining on human games
│   ├── self_play.py           # Self-play orchestrator (workers + GPU server)
│   ├── trainer.py             # Network training loop
│   ├── replay_buffer.py       # Shared-memory replay buffer
│   └── train.py               # Main entry point
├── data/
│   ├── download_sgf.sh        # Download human game records
│   └── parse_sgf.py           # SGF → training data
├── eval/
│   ├── gtp_engine.py          # GTP protocol implementation
│   ├── play_gnugo.py          # Evaluate vs GnuGo
│   └── elo_tracker.py         # Elo progression
└── scripts/
    └── setup_cloud.sh         # Cloud GPU setup
```

---

## Implementation Order

### Step 1: C++ Go Engine (~2-3 days, local)
- Board, stone placement, flood-fill capture, liberty counting
- Ko, superko (Zobrist), pass, Tromp-Taylor scoring
- Thorough unit tests; fuzz-test scoring vs GnuGo
- pybind11 bindings
- Benchmark: >100K random playouts/sec per core

### Step 2: C++ MCTS (~2 days, local)
- PUCT tree search with virtual loss
- Arena allocator for nodes (no per-node malloc)
- Batched leaf collection interface for GPU inference
- Tree reuse between moves
- Unit tests on small board (5x5)

### Step 3: Neural Network + Integration (~1 day, local)
- ResNet dual-head in PyTorch (5M/7M/23M configs)
- BF16 batched inference
- Wire up: C++ MCTS calls Python GPU → results back to C++
- Smoke test: random network + MCTS plays legal Go

### Step 4: Self-Play Training Loop (~1 day, local)
- Multi-worker self-play + shared replay buffer + training
- 8-fold symmetry augmentation
- Resign threshold, tree reuse
- Local smoke test with tiny model on 9x9

### Step 5: Cloud Training — 9x9 Pure Self-Play (~$9)
- Rent 1x RTX 4090 24GB on RunPod ($0.44/hr)
- Pure self-play from scratch: 400 sims, 10b×128ch (5M), 150K games
- Eval vs GnuGo
- ~8 runs of experimentation

### Step 6: Cloud Training — 13x13 Pure Self-Play (~$28)
- Same RTX 4090, pure self-play from scratch
- 600 sims, 15b×128ch (7M), 200K games
- Eval vs GnuGo
- ~5 runs of experimentation

### Step 7: Data Pipeline + Pretraining for 19x19 (~1 day, local)
- Download SGF records
- SGF parser → (board, move, outcome) triples
- Supervised pretraining of 20b×256ch (23M) on local GPU (~2-4 hours)
- Verify: pretrained net + MCTS plays reasonable 19x19 Go

### Step 8: Cloud Training — 19x19 Exploratory (~$7)
- Same RTX 4090, start from pretrained 19x19 weights
- 800 sims, 20b×256ch (23M), 30K games
- Verify self-play improves over pretraining baseline
- ~2 runs

### Step 9: Polish & Demo (~1 day, local)
- GTP protocol for playing vs humans / on OGS
- Elo curves, loss plots, example game SGFs
- Season 5 write-up

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Go engine bugs | Thorough tests; fuzz-test scoring vs GnuGo |
| C++ MCTS bugs | Test on 5x5 first; compare with Python MCTS reference |
| CPU bottleneck (game logic) | C++ engine + multi-worker parallelism |
| Python↔C++ overhead | Batch communication; minimize Python calls |
| Pure self-play doesn't learn | 9x9 should work (well-studied); 13x13 is the stretch goal |
| 19x19 pretraining weak | Use large pro game dataset (21M+); verify before cloud runs |
| Budget overrun | ~$39 target with ~$61 headroom; 9x9 alone proves the system |

---

## References

### Papers (in Season-5/)
- AlphaGo: `AlphaGo_Mastering_Go_with_Deep_Neural_Networks.pdf`
- AlphaGo Zero: `AlphaGo_Zero_Mastering_Go_without_Human_Knowledge.pdf`
- AlphaZero: `AlphaZero_Mastering_Chess_Shogi_by_Self_Play.pdf`

### Papers to Read
- KataGo: *"Accelerating Self-Play Learning in Go"* (2019)
- Gumbel MuZero: *"Policy improvement by planning with Gumbel"* (ICLR 2022) — useful reference even if we don't reduce sims

### Open-Source References
- **KataGo** — most efficient Go AI, C++ engine, training optimizations
- **MiniZero** (github.com/rlglab/minizero) — clean AlphaZero framework in C++
- **Leela Zero** — community distributed AlphaZero for Go

### Datasets
- featurecat/go-dataset: 21.1M games
- KGS archives: 100K+ games
- Japanese pro archives: 88K+ games
- Professional Go Dataset: 98K pro games
