# AlphaZero for Go — Technical Plan

## Mission

Build AlphaZero from scratch for the game of Go. Train on 9x9 first, then scale to 19x19.

**Budget: 9x9 < $20, 19x19 < $80, total < $100.**

Season 5 finale — from RL basics all the way to AlphaZero on Go.

---

## Why The Cost Is Affordable: Massive Parallelism

The naive cost estimate (800 sims × millions of moves = billions of NN calls = $$$) is wrong because it ignores **batched parallelism across games**.

### The GPU Batching Trick

A single A100 forward pass takes ~0.3ms whether batch=1 or batch=256. So:

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

### Hardware: RunPod A100 80GB PCIe

**$1.19/hr — 8 vCPUs, 117GB RAM, 80GB VRAM.**

8 vCPUs is lean but sufficient. C++ MCTS is fast enough that 5 worker processes handle 256 parallel games easily. The GPU is the bottleneck for larger models, not CPU.

### Process Architecture (8 vCPUs)

```
RunPod A100 80GB PCIe — 8 vCPUs, 117GB RAM

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

## Cost Estimates (8 vCPUs, 1x A100)

### CPU Work Per Leaf (C++)

| Operation | 9x9 | 19x19 |
|-----------|------|-------|
| MCTS tree traversal + VL | ~0.2μs | ~0.2μs |
| Game state clone | ~0.5μs | ~2μs |
| Node expand + backup | ~0.2μs | ~0.2μs |
| **Total per leaf** | **~1μs** | **~2.5μs** |

### Per Tick (256 games × 8 VL = 2,048 leaves)

| | 9x9 | 19x19 |
|---|---|---|
| CPU work (single thread) | 2,048 × 1μs = 2ms | 2,048 × 2.5μs = 5ms |
| CPU work (5 C++ workers) | **0.4ms** | **1ms** |
| GPU forward pass (batch=2048, BF16) | **0.5ms** | **0.5ms** |
| Sync + shared memory overhead | ~0.2ms | ~0.2ms |
| **Effective time per tick** | **~0.7ms** | **~1.2ms** |

Note: 5M model. GPU and CPU roughly balanced.
For 47M model, GPU per tick rises to ~3ms → GPU becomes bottleneck, CPU has headroom.

### 9x9 Go: 400 sims/move

| Component | Estimate |
|-----------|----------|
| Ticks per move (400 sims / 8 VL) | 50 |
| Ticks per game (50 × 60 moves) | 3,000 |
| Time per 256 games (3,000 × 0.7ms) | **2.1 sec** |
| Realistic (3x overhead: Python↔C++, mem, sync) | **~6 sec** |
| **Throughput** | **~150K games/hr** |
| 25K games self-play | ~10 min |
| + training (100 iters × 100 steps) | ~20 min |
| + eval games | ~5 min |
| **Total per run** | **~35 min** |
| **Cost: 0.6 hrs × $1.19** | **~$0.7** |
| Budget for 10+ experiment runs | **< $10** |

### 19x19 Go: 800 sims/move, 5M model

| Component | Estimate |
|-----------|----------|
| Ticks per move (800 / 8) | 100 |
| Ticks per game (100 × 220 moves) | 22,000 |
| Time per 256 games (22,000 × 1.2ms) | **26 sec** |
| Realistic (3x overhead) | **~80 sec** |
| **Throughput** | **~11.5K games/hr** |
| 80K games self-play | ~7 hours |
| + training | ~3 hours |
| + resign savings (-20%) | ~-2 hours |
| **Total per run** | **~8 hours** |
| **Cost: 8 hrs × $1.19** | **~$10** |

### 19x19 Go: 800 sims/move, 47M model

| Component | Estimate |
|-----------|----------|
| GPU per tick (batch=2048, BF16, 47M) | ~3ms (GPU bottleneck) |
| Time per 256 games (22,000 × 3ms) | **66 sec** |
| Realistic (2x — less CPU overhead since GPU is bottleneck) | **~130 sec** |
| **Throughput** | **~7K games/hr** |
| 80K games + training | **~14 hours** |
| **Cost: 14 hrs × $1.19** | **~$17** |

### Total Budget

| Phase | Cost/run | Runs | Total |
|-------|----------|------|-------|
| 9x9 (5M, 400 sims) | ~$0.7 | 10 | **~$7** |
| 19x19 (5M, 800 sims) | ~$10 | 3 | **~$30** |
| 19x19 (47M, 800 sims) | ~$17 | 2 | **~$34** |
| **Grand Total** | | | **~$71** |

Full 400-800 sim MCTS, both model sizes, 15 total runs, under $100.
The 8 vCPUs are sufficient because C++ MCTS is fast and GPU is the bottleneck for the larger model.

---

## Revised Cost Estimates (Based on Actual Benchmarks)

> Added after completing Steps 1-6 (C++ engine + MCTS). Benchmarks from Apple M1, -O3.
> RunPod x86 CPUs (AMD EPYC / Intel Xeon) expected to be within ±20% of M1 single-core.

### What Changed: Per-Leaf Cost is 3-4x Higher Than Estimated

The original estimates modeled per-leaf CPU cost as just engine operations
(clone + place_stone + check). The actual MCTS sim cost includes significant
overhead from tree traversal, game state management (deque indexing, full
Game<N> copy with history), node allocation, and backup propagation.

| | 9x9 Est. | 9x9 Actual | 19x19 Est. | 19x19 Actual |
|---|---|---|---|---|
| Per-leaf CPU cost | 1.0μs | **3.2μs** | 2.5μs | **10.8μs** |
| Ratio | | **3.2x** | | **4.3x** |

Source: MCTS benchmark, batch=8 virtual loss, DESIGN.md Step 6 results.

The 19x19 cost is dominated by Game<19> state copy (~2.9KB Board + history arrays)
and deque-based game state storage at expanded nodes.

### Revised Per Tick (256 games × 8 VL = 2,048 leaves)

| | 9x9 | 19x19 (5M) | 19x19 (47M) |
|---|---|---|---|
| CPU work (single thread) | 2,048 × 3.2μs = **6.6ms** | 2,048 × 10.8μs = **22.1ms** | 22.1ms |
| CPU work (5 workers) | **1.3ms** | **4.4ms** | **4.4ms** |
| GPU forward pass (BF16) | 0.5ms | 0.5ms | **3.0ms** |
| Sync + overhead | ~0.2ms | ~0.2ms | ~0.2ms |
| **Effective per tick** | **~2.0ms** | **~5.1ms** | **~7.6ms** |
| vs original estimate | 0.7ms (2.9x) | 1.2ms (4.3x) | 3ms (2.5x) |

**Critical finding**: CPU is the bottleneck for ALL configurations, including the
47M model. The original plan assumed GPU would be the bottleneck for 47M (3ms GPU
vs 1ms CPU). In reality, CPU per worker is 4.4ms > GPU 3ms.

### Revised 9x9 Go: 400 sims/move

| Component | Original | **Revised** |
|-----------|----------|-------------|
| Effective time per tick | 0.7ms | **2.0ms** |
| Time per 256 games (3,000 ticks) | 2.1 sec | **6.0 sec** |
| Realistic (2x overhead†) | ~6 sec | **~12 sec** |
| **Throughput** | ~150K games/hr | **~77K games/hr** |
| 25K games self-play | ~10 min | **~20 min** |
| **Total per run** | ~35 min | **~45 min** |
| **Cost per run** | ~$0.7 | **~$0.9** |
| Budget for 10 runs | < $10 | **~$9** |

†Using 2x overhead (not 3x) since C++ MCTS eliminates most Python-side overhead.

**Verdict: 9x9 costs are fine.** Modest increase, well within budget.

### Revised 19x19 Go: 800 sims/move, 5M model

| Component | Original | **Revised** |
|-----------|----------|-------------|
| Effective time per tick | 1.2ms | **5.1ms** |
| Time per 256 games (22,000 ticks) | 26 sec | **112 sec** |
| Realistic (2x overhead) | ~80 sec | **~224 sec** |
| **Throughput** | ~11.5K games/hr | **~4.1K games/hr** |
| 80K games self-play | ~7 hours | **~20 hours** |
| + training | ~3 hours | ~3 hours |
| + resign savings (-25%) | ~-2 hours | **~-5 hours** |
| **Total per run** | **~8 hours** | **~18 hours** |
| **Cost per run** | ~$10 | **~$21** |

### Revised 19x19 Go: 800 sims/move, 47M model

| Component | Original | **Revised** |
|-----------|----------|-------------|
| Bottleneck | GPU (3ms) | **CPU (4.4ms)** |
| Effective time per tick | 3ms | **7.6ms** |
| Time per 256 games | 66 sec | **167 sec** |
| Realistic (2x overhead) | ~130 sec | **~334 sec** |
| **Throughput** | ~7K games/hr | **~2.8K games/hr** |
| 80K games + training | ~14 hours | **~32 hours** |
| **Cost per run** | ~$17 | **~$38** |

### Revised Total Budget

| Phase | Orig Cost/run | **Revised** | Runs | Orig Total | **Revised Total** |
|-------|--------------|-------------|------|-----------|-------------------|
| 9x9 (5M, 400 sims) | ~$0.7 | **~$0.9** | 10 | ~$7 | **~$9** |
| 19x19 (5M, 800 sims) | ~$10 | **~$21** | 3 | ~$30 | **~$63** |
| 19x19 (47M, 800 sims) | ~$17 | **~$38** | 2 | ~$34 | **~$76** |
| **Grand Total** | | | 15 | **~$71** | **~$148** |

**The $100 budget no longer holds.** The revised estimate is ~$148, about 2.1x the original.

### Why the Estimate Was Off

1. **Per-leaf cost only counted engine ops** — tree traversal, deque indexing,
   Game<N> copy with 8-position history, and node allocation were not accounted for
2. **Game<19> is larger than Board<19>** — the Game struct stores history arrays
   (8 × 361 = 2.9KB extra), making each clone ~6KB not ~3KB
3. **deque<Game<N>> access pattern** — non-contiguous memory for game state storage
   adds cache pressure vs the theoretical minimum

### Mitigations to Stay Under $100

**Option A: Reduce scope (easiest, recommended)**

| Phase | Runs | Cost |
|-------|------|------|
| 9x9 (5M) | 10 | $9 |
| 19x19 (5M) | 2 | $42 |
| 19x19 (47M) | 1 | $38 |
| **Total** | 13 | **~$89** |

Drop from 15 to 13 runs. Skip one 19x19 5M run and one 47M run.

**Option B: Optimize C++ MCTS hot path**

The biggest cost is Game<19> state management. Potential optimizations:
- **Lazy history copy**: only copy history when encoding observations, not on every
  game state clone
- **Smaller checkpoint state**: store only Board<N> at expanded nodes, replay history
  when needed
- **Contiguous arena for game states**: replace `deque<Game<N>>` with a flat arena
  to improve cache locality
- **Increase VL batch from 8 to 16**: amortize per-tick sync overhead (may reduce
  effective per-sim cost by 10-20%)

A 2x reduction in per-sim cost (10.8μs → ~5μs for 19x19) would bring the total
back to ~$100. This is realistic — lazy history copy alone should save ~40%.

**Option C: Rent more CPU cores**

RunPod offers higher-CPU configs. 16 vCPU A100 ($1.64/hr) would give 10 workers
instead of 5, halving CPU time. Net effect: ~$110 total at higher hourly rate
but fewer hours.

**Recommended strategy**: Combine A + B. Optimize the Game state clone (Option B)
during Step 7 (SelfPlayWorker), and reduce 19x19 runs to 2+1 (Option A) as fallback.

---

## Lessons from ChessRL Candidate 4 (6 Failed Versions)

| Failure | Root Cause | Go Mitigation |
|---------|-----------|---------------|
| **Draw deadlock** — 99.5% draws, no signal | Chinese Chess draws; sparse reward | Go always has a winner (komi). No draw problem. |
| **Chicken-and-egg** — bad policy → bad MCTS → worse policy | Too few sims, too small model, no pretraining | Human pretraining gives strong init; 400-800 sims |
| **Self-play overfitting** — memorizes one trick | Buffer too small (20K) | Large buffer (500K+), 8x symmetry augmentation |
| **Policy degradation** — MCTS targets ≤ pretrained policy | Weak value head + low sims = noise | Pretrained value head + 800 sims = strong targets |

---

## Training Pipeline

### Phase 0: Pretraining on Human Games (Local, Free)

**Data sources** (freely available):
- featurecat/go-dataset: 21.1M games
- KGS archives: 100K+ games
- Japanese pro archives: 88K+ games

**Training**:
- Policy: cross-entropy on human moves (~50-55% top-1 accuracy)
- Value: MSE on game outcome (+1/-1)
- Local GPU (RTX 5060 Ti), ~2-4 hours
- Same ResNet architecture used for self-play

**Why**: Skips the random-play bootstrap. MCTS starts with competent search from day 1.

### Phase 1: 9x9 Self-Play Training (1x A100, ~$2-3/run)

**Config**:

| Parameter | Value |
|-----------|-------|
| MCTS sims/move | 400 |
| Virtual loss batch | 8 |
| Parallel games | 256 (4 workers × 64) |
| Games per iteration | 256 |
| Positions/iter (×8 symmetry) | 256 × 60 × 8 = 123K |
| Replay buffer | 500K positions |
| Train batch size | 256 |
| Train steps per iter | 100 |
| LR | 0.01 → 0.001 (cosine) |
| Optimizer | SGD + momentum 0.9 |
| L2 | 1e-4 |
| Temperature | 1.0 first 15 moves, → 0.1 |
| Dirichlet alpha/epsilon | 0.11 / 0.25 |
| Komi | 7.5 |
| Total iterations | 100-200 |

**Targets**: >95% vs Random, >80% vs pure MCTS, >60% vs GnuGo lv5.

### Phase 2: 19x19 Self-Play Training (1x A100, ~$14-21/run)

Start from 19x19 pretrained weights. Same pipeline, scaled up:

| Parameter | 9x9 | 19x19 |
|-----------|-----|-------|
| Sims/move | 400 | 800 |
| Moves/game | ~60 | ~220 |
| Parallel games | 256 | 256 |
| Games per iter | 256 | 128 |
| Total games | 25K | 80K |
| Dirichlet alpha | 0.11 | 0.03 |
| Resign threshold | none | v < -0.95 for 3+ moves |

**Targets**: >99% vs Random, >90% vs GnuGo lv1, >50% vs GnuGo lv10.

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

**9x9 model**:
```
Input: (batch, 17, 9, 9)
  → Conv2d(17→128, k=3, pad=1) + BN + ReLU
  → 10 Residual Blocks (128 channels)
  → Policy Head: Conv1x1(128→2) + BN + ReLU + FC(162→82)  [81 + pass]
  → Value Head:  Conv1x1(128→1) + BN + ReLU + FC(81→256) + ReLU + FC(256→1) + tanh
Parameters: ~5M
```

**19x19 model (option A — small, fast)**:
```
Same as 9x9 but Input: (batch, 17, 19, 19)
  → Policy FC: (722→362)
  → Value FC: (361→256)
Parameters: ~5M
```

**19x19 model (option B — large, strong)**:
```
  → 20 Residual Blocks, 256 channels
Parameters: ~47M
```

Strategy: start with option A (fast self-play), then optionally train option B.

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

25K games × 60 moves × 8 = **12M training positions** from just 25K games (9x9).
80K games × 220 moves × 8 = **140M training positions** (19x19).

### 5. Replay Buffer

- Circular buffer, 500K-2M positions
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
│   └── config.py              # Model configs (9x9, 19x19, small/large)
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
- ResNet dual-head in PyTorch
- BF16 batched inference
- Wire up: C++ MCTS calls Python GPU → results back to C++
- Smoke test: random network + MCTS plays legal Go

### Step 4: Data Pipeline + Pretraining (~2 days, local)
- Download SGF records
- SGF parser → (board, move, outcome) triples
- Supervised pretraining on local GPU (~2-4 hours)
- Verify: pretrained net + MCTS plays reasonable Go

### Step 5: Self-Play Training Loop (~1 day, local)
- Multi-worker self-play + shared replay buffer + training
- 8-fold symmetry augmentation
- Resign threshold, tree reuse
- Local smoke test with tiny model on 9x9

### Step 6: Cloud Training — 9x9 (~$12)
- Rent 1x A100 80GB on RunPod ($1.19/hr)
- Full training: 400 sims, 10 blocks/128ch, 25K games
- Eval vs GnuGo
- ~5 runs of experimentation

### Step 7: Cloud Training — 19x19 (~$42-70)
- Same A100, start from pretrained 19x19 weights
- 800 sims, 80K games
- Try both 5M and 47M model
- Eval vs GnuGo levels 1-10

### Step 8: Polish & Demo (~1 day, local)
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
| Self-play doesn't improve | Human pretraining ensures strong init; 800 sims gives quality targets |
| Budget overrun | 9x9 is a complete project; 19x19 optional |

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
