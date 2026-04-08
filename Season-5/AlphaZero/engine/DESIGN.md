# C++ Go Engine — Technical Design

## Performance Requirements

This engine runs inside the MCTS self-play loop. During training on a single A100:
- **256 parallel games**, each doing **800 MCTS simulations per move**
- Each simulation: traverse tree → reach leaf → clone board → apply move → check legality
- Per second: tens of thousands of board clones + move applications
- **Target**: board clone < 1μs, move application < 2μs, legal move check < 0.5μs per position

For reference, ChessRL's C++ engine achieved **100-200x speedup** over Python.

---

## 1. Board Representation

### Option A: Simple Flat Array + BFS

```
grid[361]  (uint8_t, EMPTY/BLACK/WHITE)
ko_point   (int16_t)
hash       (uint64_t)
```

- **Clone**: memcpy(361) = ~0.1μs (excellent)
- **Place stone + capture**: BFS to find groups, check liberties → O(group_size) per neighbor
- **is_legal**: needs to tentatively place, BFS check liberties, undo → O(group_size)
- **get_legal_moves**: for each empty point, call is_legal → O(N² × avg_group_size)

**Problem**: BFS is O(group_size) and groups can be large (50+ stones in late 19x19 games). With 800 sims × 220 moves × 256 games, BFS dominates CPU time.

### Option B: Chain-Based with Incremental Liberty Tracking (Recommended)

```cpp
struct Board {
    uint8_t color[MAX_CELLS];        // EMPTY=0, BLACK=1, WHITE=2
    int16_t chain_id[MAX_CELLS];     // which chain this stone belongs to (-1 if empty)
    int16_t chain_head[MAX_CHAINS];  // head stone of each chain (for iteration)
    int16_t chain_next[MAX_CELLS];   // next stone in same chain (linked list)
    int16_t chain_liberties[MAX_CHAINS]; // liberty count per chain
    int16_t chain_size[MAX_CHAINS];  // stone count per chain
    int16_t num_chains;              // chain counter

    int16_t ko_point;                // -1 if no ko
    uint64_t hash;                   // Zobrist hash
    uint8_t size;                    // 9, 13, or 19
};
```

- **Clone**: memcpy(~4KB for 19x19) = ~0.5μs (good)
- **Place stone**: O(4) — check 4 neighbors, decrement their chain liberties, merge own chains
- **Capture**: O(captured_stones) — but we know exactly which chain to remove (liberty=0)
- **is_legal**: O(4) — check if any adjacent opponent chain has liberty=1, check own liberties
- **get_legal_moves**: O(N) for simple cases using cached liberty info

**Key advantage**: Liberty counting is incremental — we never BFS. Placing/removing a stone only updates the 4 adjacent chains' liberty counts.

### Decision: **Option B (Chain-Based)**

The incremental approach is ~10-50x faster than BFS for the hot path (is_legal, place_stone), which matters when running billions of operations during training.

The clone cost (4KB vs 0.4KB) is acceptable — memcpy(4KB) is still < 1μs.

---

## 2. Chain Operations

### Place Stone

```
place_stone(pos, color):
    1. Set color[pos] = color
    2. Update Zobrist hash
    3. Create new chain for this stone (chain_liberties = count empty neighbors)
    4. For each neighbor:
       a. If same color: merge chains (union-find style)
       b. If opponent: decrement their chain's liberty count by 1
    5. For each adjacent opponent chain with liberties == 0:
       Remove chain (set all stones to EMPTY, update neighbor liberties)
    6. If no captures and own chain liberties == 0:
       Undo and return ILLEGAL (suicide)
    7. Handle ko: if exactly 1 stone captured and own chain has exactly 1 liberty
       → set ko_point to captured position
    8. Return captured_count
```

### Remove Chain

```
remove_chain(chain_id):
    Walk linked list: chain_head[id] → chain_next[stone] → ...
    For each stone in chain:
        color[stone] = EMPTY
        Update Zobrist hash
        For each neighbor of stone:
            If neighbor is a chain of the OTHER color:
                Increment that chain's liberty count by 1
    Free the chain_id for reuse
```

### Merge Chains

```
merge_chains(id_a, id_b):
    // Append chain B's linked list to chain A
    // Combine liberty counts (subtract shared liberties)
    // Update chain_id[] for all stones in B
    // Free id_b
```

### Is Legal (O(4) — no BFS!)

```
is_legal(pos, color):
    if color[pos] != EMPTY: return false
    if pos == ko_point: return false

    // Would this capture any opponent?
    bool captures = false
    for each neighbor of pos:
        if color[neighbor] == opponent:
            if chain_liberties[chain_id[neighbor]] == 1:
                captures = true; break

    if captures: return true  // captures → always legal (not suicide)

    // Would own group have liberties?
    // Count empty neighbors (excluding pos itself)
    int own_libs = 0
    for each neighbor of pos:
        if color[neighbor] == EMPTY: own_libs++
        if color[neighbor] == color:
            own_libs += chain_liberties[chain_id[neighbor]] - 1  // -1 because pos isn't a liberty anymore
            // (simplified — need to deduplicate shared liberties)

    return own_libs > 0  // if no liberties and no captures → suicide
```

**Note**: The exact liberty calculation for merged groups needs care to avoid double-counting. In practice, a conservative check is: "does any adjacent friendly chain have ≥ 2 liberties, OR is there any empty neighbor other than pos?" If yes, it's legal.

---

## 3. Memory Layout

### For 19x19 (MAX_CELLS=361, MAX_CHAINS=180)

| Field | Type | Size | Notes |
|-------|------|------|-------|
| color | uint8_t[361] | 361B | Stone colors |
| chain_id | int16_t[361] | 722B | Which chain each stone belongs to |
| chain_next | int16_t[361] | 722B | Linked list within chain |
| chain_head | int16_t[180] | 360B | Head of each chain's linked list |
| chain_liberties | int16_t[180] | 360B | Liberty count per chain |
| chain_size | int16_t[180] | 360B | Stone count per chain |
| num_chains | int16_t | 2B | |
| ko_point | int16_t | 2B | |
| hash | uint64_t | 8B | Zobrist hash |
| size | uint8_t | 1B | Board size |
| **Total** | | **~2.9KB** | Fits in L1 cache |

For 9x9 (MAX_CELLS=81, MAX_CHAINS=40): ~700B — trivially fits in cache.

Since we need to support both 9x9 and 19x19, we compile with MAX_CELLS=361 but use `size` to limit iteration. Alternatively, we can template on board size for zero-overhead at compile time.

### Decision: Template on Board Size

```cpp
template<int N>  // N = board size (9, 13, 19)
struct Board {
    static constexpr int CELLS = N * N;
    static constexpr int MAX_CHAINS = CELLS / 2 + 1;
    // ... fields sized by CELLS and MAX_CHAINS
};
```

This gives the compiler perfect loop bounds and enables SIMD auto-vectorization. We instantiate Board<9> and Board<19> explicitly.

---

## 4. Zobrist Hashing

```cpp
// Initialized once at startup with fixed seed (reproducible)
struct ZobristTable {
    uint64_t stone[2][N*N];   // [color-1][position]  (BLACK=0, WHITE=1)
    uint64_t side_to_move;    // XOR'd when it's BLACK's turn
};
```

**Incremental update**: XOR on each stone placement/removal.
**Used for**: ko detection, superko (optional), transposition table (optional).

---

## 5. Observation Encoding (17 Planes)

Following AlphaGo Zero, the observation is a `(17, N, N)` float32 tensor:

| Planes | Content | Description |
|--------|---------|-------------|
| 0 | Current player stones (t=0) | 1.0 where current player has a stone |
| 1 | Opponent stones (t=0) | 1.0 where opponent has a stone |
| 2-3 | Stones at t=-1 | Previous position |
| 4-5 | Stones at t=-2 | Two moves ago |
| ... | ... | ... |
| 12-13 | Stones at t=-6 | Six moves ago |
| 14-15 | Stones at t=-7 | Seven moves ago |
| 16 | Color to play | All 1.0 if BLACK, all 0.0 if WHITE |

**History storage**: The Game class stores the last 8 board positions (just the color arrays, not the full chain state). This costs 8 × 361 = 2.9KB for 19x19 — negligible.

```cpp
template<int N>
struct Game {
    Board<N> board;
    uint8_t history[8][N*N];  // Last 8 board positions (color arrays only)
    int history_len;           // How many history entries we have
    int8_t current_turn;       // BLACK=1, WHITE=2
    int8_t status;             // PLAYING, BLACK_WIN, WHITE_WIN
    int consecutive_passes;
    int move_count;
    int captured[3];           // captured[BLACK], captured[WHITE]
};
```

---

## 6. Action Space

```
Action encoding:
  action = row * N + col       (for stone placement, 0 to N²-1)
  action = N * N               (for pass)
  Total actions = N² + 1       (82 for 9x9, 362 for 19x19)
```

**Action mask**: bool array of size N²+1, true for legal moves + pass.

---

## 7. MCTS Integration

The C++ engine must support the MCTS workflow efficiently:

### 7.1 MCTS Node (C++)

```cpp
struct MCTSNode {
    int action;              // action that led to this node (-1 for root)
    float prior;             // P(s,a) from NN policy
    int visit_count;         // N(s,a)
    float value_sum;         // W(s,a)
    int virtual_loss;        // for parallel leaf selection
    int16_t children_start;  // index into children pool
    int16_t num_children;    // number of children

    float q_value() const { return visit_count > 0 ? value_sum / visit_count : 0.0f; }
};
```

### 7.2 MCTS Tree (C++)

```cpp
template<int N>
struct MCTSTree {
    // Arena allocator for nodes — no per-node malloc
    std::vector<MCTSNode> node_pool;
    int root_idx;

    // Game state per node (only stored for expanded nodes)
    // Using hash map: node_idx → Game<N>
    // Or: don't store states, replay from root each time (trades memory for compute)

    // Leaf selection: returns path from root to leaf
    std::vector<int> select_leaf(float c_puct);

    // Expansion: create children for a leaf node
    void expand(int node_idx, const float* policy, int num_actions);

    // Backup: propagate value up the path
    void backup(const std::vector<int>& path, float value);
};
```

### 7.3 Leaf Evaluation Interface (C++ → Python)

The critical boundary: C++ MCTS selects leaves, Python evaluates them on GPU.

```cpp
// C++ side: output struct for leaves needing NN evaluation
struct LeafRequest {
    float obs[17 * N * N];   // flattened observation
    int game_idx;             // which parallel game this is from
    int node_idx;             // which tree node
};

// Python calls these C++ functions:
// 1. Advance all games one MCTS tick, collect leaves
std::vector<LeafRequest> collect_leaves(int batch_size);

// 2. Feed NN results back
void process_nn_results(
    const float* policies,    // (batch, N²+1)
    const float* values,      // (batch,)
    int batch_size
);
```

### 7.4 Game State Management for MCTS

Two approaches for managing game states in the MCTS tree:

**Approach A: Store states at nodes** (memory-intensive)
- Each expanded node stores a full Game state
- Leaf selection: no state replay needed
- Memory: ~3KB × nodes_per_tree × 256_games = could be 100s of MB

**Approach B: Replay from root** (compute-intensive)
- Only root stores the game state
- Leaf selection: replay moves from root along the path
- Memory: minimal (just the path)
- Compute: O(depth) move applications per leaf

**Approach C: Hybrid — store on path** (recommended)
- Store game state at root and every K levels (checkpoints)
- Replay from nearest checkpoint
- Balance between memory and compute

**Decision: Approach A** — memory is cheap (A100 has 117GB RAM), and the speed difference is significant for 800 sims × 256 games. With ~200 nodes per tree × 256 games × 3KB = ~150MB — easily fits in RAM.

---

## 8. Multi-Worker Architecture

```
┌─────────────────────────────────────────────────┐
│              Python Orchestrator                 │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │ ...  │
│  │ (C++ lib)│  │ (C++ lib)│  │ (C++ lib)│      │
│  │ 52 games │  │ 52 games │  │ 52 games │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │              │              │            │
│       └──────────────┼──────────────┘            │
│                      ▼                           │
│            Shared Memory Buffer                  │
│         (leaf observations + results)            │
│                      ▼                           │
│              GPU Inference                       │
│           (torch BF16 batched)                   │
└─────────────────────────────────────────────────┘
```

Each worker is a Python `multiprocessing.Process` that calls into the C++ library via pybind11. Workers and the GPU process communicate via shared memory to avoid serialization overhead.

### Shared Memory Protocol

```
Layout (per tick):
  observations_buffer: float32[MAX_BATCH × 17 × N × N]  // workers write, GPU reads
  policy_buffer:       float32[MAX_BATCH × (N²+1)]       // GPU writes, workers read
  value_buffer:        float32[MAX_BATCH]                 // GPU writes, workers read
  ready_count:         int32                              // atomic counter
  done_flag:           int32                              // GPU signals completion
```

Workers fill observations_buffer with leaf observations, then wait. GPU inference process reads, runs forward pass, writes results, signals done. Workers read results and continue.

---

## 9. Scoring

**Tromp-Taylor area scoring** (same as Python engine):
- Count stones of each color
- Flood-fill empty regions, assign to color if single-color border
- White gets komi (7.5 for 19x19, 5.5 for 9x9)

This is called once per game end — performance is not critical. BFS is fine here.

---

## 10. File Structure

```
AlphaZero/engine/
├── DESIGN.md          # This document
├── go.h               # Board<N>, Game<N>, all core types and logic
├── go.cpp             # Template instantiation + non-template helpers
├── mcts.h             # MCTSNode, MCTSTree<N>
├── mcts.cpp           # MCTS implementation
├── worker.h           # SelfPlayWorker<N> — manages parallel games + MCTS
├── worker.cpp         # Worker implementation
├── zobrist.h          # Zobrist hash tables (header-only)
├── bindings.cpp       # pybind11 bindings
├── setup.py           # Build script (C++17, -O3)
├── __init__.py        # Python module
└── tests/
    ├── test_go.cpp        # C++ unit tests (board, rules, capture, ko, scoring)
    ├── test_mcts.cpp      # C++ MCTS tests
    └── test_engine.py     # Python tests (C++ vs Python engine comparison)
```

---

## 11. API Surface (pybind11)

### For Testing & Evaluation

```python
import go_engine

# Create board and game
board = go_engine.Board9()      # or Board19()
game = go_engine.Game9()        # or Game19()

# Basic operations
game.place_stone(row, col)       # returns captured count, raises on illegal
game.pass_move()
game.is_legal(row, col) -> bool
game.get_legal_moves() -> list[(row, col)]
game.score(komi) -> (black_score, white_score)
game.status -> int
game.current_turn -> int
game.board_grid() -> list[list[int]]

# Observation encoding
game.to_observation() -> numpy (17, N, N) float32
game.get_action_mask() -> numpy (N²+1,) bool
```

### For Training (Batch Self-Play)

```python
# Create worker managing multiple parallel games + MCTS
worker = go_engine.SelfPlayWorker9(
    num_games=64,
    mcts_sims=800,
    c_puct=1.5,
    dirichlet_alpha=0.11,
)

# One tick: select leaves, return observations for NN evaluation
leaf_obs = worker.collect_leaves()  # numpy (batch, 17, 9, 9) float32

# Feed NN results back
worker.process_results(policies, values)  # numpy arrays

# Check if any games are complete
completed = worker.get_completed_games()
# Returns list of (states, policies, outcome) for each completed game
```

---

## 12. Testing Strategy

### Level 1: C++ Unit Tests (test_go.cpp)

Basic rule correctness:
- Stone placement on empty board
- Simple capture (single stone, group)
- Ko detection and enforcement
- Suicide prevention
- Superko (Zobrist hash)
- Pass handling
- Tromp-Taylor scoring (various board positions)
- Chain merge correctness
- Liberty counting accuracy

### Level 2: Python Comparison Tests (test_engine.py)

Compare C++ engine against our Python Go engine (`Go/backend/engine/`):
- **Random game test**: Play 1000 random 9x9 games, verify boards match at every step
- **Legal move test**: At each step, verify legal move sets are identical
- **Scoring test**: Compare scores on various positions
- **FEN/string round-trip**: Board serialization matches

### Level 3: Performance Benchmarks

- `place_stone`: target < 2μs (19x19)
- `is_legal`: target < 0.5μs per position (19x19)
- `get_legal_moves`: target < 50μs (19x19, ~150 legal moves)
- `board_clone`: target < 1μs (19x19)
- `to_observation`: target < 20μs (19x19)
- `random_game (9x9)`: target > 50K games/sec per core
- `random_game (19x19)`: target > 5K games/sec per core

### Level 4: MCTS Integration Tests

- Single-game MCTS with random NN → verify tree statistics
- Virtual loss correctness
- Tree reuse between moves
- Batch leaf collection across parallel games

---

## 13. Build & Dependencies

```python
# setup.py
compile_args = ["-O3", "-DNDEBUG", "-std=c++17", "-march=native"]
# On Apple Silicon: "-mcpu=apple-m1"
# On x86: "-march=native" (enables AVX2 if available)

ext_modules = [
    Pybind11Extension(
        "go_engine._engine",
        ["go.cpp", "mcts.cpp", "worker.cpp", "bindings.cpp"],
        extra_compile_args=compile_args,
        include_dirs=["."],
    ),
]
```

---

## 14. Implementation Order

| Step | What | Test | Status |
|------|------|------|--------|
| 1 | `go.h/cpp`: Board<N> with chain-based groups, place_stone, is_legal | C++ unit tests | ✅ Done |
| 2 | `go.h/cpp`: Game<N> with history, pass, scoring | C++ unit tests | ✅ Done |
| 3 | `bindings.cpp` + `setup.py`: expose Board and Game to Python | Python comparison tests | ✅ Done |
| 4 | `go.h/cpp`: observation encoding, action mask | Compare with Python | ✅ Done |
| 5 | Performance benchmarks | Meet targets above | ✅ Done |
| 6 | `mcts.h/cpp`: MCTSNode, tree operations, virtual loss | MCTS unit tests | ✅ Done |
| 7 | `worker.h/cpp`: SelfPlayWorker with parallel games | Integration tests | Next |
| 8 | Shared memory interface for GPU batching | End-to-end test | |

### Actual Benchmark Results (Steps 1-5)

Apple M1, `-O3`, all 19 correctness tests passing, 178x speedup vs Python.

| Metric | Target | Actual |
|--------|--------|--------|
| `place_stone` (19x19) | < 2μs | **0.36μs** |
| `is_legal` per position (19x19) | < 0.5μs | **0.002μs** (2ns) |
| Random game (9x9 incl. get_legal_moves) | 50K games/sec | 11.6K games/sec |
| Random game (19x19 incl. get_legal_moves) | 5K games/sec | 2.3K games/sec |
| C++ vs Python speedup | 100-200x | **178x** |

Note: games/sec targets assumed MCTS-style usage without `get_legal_moves` per move.
The MCTS hot path (`place_stone` = 0.36μs) is well under the 2μs target.
Engine-only per-leaf cost: ~1.5μs (clone + place_stone + check). However, the full
MCTS sim cost measured in Step 6 is 3.2μs (9x9) / 10.8μs (19x19) — see below.

### Step 6 Results — C++ MCTS ✅

Implemented in `mcts.h` (header-only logic) + `mcts.cpp` (template instantiation):

1. **MCTSNode** — arena-allocated, PUCT with virtual loss
2. **MCTSTree<N>** — `select_leaf/leaves()`, `expand()`, `backup()`, `get_policy()`, `advance()`
3. **Dirichlet noise** at root (`apply_dirichlet_noise()`)
4. **Virtual loss** — batch leaf collection (8 leaves/tick yields 8 unique paths)
5. **Tree reuse** — `advance(action)` promotes subtree
6. **Game state storage** — `deque<Game<N>>` indexed by `MCTSNode::game_idx` (only expanded nodes)
7. **pybind11 bindings** — `MCTSTree9/13/19` exposed to Python

#### Performance optimizations (v3)

- `deque<Game<N>>` instead of `unordered_map` or `vector` — no hash overhead, no reallocation copies
- `MCTSNode::is_terminal` flag — avoids game-state lookup in `select_leaf` traversal
- `MCTSNode::game_idx` — O(1) game state access vs O(1) amortized hash lookup
- Fixed-size `LeafInfo` path buffer (64 ints) — no heap allocation per leaf
- Bulk child allocation in `expand()` — single `resize()` instead of N `emplace_back()`

#### MCTS Benchmark Results

| Metric | 9x9 | 19x19 |
|--------|-----|-------|
| Sims/sec (single leaf) | **215K** | **77K** |
| Sims/sec (batch=8) | **312K** | **92K** |
| Time per sim (single) | **4.7μs** | **13.0μs** |
| Time per sim (batch=8) | **3.2μs** | **10.8μs** |

**Note**: Per-sim cost (10.8μs for 19x19) is ~4.3x the PLAN.md per-leaf estimate of
2.5μs, which only counted engine ops (clone + place_stone). The full sim cost
includes tree traversal, Game<N> state copy with history, deque access, node
allocation, and backup. See PLAN.md "Revised Cost Estimates" for updated budget.

#### Tests (12/12 passing)

- Tree creation, root expansion
- PUCT selection (prefers high prior)
- Backup propagation (alternating sign)
- Virtual loss (8 unique paths from uniform prior)
- Full simulation cycle (100 sims)
- Tree reuse (subtree promoted, visits preserved)
- Dirichlet noise (prior smoothing)
- Policy temperature (proportional vs argmax)
- Integration (random NN plays legal 9x9 Go)
- Batch leaf collection
- Tree reuse with continued simulations

### Next: Step 7 — SelfPlayWorker

Key components to build:

1. **SelfPlayWorker<N>** — manages multiple parallel games + MCTS trees
   - Each worker owns `num_games` MCTS trees
   - `collect_leaves()` — select leaves across all games, return observation batch
   - `process_results()` — feed NN outputs back, expand + backup
   - `get_completed_games()` — return finished game data (states, policies, outcomes)

2. **Shared memory interface** for GPU batching (Step 8)
