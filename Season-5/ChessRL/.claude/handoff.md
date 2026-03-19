# ChessRL AlphaZero Project — Handoff Document
**Date: 2026-03-18**

## 1. Project Overview

Training an AlphaZero agent for **Chinese Chess (Xiangqi)** from scratch. Season 5 of user's AI/ML learning journey (Reinforcement Learning focus).

**Current phase**: Supervised pre-training on human games to bootstrap the policy network, then continue with AlphaZero self-play.

## 2. Key Architecture

### Model: `agents/alphazero/network.py`
- `AlphaZeroNet`: 5 residual blocks, 64 channels, 1,868,539 params (7.5 MB)
- Input: (15, 10, 9) observation — 14 piece planes (7 piece types × 2 colors) + 1 turn plane
- Output: policy head (8100 actions) + value head (scalar)
- Action space: 8100 = 90 source squares × 90 target squares

### C++ Game Engine: `engine_c/`
- **234x speedup** over Python engine on full game simulation
- Built with pybind11, compiled with `-O3 -march=native`
- Files: `xiangqi.h`, `xiangqi.cpp` (full engine), `bindings.cpp` (pybind11)
- Build: `cd engine_c && pip install .` (installs as `engine_c` package)
- **IMPORTANT**: Must `import engine_c` BEFORE adding ChessRL to `sys.path`, otherwise the local `engine_c/` directory shadows the installed package
- Installed in `/home/start-up/torch/` venv
- Key exports: `Board`, `Game`, `get_legal_moves`, `get_legal_action_indices`, `board_to_observation`, `get_action_mask`, `simulate_action`, `encode_move`, `decode_action`
- Correctness: 100/100 random games match Python engine (tested in `test_engine.py`)
- Performance benchmarks:
  - `get_legal_moves`: 160x faster
  - `get_action_mask`: 647x faster
  - `board_to_observation`: 20x faster
  - Full random game: 234x faster

### MCTS: `agents/alphazero/mcts.py`
- `MCTS` class: standard single-game search (used for evaluation)
- `batched_mcts_search()`: virtual loss + leaf batching for parallel self-play
- Auto-detects C++ engine via `_USE_CPP` flag
- Helper functions `_get_obs()`, `_get_mask()`, `_simulate()`, `_get_legal_actions()`, `_is_playing()`, `_get_terminal_value()` route to C++ or Python

### Python Engine (reference, slow):
- `ChineseChess/backend/engine/` — Board, Game, Rules, Pieces
- `ChessRL/env/` — observation.py, action_space.py, chess_env.py (Gymnasium wrapper)
- Piece encoding: General=1, Advisor=2, Elephant=3, Horse=4, Chariot=5, Cannon=6, Soldier=7. Red=positive, Black=negative.

## 3. Training Data

### Human Games Dataset
- **Source**: `data/community-xiangqi-games-database/` (cloned from GitHub: chasoft/community-xiangqi-games-database)
- **Format**: DhtmlXQ format files (Chinese chess web notation)
- **Raw files**: 191,234 game files (104,950 tournament, 9,573 selected, rest puzzles/openings)
- **Parsed**: 162,228 games successfully parsed → **11,025,186 training positions**
- **Results distribution**: 76,118 red wins, 39,219 black wins, 46,891 draws (27,932 unknown skipped)
- **Storage**: 23 compressed shards at `data/supervised_training_data_shard{000-022}.npz` (65.2 MB total)
- **Shard format**: Each `.npz` contains:
  - `boards`: (N, 90) int8 — flat board state
  - `actions`: (N,) int32 — action index (encode_move result)
  - `values`: (N,) float32 — game outcome from current player's perspective (+1/-1/0)
  - `turns`: (N,) int8 — current player (1=Red, -1=Black)
- **Manifest**: `data/supervised_training_data_manifest.txt`
- **Parser**: `training/parse_games.py` — parses DhtmlXQ files, saves shards

## 4. Current Status: Supervised Pre-training Running

### Script: `training/pretrain_supervised.py`
- **Running as PID 2931348** (torch venv)
- **Log**: `training/pretrain_output.log`
- **Config**: 5 epochs, batch_size=1024, lr=1e-3, CosineAnnealingLR
- **Progress at last check**: Epoch 1, batch 8800/10718 (~82% through epoch 1)
  - Policy loss: 3.27 (down from 9.0 at start)
  - Value loss: 0.575
  - Train accuracy: 23.8% (move prediction)
- **Note**: `C++ engine: False` in the log — the pre-training is running WITHOUT C++ engine for `board_to_observation` (falls back to Python). This is because the script was launched from `/tmp` but imports happen after path setup. The C++ engine import fails due to shadowing. This makes training slower but still works correctly.
- **Saves to**: `training/candidate4/az_pretrained.pt` (best val loss) and `training/candidate4/pretrain_checkpoint.pt`

### What to do after pre-training completes:
1. Load `az_pretrained.pt` weights into AlphaZero self-play training (`train_alphazero.py`)
2. Resume self-play with the bootstrapped network — it should make meaningful moves from the start instead of random play
3. The `train_alphazero.py` already has the single-process batched architecture (see section 5)

## 5. Training Script: `training/train_alphazero.py`

### Architecture: Single-process batched multi-game MCTS
- **NOT multiprocessing** — previous multi-process design had 98.5% time wasted on queue serialization
- Runs N_PARALLEL (16) games simultaneously in one process
- All MCTS leaves from all games batched into ONE GPU forward pass (avg batch size ~230)
- Zero serialization overhead, maximum GPU utilization

### Key hyperparameters:
```
NUM_BLOCKS = 5, CHANNELS = 64
NUM_SIMULATIONS = 200 (deep search, enabled by C++ engine)
VIRTUAL_LOSS_N = 16
C_PUCT = 1.5
N_PARALLEL = 16 (games running simultaneously)
MAX_GAME_STEPS = 200
GAMES_PER_ITER = 16
NUM_ITERATIONS = 300
TRAIN_STEPS_PER_ITER = 100
BATCH_SIZE = 256, LR = 2e-3
REPLAY_BUFFER_SIZE = 100,000
```

### Performance (measured):
- ~10 games/min at 200 sims (with C++ engine)
- Avg GPU batch size: 230
- ~95s per iteration (16 games)

### To add pretrained weight loading:
In `main()`, after creating the network, add:
```python
pretrained_path = os.path.join(SAVE_DIR, "az_pretrained.pt")
if os.path.exists(pretrained_path):
    network.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(f"Loaded pretrained weights from {pretrained_path}")
```

## 6. Previous Experiments (Candidates 1-3)

All used PPO (not AlphaZero). Results documented in `training/candidate2/`, `candidate2_v2/`, `candidate3/`, `candidate3_v2/`. Key insight: PPO alone couldn't learn chess — switched to AlphaZero with MCTS.

### Candidate 4 (killed runs):
1. **Multi-process Python engine, 50 sims**: ~21 games/min, all draws, no learning (GPU 10%, CPU bottleneck)
2. **Multi-process C++ engine, 200 sims**: 3.4 games/min — SLOWER because 98.5% time wasted on queue serialization between processes
3. **Single-process C++ engine, 200 sims**: ~10 games/min, avg batch 230 — much better, but still all draws after 215 iterations (no value signal)

**Root cause of all-draws**: The network starts random → plays randomly → always hits 200-step limit → value head learns nothing → policy head has no gradient signal. This is why supervised pre-training is needed to bootstrap.

## 7. Environment Setup

- **Python venv**: `/home/start-up/torch/` (Python 3.12, PyTorch 2.7.1+cu128, CUDA available)
- **Packages needed**: torch, numpy, gymnasium, pybind11, engine_c (pip install from engine_c/)
- **GPU**: NVIDIA GPU with 16GB VRAM
- **RAM**: 15.5 GB total

### To run training:
```bash
source /home/start-up/torch/bin/activate
cd /home/start-up/AI-ZeroToOne/Season-5/ChessRL
python -u training/train_alphazero.py > training/candidate4_output.log 2>&1 &
```

### To rebuild C++ engine (if modified):
```bash
source /home/start-up/torch/bin/activate
cd /home/start-up/AI-ZeroToOne/Season-5/ChessRL/engine_c
pip install .
```

## 8. Critical User Preferences (from memory)

1. **NEVER delete checkpoints, .pt files, or training artifacts** without explicit permission
2. **MUST discuss with user before changing** training algorithm, model structure, or key hyperparameters
3. **Always analyze** model size, data size, compute needs, and training time before starting any training run

## 9. File Tree (key files only)

```
Season-5/ChessRL/
├── agents/alphazero/
│   ├── mcts.py              # MCTS with C++ engine support
│   └── network.py           # AlphaZeroNet (5 blocks, 64 channels)
├── engine_c/
│   ├── xiangqi.h/cpp        # C++ game engine (234x speedup)
│   ├── bindings.cpp          # pybind11 bindings
│   ├── setup.py              # Build script
│   └── test_engine.py        # Correctness + benchmark tests
├── env/
│   ├── observation.py        # board_to_observation (Python)
│   ├── action_space.py       # encode_move, decode_action, get_action_mask
│   └── chess_env.py          # Gymnasium wrapper
├── training/
│   ├── train_alphazero.py    # Single-process batched self-play + training
│   ├── pretrain_supervised.py # Supervised pre-training on human games
│   ├── parse_games.py        # DhtmlXQ parser → sharded training data
│   ├── candidate4/           # Checkpoints (az_checkpoint.pt, az_pretrained.pt)
│   └── pretrain_output.log   # Current pre-training log
├── data/
│   ├── community-xiangqi-games-database/  # Raw DhtmlXQ game files (191K)
│   ├── supervised_training_data_shard*.npz  # 23 shards, 11M positions
│   └── supervised_training_data_manifest.txt
```

## 10. Next Steps

1. **Wait for supervised pre-training to finish** (5 epochs on 11M positions)
2. **Load pretrained weights** into `train_alphazero.py` and start self-play
3. **Monitor**: With bootstrapped policy, games should show real moves instead of random wandering → decisive outcomes → value head learns → positive feedback loop
4. **Future optimization**: If CPU-side Python MCTS becomes bottleneck again, consider implementing MCTS tree traversal in C++ too
