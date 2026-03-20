# ChessRL AlphaZero Project — Handoff Document
**Date: 2026-03-19**

## 1. Project Overview

Training an AlphaZero agent for **Chinese Chess (Xiangqi)** from scratch. Season 5 of user's AI/ML learning journey (Reinforcement Learning focus).

**Current phase**: Candidate 4 v2 — AlphaZero self-play with move banning + material adjudication to solve the draw problem.

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

### MCTS: `agents/alphazero/mcts.py`
- `MCTS` class: standard single-game search (used for evaluation)
- `batched_mcts_search()`: virtual loss + leaf batching for parallel self-play
- Auto-detects C++ engine via `_USE_CPP` flag

### Python Engine (reference, slow):
- `ChineseChess/backend/engine/` — Board, Game, Rules, Pieces
- `ChessRL/env/` — observation.py, action_space.py, chess_env.py (Gymnasium wrapper)
- Piece encoding: General=1, Advisor=2, Elephant=3, Horse=4, Chariot=5, Cannon=6, Soldier=7. Red=positive, Black=negative.

## 3. Training Data

### Human Games Dataset
- **Source**: `data/community-xiangqi-games-database/`
- **Parsed**: 162,228 games → **11,025,186 training positions** in 23 shards (65.2 MB)
- **Shard format**: Each `.npz` contains boards (N,90) int8, actions (N,) int32, values (N,) float32, turns (N,) int8
- **Parser**: `training/parse_games.py`

## 4. Current Status & Experiment History

### Supervised Pre-training: COMPLETED
- **Best result**: Epoch 7, val_pl=2.4873, val_acc=34.6% move prediction
- **Saved**: `training/candidate4/az_pretrained.pt`

### Candidate 4 v1: AlphaZero Self-Play (FAILED — all draws)
- **Problem**: 99.5% of games ended in repetition draws → no value signal → policy degraded
- **Techniques tried (all insufficient)**:
  1. Repetition detection (3-fold → draw)
  2. Material-blended MCTS (30% material eval in leaf values)
  3. Material adjudication (draws with imbalance → win for stronger side)
  4. Human-seeded replay buffer (20K positions)
  5. Repetition penalty (-0.5 for both sides)
- **Result**: 0W/2L/8D vs Random (40% score) — worse than all PPO candidates
- **Root cause**: Chicken-and-egg deadlock. Equal networks → draws → no value signal → MCTS can't guide play → policy degrades → repeat

### Candidate 4 v2: Move Banning + Material Adjudication (CURRENT)
- **Two simple rules that broke the deadlock**:
  1. **Ban repeated moves**: If position seen 2+ times, mask out the move that leads to it (forces progress)
  2. **Material adjudication at step 200**: Side with more material wins (no more draws from step limit)
- **Result**: 93% decisive self-play games (was 0.5% in v1!)
- **Issue found**: Policy still degrades during training (PL: 0.3→2.7, eval: 35%→0% vs Random over 75 iters). The pretrained policy is destroyed by MCTS policy targets from self-play.
- **Log**: `training/candidate4_output.log`

### Candidate 4 v3 (no-pretrain): Smaller Buffer + More Train Steps
- **Changes**: 200 MCTS sims, 20K buffer (was 100K), 100 train steps/iter (was 20), no pretrained weights, half reward for step_limit wins
- **Result**: Peaked at **70% vs Random** at iter 50, then degraded to 30% by iter 166
- **Diagnosis**: Model overfits to narrow self-play patterns. Iter50 has broad strategy (advance Soldiers, develop Chariots). Iter166 memorized ONE trick (Cannon captures Horse) then shuffles forever.
- **Log**: `training/candidate4_v3_nopretrain_output.log`

### Candidate 4 v4 (step penalty): CURRENT
- **Changes from v3**: Replace half-reward for step_limit with per-step penalty after step 100 (-0.001/step, accumulating to -0.1 at step 200). Penalizes BOTH sides for long games regardless of outcome.
- **Goal**: Discourage the shuffling/stalling behavior that causes degradation

### Key Unsolved Problem: Policy Degradation
Model learns useful strategies early but overfits to narrow self-play patterns during continued training.

**Ideas to try next (in priority order)**:
1. **Checkpoint pool / league training**: Instead of self-play against current self, maintain a pool of past checkpoints. Each game randomly samples an opponent from the pool. This prevents the model from overfitting to its own current strategy and forces it to generalize.
2. **KL divergence penalty**: Add a KL(current_policy || reference_policy) term to the loss. The reference can be a snapshot from N iterations ago or the best checkpoint. This prevents the policy from drifting too far too fast, acting as a regularizer against catastrophic forgetting.
3. **Expert iteration (ExIt)**: Use MCTS to improve human game positions instead of pure self-play
4. **Mixed training**: Continuously mix human data into replay buffer (not just at start)
5. **Larger model**: 1.9M params may be too small for both policy and value

## 5. Training Script: `training/train_alphazero.py`

### Architecture: Single-process batched multi-game MCTS
- Runs N_PARALLEL (16) games simultaneously in one process
- All MCTS leaves from all games batched into ONE GPU forward pass (avg batch ~230)
- Zero serialization overhead

### Key hyperparameters:
```
NUM_BLOCKS = 5, CHANNELS = 64
NUM_SIMULATIONS = 50
VIRTUAL_LOSS_N = 8, C_PUCT = 1.5, TEMP_THRESHOLD = 30
N_PARALLEL = 16, MAX_GAME_STEPS = 200
TRAIN_STEPS_PER_ITER = 20, BATCH_SIZE = 256, LR = 5e-4
REPLAY_BUFFER_SIZE = 100,000
MATERIAL_BLEND = 0.3 (blended into MCTS leaf values)
```

### Key features in current code:
- **Move banning**: After MCTS search, masks out moves leading to 3rd repetition of a position
- **Material adjudication**: At step limit, side with more material wins
- **Human-seeded buffer**: 20K positions from supervised data loaded at start
- **Material-blended MCTS**: 30% material eval in leaf values for tactical guidance

### Launch command:
```bash
cd /tmp && source /home/start-up/torch/bin/activate
nohup python -u -c "
import engine_c
import sys
sys.path.insert(0, '/home/start-up/AI-ZeroToOne/Season-5/ChessRL')
from training.train_alphazero import main
main()
" > /home/start-up/AI-ZeroToOne/Season-5/ChessRL/training/candidate4_output.log 2>&1 &
```

## 6. Environment Setup

- **Python venv**: `/home/start-up/torch/` (Python 3.12, PyTorch 2.7.1+cu128)
- **GPU**: NVIDIA GPU with 16GB VRAM
- **RAM**: 15.5 GB total
- **Rebuild C++ engine**: `cd engine_c && pip install .`

## 7. Critical User Preferences (from memory)

1. **NEVER delete checkpoints, .pt files, or training artifacts** without explicit permission
2. **MUST discuss with user before changing** training algorithm, model structure, or key hyperparameters
3. **Always analyze** model size, data size, compute needs, and training time before starting any training run

## 8. File Tree (key files only)

```
Season-5/ChessRL/
├── agents/alphazero/
│   ├── mcts.py              # MCTS with C++ engine support
│   └── network.py           # AlphaZeroNet (5 blocks, 64 channels)
├── engine_c/
│   ├── xiangqi.h/cpp        # C++ game engine (234x speedup)
│   ├── bindings.cpp          # pybind11 bindings
│   └── setup.py              # Build script
├── env/
│   ├── observation.py        # board_to_observation (Python)
│   ├── action_space.py       # encode_move, decode_action, get_action_mask
│   └── chess_env.py          # Gymnasium wrapper
├── training/
│   ├── train_alphazero.py    # Single-process batched self-play + training
│   ├── pretrain_supervised.py # Supervised pre-training on human games
│   ├── parse_games.py        # DhtmlXQ parser → sharded training data
│   ├── candidate4/           # Checkpoints (az_checkpoint.pt, az_pretrained.pt)
│   └── candidate4_output.log # Training log
├── data/
│   ├── community-xiangqi-games-database/  # Raw game files (191K)
│   └── supervised_training_data_shard*.npz  # 23 shards, 11M positions
└── README.md                 # Full experiment results for Candidates 1-4
```

## 9. Summary of All Candidates

| Candidate | vs Random | Key Finding |
|---|---|---|
| 1. PPO Self-Play | 65% | Sparse reward can't teach chess |
| 2. PPO + Reward Shaping | 51-66% | Bad reward scales hurt; balanced helps marginally |
| 3. PPO + Curriculum | 75-84% | Best PPO result, but 0% vs Minimax |
| 4v1. AlphaZero (all draws) | 40% | Self-play deadlock: no decisive games → no learning |
| 4v2. AlphaZero (move ban) | 35%→0% | Decisive games achieved but policy degrades during training |
