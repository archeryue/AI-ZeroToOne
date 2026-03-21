# ChessRL Project — Handoff Document
**Date: 2026-03-20**

## 1. Project Overview

Training a **Chinese Chess (Xiangqi) AI** from scratch. Season 5 of user's AI/ML learning journey (Reinforcement Learning focus).

**Current phase**: Moving to **NNUE** (Efficiently Updatable Neural Network) approach after 6 failed AlphaZero variants.

## 2. What Failed: Mini-AlphaZero (Candidate 4, v1-v6)

MCTS + neural network self-improvement doesn't work at small scale (1 GPU, 300 sims, 1.9M params). The MCTS policy improvement loop never bootstraps because the value head starts random → MCTS search is unguided → policy targets are noise → network doesn't improve → loop.

Best result: v3 peaked at **70% vs Random** at iter 50, then degraded to 30% via self-play overfitting. All other variants (curriculum, pure checkmate reward, material draw reward) produced 0 wins vs Random.

See README.md for full experiment details and insights.

## 3. Next Direction: NNUE

**Key idea**: Separate evaluation from search. Train a neural network to evaluate positions (supervised on human games), then plug it into alpha-beta search.

**Why NNUE:**
- No self-play loop needed — supervised training on 11M human positions we already have
- Alpha-beta search already works — our Minimax depth-3 beats all RL candidates
- NNUE replaces handcrafted eval with learned eval — strictly better
- Efficient: sparse architecture with incremental updates enables deep search on local hardware
- Proven at small scale: Stockfish NNUE runs on single CPU and dominates

**Plan:**
1. Train NNUE network to predict game outcomes from human game positions
2. Plug into alpha-beta search (C++ engine already supports this)
3. Evaluate vs Random, Greedy, Minimax

## 4. Infrastructure Available

### C++ Game Engine: `engine_c/`
- **234x speedup** over Python engine
- pybind11, compiled with `-O3 -march=native`
- Build: `cd engine_c && pip install .`
- **IMPORTANT**: Must `import engine_c` BEFORE adding ChessRL to `sys.path`
- Key exports: `Board`, `Game`, `get_legal_moves`, `get_legal_action_indices`, `board_to_observation`, `get_action_mask`, `simulate_action`

### Human Games Dataset
- **Source**: `data/community-xiangqi-games-database/`
- **Parsed**: 162,228 games → **11,025,186 training positions** in 23 shards (65.2 MB)
- **Shard format**: `.npz` with boards (N,90) int8, actions (N,) int32, values (N,) float32, turns (N,) int8
- **Parser**: `training/parse_games.py`

### Existing Model (for reference)
- `AlphaZeroNet`: 5 res blocks, 64 channels, 1.9M params — policy + value heads
- Supervised pre-training: 34.6% move prediction accuracy on human games

## 5. Environment Setup

- **Python venv**: `/home/start-up/torch/` (Python 3.12, PyTorch 2.7.1+cu128)
- **GPU**: NVIDIA GPU with 16GB VRAM
- **RAM**: 15.5 GB total
- **Rebuild C++ engine**: `cd engine_c && pip install .`

## 6. Critical User Preferences

1. **NEVER delete checkpoints, .pt files, or training artifacts** without explicit permission
2. **MUST discuss with user before changing** training algorithm, model structure, or key hyperparameters
3. **Always analyze** model size, data size, compute needs, and training time before starting any training run

## 7. Summary of All Candidates

| Candidate | vs Random | Key Finding |
|---|---|---|
| 1. PPO Self-Play | 65% | Sparse reward can't teach chess |
| 2. PPO + Reward Shaping | 51-66% | Bad reward scales hurt; balanced helps marginally |
| 3. PPO + Curriculum | 75-84% | Best PPO result, but 0% vs Minimax |
| 4. AlphaZero (6 versions) | 40-70% peak, all degraded | MCTS+NN self-improvement doesn't work at small scale |
| 5. NNUE | TODO | Supervised NN eval + alpha-beta search |
| 6. Full AlphaZero | TODO (future) | Cloud-scale MCTS+NN, needs serious compute |
