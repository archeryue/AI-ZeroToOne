# ChessRL Project — Handoff Document
**Date: 2026-03-21**

## 1. Project Overview

Training a **Chinese Chess (Xiangqi) AI** from scratch. Season 5 of user's AI/ML learning journey (Reinforcement Learning focus).

**Current phase**: **NNUE Candidate 5 complete** — strongest candidate. Next steps: optimize search or move to Candidate 6 (Full AlphaZero on cloud).

## 2. Candidate 5: NNUE (Current Best)

**Architecture**: 170K params, per-perspective piece-square features (1260 binary inputs per side), shared accumulator (Linear 1260→128), dense output (256→32→32→1→sigmoid).

**Eval**: 60% NNUE + 40% material (blended). Pure NNUE was overconfident; material gives concrete tactical signal.

**Search**: C++ alpha-beta with MVV-LVA move ordering, transposition table (Zobrist, 1M entries), quiescence search (depth 6). Depth 4 in ~0.6s.

**Results (depth 4, 100 games each)**:
| Opponent | Win | Loss | Draw | Score |
|---|---|---|---|---|
| Random | 79 | 0 | 21 | **89.5%** |
| Greedy | 50 | 0 | 50 | **75%** |
| Minimax-d3 | 0 | 0 | 20 | 50% |

**Zero losses across 220 games.** Best vs Greedy score of any candidate (previous best: 21%).

**Key files**:
- `training/candidate5/nnue_net.py` — NNUE network definition
- `training/candidate5/train_nnue.py` — Training pipeline (8 min to train)
- `training/candidate5/export_weights.py` — Export .pt → .bin for C++
- `training/candidate5/eval_nnue_cpp.py` — Evaluation with C++ search
- `training/candidate5/checkpoints/nnue_best.pt` — Best PyTorch model
- `training/candidate5/checkpoints/nnue_weights.bin` — C++ binary weights
- `engine_c/nnue_search.h` — C++ NNUE eval + alpha-beta search
- `engine_c/bindings.cpp` — Exposes `NNUESearch` class to Python

## 3. What Could Improve NNUE Further

- **Deeper search**: Depth 6 takes ~35s. Optimizations: null-move pruning, late move reductions, aspiration windows.
- **Incremental accumulator updates**: Currently recomputes full accumulator per position. With make/unmake tracking changed features, only update affected neurons.
- **Better training**: Train on centipawn-like targets instead of game outcomes. Or train a separate model on endgame positions.
- **Endgame play**: The draws vs Minimax are due to inability to convert advantages to checkmate. Endgame tablebases or specialized training could help.

## 4. Infrastructure

### C++ Game Engine: `engine_c/`
- **234x speedup** over Python engine
- pybind11, compiled with `-O3 -march=native`
- Build: `cd engine_c && pip install .`
- **IMPORTANT**: After rebuild, copy .so to local dir: `cp /home/start-up/torch/lib/python3.12/site-packages/engine_c/_xiangqi*.so engine_c/`
- Key exports: `Board`, `Game`, `get_legal_moves`, `NNUESearch`

### Human Games Dataset
- 162,228 games → 11M positions in 23 shards
- After 30% draw subsampling: ~8.7M training positions

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

| Candidate | vs Random | vs Greedy | Key Finding |
|---|---|---|---|
| 1. PPO Self-Play | 65% | 12.5% | Sparse reward can't teach chess |
| 2. PPO + Reward Shaping | 51-66% | 9-12.5% | Bad reward scales hurt |
| 3. PPO + Curriculum | 75-84% | 17.5-21% | Best PPO, but 0% vs Minimax |
| 4. AlphaZero (6 versions) | 40-70% peak | — | MCTS+NN fails at small scale |
| **5. NNUE** | **89.5%** | **75%** | **Best candidate. 0 losses.** |
| 6. Full AlphaZero | TODO | TODO | Cloud-scale, needs serious compute |
