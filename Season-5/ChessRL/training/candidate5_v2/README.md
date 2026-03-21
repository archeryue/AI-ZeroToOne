# Candidate 5 v2: NNUE with TD(lambda) + Cross-Entropy

Goal: **Beat minimax depth-3** (pure material eval).

## Problems with v1

1. **Crude credit assignment**: Every position in a game gets the same label (win/loss/draw outcome). A position at move 5 gets the same target as move 80. The network learns an average, not positional truth.
2. **MSE loss on WDL targets**: MSE treats the distance between 0.5 (draw) and 1.0 (win) the same as 0.0 and 0.5. Cross-entropy gives sharper gradients for confident predictions.
3. **Wasted feature space**: v1 uses 1260 features (7 types x 90 squares x 2 colors), but pieces have restricted movement zones. General can only be in 9 squares, not 90.
4. **Material blending (60/40)**: If the NNUE eval is only 77% sign-accurate, blending 40% material means the network barely contributes beyond what material already gives. The NNUE needs to be good enough to stand on its own, or at least dominate the blend.

## Changes from v1

### 1. Feature Space: 1260 -> 692

Per perspective, per color (friendly/enemy):

| Piece Type | Reachable Squares | Count |
|------------|-------------------|-------|
| General    | Palace (3x3)      | 9     |
| Advisor    | Palace diagonals  | 5     |
| Elephant   | Own-half points   | 7     |
| Horse      | Any               | 90    |
| Chariot    | Any               | 90    |
| Cannon     | Any               | 90    |
| Soldier    | Forward + cross-river | 55 |

**Per color total**: 346, **per perspective**: 692

Board is normalized so perspective's pieces are always at bottom (color-invariant features).

### 2. Loss: BCE instead of MSE

Binary cross-entropy gives stronger gradients for confident predictions near 0 and 1.

### 3. Training: TD(lambda) with Search Bootstrapping

**Self-play data generation**:
- v1 NNUE engine plays both sides at depth 4
- Epsilon-greedy noise for game diversity:
  - Moves 1-6: fully random (diverse openings)
  - Moves 7-16: search + epsilon=0.15
  - Moves 17+: search + epsilon=0.05
- Each position records the search score (minimax-backed eval)
- 3000 games -> ~339k positions, W/L/D ~27%/48%/25%

**TD(lambda=0.8) target computation**:
- Process each game backward from the final position
- `target_t = (1-lambda) * search_score_t + lambda * (1 - target_{t+1})`
- Propagates credit backward: positions that led to strong positions get higher values

### 4. Pure NNUE Eval (no material blending)

NNUE_WEIGHT = 1.0 (100% neural, 0% material). The network learns material value implicitly.

### 5. Architecture

Same structure, smaller input:
```
Input(692) -> Accumulator(692, 128) [shared, per perspective]
Concat(128, 128) = 256
-> ClippedReLU -> FC(256, 32) -> ClippedReLU
-> FC(32, 32) -> ClippedReLU -> FC(32, 1) -> Sigmoid
```
~98K parameters (down from ~170K).

## Results

### Training
- 30 epochs, ~30 seconds total on GPU
- Best val BCE: 0.5684
- **Sign accuracy: 99.8%** (vs 77% in v1)

### Benchmark: NNUE v2 (depth 4) vs opponents

| Opponent | Games | Wins | Losses | Draws | Win Rate |
|----------|-------|------|--------|-------|----------|
| Minimax depth-3 | 50 | 50 | 0 | 0 | **100%** |
| Minimax depth-4 | 20 | 10 | 0 | 10 | **75%** |

**v1 comparison**: v1 was 0-0-20 (all draws) against minimax-d3. v2 is 50-0-0.

Key: v2 beats minimax-d4 (same depth, but material-only eval) with 50% wins, 0 losses.
The draws against d4 suggest room for improvement — more training data or deeper search could help.

## Files

- `nnue_net_v2.py` — Model with 692-feature mapping table
- `gen_td_data.py` — Parallel self-play data generation (multiprocessing)
- `train_nnue_v2.py` — TD(lambda) training with BCE loss
- `export_weights_v2.py` — Export to binary for C++ engine
- `eval_nnue_v2.py` — Benchmark script
- `../../../engine_c/nnue_search_v2.h` — C++ search engine for v2

## Next Steps

- Train on full 10k games (~1.15M positions) for stronger eval
- Iterate: use v2 model for self-play -> retrain -> repeat
- Try deeper search (depth 5-6) with the improved eval
- Add search enhancements: null-move pruning, LMR for deeper effective depth
