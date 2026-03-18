# ChessRL — Reinforcement Learning for Chinese Chess

## Goal

Train an AI to play Chinese Chess (Xiangqi) using RL. We progressively improve through 5 candidates, each building on the last, and benchmark every candidate against the 3 AIs from our ChineseChess game (Random, Greedy, Minimax).

## Candidate Roadmap

| # | Candidate | Key Idea | Device | Status |
|---|-----------|----------|--------|--------|
| 1 | PPO Self-Play | Baseline — pure self-play, sparse reward | Local GPU | Done |
| 2 | PPO + Reward Shaping | Material-based intermediate rewards | Local GPU | TODO |
| 3 | PPO + Reward Shaping + Curriculum | + train vs increasingly strong opponents | Local GPU | TODO |
| 4 | Mini MCTS + NN | Lightweight search (50 sims) + small ResNet | Local GPU | TODO |
| 5 | Full AlphaZero | Full MCTS (200 sims) + ResNet | Cloud H100 | TODO |

## Benchmark Opponents

All candidates are evaluated against 3 AIs from `ChineseChess/backend/ai/`:

| AI | Strategy | Strength |
|---|---|---|
| **Random** | Picks a random legal move | Baseline, no strategy |
| **Greedy** | 1-move lookahead, maximizes material value | Captures well, no depth |
| **Minimax (depth=3)** | Alpha-beta pruning, material + positional eval | Tactical, plans 3 moves ahead |

**Evaluation format:** 20 games per opponent (10 as Red, 10 as Black), 300-step max.

---

## Project Structure

```
ChessRL/
├── env/                        # Gymnasium environment wrapper
│   ├── chess_env.py            # ChineseChessEnv (obs, action, reward, masking)
│   ├── observation.py          # Board → (15, 10, 9) tensor encoding
│   └── action_space.py         # Move ↔ action index (8100 total)
├── agents/
│   ├── model.py                # CNN Actor-Critic (Candidates 1-3)
│   ├── ppo_agent.py            # PPO agent with action masking
│   └── alphazero/              # AlphaZero agent (Candidates 4-5)
│       ├── network.py          # ResNet policy-value network
│       └── mcts.py             # Monte Carlo Tree Search
├── training/
│   ├── train_ppo.py            # Self-play PPO training (Candidate 1)
│   └── evaluate_vs_minimax.py  # Evaluation script (all opponents)
└── serve/
    └── model_server.py         # Integration with ChineseChess backend
```

## Environment Design

- **Observation:** (15, 10, 9) float32 — 14 piece planes (7 types × 2 colors) + 1 turn indicator
- **Action space:** Discrete(8100) — flat encoding of all (from_pos, to_pos) pairs on a 10×9 board
- **Action masking:** Boolean mask over 8100 actions, True = legal move
- **Rewards:** +1 win, -1 loss, 0 otherwise, -1 illegal move
- **Max steps:** 200 per game (truncated as draw)
- **Engine:** Wraps `ChineseChess/backend/engine/` (Board, Game, rules, pieces)

---

## Candidate 1: PPO Self-Play ✅

### Idea

Baseline approach — the agent plays both Red and Black, learning entirely from self-play with only terminal rewards (+1/-1 at game end). No domain knowledge injected.

### Architecture

CNN Actor-Critic with action masking:
```
Input: (batch, 15, 10, 9)
  → Conv2d(15→64, k=3, pad=1) → ReLU
  → Conv2d(64→128, k=3, pad=1) → ReLU
  → Conv2d(128→128, k=3, pad=1) → ReLU
  → Flatten (128×10×9 = 11520)
  → Linear(11520→512) → ReLU
  → Actor: Linear(512→8100)    [masked before softmax]
  → Critic: Linear(512→1)

Parameters: 10,284,709 (~10.3M)
```

### Training Config

| Param | Value |
|---|---|
| Total steps | 3,000,000 |
| Games per batch | 64 |
| Max steps/game | 200 |
| Learning rate | 3e-4 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Clip ratio | 0.2 |
| Update epochs / Minibatches | 4 / 4 |
| Entropy coef | 0.01 |
| Device | CUDA (local GPU) |

### Training Results

**Training:** 3M steps, 18,368 games, 2.9 hours, 283 FPS

| Metric | Value |
|---|---|
| Self-play results | R=6033 B=741 D=0 T=11594 |
| Best win rate vs Random | **90%** (18W/0L/2D) |
| Final win rate vs Random | 50% (10W/0L/10D) |

- Red wins much more often in self-play (6033 vs 741) — first-move advantage is significant
- 63% of games hit the 200-step truncation limit — many games end in stalemate-like loops
- Entropy dropped from ~3.4 → ~1.7 over training, showing the policy became more focused

### Benchmark vs Game AIs

**PPO vs Random (20 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Random (Black) | 7W / 0L / 3D |
| PPO (Black) vs Random (Red) | 0W / 1L / 9D |
| **Total** | **PPO 7W / Random 1W / 12D (35% win rate)** |

**PPO vs Greedy (20 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Greedy (Black) | 0W / 8L / 2D |
| PPO (Black) vs Greedy (Red) | 0W / 6L / 4D |
| **Total** | **PPO 0W / Greedy 15W / 5D (0% win rate)** |

**PPO vs Minimax depth-3 (10 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Minimax (Black) | 0W / 3L / 2D |
| PPO (Black) vs Minimax (Red) | 0W / 5L / 0D |
| **Total** | **PPO 0W / Minimax 8W / 2D (0% win rate)** |

### Analysis

- PPO as Red can beat Random (7/10), but as Black it mostly draws — learned a Red-biased strategy
- Cannot win a single game against Greedy or Minimax
- Even Greedy (1-move lookahead) dominates — PPO has no concept of material value
- The many draws (300-step truncation) show PPO learned to "not lose quickly" but can't convert to wins
- **Root cause:** Sparse reward (+1/-1 only at game end) makes credit assignment over 100+ moves nearly impossible

---

## Candidate 2: PPO + Reward Shaping ✅

### Idea

The biggest problem with Candidate 1 is **sparse reward** — the agent only learns something when the game ends. By adding material-based intermediate rewards, every capture gives immediate feedback, teaching the agent that taking pieces is good and losing pieces is bad.

### What Changes

- **Reward function:** After each move, reward = `material_delta * 0.01`
  - Capturing a Chariot (value 90): +0.9 immediate reward
  - Losing a Soldier (value 10): -0.1 immediate reward
  - Terminal rewards remain: +1 win, -1 loss
- **Material values:** Same as Greedy AI (General=10000, Chariot=90, Cannon=45, Horse=40, Advisor=20, Elephant=20, Soldier=10)
- **Everything else stays the same** — same architecture, same PPO hyperparams, same self-play

### Training Results

**Training:** 3M steps, 16,448 games, 3.1 hours, 268 FPS

| Metric | Value |
|---|---|
| Self-play results | R=2345 B=662 D=0 T=13441 |
| Best win rate vs Random | **60%** |

- Even more truncated games (82% vs 63% in Candidate 1) — reward shaping didn't help games finish faster
- Entropy dropped to ~1.0, similar to Candidate 1

### Benchmark vs Game AIs

**Candidate 2 vs Random (20 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Random (Black) | 4W / 0L / 6D |
| PPO (Black) vs Random (Red) | 1W / 2L / 7D |
| **Total** | **PPO 5W / Random 2W / 13D (25% win rate)** |

**Candidate 2 vs Greedy (20 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Greedy (Black) | 0W / 7L / 3D |
| PPO (Black) vs Greedy (Red) | 0W / 10L / 0D |
| **Total** | **PPO 0W / Greedy 17W / 3D (0% win rate)** |

**Candidate 2 vs Minimax depth-3 (10 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Minimax (Black) | 0W / 5L / 0D |
| PPO (Black) vs Minimax (Red) | 0W / 5L / 0D |
| **Total** | **PPO 0W / Minimax 10W / 0D (0% win rate)** |

### Analysis

- Reward shaping did NOT improve performance — actually slightly worse than Candidate 1 (25% vs 35% vs Random)
- Still 0% against Greedy and Minimax
- The shaped rewards may have caused the agent to optimize for captures rather than winning
- Minimax wins even faster (19-77 steps) — no draws this time, total domination
- **Conclusion:** Dense rewards alone aren't enough. The problem isn't just credit assignment — the agent fundamentally lacks look-ahead ability

---

## Candidate 3: PPO + Reward Shaping + Curriculum (TODO)

### Idea

Even with reward shaping, pure self-play has a weakness: if the agent develops bad habits early, both sides reinforce them. **Curriculum training** fixes this by mixing in games against known opponents of increasing strength, forcing the agent to learn from stronger play patterns.

### What Changes (on top of Candidate 2)

- **Training curriculum:**
  - Phase A (0-1M steps): 50% self-play + 50% vs Random
  - Phase B (1M-2M steps): 50% self-play + 25% vs Random + 25% vs Greedy
  - Phase C (2M-3M steps): 50% self-play + 25% vs Greedy + 25% vs Minimax
- Agent always learns from both sides (Red and Black trajectories)
- Reward shaping from Candidate 2 is kept

### Why This Should Help

- Playing against Greedy teaches material awareness from a different angle (opponent punishes loose pieces)
- Playing against Minimax exposes the agent to tactical patterns (forks, pins, discovered attacks)
- Self-play component still allows exploration beyond what fixed opponents show

### Training Plan

- Same architecture (10.3M params)
- 3M steps with curriculum schedule
- ~3 hours on local GPU
- Evaluate against Random, Greedy, Minimax (20 games each)

### Benchmark vs Game AIs

| Opponent | PPO Wins | Opponent Wins | Draws | PPO Win Rate |
|---|---|---|---|---|
| Random | | | | |
| Greedy | | | | |
| Minimax (d=3) | | | | |

---

## Candidate 4: Mini MCTS + NN (TODO)

### Idea

The core insight from AlphaZero: **search at inference time**. Instead of picking the action with the highest policy probability, run a lightweight Monte Carlo Tree Search (MCTS) using the neural network to evaluate positions. Even 50 simulations per move gives the agent look-ahead that pure policy cannot achieve.

### Architecture

Small ResNet + dual heads (policy + value):
```
Input: (15, 10, 9)
  → Conv2d(15→64, k=3, pad=1) + BN + ReLU
  → 5 Residual Blocks (64 channels each)
  → Policy Head: Conv1x1 → FC → 8100 logits
  → Value Head:  Conv1x1 → FC → tanh → scalar

Estimated params: ~3-5M
```

### MCTS Design

- **Simulations per move:** 50 (lightweight, fast on local GPU)
- **Selection:** PUCT (Upper Confidence bound for Trees, guided by policy head)
- **Expansion:** Use NN policy as prior probabilities
- **Evaluation:** Use NN value head (no rollouts)
- **Temperature:** 1.0 for first 30 moves (exploration), 0.1 after (exploitation)

### Training Loop (AlphaZero-style)

1. Play N self-play games using MCTS to select moves
2. Collect (board_state, MCTS_policy, game_result) tuples
3. Train network to predict MCTS policy (cross-entropy) and game result (MSE)
4. Repeat

### Training Plan

- 5 res blocks, 64 channels (~3-5M params)
- 50 MCTS sims per move
- ~100K-500K self-play games
- Estimated: 4-8 hours on local GPU (bottleneck: MCTS inference)
- Evaluate against Random, Greedy, Minimax (20 games each)

### Benchmark vs Game AIs

| Opponent | Mini-AZ Wins | Opponent Wins | Draws | Win Rate |
|---|---|---|---|---|
| Random | | | | |
| Greedy | | | | |
| Minimax (d=3) | | | | |

---

## Candidate 5: Full AlphaZero (TODO)

### Idea

Scale up Candidate 4: deeper network, more MCTS simulations, more training games. This is the full AlphaZero recipe applied to Chinese Chess, trained on cloud GPU.

### Architecture

ResNet + dual heads (policy + value):
```
Input: (15, 10, 9)
  → Conv2d(15→128, k=3, pad=1) + BN + ReLU
  → 10 Residual Blocks (128 channels each)
  → Policy Head: Conv1x1 → FC → 8100 logits
  → Value Head:  Conv1x1 → FC → tanh → scalar

Parameters: ~26.3M
```

### MCTS Design

- **Simulations per move:** 200
- **PUCT exploration constant:** c=1.5
- **Dirichlet noise at root:** alpha=0.3, epsilon=0.25
- **Temperature:** 1.0 for first 30 moves, then 0.1

### Training Plan

- 10 res blocks, 128 channels (26.3M params)
- 200 MCTS sims per move
- Target: 100K-500K self-play games
- Device: Cloud H100 (~$3/hr)
- Estimated: 3-12 hours, ~$10-$36
- Evaluate against Random, Greedy, Minimax (20 games each)

### Benchmark vs Game AIs

| Opponent | AlphaZero Wins | Opponent Wins | Draws | Win Rate |
|---|---|---|---|---|
| Random | | | | |
| Greedy | | | | |
| Minimax (d=3) | | | | |

---

## Summary Scoreboard

| Candidate | vs Random | vs Greedy | vs Minimax (d=3) |
|---|---|---|---|
| 1. PPO Self-Play | 7W/1L/12D (35%) | 0W/15L/5D (0%) | 0W/8L/2D (0%) |
| 2. PPO + Reward Shaping | 5W/2L/13D (25%) | 0W/17L/3D (0%) | 0W/10L/0D (0%) |
| 3. PPO + Reward + Curriculum | — | — | — |
| 4. Mini MCTS + NN | — | — | — |
| 5. Full AlphaZero | — | — | — |

---

## Experiments Log

| Date | Experiment | Result |
|---|---|---|
| 2026-03-17 | Candidate 1: PPO 3M self-play training | 90% win rate vs random (during training eval) |
| 2026-03-17 | Candidate 1 vs Random (20 games) | 7W/1L/12D — wins as Red, mostly draws as Black |
| 2026-03-17 | Candidate 1 vs Greedy (20 games) | 0W/15L/5D — can't compete with 1-move lookahead |
| 2026-03-17 | Candidate 1 vs Minimax depth-3 (10 games) | 0W/8L/2D — no tactical awareness |
| 2026-03-17 | Candidate 2: PPO+RewardShaping 3M training | 60% best win rate vs random (during training eval) |
| 2026-03-17 | Candidate 2 vs Random (20 games) | 5W/2L/13D (25%) — worse than Candidate 1 |
| 2026-03-17 | Candidate 2 vs Greedy (20 games) | 0W/17L/3D — still can't compete |
| 2026-03-17 | Candidate 2 vs Minimax depth-3 (10 games) | 0W/10L/0D — total domination by Minimax |
