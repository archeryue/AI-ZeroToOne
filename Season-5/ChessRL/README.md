# ChessRL — Reinforcement Learning for Chinese Chess

## Goal

Train an AI to play Chinese Chess (Xiangqi) using RL. We progressively improve through 5 candidates, each building on the last, and benchmark every candidate against the 3 AIs from our ChineseChess game (Random, Greedy, Minimax).

## Candidate Roadmap

| # | Candidate | Key Idea | Device | Status |
|---|-----------|----------|--------|--------|
| 1 | PPO Self-Play | Baseline — pure self-play, sparse reward | Local GPU | Done |
| 2 | PPO + Reward Shaping | Material-based intermediate rewards | Local GPU | Done |
| 3 | PPO + Reward Shaping + Curriculum | + train vs increasingly strong opponents | Local GPU | Done |
| 4 | Mini MCTS + NN | Lightweight search (50 sims) + small ResNet | Local GPU | TODO |
| 5 | Full AlphaZero | Full MCTS (200 sims) + ResNet | Cloud H100 | TODO |

## Benchmark Opponents

All candidates are evaluated against 3 AIs from `ChineseChess/backend/ai/`:

| AI | Strategy | Strength |
|---|---|---|
| **Random** | Picks a random legal move | Baseline, no strategy |
| **Greedy** | 1-move lookahead, maximizes material value | Captures well, no depth |
| **Minimax (depth=3)** | Alpha-beta pruning, material + positional eval | Tactical, plans 3 moves ahead |

**Evaluation format:** 20 games per side (40 total) per opponent, 300-step max.

**Scoring:** Chess tournament scoring — Win=1, Draw=0.5, Loss=0. Score% = (W + 0.5×D) / Total × 100.

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
| **Total** | **PPO 7W / Random 1W / 12D — Score: 65%** |

**PPO vs Greedy (20 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Greedy (Black) | 0W / 8L / 2D |
| PPO (Black) vs Greedy (Red) | 0W / 6L / 4D |
| **Total** | **PPO 0W / Greedy 15W / 5D — Score: 12.5%** |

**PPO vs Minimax depth-3 (10 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Minimax (Black) | 0W / 3L / 2D |
| PPO (Black) vs Minimax (Red) | 0W / 5L / 0D |
| **Total** | **PPO 0W / Minimax 8W / 2D — Score: 10%** |

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

**Candidate 2 vs Random (40 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Random (Black) | 5W / 3L / 12D |
| PPO (Black) vs Random (Red) | 0W / 1L / 19D |
| **Total** | **PPO 5W / Random 4W / 31D — Score: 51%** |

**Candidate 2 vs Greedy (40 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Greedy (Black) | 0W / 15L / 5D |
| PPO (Black) vs Greedy (Red) | 0W / 18L / 2D |
| **Total** | **PPO 0W / Greedy 33W / 7D — Score: 9%** |

**Candidate 2 vs Minimax depth-3 (40 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Minimax (Black) | 0W / 20L / 0D |
| PPO (Black) vs Minimax (Red) | 0W / 20L / 0D |
| **Total** | **PPO 0W / Minimax 40W / 0D — Score: 0%** |

### Analysis

- Reward shaping did NOT improve performance — slightly worse than Candidate 1 (51% vs 65% score vs Random)
- Still 0% against Greedy and Minimax
- The shaped rewards may have caused the agent to optimize for captures rather than winning
- Minimax wins even faster (19-77 steps) — no draws this time, total domination
- **Conclusion:** Dense rewards alone aren't enough. The problem isn't just credit assignment — the agent fundamentally lacks look-ahead ability

### Candidate 2 v2: Balanced Reward Values

Retrained with balanced piece values (win=1.0, chariot=0.333, cannon/horse=0.167, advisor/elephant=0.1, soldier=0.033) instead of the original scale (win=1.0, chariot=0.9, cannon=0.45). The old values made two Chariots worth more than winning.

**Training:** 3M steps, 16,960 games, 2.2 hours, 388 FPS. Best win rate vs random: **90%**.

| Opponent | PPO Wins | Opp Wins | Draws | Score |
|---|---|---|---|---|
| Random (40g) | 14 (13R+1B) | 1 (0R+1B) | 25 | **66%** |
| Greedy (40g) | 0 | 30 (14R+16B) | 10 | **12.5%** |
| Minimax d=3 (40g) | 0 | 38 (18R+20B) | 2 | **2.5%** |

- Big improvement over v1 (51% → 66% vs Random) — balanced rewards helped significantly
- 2 draws vs Minimax (vs 0 in v1) — slightly more resilient
- Still 0% vs Greedy and Minimax — confirms PPO without search can't compete with tactical AIs

---

## Candidate 3: PPO + Reward Shaping + Curriculum ✅

### Idea

Even with reward shaping, pure self-play has a weakness: if the agent develops bad habits early, both sides reinforce them. **Curriculum training** fixes this by mixing in games against known opponents of increasing strength, forcing the agent to learn from stronger play patterns.

### What Changes (on top of Candidate 2)

- **Training curriculum:**
  - Phase A (0-1M steps): 50% self-play + 50% vs Random
  - Phase B (1M-2M steps): 50% self-play + 25% vs Random + 25% vs Greedy
  - Phase C (2M-3M steps): 50% self-play + 25% vs Greedy + 25% vs Minimax
- Agent always learns from both sides (Red and Black trajectories)
- Reward shaping from Candidate 2 is kept

### Training Results

**Training:** 3M steps, 18,624 games, 2.3 hours, 356 FPS

| Metric | Value |
|---|---|
| Self-play results | R=2386 B=361 T=6565 |
| vs AI results | W=2212 L=2769 |
| Best win rate vs Random | **100%** |

- Phase A (vs Random): Win rate climbed to 90-100%
- Phase B (vs Greedy): Win rate vs random stayed 85-90%
- Phase C (vs Minimax): Win rate vs random fluctuated 20-90% — Minimax opponent was destabilizing

### Benchmark vs Game AIs

**Candidate 3 vs Random (40 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Random (Black) | 14W / 0L / 6D |
| PPO (Black) vs Random (Red) | 6W / 0L / 14D |
| **Total** | **PPO 20W / Random 0W / 20D — Score: 75%** |

**Candidate 3 vs Greedy (40 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Greedy (Black) | 1W / 14L / 5D |
| PPO (Black) vs Greedy (Red) | 0W / 10L / 10D |
| **Total** | **PPO 1W / Greedy 24W / 15D — Score: 21%** |

**Candidate 3 vs Minimax depth-3 (40 games):**

| Matchup | Result |
|---|---|
| PPO (Red) vs Minimax (Black) | 0W / 20L / 0D |
| PPO (Black) vs Minimax (Red) | 0W / 20L / 0D |
| **Total** | **PPO 0W / Minimax 40W / 0D — Score: 0%** |

### Analysis

- Big improvement over Candidate 2 vs Random: 75% score (vs 51%) — never loses, many draws
- First win against Greedy (1W) — progress but still dominated
- Still 0% vs Minimax — no draws at all, gets crushed in ~20-30 moves
- As Black, agent is much more defensive (draws frequently) — learned to survive but not attack
- Curriculum helped learn from stronger opponents, but PPO policy alone still lacks look-ahead
- **Conclusion:** Curriculum training is the best improvement so far for pure PPO, but hitting a ceiling — the agent needs search (MCTS) to compete with tactical AIs

### Candidate 3 v2: Balanced Reward Values

Retrained with same balanced rewards as Candidate 2 v2 (win=1.0, chariot=0.333, etc.) + curriculum.

**Training:** 3M steps, 19,136 games, 2.8 hours, 304 FPS. Best win rate vs random: **100%**.

| Opponent | PPO Wins | Opp Wins | Draws | Score |
|---|---|---|---|---|
| Random (40g) | 27 (20R+7B) | 0 | 13 | **84%** |
| Greedy (40g) | 1 | 27 | 12 | **17.5%** |
| Minimax d=3 (40g) | 0 | 40 | 0 | **0%** |

- Best result yet: 84% score vs Random, **perfect 20/20 as Red**
- As Black still draws often (13/20) — defensive but not aggressive enough
- Balanced rewards + curriculum is the winning combo for pure PPO
- Still 0% vs Minimax — confirms PPO without search fundamentally can't compete

---

## Candidate 4: Mini AlphaZero ✅

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

Parameters: 1,868,539 (~1.9M, 7.5 MB)
```

### Infrastructure Built

1. **C++ Game Engine (pybind11)**: 234x speedup over Python for game simulation. Uses make/unmake pattern with targeted check detection.
2. **Single-process batched multi-game MCTS**: 16 games run simultaneously, all MCTS leaves batched into one GPU call (avg batch=230). Eliminates multiprocessing queue serialization that wasted 98.5% of time in the first implementation.
3. **Supervised pre-training pipeline**: Parser for DhtmlXQ game format + sharded training data (11M positions from 162K human games, 65MB). Pre-trained policy to 34.6% move prediction accuracy.

### Candidate 4 v1: AlphaZero Self-Play

**Training Config:**

| Param | Value |
|---|---|
| MCTS sims per move | 50 |
| Virtual loss batch | 8 |
| c_puct | 1.5 |
| Temperature threshold | 30 moves |
| Parallel games | 16 |
| Max game steps | 200 |
| Learning rate | 5e-4 |
| Train steps per iter | 20 |
| Batch size | 256 |
| Replay buffer | 100,000 |
| Iterations | 300 (ran 212) |

**Techniques to combat the all-draws problem:**
1. **Repetition detection**: 3-fold repetition → draw (ends games at ~55 steps instead of 200)
2. **Material-blended MCTS**: 30% material eval blended into leaf values for tactical guidance
3. **Material adjudication**: Draws with material imbalance (>1.5 points) treated as wins for stronger side
4. **Human-seeded replay buffer**: 20K positions from human games loaded at start (provides value signal with known game outcomes)
5. **Repetition penalty**: Equal-material repetition draws penalized (-0.5) for both sides

**Training Results (212 iterations, 3418 games, 63 min):**

| Metric | Iter 1 | Iter 50 | Iter 100 | Iter 150 | Iter 200 |
|---|---|---|---|---|---|
| Policy Loss | 0.15 | 1.95 | 2.72 | 2.79 | 2.78 |
| Value Loss | 0.53 | 0.17 | 0.13 | 0.11 | 0.10 |
| Avg Game Length | 72 | 60 | 65 | 56 | 50 |
| Throughput (g/m) | 62 | 56 | 69 | 65 | 66 |

| Eval vs Random | Iter 25 | Iter 50 | Iter 75 | Iter 100 | Iter 125 | Iter 150 | Iter 175 | Iter 200 |
|---|---|---|---|---|---|---|---|---|
| Result | 0W/0L/10D | 0W/1L/9D | 0W/0L/10D | 0W/0L/10D | 0W/1L/9D | 0W/1L/9D | 0W/2L/8D | 0W/2L/8D |
| Score | 50% | 45% | 50% | 50% | 45% | 45% | 40% | 40% |

**Self-play decisive games:** 13 Red wins + 5 Black wins out of 3418 games total (**0.5% decisive rate**)

### Analysis

**What worked:**
- **C++ engine + batched MCTS**: Throughput of 65 games/min with 50 MCTS sims (234x engine speedup + zero serialization overhead)
- **Repetition detection**: Cut average game length from 200 to 55 steps, 3.6x throughput improvement
- **Human-seeded replay buffer**: Gave value head real signal immediately (VL went from 0.53→0.10, showing learning)

**What failed:**
- **0 wins against random player** in evaluation — worse than every PPO candidate
- **Policy loss increased** from 0.15→2.78 — the pretrained human-game policy was destroyed by MCTS self-play training, and nothing better replaced it
- **Only 0.5% decisive self-play games** — 99.5% end in repetition draws despite all the countermeasures
- **Material adjudication** decreased over time (adj went from ~12 to ~7 per 16 games) — the network learned to maintain material balance, reducing the signal
- **Value head learned from human data but couldn't transfer** — it predicts values well on training data (VL=0.10) but can't guide MCTS to find checkmate sequences

**Root cause diagnosis — Why self-play produces all draws:**

The problem is a **chicken-and-egg deadlock**:

1. Both sides use the **same network** → equally strong → no side can create advantage
2. Equal play → games end in repetition → no decisive outcomes
3. No decisive outcomes → value head learns constant values → flat position evaluation
4. Flat evaluation → MCTS can't distinguish winning from losing positions
5. MCTS without value guidance → poor policy targets → pretrained policy degrades
6. Weaker policy → even less able to find winning moves → back to step 1

This is fundamentally different from the original AlphaZero which used **5000 TPUs, 800 MCTS sims, 30M parameter network**, and millions of games. At our scale (1 GPU, 50 sims, 1.9M params), the system doesn't generate enough variance to bootstrap learning.

### Candidate 4 v2: Move Banning + Material Adjudication

**Key insight**: Instead of detecting repetition and declaring draw, **prevent repetition from happening** and **eliminate draws entirely**.

Two simple rules:
1. **Ban repeated moves**: If a position has been seen 2+ times, mask out the move that leads to it. Forces the agent to try new moves instead of shuffling.
2. **Material adjudication at step 200**: If the game hits the step limit, the side with more material wins (not a draw).

Same config as v1, plus human-seeded replay buffer (20K positions) and material-blended MCTS (30%).

**Training Results (77 iterations, 1232 games, 45 min):**

| Metric | Iter 1 | Iter 25 | Iter 50 | Iter 75 |
|---|---|---|---|---|
| Policy Loss | 0.32 | 2.08 | 2.64 | 2.71 |
| Value Loss | 0.56 | 0.23 | 0.15 | 0.15 |
| R/B/D (per 16) | 12/3/1 | 10/5/1 | 10/4/2 | 10/4/2 |
| Avg Game Length | 186 | 200 | 200 | 200 |
| Throughput (g/m) | 38 | 30 | 29 | 29 |

| Eval vs Random | Iter 25 | Iter 50 | Iter 75 |
|---|---|---|---|
| Result | 3W/6L/1D | 1W/8L/1D | 0W/10L/0D |
| Score | 35% | 15% | 0% |

**Self-play decisive rate: ~93%** (was 0.5% in v1) — the two rules completely solved the draw problem.

### Analysis

**What worked:**
- **Move banning + material adjudication**: Completely eliminated the draw deadlock. 93% of self-play games are now decisive (12-13 Red wins + 3-5 Black wins per 16 games)
- **Value head learning**: VL went from 0.56→0.15 with real win/loss signals — much stronger learning than v1
- **Red/Black asymmetry visible**: Red wins ~75% of self-play games (first-move advantage), providing diverse value targets

**What failed:**
- **Policy degradation**: PL rose from 0.32→2.71 — the pretrained human policy was destroyed, same as v1
- **Eval collapsed**: 35%→0% over 75 iterations — the network went from winning 3/10 to losing 10/10 against random
- **MCTS policy targets from self-play are worse than pretrained policy**: Even though games are decisive, the MCTS search with a degrading value head produces poor move recommendations. Training on these targets destroys the pretrained knowledge faster than the network can learn from self-play outcomes

**Root cause — Policy degradation despite decisive games:**

The draw problem is solved, but a new problem is exposed: **the MCTS policy improvement loop doesn't work at small scale**. In AlphaZero, MCTS is supposed to produce better moves than the raw policy (because search looks ahead). But with only 50 sims and a 1.9M param network:
1. The value head is too weak to guide search effectively
2. MCTS with bad value estimates produces mediocre policy targets
3. Training on these targets degrades the pretrained policy (which was trained on 11M human positions)
4. Worse policy → even worse MCTS → accelerating degradation

The pretrained policy at 34.6% accuracy is actually better than what MCTS can produce at this scale. The training loop is replacing good knowledge with bad knowledge.

### Candidate 4 v3 (no-pretrain): Smaller Buffer + More Train Steps

**Key changes from v2:**
- 200 MCTS sims (was 50) — better policy targets from search
- 20K replay buffer (was 100K) — more consistent training data
- 100 train steps/iter (was 20) — actually learn from each batch
- No pretrained weights — start from random initialization
- Half reward (0.5) for step_limit wins — discourage passive play

**Why no pretrain?** Testing showed pretrained weights get destroyed within ~10 iterations of self-play. The no-pretrain model actually performed better (50% at iter 25 vs 35% for pretrained v2).

**Training Results (166 iterations, ~2660 games):**

| Eval vs Random | Iter 25 | Iter 50 | Iter 75 | Iter 100 | Iter 125 | Iter 150 |
|---|---|---|---|---|---|---|
| Score | 55% | **70%** | 65% | 55% | 45% | 30% |

- **Peak: 70% vs Random at iter 50** — best AlphaZero result, exceeds PPO Candidate 1 (65%)
- Degraded to 30% by iter 166 — same pattern as v2, just slower

**Deep Investigation — Why Degradation Happens:**

Ran comprehensive diagnostics comparing the peak model (iter 50, 70%) vs degraded model (iter 166, 30%):

1. **MCTS visit distribution is NOT the problem**: Even with near-uniform policy priors (96-97% entropy), 200 MCTS sims produce concentrated visit distributions (top move gets 40-60% of visits). Training targets are reasonably sharp (~33% top action probability).

2. **The problem is self-play overfitting**:
   - **Iter 50 model (70%)**: Has broad strategic understanding — advances Soldiers down the board, develops Chariots, actively captures pieces. Wins by material +19 in aggressive games.
   - **Iter 166 model (30%)**: Memorized ONE trick — Cannon captures Horse opening (+0.40 material). Then gets STUCK shuffling Cannon back and forth (0,1)↔(1,1) for 20+ moves. Has 9 position repetitions per game vs 2 for iter 50.

3. **Root cause**: As the model plays against itself, it converges to a narrow set of patterns. Early training discovers useful strategies by exploring randomly. Later training overfits to exploiting its own weaknesses, producing increasingly specialized (and fragile) play.

**Log**: `training/candidate4_v3_nopretrain_output.log`

### Candidate 4 v4 (step penalty): ABANDONED

**Experiments tried (all failed to produce decisive games):**

1. **Step penalty only, no material adjudication**: -0.001/step after step 100, accumulating to -0.1 at step 200. Result: 0/0/16 draws every iteration. Without material adjudication, all 200-step games are draws → zero signal.

2. **Material adjudication (0.5) + strong step penalty (-0.5 at 200)**: Material adj wins get +0.5 reward, step penalty -0.005/step. At step 200: winner gets +0.5-0.5=**0.0 net reward**, loser gets -0.5-0.5=-1.0. Problem: winning by material at step limit gives zero positive signal. The model can't learn because the only way to get positive reward is checkmate, but it doesn't know how to checkmate yet.

**Key insight**: Material adjudication + step penalty cancel each other out at step 200. The model needs positive reward signal from material wins to bootstrap learning toward checkmate. This led to v5's curriculum approach.

**Log**: `training/candidate4_v4_output.log`

### Candidate 4 v5 (curriculum): IN PROGRESS

**Major redesign** — instead of pure self-play, use curriculum training with progressively stronger opponents:

**Curriculum phases**: Random → Greedy → Minimax (depth 2) → Self-play. Promote when eval score ≥ 75% vs current opponent.

**Key features:**
- **Curriculum opponents**: Greedy (1-ply material lookahead) and Minimax (alpha-beta depth 2) implemented using C++ engine primitives
- **Agent plays both sides**: Half Red, half Black per batch — eliminates color bias
- **Only agent MCTS positions** stored as training examples (opponent moves have no policy targets)
- **Check bonus (+0.15)**: MCTS leaf evaluation penalizes positions where current player is in check, guiding search toward checking sequences and checkmate
- **Endgame starting positions (25%)**: Games randomly start from mid/endgame positions (40-80 random moves played first), exposing model to positions where checkmate is achievable
- **Higher material adjudication threshold (0.3)**: Need ~Horse/Cannon advantage to win by adjudication, forcing more actual checkmates
- **Checkmate boost**: Rare checkmate games upweighted in replay buffer (up to 10x when checkmate rate < 5%, scaling down as rate increases)
- **Step penalty**: -0.005/step after step 100, -0.5 at step 200 for both sides
- **Half reward (0.5)** for material adjudication wins
- **200 MCTS sims for eval** (was 50) — matches training strength

**Training Config:**

| Param | Value |
|---|---|
| MCTS sims | 200 (train and eval) |
| Parallel games | 16 |
| Replay buffer | 20,000 |
| Train steps/iter | 100 |
| Eval every | 10 iterations |
| Total iterations | 500 |
| Curriculum | random → greedy → minimax → self_play |
| Promote threshold | 75% score |

**Log**: `training/candidate4_v5_output.log`

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

| Opponent | AlphaZero Wins | Opponent Wins | Draws | Score |
|---|---|---|---|---|
| Random | | | | |
| Greedy | | | | |
| Minimax (d=3) | | | | |

---

## Summary Scoreboard

| Candidate | vs Random | vs Greedy | vs Minimax (d=3) |
|---|---|---|---|
| 1. PPO Self-Play | 7W/1L/12D (65%) | 0W/15L/5D (12.5%) | 0W/8L/2D (10%) |
| 2. PPO + Reward Shaping (v1) | 5W/4L/31D (51%) | 0W/33L/7D (9%) | 0W/40L/0D (0%) |
| 2. PPO + Reward Shaping (v2) | 14W/1L/25D (66%) | 0W/30L/10D (12.5%) | 0W/38L/2D (2.5%) |
| 3. PPO + Reward + Curriculum (v1) | 20W/0L/20D (75%) | 1W/24L/15D (21%) | 0W/40L/0D (0%) |
| 3. PPO + Reward + Curriculum (v2) | 27W/0L/13D (84%) | 1W/27L/12D (17.5%) | 0W/40L/0D (0%) |
| 4v1. AlphaZero (all draws) | 0W/2L/8D (40%) | — | — |
| 4v2. AlphaZero (move ban) | 3W/6L/1D→0W/10L (35%→0%) | — | — |
| 4v3. AlphaZero (no-pretrain) | Peak 70%→30% | — | — |
| 4v4. AlphaZero (step penalty) | Abandoned (zero signal) | — | — |
| 4v5. AlphaZero (curriculum) | In progress | — | — |
| 5. Full AlphaZero | — | — | — |

---

## Insights

### 1. Reward Shaping: Not Helpful, Bad Design is Harmful

| Comparison | vs Random | vs Greedy | vs Minimax |
|---|---|---|---|
| Candidate 1 (no shaping) | 65% | 12.5% | 10% |
| Candidate 2 v1 (bad scale) | 51% | 9% | 0% |
| Candidate 2 v2 (balanced) | 66% | 12.5% | 2.5% |

- Adding reward shaping with bad proportions (v1: chariot=0.9 vs win=1.0) made the agent **worse** than no shaping at all — it learned to chase captures instead of checkmate
- Even with balanced proportions (v2), reward shaping barely improved over the sparse-reward baseline (66% vs 65%)
- **Takeaway:** Reward shaping is a double-edged sword. If piece values are poorly calibrated relative to the terminal reward, the agent optimizes for the wrong objective. When done right, the improvement is marginal — the real bottleneck is elsewhere

### 2. Curriculum: Overfits Easy AI, Underfits Hard AI

| Comparison | vs Random | vs Greedy | vs Minimax |
|---|---|---|---|
| Candidate 2 v2 (no curriculum) | 66% | 12.5% | 2.5% |
| Candidate 3 v2 (curriculum) | 84% | 17.5% | 0% |

- Curriculum dramatically improves vs Random (66% → 84%) because Phase A dedicates 1M steps to playing Random — the agent gets very good at exploiting weak play
- vs Greedy improves modestly (12.5% → 17.5%), but still dominated
- vs Minimax actually got **worse** (2.5% → 0%) — Phase C only allocates 1M steps with 25% Minimax games, far too little to learn tactical play
- **Takeaway:** Curriculum creates a specialization trap. The agent overfits to the opponents it trains against most. To improve vs Minimax, it likely needs much more training time in Phase C, or a more gradual curriculum that doesn't move on from hard opponents too quickly

### 3. Pure PPO Has a Fundamental Ceiling

- No PPO variant scored above 0% wins against Minimax (depth 3) across all experiments
- Even the best candidate (3v2) only manages 17.5% score vs Greedy (1-move lookahead)
- PPO learns reactive patterns (what to do given a board state) but cannot plan ahead
- **Takeaway:** Policy-only RL cannot compete with search-based AI in chess. The agent needs look-ahead ability at inference time (MCTS) to bridge this gap — which is exactly what Candidates 4 and 5 aim to provide

### 4. Red vs Black Asymmetry

- Across all candidates, the agent is significantly stronger as Red (first mover) than Black
- Candidate 3 v2: 20/20 wins as Red vs Random, but only 7/20 as Black
- As Black, the agent defaults to defensive play that draws but rarely wins
- **Takeaway:** Self-play training naturally develops a Red-biased strategy due to first-move advantage. Future training could explicitly balance Red/Black experience or add Black-specific objectives

### 5. AlphaZero Self-Play Has Two Problems at Small Scale

| Metric | AlphaZero v1 | AlphaZero v2 | AlphaZero v3 | Best PPO (3v2) |
|---|---|---|---|---|
| vs Random (best) | 40% | 35% | **70%** | **84%** |
| Decisive self-play rate | 0.5% | **93%** | ~93% | ~18% |
| Training time | 63 min | 45 min | ~4 hrs | 2.8 hrs |

**Problem 1 (v1): All-draws deadlock.** Equal networks produce no decisive games → no value signal → no learning. Fixed in v2 by banning repeated moves and material adjudication.

**Problem 2 (v2): Policy degradation.** Even with 93% decisive games, MCTS at small scale (50 sims, 1.9M params) produces worse policy targets than the pretrained human policy. Training replaces good knowledge with bad knowledge, and eval collapses (35%→0%).

**Problem 3 (v3): Self-play overfitting.** More sims (200) + more training (100 steps) + smaller buffer (20K) peaks at 70% — best AlphaZero result! But the model eventually overfits to narrow self-play patterns. The iter 166 model memorized a single Cannon trick and shuffles pieces endlessly, while the iter 50 model had diverse strategic play.

- **Takeaway:** AlphaZero has three barriers at small scale: (1) draw deadlock — solved with move banning, (2) policy degradation from pretrain — solved by starting from random init, (3) self-play overfitting — the model converges to narrow patterns. Potential fixes: checkpoint pool/league training (diversify opponents), KL divergence penalty (prevent policy drift), or expert iteration on human data

---

## Experiments Log

| Date | Experiment | Result |
|---|---|---|
| 2026-03-17 | Candidate 1: PPO 3M self-play training | 90% win rate vs random (during training eval) |
| 2026-03-17 | Candidate 1 vs Random (20 games) | 7W/1L/12D (65%) — wins as Red, mostly draws as Black |
| 2026-03-17 | Candidate 1 vs Greedy (20 games) | 0W/15L/5D (12.5%) — can't compete with 1-move lookahead |
| 2026-03-17 | Candidate 1 vs Minimax depth-3 (10 games) | 0W/8L/2D (10%) — no tactical awareness |
| 2026-03-17 | Candidate 2: PPO+RewardShaping 3M training | 60% best win rate vs random (during training eval) |
| 2026-03-17 | Candidate 2 vs Random (40 games) | 5W/4L/31D (51%) — worse than Candidate 1 |
| 2026-03-17 | Candidate 2 vs Greedy (40 games) | 0W/33L/7D (9%) — still can't compete |
| 2026-03-17 | Candidate 2 vs Minimax depth-3 (40 games) | 0W/40L/0D (0%) — total domination by Minimax |
| 2026-03-18 | Candidate 2 v2: Retrain with balanced rewards (win=1, chariot=0.333) | 90% best eval win rate, 66% score vs random |
| 2026-03-17 | Candidate 3: PPO+RewardShaping+Curriculum 3M training | 100% best eval win rate vs random, 2.3hrs |
| 2026-03-17 | Candidate 3 vs Random (40 games) | 20W/0L/20D (75%) — never loses |
| 2026-03-17 | Candidate 3 vs Greedy (40 games) | 1W/24L/15D (21%) — first win vs Greedy! |
| 2026-03-17 | Candidate 3 vs Minimax depth-3 (40 games) | 0W/40L/0D (0%) — still no chance |
| 2026-03-18 | Candidate 3 v2: Retrain with balanced rewards + curriculum | 100% best eval win rate, 84% score vs random — best pure PPO |
| 2026-03-18 | Candidate 3 v2 vs Random (40 games) | 27W/0L/13D (84%) — perfect 20/20 as Red |
| 2026-03-18 | Candidate 3 v2 vs Greedy (40 games) | 1W/27L/12D (17.5%) — still dominated |
| 2026-03-18 | Candidate 3 v2 vs Minimax depth-3 (40 games) | 0W/40L/0D (0%) — PPO ceiling reached |
| 2026-03-18 | Built C++ game engine (pybind11) | 234x speedup over Python engine |
| 2026-03-18 | Supervised pre-training on 11M human positions | 34.6% move prediction accuracy (epoch 7) |
| 2026-03-19 | Candidate 4 v1: AlphaZero self-play (212 iters, 3418 games) | 0W/2L/8D vs Random (40%) — worse than all PPO candidates |
| 2026-03-19 | Candidate 4 v1 analysis | 99.5% draws in self-play. Chicken-and-egg deadlock: no decisive games → no value signal → no improvement |
| 2026-03-19 | Candidate 4 v2: Move banning + material adjudication (77 iters, 1232 games) | 93% decisive rate (solved draws!), but eval 35%→0% — policy degradation |
| 2026-03-19 | Candidate 4 v2 analysis | Two separate problems: (1) draw deadlock = solved, (2) MCTS policy targets worse than pretrained policy at small scale |
| 2026-03-19 | Candidate 4 v3: No-pretrain, 200 sims, 20K buffer, 100 train steps | Peak 70% vs Random at iter 50 — best AlphaZero result |
| 2026-03-19 | Candidate 4 v3 degradation investigation | Iter 50 has diverse strategy; iter 166 memorized ONE trick (Cannon captures Horse) then shuffles forever. Self-play overfitting. |
| 2026-03-20 | Candidate 4 v4: Step penalty experiments | Abandoned — step penalty + material adj cancel out, zero positive signal for model |
| 2026-03-20 | Candidate 4 v5: Curriculum (random→greedy→minimax→self-play) | In progress — check bonus, endgame starts, checkmate boost, higher adj threshold |
