# ChessRL — Reinforcement Learning for Chinese Chess

## Goal

Train an AI to play Chinese Chess (Xiangqi) using RL and supervised learning. We progressively improve through 6 candidates, each building on the last, and benchmark every candidate against the 3 AIs from our ChineseChess game (Random, Greedy, Minimax).

## Candidate Roadmap

| # | Candidate | Key Idea | Device | Status |
|---|-----------|----------|--------|--------|
| 1 | PPO Self-Play | Baseline — pure self-play, sparse reward | Local GPU | Done |
| 2 | PPO + Reward Shaping | Material-based intermediate rewards | Local GPU | Done |
| 3 | PPO + Reward Shaping + Curriculum | + train vs increasingly strong opponents | Local GPU | Done |
| 4 | Mini AlphaZero | MCTS + small ResNet self-play (6 versions, all failed) | Local GPU | Done |
| 5 | NNUE | Supervised NN eval + alpha-beta search | Local GPU | Done |
| 5v2 | NNUE + TD(lambda) | TD learning + BCE + 692 features, pure NNUE eval | Local GPU | Done |
| 6 | Full AlphaZero | Full MCTS (800 sims) + large ResNet | Cloud H100 | TODO |

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

### Candidate 4 v5 (curriculum): FAILED

**Major redesign** — instead of pure self-play, use curriculum training with progressively stronger opponents.

**Curriculum phases**: Random → Greedy → Minimax (depth 2) → Self-play. Promote when eval score ≥ 75% vs current opponent.

**Key features (evolved across multiple restarts):**
- **Curriculum opponents**: Greedy (1-ply material lookahead) and Minimax (alpha-beta depth 2) implemented using C++ engine primitives
- **Agent plays both sides**: Half Red, half Black per batch — eliminates color bias
- **Only agent MCTS positions** stored as training examples (opponent moves have no policy targets)
- **Check bonus (+0.15)**: MCTS leaf evaluation penalizes positions where current player is in check
- **Endgame starting positions (25%)**: Games randomly start from mid/endgame positions (40-80 random moves played first)
- **Checkmate boost**: Rare checkmate games upweighted in replay buffer (up to 10x when checkmate rate < 5%)
- **Pure checkmate reward**: +1/-1 for checkmate only, 0 for everything else (no material adjudication, no step penalty)
- **Draw downsampling**: Only 25% of draw game examples kept in replay buffer
- **Increased exploration**: Dirichlet epsilon 0.4 (was 0.25), temperature threshold 60 moves (was 30)
- **300 MCTS sims** (increased from 200)

**Training Config:**

| Param | Value |
|---|---|
| MCTS sims | 300 (train and eval) |
| Parallel games | 16 |
| Replay buffer | 20,000 |
| Train steps/iter | 100 |
| Eval every | 10 iterations |
| Total iterations | 500 |
| Curriculum | random → greedy → minimax → self_play |
| Promote threshold | 75% score |

**Training Results (208+ iterations, 3343+ games, ~9 hours):**

Never promoted past random phase. All 18 eval checkpoints showed 0 wins:

| Eval vs Random | Score |
|---|---|
| Iter 30 | 0W/2L/8D (40%) |
| Iter 40 | 0W/3L/7D (35%) |
| Iter 80 | 0W/0L/10D (50%) |
| Iter 100 | 0W/1L/9D (45%) |
| Iter 150 | 0W/1L/9D (45%) |
| Iter 200 | 0W/1L/9D (45%) |

Checkmate rate in self-play: ~15-30% per iteration (vs random opponent), but model never learned to win in eval.

**Diagnostic — Raw Network Output on Full Game (iter 200):**

Played model (greedy from raw network, no MCTS) vs random opponent for 200 moves:

1. **Value head is broken**: Start position value = **-0.826** (model thinks it's heavily losing from move 1). Values oscillate wildly: -0.826 → -0.612 → -0.190 → -0.345 → -0.998 → -1.000 within the first 12 moves. No stable evaluation.

2. **Policy is diffuse**: Top move gets only ~10-17% probability. No strong preferences — the model treats most legal moves as roughly equal.

3. **No strategic plan**: Model moves pieces aimlessly — Soldier advance, Cannon repositioning, Horse shuffle — with no coordination or attack plan. Captures are incidental, not deliberate.

4. **Piece shuffling in late game**: Elephant moves (9,2)→(7,4)→(9,2)→(7,4) repeatedly. General oscillates between (9,3) and (9,4). Model runs out of ideas and loops.

5. **Result**: Step limit (200) — draw. Model has material advantage (captured pieces via random opponent blunders) but cannot convert to checkmate.

**Root cause — Pure checkmate reward is too sparse:**

With only +1/-1 for checkmate and 0 for everything else:
- ~75-80% of games end in draws (step limit) → value=0 for all positions → value head collapses toward 0 or learns noise
- The ~20% checkmate games are mostly the random opponent blundering into checkmate — not the model learning to force checkmate
- Even with 10x checkmate boost and 25% draw keep rate, the signal-to-noise ratio is too low
- Value head converges to near-zero/negative for all positions → MCTS has no value guidance → search is effectively random → policy targets are noise

**Log**: `training/candidate4_v5_output.log`

### Candidate 4 v6 (material draw reward): FAILED

**Key changes from v5:**
- **Material-based draw reward**: Draw at step limit → `clamp(-0.2 + material_diff, -0.7, 0.9)` instead of pure 0
  - Material scoring: 车=0.33, 马/炮=0.16, 士/象=0.08, 兵=0.03
  - Equal material draw = -0.2 (discourages draws)
  - Material advantage up to +0.9, deficit down to -0.7
- **Checkmate**: Still +1/-1 (unchanged)
- **Shorter games**: 120 max steps (was 200) — faster iterations
- **Smaller check bonus**: 0.03 (was 0.15)
- **Removed**: Checkmate boost upsampling, draw downsampling
- **Kept**: Curriculum, endgame starts (25%), Dirichlet epsilon 0.4, temp threshold 60, 300 MCTS sims

**Training Results (58 iterations, 928 games, ~1.5 hours):**

| Eval vs Random | Score |
|---|---|
| Iter 10 | 0W/1L/9D (45%) |
| Iter 20 | 0W/0L/10D (50%) |
| Iter 30 | 0W/0L/10D (50%) |
| Iter 40 | 0W/0L/10D (50%) |
| Iter 50 | 0W/1L/9D (45%) |

Checkmate rate dropped from ~12% early to mostly 0% by iter 40+. Speed improved to 12 games/min (was 8 g/m with 200 steps).

**Diagnostic — Raw Network Output (iter 50):**

1. **Value head stuck negative**: Start position = **-0.55**. All positions evaluated between -0.4 and -0.7. The model thinks every position is losing — the draw base of -0.2 biased the value head negative, and it overshot to ~-0.5.

2. **Policy still diffuse**: Top move gets 10-27%. No strong preferences.

3. **Lost material to random**: Final material Red=1.14, Black=1.41 (diff=-0.27). The model can't even accumulate material advantage against random — it doesn't prioritize captures.

4. **No improvement over v5**: Same symptoms — aimless piece movement, no attack coordination, piece shuffling.

**Root cause — same as all v3-v6 variants:**

The MCTS policy improvement loop doesn't work at this scale. With a broken value head, MCTS search is effectively random → produces noise policy targets → training on noise doesn't improve the network → value head stays broken. The material draw reward gives *some* signal (VL=0.01 vs 0.00), but it's not enough to break the cycle.

**Log**: `training/candidate4_v6_output.log`

---

## Candidate 5: NNUE

### Idea

**Separate evaluation from search.** Instead of MCTS + NN self-play (which failed at small scale), use a neural network trained on human games as the evaluation function inside traditional alpha-beta search. This is the approach that made Stockfish the strongest chess engine.

### Architecture (~170K params)

```
Per-perspective input: 2 colors × 7 piece types × 90 squares = 1260 binary features

Perspective A (side to move):
  1260 features → Linear(1260, 128) → ClippedReLU(0,1)  [accumulator]

Perspective B (opponent, mirrored board):
  1260 features → Linear(1260, 128) → ClippedReLU(0,1)  [accumulator]

Concat [A, B] = 256
  → Linear(256, 32) → ClippedReLU
  → Linear(32, 32) → ClippedReLU
  → Linear(32, 1) → sigmoid → [0, 1] eval

Parameters: ~170,721 (~0.17M)
```

Final eval = 60% NNUE + 40% material (blended for concrete tactical signal).

### Training

- **Supervised** on 8.7M positions (11M with 30% draw subsampling) from 162K human games
- Loss: MSE on sigmoid output vs game outcome [0=loss, 0.5=draw, 1=win from STM]
- 20 epochs, batch size 8192, Adam 1e-3 with LR drop at epoch 15
- **Training time: 8 minutes** on local GPU
- **Best val loss: 0.1408, sign accuracy: 77.3%** (correctly predicts winner on decisive positions)

### Search: C++ Alpha-Beta

Implemented in C++ (`engine_c/nnue_search.h`) with pybind11 bindings:
- Negamax alpha-beta with iterative deepening
- MVV-LVA move ordering (captures first, high-value victims prioritized)
- Transposition table (Zobrist hashing, 1M entries)
- Quiescence search (capture-only extension, depth 6)
- NNUE weights exported to binary, loaded in C++ for fast forward pass

Performance: **~350K positions/sec training, depth 4 search in ~0.6s** (vs 7.6s in Python — 760x speedup from C++).

### Benchmark vs Game AIs (depth 4, 100 games)

| Opponent | NNUE Wins | Opponent Wins | Draws | Score |
|---|---|---|---|---|
| Random | 79 | 0 | 21 | **89.5%** |
| Greedy | 50 | 0 | 50 | **75.0%** |
| Minimax (d=3) | 0 | 0 | 20 | **50.0%** |

**Key results:**
- **Zero losses across all 220 games** — NNUE never loses to any opponent
- **79% win rate vs Random** — matches best PPO result (Candidate 3 v2: 84%)
- **50% win rate vs Greedy** — first candidate to beat Greedy consistently!
- Previous best vs Greedy was Candidate 3 at 21% (1W/24L/15D)
- Draws vs Minimax due to similar depth and material-preserving play on both sides

---

## Candidate 5 v2: NNUE + TD(lambda) + BCE

### Idea

Fix 4 problems with v1:
1. **Crude credit assignment** — v1 labels every position with the game outcome. TD(lambda) propagates credit backward using search scores.
2. **MSE loss** — BCE gives sharper gradients for confident predictions near 0 and 1.
3. **Wasted features** — v1 maps all 7 piece types to 90 squares (1260 features). But General can only be in 9 squares, Advisor in 5, etc. v2 uses 692 features.
4. **Material blending** — v1 needs 40% material because the NNUE is too weak alone. v2 is pure NNUE.

### Architecture (~98K params)

```
Per-perspective input: 692 binary features (piece-aware square mapping)
  General: 9 sq, Advisor: 5, Elephant: 7, Horse: 90, Chariot: 90, Cannon: 90, Soldier: 55
  × 2 colors (friendly/enemy) = 346 × 2 = 692

Board normalized so perspective's pieces are always at bottom (color-invariant).

Perspective A (side to move):
  692 features → Linear(692, 128) → ClippedReLU  [accumulator]

Perspective B (opponent):
  692 features → Linear(692, 128) → ClippedReLU  [accumulator]

Concat [A, B] = 256
  → Linear(256, 32) → ClippedReLU
  → Linear(32, 32) → ClippedReLU
  → Linear(32, 1) → sigmoid → [0, 1] eval

Parameters: ~98,017 (~98K)
```

Eval = **100% NNUE** (no material blending).

### Training: TD(lambda) on Self-Play Search Scores

**Phase 1 — Self-play data generation:**
- v1 NNUE engine plays itself at depth 4 with epsilon-greedy noise
  - Moves 1-6: fully random (diverse openings)
  - Moves 7-16: search + epsilon=0.15
  - Moves 17+: search + epsilon=0.05
- Records (board, search_score) at each position
- 3000 games → 339K positions, W/L/D ~27%/48%/25%
- 12-worker parallel generation

**Phase 2 — TD(lambda=0.8) target computation:**
- Process each game backward: `target_t = (1-λ) × score_t + λ × (1 - target_{t+1})`
- Propagates credit: positions leading to strong positions get higher values

**Phase 3 — Training:**
- 30 epochs, batch 4096, Adam 5e-4, BCE loss
- LR drop 10x at epoch 20
- **30 seconds** total on local GPU
- **Best val BCE: 0.5684, sign accuracy: 99.8%** (vs 77% in v1)

### Benchmark (depth 4)

| Opponent | v2 Wins | Opponent Wins | Draws | Score |
|---|---|---|---|---|
| Minimax (d=3) | 50 | 0 | 0 | **100%** |
| Minimax (d=4) | 10 | 0 | 10 | **75%** |

**Key results:**
- **50-0 vs Minimax-d3** — v1 was 0-0-20 (all draws). v2 is a complete breakthrough.
- **10-0-10 vs Minimax-d4** — beats same-depth material eval with 0 losses
- Pure NNUE (no material blending) works because TD training gives accurate position values
- Sign accuracy jumped from 77% → 99.8% thanks to search-bootstrapped targets

---

## Candidate 5 v3: NNUE Self-Improvement Iteration

### Idea

Use the v2 engine to generate new self-play data, then retrain with the same TD(lambda) pipeline. If the self-improvement loop works, v3 should be stronger than v2 because it learns from higher-quality games.

### Experiments

**Experiment A: Train on v2 self-play data only (3k games, 334K positions)**

- v2 engine plays itself at depth 4, same epsilon-greedy noise as v2
- Data: 3000 games, 333,600 positions, W/L/D = 1236/1129/635 (41%/38%/21%)
- Board diversity: 100% unique at moves 5/10/20, 76.5% unique overall
- Training: 30 epochs, best val BCE = 0.5509, sign accuracy = 99.7%

| Opponent | v3 Wins | Opponent Wins | Draws | Score |
|---|---|---|---|---|
| Minimax (d=3) | 50 | 0 | 0 | **100%** |
| Minimax (d=4) | 0 | 0 | 20 | **50%** |
| v2 (depth 4, 20g) | 0 | 0 | 20 | **50%** |

**Result:** Same as v2 vs minimax-d3 (100%), but regressed vs minimax-d4 (50% vs 75%). No improvement over v2 in head-to-head — all draws.

**Experiment B: Train on mixed v1 + v2 data (6k games, 685K positions)**

- All v1-generated data (5000 games, 571K positions) + first 1000 v2-generated games (113K positions)
- Hypothesis: more diverse data from two different engine strengths = better generalization
- Training: 30 epochs, best val BCE = 0.5745, sign accuracy = 99.5%

| Opponent | v3-mixed Wins | Opponent Wins | Draws | Score |
|---|---|---|---|---|
| Minimax (d=3) | 25 | 0 | 25 | **50%** |
| v2 (depth 4, 20g) | 0 | 0 | 20 | **50%** |

**Result:** Regression — back to v1-level performance vs minimax-d3 (50% vs v2's 100%). The weaker v1 data diluted the stronger v2 signals.

### Analysis

The self-improvement iteration did **not** produce a stronger model:

1. **v3 (v2-only data):** Matched v2 vs minimax-d3 but lost ground vs minimax-d4. The v2 engine's self-play data doesn't contain enough new information beyond what v2 already learned — the model is essentially learning to replicate itself.

2. **v3 (mixed data):** Mixing weaker v1 data actively hurt performance. The v1 engine (with material blending) generates positions evaluated from a fundamentally different perspective than v2 (pure NNUE). The model learns a compromise that's worse than either.

3. **Why self-improvement stalled:** The v2 engine at depth 4 plays deterministic (within epsilon noise) games against itself. The training data captures v2's existing knowledge but not new knowledge. To break through, we likely need either: (a) deeper search (depth 5-6) to generate higher-quality targets, (b) fundamentally different positions (e.g., from human games or different opening books), or (c) search enhancements (null-move pruning, LMR) that increase effective search depth without more compute.

---

## Candidate 6: Full AlphaZero (TODO)

### Idea

Scale up Candidate 4: deeper network, more MCTS simulations, more training games. This is the full AlphaZero recipe applied to Chinese Chess, trained on cloud GPU with enough compute to make the self-play loop work.

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

### Training Plan

- 10 res blocks, 128 channels (26.3M params)
- 800 MCTS sims per move
- Target: 100K-500K self-play games
- Device: Cloud H100 (~$3/hr)

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
| 4v5. AlphaZero (curriculum) | 0W, 45% best (failed) | — | — |
| 4v6. AlphaZero (material draw) | 0W, 50% best (failed) | — | — |
| **5. NNUE (depth 4)** | **79W/0L/21D (89.5%)** | **50W/0L/50D (75%)** | **0W/0L/20D (50%)** |
| **5v2. NNUE+TD (depth 4)** | — | — | **50W/0L/0D (100%)** |
| 5v3. NNUE self-improve (v2 data) | — | — | 50W/0L/0D (100%) vs d3, 0W/0L/20D (50%) vs d4 |
| 5v3. NNUE self-improve (mixed) | — | — | 25W/0L/25D (50%) vs d3 — regression |
| 6. Full AlphaZero | — | — | — |

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

### 6. Reward Signal is Everything — Too Sparse or Too Dense Both Fail

| Reward Design | Result | Problem |
|---|---|---|
| Material adj (0.5) + step penalty (-0.5) | Zero signal | Rewards cancel: +0.5 - 0.5 = 0 at step 200 |
| Pure checkmate (+1/-1 only) | Broken value head | ~80% draws produce value=0 everywhere; 20% checkmates from random blunders, not learned skill |
| Material adj (v3, 0.5) no penalty | Peak 70% | Best result — material wins give clear positive signal to bootstrap |

- The v3 result (peak 70%) used material adjudication at step limit to give the model clear win/loss signal from the start
- v5 tried to remove this "crutch" and use pure checkmate reward, but the signal was too sparse for the value head to learn anything meaningful
- **Takeaway:** At small scale, the model needs intermediate reward signal (like material adjudication) to bootstrap. Pure checkmate reward requires either (a) much more games to find checkmates, or (b) a way to explicitly teach checkmate patterns. The v3 approach of material adjudication + move banning remains the best foundation

### 7. MCTS + Neural Network Self-Improvement Doesn't Work at Small Scale

Across 6 versions of Candidate 4, the AlphaZero approach consistently failed to beat even a random player:

| Version | Best vs Random | Core Issue |
|---|---|---|
| v1 (self-play) | 40% | All-draws deadlock |
| v2 (move ban) | 35%→0% | Policy degradation |
| v3 (no-pretrain) | **70%→30%** | Self-play overfitting |
| v4 (step penalty) | Abandoned | Zero reward signal |
| v5 (curriculum) | 45% | Value head broken |
| v6 (material draw) | 50% | Value head stuck negative |

The fundamental problem: **AlphaZero requires the MCTS policy improvement loop to work** — MCTS produces better moves than raw policy → train on MCTS targets → better policy → even better MCTS. At small scale (1 GPU, 300 sims, 1.9M params), this loop never bootstraps:

1. Value head is random at init → MCTS search has no guidance → visit distribution ≈ random
2. Training on random-quality policy targets → network doesn't improve
3. No improvement → value head stays random → back to step 1

The original AlphaZero used 5000 TPUs, 800 sims, 30M+ params, and millions of games. The massive compute creates enough search quality even with a weak value head to get the flywheel spinning. On a single local GPU, we don't have that luxury.

**Conclusion:** Mini-AlphaZero is not viable for Xiangqi on local hardware. The best local-scale result remains PPO Candidate 3v2 (84% vs Random). For a stronger Xiangqi AI on local device, a fundamentally different approach is needed.

### 8. Next Direction: NNUE (Efficiently Updatable Neural Network)

Instead of trying to make MCTS + NN self-play work at small scale, a more promising approach is **NNUE** — the technique that made Stockfish the strongest chess engine:

**Key idea:** Separate evaluation from search. Instead of using a neural network inside MCTS (which requires the self-improvement loop to work), use a neural network as the **evaluation function** inside traditional **alpha-beta search** (which already works — our Minimax opponent proves it).

**Why NNUE fits our constraints:**
- **No self-play loop needed**: Train the NN to predict game outcomes from human game data (supervised learning). We already have 11M positions from 162K human games.
- **Alpha-beta search already works**: Our Minimax depth-3 opponent beats all PPO and AlphaZero candidates. NNUE replaces the handcrafted eval with a learned eval — strictly better.
- **Efficient inference**: NNUE uses a sparse architecture with incremental updates — only recomputes the neurons affected by a move, not the whole network. This enables deeper search on local hardware.
- **Proven at small scale**: Stockfish NNUE runs on a single CPU and beats every AlphaZero-style engine at equivalent hardware.

**Architecture sketch:**
```
Input: piece-square features (which piece is on which square)
  → Sparse linear layer (incrementally updated on each move)
  → Dense layers (small, fast)
  → Output: position evaluation scalar

Search: Alpha-beta with NNUE eval instead of material + positional heuristic
```

**Training plan:**
1. Generate training data: human game positions + game outcomes (win/loss/draw)
2. Train NNUE network to predict outcomes (supervised, no RL needed)
3. Plug into alpha-beta search engine (our C++ engine already supports this)
4. Evaluate vs Random, Greedy, Minimax

### 9. NNUE Validates the "Separate Eval from Search" Hypothesis

| Approach | vs Random | vs Greedy | vs Minimax | Training Time |
|---|---|---|---|---|
| Best RL (PPO Candidate 3v2) | 84% | 17.5% | 0% | 2.8 hours |
| Best AlphaZero (Candidate 4v3) | 70% peak | — | — | 4+ hours |
| NNUE v1 (Candidate 5, depth 4) | 89.5% | 75% | 50% | 8 minutes |
| **NNUE v2 (Candidate 5v2, depth 4)** | **—** | **—** | **100%** | **30 seconds** |

NNUE v2 is the strongest candidate:
- **100% vs Minimax-d3** — v1 was 50% (all draws). v2 wins every game.
- **75% vs Minimax-d4** — beats same-depth material eval (10W/0L/10D).
- **30 seconds training** — TD(lambda) on 339K search-bootstrapped positions.
- **99.8% sign accuracy** — up from 77% in v1, thanks to search-derived targets.
- **Pure NNUE eval** — no material blending needed. The network learns material value implicitly.

v1 needed blended eval (60% NNUE + 40% material) because pure NNUE predicted "slightly winning" for everything. v2 eliminated this crutch entirely — TD(lambda) with search-bootstrapped targets teaches the network to encode material value implicitly, making pure NNUE eval viable.

**Key v1→v2 improvements:** (1) TD(lambda) credit assignment instead of game-outcome labels, (2) BCE loss for sharper gradients, (3) 692 piece-aware features instead of 1260, (4) self-play with epsilon-greedy noise for diverse training data.

### 10. Self-Improvement Iteration Stalls Without New Information

v3 attempted to improve on v2 by using v2's own engine to generate self-play data and retrain. Two experiments:

| Experiment | Data | vs Minimax-d3 | vs Minimax-d4 | vs v2 |
|---|---|---|---|---|
| v2 (baseline) | 3k v1-engine games, 339K pos | **100%** | **75%** | — |
| v3 (v2 data only) | 3k v2-engine games, 334K pos | 100% | 50% ↓ | 0-0-20 |
| v3 (mixed v1+v2) | 6k games, 685K pos | 50% ↓↓ | — | 0-0-20 |

- **v2-only data:** The model learns to replicate v2 but doesn't surpass it. The self-play data captures v2's existing knowledge, not new knowledge. Performance vs minimax-d4 actually regressed (75% → 50%).
- **Mixed data:** Adding weaker v1 data actively hurt. The v1 engine (material-blended) and v2 engine (pure NNUE) evaluate positions from fundamentally different perspectives — mixing creates conflicting training signals.
- **Takeaway:** Self-improvement requires new information the current model doesn't have. At fixed search depth, the engine generates games within its existing capability. To break through, we need either: (a) deeper search to discover longer tactical combinations, (b) search enhancements (null-move pruning, LMR) for greater effective depth, or (c) external knowledge (human games, opening books) to introduce positions outside the engine's experience.

### Current Best: NNUE v2 (Candidate 5v2)

**NNUE v2 is our strongest Xiangqi AI** — the new baseline for all future work.

| Metric | Value |
|---|---|
| Architecture | 692 features → 128 accumulator → 256 → 32 → 32 → 1 (98K params) |
| Training | TD(lambda=0.8) + BCE on 339K self-play positions, 30 seconds |
| Eval | 100% NNUE (no material blending) |
| Sign accuracy | 99.8% |
| vs Minimax-d3 | **50W/0L/0D (100%)** |
| vs Minimax-d4 | **10W/0L/10D (75%)** |
| vs all opponents | **Zero losses across all benchmarks** |

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
| 2026-03-20 | Candidate 4 v5: Curriculum with pure checkmate reward (208+ iters, 3343+ games) | 0W vs Random across all evals. Pure checkmate reward too sparse — value head broken (start pos = -0.83), policy diffuse, model shuffles pieces aimlessly |
| 2026-03-20 | Candidate 4 v6: Material draw reward + 120 step limit (58 iters, 928 games) | 0W vs Random. Value head stuck at -0.55, lost material to random player. Material draw signal too weak to bootstrap MCTS loop |
| 2026-03-20 | Candidate 5: NNUE supervised on 8.7M human positions, MSE loss | 77.3% sign accuracy, 8 min training. 89.5% vs Random, 75% vs Greedy, 50% vs Minimax-d3 (all draws) |
| 2026-03-20 | Candidate 5: NNUE eval analysis | Pure NNUE too weak alone → 60/40 blend with material needed. Cannot beat Minimax-d3 — similar depth + both preserve material |
| 2026-03-21 | Candidate 5 v2: 692 features + TD(lambda=0.8) + BCE + pure NNUE | 99.8% sign accuracy, 30s training. Self-play data: 3k games, 339K positions with epsilon-greedy noise |
| 2026-03-21 | Candidate 5 v2 vs Minimax-d3 (50 games) | **50W/0L/0D (100%)** — complete breakthrough, v1 was 0-0-20 |
| 2026-03-21 | Candidate 5 v2 vs Minimax-d4 (20 games) | **10W/0L/10D (75%)** — beats same-depth material eval with 0 losses |
| 2026-03-21 | Candidate 5 v3 (v2 data): 3k v2 self-play games, TD(lambda) training | 99.7% sign acc, 100% vs d3, 50% vs d4 (regression from v2's 75%), 0-0-20 vs v2 |
| 2026-03-21 | Candidate 5 v3 (mixed): 5k v1 + 1k v2 games combined | 99.5% sign acc, 50% vs d3 (regression to v1-level). Weaker v1 data dilutes v2 signal |
| 2026-03-21 | v3 vs v2 head-to-head (20 games, depth 4) | 0-0-20 all draws. Self-improvement iteration did not produce a stronger model |
