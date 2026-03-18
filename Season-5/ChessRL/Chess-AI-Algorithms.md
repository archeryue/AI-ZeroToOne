# Chess AI Algorithms — From Random to AlphaZero

A guide to the major approaches for building chess-playing AIs, ordered from simplest to most powerful. Each section explains the core idea, how it works, and its strengths/weaknesses. All examples reference our Chinese Chess (Xiangqi) implementation.

---

## 1. Random AI

### Core Idea
Pick a random legal move. No strategy at all.

### How It Works
```
1. Get all legal moves for the current position
2. Pick one uniformly at random
3. Done
```

### Strength
- Baseline for comparison. Any useful AI should beat this easily.
- Occasionally stumbles into wins against other weak players by accident.

### Weakness
- No concept of good or bad moves
- Will happily walk its General into a capture

---

## 2. Greedy AI (1-Move Lookahead)

### Core Idea
Look at every legal move, simulate it, evaluate the resulting board by material count, and pick the best one.

### How It Works
```
For each legal move:
    1. Simulate the move on a copy of the board
    2. Count material: sum up piece values for both sides
    3. Score = (my material) - (opponent's material)
Pick the move with the highest score.
```

### Piece Values (Our Implementation)
| Piece | Value |
|---|---|
| General | 10000 (effectively infinite) |
| Chariot | 90 |
| Cannon | 45 |
| Horse | 40 |
| Advisor | 20 |
| Elephant | 20 |
| Soldier | 10 |

### Strength
- Will always capture a free piece
- Won't blunder its own pieces for nothing
- Very fast (evaluates ~30-40 moves per turn)

### Weakness
- Only sees 1 move ahead — can't see that a capture leads to losing a bigger piece next turn
- No concept of tactics (forks, pins, discovered attacks)
- No positional understanding

---

## 3. Minimax + Alpha-Beta Pruning

### Core Idea
Think ahead multiple moves. Build a game tree of all possible move sequences, assume the opponent plays optimally (minimizes your score), and pick the move that leads to the best worst-case outcome.

### How Minimax Works

The name "minimax" means: **maximize** my score while assuming my opponent will **minimize** it.

```
function minimax(position, depth, is_maximizing):
    if depth == 0 or game is over:
        return evaluate(position)    # leaf node — static evaluation

    if is_maximizing:     # My turn — I want the highest score
        best = -infinity
        for each legal move:
            child = simulate(move)
            score = minimax(child, depth - 1, False)  # opponent's turn next
            best = max(best, score)
        return best

    else:                 # Opponent's turn — they want the lowest score
        best = +infinity
        for each legal move:
            child = simulate(move)
            score = minimax(child, depth - 1, True)   # my turn next
            best = min(best, score)
        return best
```

**Example at depth 2:**
```
My turn: I have 3 moves (A, B, C)
  Move A → opponent has 2 responses:
    Response 1 → evaluate: +5
    Response 2 → evaluate: +2
    Opponent picks min → +2
  Move B → opponent has 2 responses:
    Response 1 → evaluate: +8
    Response 2 → evaluate: +1
    Opponent picks min → +1
  Move C → opponent has 2 responses:
    Response 1 → evaluate: +4
    Response 2 → evaluate: +3
    Opponent picks min → +3

I pick max of {+2, +1, +3} → Move C (score +3)
```

### The Problem: Exponential Blowup

Chinese Chess has ~30-40 legal moves per position. At depth d:
- Depth 1: ~35 positions to evaluate
- Depth 2: ~1,225
- Depth 3: ~42,875
- Depth 4: ~1,500,625
- Depth 6: ~1.8 billion

This is where **alpha-beta pruning** comes in.

### Alpha-Beta Pruning

Alpha-beta is an optimization that skips branches of the game tree that **cannot possibly affect the final decision**. It maintains two values:

- **Alpha:** the best score the maximizer can guarantee so far
- **Beta:** the best score the minimizer can guarantee so far

When alpha >= beta, we can **prune** (skip) the remaining moves in that branch.

```
function alphabeta(position, depth, alpha, beta, is_maximizing):
    if depth == 0 or game is over:
        return evaluate(position)

    if is_maximizing:
        for each legal move:
            score = alphabeta(child, depth-1, alpha, beta, False)
            alpha = max(alpha, score)
            if alpha >= beta:
                break    # PRUNE: opponent would never allow this branch
        return alpha

    else:
        for each legal move:
            score = alphabeta(child, depth-1, alpha, beta, True)
            beta = min(beta, score)
            if alpha >= beta:
                break    # PRUNE: I would never choose this branch
        return beta
```

**Why it works:** If I already found a move scoring +5, and while exploring another move I discover the opponent can force a score of +3 in that branch, I don't need to keep looking at that branch — I already have something better.

**Speedup:** In the best case (moves ordered perfectly), alpha-beta evaluates sqrt(N) nodes instead of N. For depth 4, that's ~1,225 instead of ~1.5 million. In practice, it roughly doubles the searchable depth.

### Move Ordering

Alpha-beta works best when good moves are examined first (more pruning opportunities). Our implementation orders moves by:
1. Captures first (sorted by captured piece value — capturing a Chariot is examined before capturing a Soldier)
2. Non-captures after

### Evaluation Function

At leaf nodes (depth = 0), we need a static evaluation of the position. Our minimax uses:

**Material score** (same as Greedy):
```
score = sum(my_piece_values) - sum(opponent_piece_values)
```

**Positional bonuses:**
- Soldiers get bonus for advancing across the river
- Horses get bonus for being near the center
- Chariots get bonus for being on open files (no blocking Soldiers)

**Check bonus:** +5 for delivering check to the opponent

### Strengths
- Can see tactics several moves deep (forks, pins, skewers)
- At depth 3: beats any 1-move-lookahead AI consistently
- At depth 5-6: plays at a reasonable amateur level
- Deterministic and explainable — you can trace exactly why it chose a move

### Weaknesses
- Depth is limited by computation — each extra level is ~35x more work
- The evaluation function is hand-crafted and inevitably misses subtle positional factors
- No learning — it doesn't improve from experience
- Horizon effect: can't see threats just beyond its search depth

### Our Implementation
- Depth 3 (default), with alpha-beta pruning and move ordering
- Takes ~1-3 seconds per move
- Beats PPO Candidate 1 easily (8W/0L/2D)

---

## 4. PPO Self-Play (Pure RL)

### Core Idea
Train a neural network via self-play. The network directly outputs move probabilities (policy) and position evaluation (value). No explicit search tree — just "pattern recognition" learned from millions of games.

### How It Works
```
1. Network sees the board → outputs policy (move probabilities) + value (who's winning)
2. Sample a move from the policy distribution
3. Play the game to completion
4. Use the outcome to compute advantages (via GAE)
5. Update the network with PPO (clipped surrogate objective)
6. Repeat for millions of games
```

### Architecture (Our Implementation)
```
Board (15×10×9) → 3-layer CNN → FC(512) → Actor(8100) + Critic(1)
                                           ↓ action masking
                                    Policy distribution over legal moves
```

### Strengths
- Learns entirely from experience — no hand-crafted evaluation needed
- Can discover unconventional strategies that humans might not encode
- Scales with more compute (more games = more learning)

### Weaknesses
- **No look-ahead:** picks moves based on pattern recognition only
- **Sparse reward problem:** only +1/-1 at game end makes credit assignment over 100+ moves very hard
- **Self-play echo chamber:** if the agent develops a bad habit early, both sides reinforce it
- **No tactical awareness:** can't calculate "if I move here, they take there, then I take back"

### Our Results
- Candidate 1 (pure PPO): 35% vs Random, 0% vs Greedy, 0% vs Minimax
- The agent learned to move pieces and not make illegal moves, but has no tactical understanding
- Even Greedy (1-move lookahead) dominates it — PPO doesn't understand material value

### Improvements We're Trying
- **Candidate 2:** Add reward shaping (material-based intermediate rewards)
- **Candidate 3:** Add curriculum training (play vs Random → Greedy → Minimax)
- These address the sparse reward and echo chamber problems, but NOT the lack of search

---

## 5. Monte Carlo Tree Search (MCTS)

### Core Idea
Instead of exhaustively searching all branches (like minimax), **sample** promising lines of play using random simulations (rollouts). Build a search tree incrementally, guided by statistics about which moves have worked well in past simulations.

### How It Works (4 Phases per Simulation)

```
1. SELECTION:   Start at root. Traverse the tree, picking the "best" child
                at each node using UCB1 (Upper Confidence Bound):

                UCB1 = win_rate + C * sqrt(ln(parent_visits) / child_visits)
                       ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       exploitation              exploration

                This balances exploiting known good moves vs exploring unknown ones.

2. EXPANSION:   When we reach a node that hasn't been fully explored,
                add a new child node for an untried move.

3. SIMULATION:  From the new node, play a random game to completion
                (called a "rollout" or "playout").

4. BACKPROPAGATION: Propagate the result (win/loss) back up the tree,
                    updating visit counts and win rates at each node.
```

**After N simulations:** Pick the root's most-visited child as the actual move.

### Strengths
- No need for a hand-crafted evaluation function (rollouts give an unbiased estimate)
- Focuses computation on promising branches (unlike minimax which explores everything to a fixed depth)
- Naturally handles large branching factors
- Works well for games where evaluation is hard (like Go)

### Weaknesses
- Random rollouts are noisy — need many simulations for good estimates
- Slow for games where random play is very uninformative
- Without neural network guidance, the search can waste time on bad branches

### Pure MCTS vs AlphaZero MCTS
| Aspect | Pure MCTS | AlphaZero MCTS |
|---|---|---|
| Rollout policy | Random moves | **No rollouts** — use NN value head |
| Prior probabilities | Uniform | **NN policy head** (learned priors) |
| Leaf evaluation | Random playout result | **NN value prediction** |
| Quality | Decent with many sims | Much stronger with fewer sims |

---

## 6. AlphaZero (MCTS + Neural Network)

### Core Idea
Combine the **search depth** of MCTS with the **pattern recognition** of a neural network. The network provides two things:
1. **Policy prior:** which moves are likely good (guides MCTS exploration)
2. **Value estimate:** who's winning from this position (replaces random rollouts)

This eliminates both weaknesses of pure RL (no search) and pure MCTS (random rollouts).

### Architecture

```
Board → ResNet (residual blocks) → Policy Head (move probabilities)
                                 → Value Head  (position evaluation, [-1, +1])
```

The ResNet learns board features through self-play. Residual connections allow deep networks (10-40 blocks) to train effectively.

### MCTS with Neural Network (PUCT)

AlphaZero replaces UCB1 with PUCT (Predictor + UCT):

```
PUCT(s, a) = Q(s, a) + c * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
              ^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              mean value    exploration bonus weighted by NN policy prior
              from search
```

Where:
- `Q(s, a)`: average value from simulations that went through this move
- `P(s, a)`: neural network's prior probability for this move
- `N(s)`: parent visit count
- `N(s, a)`: this move's visit count
- `c`: exploration constant (~1.5)

**Key insight:** The NN policy prior `P(s,a)` guides the search toward promising moves. A move the NN thinks is good gets explored more, but if the search discovers it's actually bad (low Q), it's abandoned.

### Training Loop

```
Repeat forever:
    1. SELF-PLAY: Play N games using MCTS + current network
       - Each position: run 200 MCTS simulations → get visit count distribution π
       - Store (board_state, π, game_result) for training

    2. TRAIN: Update the network on collected data
       - Policy loss:  cross-entropy(network_policy, MCTS_policy π)
       - Value loss:   MSE(network_value, actual_game_result)
       - The network learns to predict what MCTS would do (policy)
         and who will win (value)

    3. EVALUATE: New network vs old network
       - If the new network is stronger, it replaces the old one
       - This ensures the training data keeps improving
```

**The beautiful feedback loop:**
- Better network → better MCTS (more accurate priors and values) → better training data → even better network → ...

### Dirichlet Noise

At the root node, AlphaZero adds random noise to the NN policy:

```
P(root, a) = (1 - ε) * NN_policy(a) + ε * Dirichlet(α)
```

With ε=0.25 and α=0.3 (for chess). This ensures exploration of moves the network might not initially favor — important for discovering new strategies.

### Temperature

During training games, move selection uses temperature:
```
π(a) = N(a)^(1/τ) / Σ N(b)^(1/τ)
```

- **τ = 1.0** (first 30 moves): proportional to visit counts → more exploration
- **τ → 0** (after 30 moves): always pick most-visited move → more exploitation

### Why AlphaZero Is So Strong

1. **Search + Learning:** MCTS gives tactical depth (like minimax), NN gives positional understanding (like pattern recognition)
2. **No human knowledge:** learns entirely from self-play — no opening books, no hand-crafted evaluation
3. **The network bootstraps the search:** as the NN improves, MCTS becomes more efficient, which generates better training data
4. **Self-improvement:** each generation is trained on games from better and better players (itself)

### Computational Cost
- Original AlphaZero: 5,000 TPUs, 9 hours for chess
- Our plan: 1 H100, 3-12 hours for Chinese Chess (smaller board, simpler game)
- Mini version: local GPU, 50 MCTS sims, 5 ResNet blocks (Candidate 4)

---

## Algorithm Comparison

| Algorithm | Search | Learning | Key Limitation |
|---|---|---|---|
| Random | None | None | No strategy |
| Greedy | 1 move | None | No depth |
| Minimax (α-β) | d moves deep | None | Hand-crafted eval, exponential cost |
| Pure MCTS | Adaptive | None | Random rollouts are noisy |
| PPO Self-Play | None | Yes | No look-ahead |
| AlphaZero | MCTS | Yes | High compute cost |

### The Key Insight

**Search and learning are complementary:**
- Search without learning (Minimax): strong tactics, weak intuition
- Learning without search (PPO): recognizes patterns, no tactical depth
- Search + learning (AlphaZero): best of both worlds

This is exactly what we observed in our experiments: PPO (Candidate 1) learned to move pieces but couldn't compete with Minimax's 3-move lookahead. AlphaZero combines the NN's pattern recognition with MCTS's search to achieve both.

---

## Further Reading

- **Minimax & Alpha-Beta:** Russell & Norvig, *Artificial Intelligence: A Modern Approach*, Chapter 5
- **MCTS:** Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
- **AlphaGo:** Silver et al., "Mastering the game of Go with deep neural networks and tree search" (2016)
- **AlphaZero:** Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (2018)
