# RL Training Plan

## Overview
Hands-on RL training journey: from DQN basics to PPO mastery, using LunarLander and Atari games.

---

## Phase 1: DQN + LunarLander (First RL Model)

**Goal:** Train your very first RL agent using DQN on LunarLander-v2.

**Why DQN first:**
- Natural extension of value-based RL you already studied
- Simple to implement and debug
- Understand experience replay and target networks in practice

**Steps:**
1. Set up the environment (`gymnasium`)
2. Implement DQN from scratch in PyTorch
   - Replay buffer
   - Epsilon-greedy exploration
   - Target network with periodic sync
3. Train on `LunarLander-v2` (discrete, 4 actions, 8-dim state)
4. Plot reward curves, observe convergence
5. Experiment: tune hyperparameters (lr, buffer size, epsilon decay, batch size)

**Deliverable:** A working DQN agent that consistently lands the lunar lander (reward > 200).

---

## Phase 2: PPO + LunarLander

**Goal:** Implement PPO and compare it against your DQN on the same environment.

**Why PPO second:**
- You already know the theory (policy gradient → A2C → TRPO → PPO)
- Same environment lets you do a fair comparison with DQN
- Builds intuition for when value-based vs policy-based methods shine

**Steps:**
1. Implement PPO from scratch in PyTorch
   - Actor-Critic network
   - GAE (Generalized Advantage Estimation)
   - Clipped surrogate objective
   - Value function loss + entropy bonus
2. Train on `LunarLander-v2`
3. Compare learning curves: DQN vs PPO
4. Experiment: clip ratio, GAE lambda, number of epochs per update

**Deliverable:** A working PPO agent on LunarLander + a comparison report (DQN vs PPO).

---

## Phase 3: DQN + Atari (Pixel-based RL)

**Goal:** Scale DQN to pixel observations — the classic DeepMind Atari setup.

**Why this matters:**
- Moves from low-dim state to raw pixels (need CNN)
- Real test of whether you understand preprocessing and training at scale
- Directly connects to the DQN paper you already have

**Steps:**
1. Set up Atari environment (`gymnasium[atari]`, `ale-py`)
2. Implement Atari preprocessing
   - Frame grayscale + resize (84x84)
   - Frame stacking (4 frames)
   - Reward clipping
   - Episode life wrapper
3. Add CNN feature extractor to your DQN
4. Train on `Pong` first (easiest Atari game, fast convergence)
5. Then try `Breakout` or `SpaceInvaders`

**Deliverable:** A DQN agent that beats Atari Pong (reward > 18).

---

## Phase 4: PPO + Atari

**Goal:** Apply PPO to Atari and compare with DQN on pixel-based tasks.

**Steps:**
1. Adapt PPO with CNN feature extractor
2. Use vectorized environments for parallel data collection
3. Train on `Pong`, then `Breakout`
4. Compare sample efficiency and final performance: DQN vs PPO on Atari

**Deliverable:** PPO Atari agent + full DQN vs PPO comparison on both LunarLander and Atari.

---

## Phase 5: Chinese Chess AI (Boss Level)

**Goal:** Apply RL to train a Chinese Chess agent, building toward AlphaZero-style self-play.

**Prerequisites from earlier phases:**
- Solid DQN and PPO implementations
- Experience with CNN feature extraction
- Understanding of training dynamics and debugging RL

### Stage 1: Gymnasium Env Wrapper [DONE]
- Observation: (15, 10, 9) tensor — 14 piece planes + 1 turn plane
- Action space: Discrete(8100) with legal move masking
- Wraps existing ChineseChess game engine

### Stage 2: PPO Self-Play (Local — RTX 5060 Ti)

**Model:** CNN Actor-Critic
- 3-layer CNN (15→64→128→128) + FC(11520→512) + actor(8100) + critic(1)
- ~10.3M params, ~39 MB
- GPU memory: ~130 MB (batch=64) — trivial for RTX 5060 Ti (16GB)

**Training:**
- Self-play: agent plays both Red and Black
- Env speed: ~672 steps/sec (bottleneck is legal move generation)
- With 8 parallel envs: ~3000-4000 steps/sec estimated
- Target: 5-10M steps of self-play
- **Estimated time: 1-3 hours on RTX 5060 Ti**
- Expected strength: should beat RandomAI and GreedyAI, maybe weak MinimaxAI

**Why PPO here:**
- Reuses our existing PPO code from Phases 2/4
- Good baseline before going to AlphaZero
- Action masking straightforward with PPO (mask logits before softmax)

### Stage 2 Results

90% win rate vs random, but 0 wins vs Minimax depth-3 (0W/8L/2D in 10 games).
Pure PPO lacks tactical depth — search (MCTS) is essential. See [ChessRL/README.md](ChessRL/README.md) for full analysis.

### Stage 3: AlphaZero (Cloud — H100)

**Model:** ResNet + dual heads (policy + value)
- Option A: 10 res blocks, 128 channels — 26.3M params, 100 MB, ~430 MB GPU
- Option B: 20 res blocks, 256 channels — 47.0M params, 179 MB, ~800 MB GPU
- Recommend starting with Option A

**Training:**
- AlphaZero loop: self-play with MCTS → collect games → train network → repeat
- Each self-play game needs ~200 MCTS simulations per move × ~100 moves = 20,000 NN forward passes per game
- MCTS is sequential per move (hard to parallelize within a game)
- Parallelism: run multiple self-play games concurrently

**Compute estimate (Option A on H100):**

| Component | Estimate |
|---|---|
| NN forward pass (H100) | ~0.05ms per batch of 1 |
| MCTS sims per move | 200 |
| Moves per game | ~100 |
| NN evals per game | ~20,000 |
| Time per game (1 thread) | ~2-3 sec |
| Parallel games (H100 80GB) | 32-64 concurrent |
| Games per hour | ~40,000-80,000 |
| Target total games | 100,000-500,000 |
| **Estimated training time** | **3-12 hours on H100** |
| H100 cloud cost (~$3/hr) | **~$10-$36** |

**Why H100 not RTX 5060 Ti:**
- MCTS needs massive NN inference throughput (millions of forward passes)
- H100 has ~3x memory bandwidth and ~4x compute vs 5060 Ti
- 80GB VRAM allows larger batch sizes and more parallel games
- On RTX 5060 Ti the same training would take ~12-48 hours

**Expected strength:** Should beat MinimaxAI (depth 3-4), approach amateur human level

---

## Phase 6: Train a Small LLM (NanoChat)

**Goal:** Pre-train a small language model from scratch, then fine-tune it with RLHF/PPO to create a chat model.

**Why this matters:**
- Connects RL (PPO) back to LLMs — the RLHF loop
- End-to-end experience: data → pre-training → SFT → reward model → PPO
- Ties together Season 4 (Language Models) and Season 5 (RL)

**Steps:**
1. Build a minimal GPT architecture from scratch (nanoGPT-style)
   - Transformer decoder, tokenizer, positional encoding
2. Pre-train on a small text corpus (e.g., TinyStories, small Wikipedia subset)
3. Supervised Fine-Tuning (SFT) on chat/instruction data
4. Train a reward model on preference data
5. PPO fine-tuning (RLHF) — reuse your PPO implementation from Phase 2/4
6. Evaluate: compare base model vs SFT vs RLHF outputs

**Deliverable:** A small chat model trained end-to-end, demonstrating the full RLHF pipeline.

---

## Tech Stack
- **Python 3.10+**
- **PyTorch** (model implementation)
- **Gymnasium** (environments)
- **Matplotlib / TensorBoard** (training visualization)
- **stable-baselines3** (reference/benchmark only)

## Reference Papers (already in Season-5)
- DQN: `DQN_Playing_Atari_with_Deep_Reinforcement_Learning.pdf`
- PPO: `PPO_Proximal_Policy_Optimization.pdf`
- TRPO: `TRPO_Trust_Region_Policy_Optimization.pdf`
- AlphaGo/AlphaZero: `AlphaGo_*.pdf`, `AlphaZero_*.pdf`
