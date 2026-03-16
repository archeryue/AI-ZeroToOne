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

**Steps:**
1. Build or finalize the Chinese Chess environment (already started in `ChineseChess/`)
2. Start with supervised learning on existing game databases
3. Implement MCTS (Monte Carlo Tree Search)
4. Combine neural network + MCTS (AlphaZero approach)
5. Self-play training loop

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
