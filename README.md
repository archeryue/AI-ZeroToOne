# AI-ZeroToOne

## Goal
 - Understand the fundamentals of AI
 - Master the AI-related tools and frameworks
 - Know the current state of the art, and How it works

## Plan
### Season 1: Learning Basics [Videos](https://space.bilibili.com/20942052/lists/4258679)
 - Machine Learning Basics
 - Deep Learning Basics
 - GPU & CUDA
 - PyTorch

### Season 2: Network Structure [Videos](https://space.bilibili.com/20942052/lists/5164326)
 - CNN(Convolutional Neural Network)
 - RNN(Recurrent Neural Network)
 - LSTM & GRU (Long Short Memories)
 - Computer Vision & Machine Translation
 - Seq2Seq & Attention Mechanism
 - ResNet (Shortcut Connections)
 - Transformer

### Season 3: Content Generation [Videos](https://space.bilibili.com/20942052/lists/5853725)
 - VAE(Variational Auto-Encoder)
 - GAN(Generative Adversarial Nets)
 - Diffusion Models(VI & Score-based)
 - Flow Matching & Diffusion(ODE & SDE)
 - Conditional Generation (Text-To-Image)
 - DiT(Diffusion Transformer)
 - Video Generation

### Season 4: Language Models [Videos](https://space.bilibili.com/20942052/lists/6900780)
 - Review: RNN -> Seq2Seq -> Transformer
 - From Word2vec to BERT (Representation Learning)
 - GPT Series (Next-token Prediction is Intelligence)
 - BERT -> T5 (How to scale-up BERT?)
 - From CLIP to Flamingo (Way to Multimodality)

### Season 5: Reinforcement Learning [Videos](https://space.bilibili.com/20942052/lists/7264066)
 - RL Basics (Agent, Value Function, Policy)
 - Markov Decision Process (MP, MRP, MDP, & DP)
 - Traditional RL (Model-free Prediction, Model-free Control)
 - Deep RL (Value-based RL, Policy Gradient Methods, Actor-critic Methods)
 - Learning -> Optimization (TRPO -> PPO)
 - RL in Action (LunarLander & Atari)
 - RL in Action (Chess AI)
 - RL in Action (AlphaGo)

### Season 6: Build LLM

#### Part A: LLM System Concepts

* Engineering Foundation (Compute & Memory & Communication)
* Modern LLM Architecture (RoPE, GQA, SWA, MLA, MoE, AttnRes)
* Data Pipeline (Corpus Cleaning, Tokenization, Chat Templates)
* Distributed Training (DP → PP → TP → ZeRO)
* Post-Training & Alignment (RLHF, DPO, GRPO, LoRA)
* Evaluation (Objective Benchmarks, LLM-as-a-Judge)
* Inference Optimization (KV Cache, FlashAttention, PagedAttention, Quantization)

#### Part B: Rebuilding NanoChat from Scratch

* Codebase Walkthrough
* Core Module Rewrites (I): Muon Optimizer & Compute-Optimal Scaling
* Core Module Rewrites (II): Sliding Window Attention, Dataloader, KV Cache Engine
* Post-Training Rewrite: SFT Packing-to-Padding Transition
* Local Training: d12 Full Pipeline on Consumer GPU
* Cloud Speedrun: Beating GPT-2 under $100
* Evaluation & Retrospective
