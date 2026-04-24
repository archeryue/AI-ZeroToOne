# Season 6: Build LLM

## Part A: LLM System Concepts

* Engineering Foundation (Compute & Memory & Communication)
* Modern LLM Architecture (RoPE, GQA, SWA, MLA, MoE, AttnRes)
* Data Pipeline (Corpus Cleaning, Tokenization, Chat Templates)
* Distributed Training (DP → PP → TP → ZeRO)
* Post-Training & Alignment (RLHF, DPO, GRPO, LoRA)
* Evaluation (Objective Benchmarks, LLM-as-a-Judge)
* Inference Optimization (KV Cache, FlashAttention, PagedAttention, Quantization)

## Part B: Rebuilding NanoChat from Scratch

* Codebase Walkthrough
* Core Module Rewrites (I): Muon Optimizer & Compute-Optimal Scaling
* Core Module Rewrites (II): Sliding Window Attention, Dataloader, KV Cache Engine
* Post-Training Rewrite: SFT Packing-to-Padding Transition
* Local Training: d12 Full Pipeline on Consumer GPU
* Cloud Speedrun: Beating GPT-2 under $100
* Evaluation & Retrospective
