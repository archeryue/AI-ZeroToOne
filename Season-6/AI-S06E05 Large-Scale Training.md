## Large-Scale Training: DP, TP, PP, and ZeRO

> **Format:** Lecture handout
> **Companion code:** [`karpathy/nanochat`](https://github.com/karpathy/nanochat) (single-node baseline) + Megatron-LM / DeepSpeed reference snippets
> **Strategy:** Start from a single GPU that can't fit the model. Walk DP, then the two model-sharding axes (TP, PP), then ZeRO as the alternative that keeps DP and shards the redundancy instead. End at 3D parallelism, the recipe that actually trains frontier models.

---

## Learning Objectives

By the end of this episode you should be able to:

1. Diagnose *which* memory or compute wall a given model hits on a given cluster, and pick the parallelism axis that addresses it.
2. Derive the communication cost of Ring AllReduce and explain why DP scales well on a single node but degrades across nodes.
3. Explain why TP shards activations along the *hidden* dimension and why this constrains it to NVLink-tier interconnects.
4. Draw a GPipe / 1F1B / interleaved pipeline schedule and quantify bubble overhead.
5. Walk the three ZeRO stages (1 / 2 / 3) and compute the per-GPU memory of each from first principles — then argue when ZeRO replaces TP+PP and when it complements them.
6. Compose DP × TP × PP into a 3D mesh and reason about *which axis goes where* in a real cluster.

## Prerequisites

- Episode 6.1 — Engineering Foundation (the 16×P rule, activation memory, NVLink vs IB).
- Episode 6.2 — Modern LLM Architecture (we'll reuse `d_model`, `d_ff`, `n_layers`, `H`).
- Comfort with PyTorch and `torch.distributed` collectives (`all_reduce`, `all_gather`, `reduce_scatter`).

---

## 0. The Setup: One GPU Isn't Enough

We anchor on a concrete question: **how do you train a 70B-parameter model when no single GPU can hold it?**

A back-of-envelope check with the 16×P rule (Episode 6.1):

```
Static state for 70B @ Adam mixed precision:
  fp32 weights      : 4 ·  70e9 =  280 GB
  fp32 grads        : 4 ·  70e9 =  280 GB
  Adam m, v         : 8 ·  70e9 =  560 GB
  bf16 weights copy : 2 ·  70e9 =  140 GB
  ─────────────────────────────────────────
  Total             :          ≈ 1.26 TB
```

An H100 has **80 GB**. The static state alone is **16× too large** for one GPU, before we touch activations.

Four classical answers, each a different cut:

| Technique | What it shards | Communication | Memory savings |
|-----------|----------------|---------------|----------------|
| **DP** (Data Parallelism) | Data across replicas | AllReduce of gradients | None (full replica per GPU) |
| **TP** (Tensor Parallelism) | Individual matmuls across GPUs | AllReduce *inside* each layer | Linear in TP degree |
| **PP** (Pipeline Parallelism) | Layers across GPUs | Point-to-point activation passes | Linear in PP degree |
| **ZeRO** | Optimizer / grad / param state across DP replicas | Extra ReduceScatter + AllGather | Up to N× (linear in DP world size) |

DP and ZeRO live on the *data* axis; TP and PP live on the *model* axis. We walk DP first (it doesn't fix memory), then TP and PP (the two ways to cut the model), then ZeRO (the alternative that keeps DP and just removes its redundancy). At the end they compose: **3D parallelism** is DP × TP × PP, with ZeRO bolted onto the DP axis and EP (expert parallelism) added for MoE.

---

## 1. Data Parallelism (DP)

### 1.1 The Mechanism

Every GPU holds a **full copy** of the model. The global batch is split N ways; each GPU computes its forward + backward on its shard, then gradients are summed across replicas before the optimizer step.

```
GPU 0:  batch[0:B/N]  →  fwd → bwd → grads_0  ┐
GPU 1:  batch[B/N:2B/N] →  fwd → bwd → grads_1 ├── AllReduce → mean(grads) → step
  ...                                          │
GPU N-1: batch[(N-1)B/N:B] → fwd → bwd → ...   ┘
```

After AllReduce all replicas have identical gradients, so the optimizer step produces identical weight updates everywhere. Replicas stay in sync.

### 1.2 Ring AllReduce: Cost Model

Naive AllReduce (gather everything to rank 0, broadcast back) is `O(N · M)` per GPU. Ring AllReduce hits the bandwidth lower bound:

```
Per-GPU bytes moved (Ring AllReduce) = 2 · (N − 1) / N · M
                                          ↑
                                     ReduceScatter + AllGather
```

For large `N`, this approaches `2 · M` — **independent of world size**. That's why DP scales gracefully.

### 1.3 Where DP Hits a Wall

- **Memory.** Every GPU still holds the full 1.26 TB of state for 70B. DP alone never fixes this.
- **Cross-node bandwidth.** AllReduce on every step. Inside a node (NVLink ~900 GB/s) this is cheap. Across nodes (InfiniBand ~50 GB/s) it dominates step time.
- **Effective batch size.** Scaling DP means scaling global batch. At some point you exceed the critical batch size and per-token efficiency drops.

### 1.4 PyTorch Reality

`torch.nn.parallel.DistributedDataParallel` (DDP) is the standard implementation. Key engineering details:

- **Gradient bucketing.** Gradients are coalesced into ~25 MB buckets so that AllReduce overlaps with the backward pass — the last bucket finishes shortly after the backward does.
- **`find_unused_parameters=False`** by default; turn on only when the graph is dynamic.
- **NCCL** is the actual collective; pinned by `torch.distributed.init_process_group("nccl")`.

> **Takeaway:** DP is the cheapest parallelism to reason about and the only one that scales the global batch. It doesn't help with the memory wall. Use it as the outer axis of any composite scheme.

---

## 2. Tensor Parallelism (TP)

DP didn't shard the model. The two model-sharding axes are TP (cut *inside* a layer) and PP (cut *between* layers). Start with TP.

### 2.1 The Mechanism

Instead of replicating each linear layer, **shard the matmul itself** across GPUs. Each GPU owns a slice of the weight matrix; partial outputs are combined with an AllReduce inside the layer.

Two canonical patterns from Megatron-LM (Shoeybi et al., 2019):

**Column-parallel linear** — shard `W` along output dim:

```
Y = X · W              W ∈ ℝ^(d_in × d_out)
                       Split W along columns: W = [W₁ | W₂ | ... | W_T]
Each GPU computes:     Y_i = X · W_i             ←  X is replicated
Output:                Y   = [Y₁ | Y₂ | ... | Y_T]  (concatenated, no comm)
```

**Row-parallel linear** — shard `W` along input dim:

```
Y = X · W              X is split along feature dim: X = [X₁; X₂; ...; X_T]
                       W is split along rows: W = [W₁; W₂; ...; W_T]
Each GPU computes:     Y_i = X_i · W_i           ←  partial sum
Output:                Y = Σ Y_i  via AllReduce  ←  one collective
```

### 2.2 Transformer Block, TP-Sharded

The classical pattern: **Column-parallel → Row-parallel** for each pair of linears, AllReduce at the end.

```
Attention:
  W_Q, W_K, W_V       :  column-parallel  (shard heads across GPUs)
  W_O                 :  row-parallel     (AllReduce on output)

FFN:
  W_up   / W_gate     :  column-parallel
  W_down              :  row-parallel     (AllReduce on output)
```

Two AllReduces per layer (one for attention output, one for FFN output). Activations along the *hidden* dimension are sharded; the residual stream is replicated.

### 2.3 Sequence Parallelism (Bonus Axis)

The norms and dropouts between sublayers aren't naturally TP-friendly — they need the full hidden vector. **Sequence parallelism** (Korthikanti et al., 2022) shards *those* along the sequence axis instead, eliminating the need to replicate activations for norms. Megatron-LM ships this by default now.

### 2.4 Why TP Wants NVLink

TP fires **two AllReduces per layer**, on tensors of shape `[B, T, d_model]`. For LLaMA-3-70B at `T=8192, B=1`:

```
Activation per AllReduce: 1 · 8192 · 8192 · 2 bytes ≈ 128 MB
× 2 collectives × 80 layers          ≈ 20 GB / step / GPU of TP traffic
```

At NVLink bandwidth (~900 GB/s) this is ~22 ms per step — tolerable.
At IB bandwidth (~50 GB/s) it would be **400 ms** — pure overhead.

**Rule of thumb:** keep TP degree ≤ GPUs per NVLink domain. For H100 SXM, that's 8.

### 2.5 What TP Buys You

- Linear memory split on weights, gradients, optimizer state, and *most* activations.
- Splits the matmul compute: a TP=8 group does each layer's matmul 8× faster (modulo overhead).
- The only parallelism axis that reduces both memory and per-step latency.

> **Takeaway:** TP is the inside-node hammer. It shards the layer; both memory and compute drop linearly with TP degree. The cost is two AllReduces per layer, which is why TP doesn't cross node boundaries.

---

## 3. Pipeline Parallelism (PP)

### 3.1 The Mechanism

Split the model's **layers** across GPUs. Each GPU holds a contiguous *stage* of `L/P` layers. Activations flow forward from stage to stage; gradients flow backward.

```
GPU 0  : [layers  0..11 ]  ──fwd──▶
GPU 1  : [layers 12..23 ]  ──fwd──▶
GPU 2  : [layers 24..35 ]  ──fwd──▶
GPU 3  : [layers 36..47 ]  ──fwd──▶ loss
                          ◀──bwd── back through stages 3→0
```

Communication is *point-to-point*: send the activation tensor `[B, T, d]` from stage `i` to stage `i+1`. **Tiny** compared to TP's AllReduces — one tensor per stage transition per microbatch.

### 3.2 The Bubble Problem

Naive PP (full batch through stage 0, then stage 1, ...) leaves every other GPU idle most of the time. The classical fix: split the batch into **microbatches** and feed them in pipeline.

**GPipe schedule (Huang et al., 2018):**

```
time →
GPU 0:  F₀ F₁ F₂ F₃ . . . . . . . . B₃ B₂ B₁ B₀
GPU 1:     F₀ F₁ F₂ F₃ . . . . . B₃ B₂ B₁ B₀ .
GPU 2:        F₀ F₁ F₂ F₃ . . B₃ B₂ B₁ B₀ . . .
GPU 3:           F₀ F₁ F₂ F₃ B₃ B₂ B₁ B₀ . . . . .
                ↑           ↑           ↑
              fill        steady     drain
              bubble      state      bubble
```

**Bubble fraction:**

```
bubble = (P − 1) / (M + P − 1)
```

where `P` = pipeline stages, `M` = number of microbatches. Push `M ≫ P` and the bubble shrinks; `M ≈ 4·P` is typical.

### 3.3 1F1B and Interleaved Schedules

**1F1B (one forward, one backward)** — used by PipeDream and Megatron-LM. Once the pipeline fills, alternate one forward and one backward per step:

- Activation memory per stage drops from `M × per_microbatch` (GPipe) to `~P × per_microbatch` (1F1B). Big win at long pipelines.

**Interleaved 1F1B (Megatron-LM v2)** — split each stage further into `V` *virtual stages*, interleaved on the same GPU. Bubble shrinks by `1/V` at the cost of `V×` more communication.

```
bubble (interleaved) = (P − 1) / (V · M + P − 1)
```

Megatron commonly uses V=2 or V=4 on long pipelines.

### 3.4 What PP Buys You

- Linear memory split on **weights, grads, optimizer state, and per-layer activations**.
- Minimal communication — only a single tensor per stage boundary.
- Survives across IB links because point-to-point is cheap.

### 3.5 What PP Costs You

- Bubble overhead, mitigated but never eliminated.
- Activations of all in-flight microbatches must live in memory.
- Schedule complexity. Frameworks (Megatron-LM, DeepSpeed, FairScale) hide it; building it yourself is non-trivial.

> **Takeaway:** PP is the outside-node hammer. It costs almost no bandwidth but introduces a scheduling bubble. Use it to split layers across nodes once TP has filled the NVLink domain.

---

## 4. ZeRO: Removing the DP Replica Redundancy

TP and PP shard the model. ZeRO takes the other path: **keep plain DP, and shard the duplication DP creates.** It lives on the *data* axis, not the model axis — which is what makes it a sibling alternative to TP+PP, not a refinement of either.

### 4.1 The Insight

Plain DP wastes memory: every replica stores identical optimizer state, gradients, and parameters. ZeRO (Rajbhandari et al., 2020) shards these *across* the DP group. Each GPU owns `1/N` of each tensor; collectives reconstruct what's needed on demand.

### 4.2 The Three Stages

Build them up as a ladder — each stage removes one more form of duplication.

| Stage | Sharded | Per-GPU memory (16×P → ?) | Extra communication |
|-------|---------|---------------------------|---------------------|
| **ZeRO-1 (Pₒₛ)** | Optimizer state | 4P + 4P + (8P)/N | Same as DDP (1 AllReduce of grads) |
| **ZeRO-2 (Pₒₛ+g)** | Optimizer state + gradients | 4P + (4P + 8P)/N | ReduceScatter of grads (same bandwidth as AllReduce halved) |
| **ZeRO-3 (Pₒₛ+g+p)** | Optimizer state + gradients + parameters | 16P/N | AllGather of params each fwd/bwd + ReduceScatter of grads |

In the limit `N → ∞`, ZeRO-3 drives per-GPU memory toward zero. This is the engine behind **FSDP** (Fully Sharded Data Parallel) — PyTorch's native ZeRO-3 implementation.

### 4.3 ZeRO-3 / FSDP: What Actually Happens Per Step

```
For each transformer block (during forward):
  AllGather params for this block      ←  ~M_block bytes across DP group
  Compute block forward
  Discard the gathered params
For each block (during backward, reverse order):
  AllGather params again
  Compute block backward
  ReduceScatter the local grads        ←  M_block bytes
  Discard gathered params
Optimizer step on the locally-owned 1/N shard
```

Memory drops to `O(M / N)` static + the largest single block's activations in flight. Communication roughly **1.5×** plain DDP: ReduceScatter (= half AllReduce) + two AllGathers (one fwd, one bwd, but bwd's can overlap with the previous gather's compute).

### 4.4 ZeRO-Offload and ZeRO-Infinity

- **ZeRO-Offload:** push optimizer state to **CPU RAM**; do the Adam update on CPU. Useful when DRAM is plentiful and PCIe isn't the bottleneck. Trains a 13B model on a single GPU with 32 GB of HBM.
- **ZeRO-Infinity:** push state further, to **NVMe**. Now you can prefetch shards from SSD between layers. Frontier "single-node" record-breakers (e.g., training a 175B model on one DGX) use this. Real throughput is modest, but the *capability threshold* moves.

### 4.5 FSDP-Only: The Modern Shortcut

The reason ZeRO has eaten so much of the mid-scale playbook: for many models, **you can skip TP and PP entirely**.

- No TP code (no column/row-parallel linears, no sequence-parallelism kernel quirks).
- No PP code (no schedule, no microbatch bookkeeping, no bubble).
- Just `FSDP(model)` plus DDP's mental model.

This works cleanly up to roughly the 70B range on a fast-interconnect cluster. Beyond that, ZeRO-3's per-block AllGather across IB becomes the bottleneck, and you go back to TP+PP+ZeRO-1.

### 4.6 Practical Tradeoffs

- ZeRO-1 is "free" — no extra collective, ~3.75× memory savings on Adam state. Use it always.
- ZeRO-2 adds no bandwidth, halves grad memory. Use it whenever DP > 1.
- ZeRO-3 / FSDP is the workhorse for large models without bespoke TP/PP code. Cost: an extra round of AllGathers per step.
- **Inside-node only** for ZeRO-3 ideally — crossing IB on every block AllGather is painful.

> **Takeaway:** ZeRO is "DP minus duplication." Same effective parallelism, much smaller per-GPU footprint, at a modest communication tax that overlaps well with compute. At mid scale it replaces TP+PP entirely; at frontier scale it complements them by riding the DP axis.

---

## 5. 3D Parallelism: Putting It Together

### 5.1 The Hierarchy

A real training mesh is a **3D grid**: `DP × TP × PP`. Each axis is placed onto the hardware tier whose communication cost it can afford.

```
              Communication cost    Where it lives
TP axis     :  2 AllReduces/layer    inside NVLink domain (1 node)
PP axis     :  P2P at boundaries     across nodes (IB)
DP axis     :  1 AllReduce/step      across nodes (IB) — overlaps with bwd
```

A worked layout for a 64-GPU cluster (8 nodes × 8 GPUs):

```
TP = 8   :  one TP group = one node (NVLink)
PP = 4   :  4 pipeline stages spanning 4 "blocks of nodes" (IB)
DP = 2   :  2 data replicas across the remaining axis
Total    :  8 × 4 × 2 = 64 GPUs
```

### 5.2 Megatron's Rules of Thumb

From the Megatron-LM scaling paper (Narayanan et al., 2021):

1. **Fill TP first**, up to `nvlink_domain_size` (usually 8).
2. **Then PP**, sized so each stage fits in memory after TP sharding.
3. **DP gets whatever is left.** Add ZeRO-1 on the DP axis "for free."
4. Tune microbatch count `M` so the pipeline bubble is < 10%.

### 5.3 Where ZeRO Plugs In

ZeRO shards along the **DP axis**. The DP world size is the number of replicas, and ZeRO partitions optimizer / grad / param state across them. With (TP=8, PP=4, DP=2) you have only `N_DP=2` shards — small benefit. Big DP groups (no PP, no TP) get the most out of ZeRO-3 / FSDP.

A common modern composition:

- **TP + ZeRO-1 (small/medium models):** fill the node with TP, shard optimizer state across DP groups. Used by Llama-3 405B training internally at Meta.
- **TP + PP + ZeRO-1 (huge dense):** Megatron's recipe for GPT-3 and beyond.
- **FSDP-only (pure ZeRO-3):** PyTorch-native, no TP/PP code. Scales to ~100B before TP/PP becomes mandatory.

### 5.4 Expert Parallelism (MoE Footnote)

When the FFN becomes a MoE (Episode 6.2 §6.2), a new axis appears: **EP** (expert parallelism). Different experts live on different GPUs. Token routing introduces all-to-all communication, which dominates MoE training cost. DeepSeek-V3 trains with EP=64 in addition to TP/PP/DP. We don't unpack this here — own episode later.

---

## 6. Where nanochat Sits

nanochat trains on a **single 8×H100 node**. The parallelism story is therefore deliberately minimal:

| Axis | nanochat | Production frontier |
|------|----------|---------------------|
| DP   | DDP across 8 GPUs | DP × N nodes |
| TP   | None | TP=8 typical, fills NVLink |
| PP   | None | PP=4–32 |
| ZeRO | None — model fits in 80 GB | ZeRO-1 minimum, often ZeRO-3 / FSDP |
| EP   | None (dense FFN) | EP=8–64 for MoE |

For nanochat-sized models (~560M params), 8× DDP with bf16 mixed precision and FlashAttention is enough. The compositional machinery in this episode is **what you reach for when nanochat-style speedruns no longer fit.**

> **Pedagogical note:** the techniques here are *invisible* in nanochat's source, but every line of `torchrun --nproc-per-node=8` is implicitly a DP world. Understanding TP/PP/ZeRO is what lets you reason about the next step up — turning the speedrun into something that trains a 70B model.

---

## 7. Synthesis

| Axis | What it shards | Bandwidth tier | Memory savings | Compute savings |
|------|---------------|----------------|----------------|-----------------|
| **DP** | Data (none of the model) | IB (1 AllReduce/step) | None | None |
| **TP** | Matmul | NVLink (2 AllReduces/layer) | Linear in TP | Linear in TP |
| **PP** | Layers | IB (P2P at boundaries) | Linear in PP | None (and adds bubble) |
| **ZeRO-1/2/3** | Opt state / + grads / + params | IB (overlaps with bwd) | Linear in DP | None |

The unifying picture:

> **Pick the axis whose communication you can afford on the link it has to cross.**
>
> NVLink → TP. IB → PP and DP. CPU/NVMe → ZeRO-Offload/Infinity.

Every memory wall has a parallelism axis tuned to it. The art is matching the axis to the cluster topology.

### Decision Tree

```
Model fits in one GPU?
  → DDP for throughput. Done.
Model fits across one node but not on one GPU?
  → TP (degree = nvlink_size), DP across nodes, ZeRO-1.
Model doesn't fit across one node?
  → Option A: FSDP (ZeRO-3) only. Simplest code, works up to ~70B.
  → Option B: TP inside node, PP across nodes, DP on what's left, ZeRO-1 on DP.
Model is MoE?
  → Add EP axis; expect all-to-all to dominate.
Model is bigger than your whole cluster?
  → ZeRO-Infinity (NVMe offload) buys capability at low throughput.
```

---

## References

### Core Papers

- Sergeev & Del Balso 2018 — *Horovod* (ring AllReduce in PyTorch/TF).
- Shoeybi et al. 2019 — *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (TP).
- Huang et al. 2018 — *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism*.
- Narayanan et al. 2019 — *PipeDream: Generalized Pipeline Parallelism for DNN Training* (1F1B).
- Rajbhandari et al. 2020 — *ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models*.
- Rajbhandari et al. 2021 — *ZeRO-Infinity: Breaking the GPU Memory Wall*.
- Korthikanti et al. 2022 — *Reducing Activation Recomputation in Large Transformer Models* (Sequence Parallelism).
- Narayanan et al. 2021 — *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (3D parallelism, interleaved schedules).
- Zhao et al. 2023 — *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*.

### Code

- [`NVIDIA/Megatron-LM`](https://github.com/NVIDIA/Megatron-LM) — canonical TP + PP reference.
- [`microsoft/DeepSpeed`](https://github.com/microsoft/DeepSpeed) — ZeRO + offload.
- [`pytorch/pytorch`](https://github.com/pytorch/pytorch) — `torch.distributed.fsdp`, `torch.distributed.pipelining`.
- [`karpathy/nanochat`](https://github.com/karpathy/nanochat) — single-node DDP baseline (what we'll diff against).

### Further Reading

- HuggingFace — *The Ultra-Scale Playbook* (interactive walkthrough of every axis in this episode).
- Meta AI Engineering — Llama-3 training infrastructure post-mortem.

---

## Part B Code-Rewrite Modules (Preview)

1. **Module 1** — Wrap a nanochat training step in DDP across 2 GPUs; measure AllReduce overlap with backward.
2. **Module 2** — Manually implement column-parallel + row-parallel linears for a transformer block; verify forward matches single-GPU reference numerically.
3. **Module 3** — Implement a 2-stage GPipe pipeline by hand; plot bubble fraction vs `M`.
4. **Module 4** — Enable FSDP (ZeRO-3); compare per-GPU memory and step time vs DDP.
5. **Module 5** *(stretch)* — Compose TP=2 × PP=2 × DP=2 on 8 GPUs. Match Megatron's recipe at a tiny scale.
