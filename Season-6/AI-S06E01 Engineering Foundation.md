
---

## Episode Structure

| Section | Duration | Topic |
|---|---|---|
| Hook | 2 min | The $43,000 question |
| Setup | 3 min | GPT-2 as the reference model |
| Section 1 | 15 min | Compute |
| Section 2 | 20 min | Memory |
| Section 3 | 15 min | Communication |
| Closing | 5 min | The three-way tradeoff triangle + season preview |

---

## Hook (2 min)

> "In February 2019, OpenAI released GPT-2. Training it cost approximately $43,000 and took 168 hours on 32 TPU v3 chips. Seven years later, we can train a model that beats it on a single 8×H100 node in under 3 hours for under $100. That's a 600× cost reduction. Where did those savings come from?
>
> The answer isn't one thing. It's dozens of improvements stacked across three physical constraints: compute, memory, and communication. By the end of Season 6, you'll have rebuilt each of those improvements yourself, and trained a model that beats GPT-2 — for real, on the same CORE benchmark OpenAI used."

The "600×" decomposition, previewed here and paid off in the closing:

- ~2× from hardware (TPU v3 → H100)
- ~2.5× from utilization / MFU (better software)
- ~3× from data quality (FineWeb-EDU vs WebText)
- ~3× from algorithms (Muon, modern architecture, better schedules)
- ~5–10× from GPU rental market dynamics

Exact attribution is hand-wavy, but the decomposition itself is the teaching point.

---

## Setup: GPT-2 as the Reference Model (3 min)

Pin down the numbers viewers will see referenced throughout the season:

| Attribute | Value |
|---|---|
| Parameters | 1.5B (n_layer=48, n_embd=1600, n_head=25) |
| Training tokens | ~40B (WebText) |
| Context length | 1024 |
| Tokenizer | 50,257 BPE vocab |
| Hardware | 32× TPU v3 chips (256 cores) |
| Training time | ~168 hours (~1 week) |
| Training cost | ~$43,000 (2019 TPU prices) |
| CORE score | 0.2565 |

Note on the "32 chips vs 256 cores" discrepancy: TPU v3 has 8 cores per chip. Both numbers are correct; mention it briefly so search-savvy viewers can reconcile sources.

---

## Section 1: Compute (15 min)

### The 6ND FLOPs Budget

Training FLOPs ≈ **6 × N × D** where N = params, D = training tokens.

Derivation:
- Forward pass: 2 FLOPs per parameter per token (one multiply + one add per weight)
- Backward pass: 4 FLOPs per parameter per token (activation gradients + weight gradients, each roughly one forward's cost)
- Total: 6 FLOPs per parameter per token

**GPT-2 worked example**: 6 × 1.5e9 × 40e9 = **3.6 × 10²⁰ FLOPs**
**GPT-3 for scale**: 3.14 × 10²³ FLOPs (~1000× more)

### Hardware Evolution: TPU v3 → H100

| Hardware | bf16 TFLOPS (peak) | Year |
|---|---|---|
| TPU v3 (single chip) | 123 | 2018 |
| 32× TPU v3 pod | ~3,900 | 2018 |
| 8× H100 SXM | ~7,900 dense / ~15,800 sparse | 2023 |

Raw hardware gives ~2×. The rest of the 600× must come from software and algorithms.

### MFU: Model FLOPs Utilization

The honest metric for real-world compute efficiency:

```
MFU = (6·N·D) / (training_seconds × peak_FLOPS × num_GPUs)
```

- GPT-2 on TPU v3: estimated ~15–20% MFU
- Modern training on H100: 40–55% MFU
- Another ~2.5× "for free" from better software

Peak numbers (989 TFLOPS bf16 on H100) are marketing. Real workloads hit 30–60% of peak.

| **GPU Model**  | **Generation / Target** | **VRAM & Memory Type** | **Memory Bandwidth** | **FP16 / BF16 (Dense TFLOPS)** | **FP8 (Dense TFLOPS)** | **Compute Exchange Rate (vs. A100)** |
| -------------- | ----------------------- | ---------------------- | -------------------- | ------------------------------ | ---------------------- | ------------------------------------ |
| **Tesla V100** | Volta (Legacy Server)   | 32GB HBM2              | 900 GB/s             | 125                            | N/A                    | **~0.4x**                            |
| **RTX 4090**   | Ada (Local R&D)         | 24GB GDDR6X            | 1.0 TB/s             | ~330                           | ~660                   | **~1.0x** (Compute only)             |
| **L20**        | Ada (PCIe, Export)      | 48GB GDDR6             | 864 GB/s             | 120                            | 239                    | **~0.4x**                            |
| **L40**        | Ada (PCIe Server)       | 48GB GDDR6a            | 864 GB/s             | 181                            | 362                    | **~0.6x**                            |
| **A100 80GB**  | Ampere (SXM Standard)   | 80GB HBM2e             | 2.0 TB/s             | **312**                        | N/A                    | **1.0x (Baseline)**                  |
| **H20**        | Hopper (SXM, Export)    | 96GB HBM3              | **4.0 TB/s**         | 148                            | 296                    | **~0.5x**                            |
| **H100**       | Hopper (SXM Flagship)   | 80GB HBM3              | 3.35 TB/s            | 989                            | 1,979                  | **~3.1x**                            |
| **B200**       | Blackwell (Next-Gen)    | 192GB HBM3e            | 8.0 TB/s             | 2,250                          | 4,500                  | **~7.2x**                            |

### Arithmetic Intensity and the Roofline Model

- Arithmetic intensity = FLOPs ÷ bytes moved from HBM
- Attention is **memory-bound** at small batch sizes, **compute-bound** at large batch sizes
- This motivates FlashAttention (revisited in Episode 6.7)

### Preview of NanoChat's Numbers

- NanoChat d24 on 8×H100: ~3 hours, beats GPT-2 CORE
- Achieved MFU: ~45% (from leaderboard wandb logs)
- Total training FLOPs: ~4.3 × 10¹⁹

**Punchline**: NanoChat d24 beats GPT-2 using ~1/8 the FLOPs. Not hardware, not utilization — fundamentally better training recipes.

### Required Visual

A single slide: "What 1 H100-hour buys you" showing tokens processed for d12, d20, d24, d32.

---

## Section 2: Memory (20 min)

### The 16×P Rule for Standard Adam Training

Byte-by-byte decomposition:
- fp32 weights: 4 bytes/param
- fp32 gradients: 4 bytes/param
- Adam m: 4 bytes/param
- Adam v: 4 bytes/param
- **Total: 16 bytes/param**

**GPT-2 at 1.5B params**: 24 GB of static memory. TPU v3 had 16 GB HBM — so GPT-2's optimizer state didn't even fit on one chip. Model parallelism wasn't optional.

### Modern Static Memory: Muon + Mixed Precision

- fp32 master weights: 4 bytes/param
- fp32 gradients: 4 bytes/param
- Muon momentum (matrix params): 4 bytes/param (no second moment)
- AdamW (embeddings only, small fraction): 8 bytes × small

Net: ~**12 bytes/param** for nanochat-style training. A modest improvement for GPT-2 specifically, but the real memory breakthrough is activations.

### Activation Memory: The B·T·d·L Formula

The critical insight: **activation memory scales with tokens-in-flight, not param count.**

```
activation_memory ≈ c · B · T · d · L · bytes_per_element
```

Where:
- **B** = batch size per GPU
- **T** = sequence length
- **d** = hidden dimension
- **L** = number of layers
- **c** ≈ 20–34 (constant depending on FlashAttention and compiler fusion)

**Why these four dimensions**: every activation is a d-dimensional vector at some (batch, sequence position, layer) coordinate. Total count is B·T·L, total size is B·T·L·d, constant c counts how many such vectors per token per layer need to be kept alive for backward.

**Where c comes from** (per transformer block):
- Attention sub-block: ~7d per token (pre-norm input, QKV outputs, attention output, post-projection, FA LSE scalar)
- FFN sub-block: ~12–14d per token (pre-norm, FFN intermediate at 4d or SwiGLU variants, output)
- Plus residual stream and norms: ~25–30d total per token per layer

### The Pre-FlashAttention Formula (for contrast)

Classic Megatron-LM formula per layer, per sample:

```
s · b · h · (34 + 5·a·s/h) bytes
```

Two parts:
- **34·s·b·h** — linear in all dimensions (the stuff above)
- **5·a·s²·b** — quadratic in T: the attention scores matrix `softmax(Q·Kᵀ)` of shape (B, heads, T, T)

The T² term is what made long context intractable pre-FlashAttention. At T=2048 it's ~15% of activation memory; at T=8192 it dominates; at T=32k it's catastrophic.

**FlashAttention's achievement**: eliminates the O(T²) term by never materializing the full attention matrix. Only stores a per-row logsumexp scalar for backward. This is why modern long-context training is possible.

### GPT-2 Activation Memory Worked Example

Per sample (B=1), without FlashAttention:
- Linear part: 25 × 1 × 1024 × 1600 × 48 × 2 bytes ≈ **6.3 GB**
- Quadratic part: 5 × 25 × 1 × 1024² × 2 bytes ≈ **0.26 GB**
- Modest at T=1024; would explode at modern context lengths

### The Four Dimensional Levers

When you OOM, the formula tells you exactly what to cut:

| Change | Activation memory |
|---|---|
| Halve B | 50% |
| Halve T | 50% |
| Halve d | 50% (also halves model) |
| Halve L | 50% (also halves model) |

**B is the first knob to touch** — doesn't change model identity, compensate with gradient accumulation. T is often fixed by the task. d and L are locked once you've picked a depth.

### Activation Checkpointing

- Without: O(L) activation memory, 1× compute
- With checkpointing every √L layers: O(√L) activation memory, 1.33× compute
- For L=20 (nanochat d20): ~78% memory savings at 33% compute cost
- NanoChat doesn't enable by default — speedrun prioritizes wall-clock time

### Before/After Memory Table

| Technique | 2019 (GPT-2) | 2026 (nanochat) |
|---|---|---|
| Precision | fp32 everywhere | bf16 compute, fp32 master |
| Optimizer state | Adam (8 bytes/param) | Muon (4 bytes/param, matmul params) |
| Attention memory | Quadratic in T | Linear in T (FlashAttention) |
| Architecture | MHA, LayerNorm | GQA, SWA, RMSNorm |

Each row previews a future Season 6 episode.

### Required Visuals

1. Stacked bar chart: memory breakdown (weights/grads/opt/activations) for different model sizes
2. Activation memory formula decomposition showing the 4 dimensional scaling
3. Before/after memory comparison table

---

## Section 3: Communication (15 min)

### The Three-Layer Framing

Networking technologies used in training clusters span three levels of the OSI model, but for LLM purposes we care about three physical hierarchy tiers:

**Tier 1: Intra-chip** (SRAM ↔ HBM)
- ~3.35 TB/s HBM3 on H100 SXM
- FlashAttention's design exploits this tier's speed

**Tier 2: Intra-node** (GPU ↔ GPU, same server)
- NVLink 4: 900 GB/s bidirectional per GPU (H100 SXM)
- PCIe Gen5: 128 GB/s (used when no NVLink, e.g., consumer GPUs)

**Tier 3: Inter-node** (server ↔ server, same datacenter)
- InfiniBand 400G (NDR): 50 GB/s per NIC, ~1 μs latency
- High-speed Ethernet (with RoCE): comparable bandwidth, slightly worse latency
- ~18× slower than NVLink

**Tier 4: Inter-datacenter** (long-haul, DCI)
- Dedicated fiber / leased wavelengths
- Millisecond-scale latency
- Relevant only for hyperscale training (GPT-4+); nanochat never touches this

### The 10× Rule

Each tier is roughly 3–10× slower than the one above. That multiplicative hierarchy is what drives every distributed training decision.

### Ethernet vs InfiniBand vs Wi-Fi: Layer 2 Alternatives

All three are Layer 2 networking technologies. They all carry IP packets (Layer 3) on top.

| Technology | Use case | Speed | Latency |
|---|---|---|---|
| Ethernet | Wired LAN, datacenters | up to 800 Gb/s | ~1 μs |
| Wi-Fi | Wireless LAN | up to 10 Gb/s | ~1–10 ms |
| InfiniBand | HPC / AI clusters | up to 800 Gb/s | ~0.5 μs |

In a real AI cluster, a node runs **two parallel Layer 2 fabrics**:
- IB (or RoCE over Ethernet) for GPU training traffic
- Standard Ethernet for management, storage, SSH, monitoring

They're physically separate — different NICs, switches, subnets. Not bridged.

### RDMA and GPUDirect: Why Inter-Node Communication Is Usable

Naive cross-node transfer would be: GPU → CPU → NIC → wire → NIC → CPU → GPU. Two PCIe hops, two CPU memory copies, kernel syscalls. ~50+ μs per transfer.

**RDMA (Remote Direct Memory Access)**: NIC hardware reads/writes remote memory without CPU involvement.

**GPUDirect RDMA**: NIC directly reads/writes GPU memory without CPU OR system memory involvement. Data path becomes `GPU → NIC → wire → NIC → GPU`. Latency drops to ~2 μs for small messages; bandwidth hits 95%+ of link speed.

Without RDMA, distributed LLM training wouldn't be viable at scale. NCCL (NVIDIA's collective communication library) builds on RDMA and is what PyTorch's `all_reduce` ultimately calls.

### Collective Operations: AllReduce

Every data-parallel training step requires AllReduce of gradients across GPUs.

**Ring AllReduce** cost: 2(N-1)/N × model_size bytes per GPU, regardless of N. Decomposes as ReduceScatter + AllGather. This efficiency is why DP scales well.

**GPT-2 example**: 32 TPU v3 chips, 6 GB of fp32 gradients per step. AllReduce across 32 chips: ~11.6 GB per chip moved, at ~650 GB/s → ~18 ms per step just for gradient sync.

### The Communication Ceiling

Modeling communication as a bottleneck on a hypothetical cross-node 7B training:
- Intra-node reduce (NVLink): ~30 ms
- Inter-node AllReduce (IB): ~280 ms
- Total communication: ~340 ms per step

If compute step is 400 ms, communication imposes an 85% MFU ceiling. **This is why inter-node bandwidth is THE bottleneck at scale.**

For nanochat specifically (single 8×H100 node), none of this applies — you never leave NVLink. A deliberate simplification of the single-node-training design.

### Required Visuals

1. Latency ladder: SRAM → HBM → NVLink → PCIe → IB → Ethernet with 10^x numbers
2. Parallel-fabric diagram: node with both IB NICs and Ethernet NICs, serving different traffic
3. AllReduce cost model: compute time vs communication time as bar chart

---

## Closing (5 min)

### The Three-Way Tradeoff Triangle

Draw compute / memory / communication as triangle vertices. Every design decision is a point inside:

- **Gradient accumulation**: more compute time, less memory
- **Activation checkpointing**: more compute, less activation memory
- **ZeRO**: more communication, less memory
- **Tensor parallelism**: more communication, less memory AND splits compute
- **MoE**: more communication, more effective capacity per FLOP

Every technique in Season 6 lands somewhere in this triangle.

### Decomposing the 600× Cost Reduction

Back to the opening claim. Roughly:

| Factor | Size | Driven by | Episode |
|---|---|---|---|
| Hardware (TPU v3 → H100) | ~2× | Raw FLOPs | 6.1 |
| MFU (utilization) | ~2.5× | Software stack | 6.1, 6.4, 6.7 |
| Data quality | ~3× | FineWeb-EDU vs WebText | 6.3 |
| Algorithms | ~3× | Muon, architecture, schedules | 6.2, 6.5, Part B |
| Market pricing | ~5–10× | GPU rental economics | non-technical |

Multiplied: ~600×.

**Be epistemically honest on video**: exact attribution is fuzzy, different analyses give different numbers, but order of magnitude of each factor is roughly right. And acknowledge that price-per-FLOP market dynamics — not pure engineering — contribute a big chunk.

### Part B Hook

> "By the time we finish Season 6, you'll have implemented each of these improvements yourself. And in Part B, we'll take the complete stack and train a model on an 8×H100 node that beats GPT-2's CORE score — for real. Same benchmark OpenAI used. Same capability threshold. A fraction of the cost. That's the Season 6 promise."

---

## Key Numbers Cheatsheet for the Episode

| Quantity | Value |
|---|---|
| GPT-2 params | 1.5B |
| GPT-2 training tokens | 40B |
| GPT-2 FLOPs | 3.6 × 10²⁰ |
| GPT-2 CORE | 0.2565 |
| GPT-2 training cost | $43,000 |
| NanoChat d24 FLOPs | 4.3 × 10¹⁹ (~1/8 of GPT-2) |
| NanoChat d24 cost | ~$75 |
| H100 bf16 peak | 989 TFLOPS |
| H100 HBM3 | 3.35 TB/s |
| NVLink 4 | 900 GB/s |
| PCIe Gen5 | 128 GB/s |
| InfiniBand NDR | 50 GB/s |
| Static memory formula | 12–16 bytes/param |
| Activation formula | c · B · T · d · L · bytes, c ≈ 20–34 |
| AllReduce cost | 2(N-1)/N × model_size bytes/GPU |
