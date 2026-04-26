## The Modern LLM Architecture: A Diff Against GPT-2

> **Format:** Lecture handout
> **Companion code:** [`karpathy/nanochat`](https://github.com/karpathy/nanochat)
> **Strategy:** Anchor on GPT-2 (2019). Walk through what changed, why, and what nanochat adopts vs. simplifies.

-----

## Learning Objectives

By the end of this episode you should be able to:

1. Identify the core architectural shifts between GPT-2 and modern LLMs (LLaMA-3, Qwen3, DeepSeek-V3).
2. Derive RoPE from first principles and explain why it captures *relative* position despite using absolute rotations.
3. Articulate the relationship between Pre-norm, RMSNorm, and QK-Norm — and why all three coexist in nanochat.
4. Compute KV-cache savings for MHA → GQA → MQA, and explain when SWA helps.
5. Read nanochat’s `gpt.py` and identify each modern component, *including the ones nanochat deliberately rejects*.

## Prerequisites

- Episode 1 / *“Attention Is All You Need”* fluency.
- Linear algebra: rotations, dot products, matrix decomposition.
- PyTorch comfort — we’ll show real nanochat snippets.

-----

## 0. The Diff Strategy

Modern LLMs share GPT-2’s skeleton: stacked decoder-only blocks, each containing self-attention and an FFN, wired with residual connections and normalization. Rather than re-derive the transformer, **we treat GPT-2 as the baseline and walk the diff.**

Five components changed substantially:

| #   | Component           | Pressure                    |
| --- | ------------------- | --------------------------- |
| 1   | Positional encoding | Length generalization       |
| 2   | Normalization       | Training stability          |
| 3   | Attention           | KV-cache memory             |
| 4   | FFN                 | Expressivity / sparse scale |
| 5   | Residuals           | Depth dilution (frontier)   |

Plus a quieter shift in **proportions**: aspect ratio, head dimension, FFN expansion, vocab size.

**Deep dives this episode:** (1), (2), (3). **Mentions:** (4), (5), and the proportions segment.

-----

## 1. The GPT-2 1.5B Baseline

We anchor on **GPT-2 XL (1.5B)** — the season’s capability target and the reference our nanochat speed-runs aim to beat.

|Property       |Value                           |
|---------------|--------------------------------|
|Total params   |~1.5B                           |
|`d_model`      |1600                            |
|`n_layers`     |48                              |
|`n_heads`      |25                              |
|`d_head`       |64                              |
|`d_ff`         |6400 (= 4 × `d_model`)          |
|Vocab          |50,257                          |
|Max context    |1024 tokens                     |
|Position       |Learned absolute                |
|Norm           |LayerNorm, post-norm placement  |
|Attention      |MHA (25 query, 25 K, 25 V heads)|
|Activation     |GeLU                            |
|Tied embeddings|Yes (`wte == lm_head`)          |

Every choice above gets revisited.

> **Recap from Episode 1:** OpenAI’s original 1.5B run in 2019 cost roughly $43K on 32 TPU v3 chips. The nanochat speed-run reproduces this capability for under $100 on an 8×H100 node — an ~600× cost reduction over six years, driven *partly* by hardware but mostly by the architectural and training-recipe upgrades we walk through this episode.

-----

## 2. The Shape of the Network

Before we touch components, modern models reshape the network itself.

### 2.1 Aspect Ratio (`d_model / n_layers`)

|Model                        |Ratio                                   |
|-----------------------------|----------------------------------------|
|**GPT-2 1.5B (our baseline)**|**1600 / 48 ≈ 33**                      |
|GPT-3 175B                   |12288 / 96 = 128                        |
|LLaMA-3-8B                   |4096 / 32 = 128                         |
|Qwen2.5-7B                   |3584 / 28 = 128                         |
|nanochat depth-26            |~64 (single-node speed-run optimization)|

Post-Chinchilla consensus pushed dense models toward **wider and shallower** (ratio ≈ 128). GPT-2 XL’s ratio of 33 is *very* tall-and-narrow by modern standards — almost 4× deeper than the modern optimum at the same width.

The AttnRes paper hints this may shift back toward deeper/narrower if residuals become learned — interesting footnote.

### 2.2 Head Dimension

GPT-2 fixed `d_head = 64`. Modern models commonly use `d_head = 128`. Larger heads → better tensor-core utilization on H100, no measurable quality cost.

### 2.3 FFN Expansion Ratio

- GPT-2: `d_ff = 4 · d_model` (canonical)
- LLaMA-3 with SwiGLU: `d_ff ≈ 2.67 · d_model` per matrix (because SwiGLU has *two* up-projections — gate + value)
- nanochat: keeps **4×** because it uses ReLU² (single up-projection)

### 2.4 Vocabulary

- GPT-2: 50K
- LLaMA-3: 128K
- Qwen3: 150K+

Driven by multilingual coverage and code-token efficiency.

> **Pedagogical note:** nanochat’s `--depth` dial encodes a specific aspect ratio. This is a design choice optimized for a single-node speed-run, not a universal truth. **Architectural choices are workload-dependent.**

-----

## 3. Deep Dive — Positional Encoding: RoPE

### 3.1 The Problem with Learned Absolute Positions

GPT-2’s learned positional embedding table has three issues:

1. **Hard cap on context length** — table size = max position.
2. **No length extrapolation** — positions never seen in training are essentially random vectors.
3. **Encodes absolute position**, but attention usually cares about *relative* position.

### 3.2 The RoPE Idea

Don’t *add* positional information to the embedding — **rotate** the query and key vectors by an angle proportional to position.

For a 2D pair `(q^(2i), q^(2i+1))` at position `m`, define angle `θ_i = base^(-2i/d)` (base typically 10000):

```
[ q'_m^(2i)   ]   [ cos(m·θ_i)  -sin(m·θ_i) ] [ q^(2i)   ]
[ q'_m^(2i+1) ] = [ sin(m·θ_i)   cos(m·θ_i) ] [ q^(2i+1) ]
```

Apply the same rotation to `k`. Group dimensions into pairs; each pair rotates at its own frequency.

### 3.3 Why It Captures Relative Position

The key algebraic property of rotation matrices:

```
R(α)ᵀ · R(β) = R(β − α)
```

So when computing the attention score between query at position `m` and key at position `n`:

```
⟨R_m q, R_n k⟩ = qᵀ R_mᵀ R_n k = qᵀ R_(n−m) k
```

The score depends only on `(n − m)`. **Relative position emerges from absolute rotations.** No learnable parameters, no max-length cap, and frequencies spread geometrically across dimensions:

- **Low-index dims** (small `i`): high frequency → short-period features.
- **High-index dims** (large `i`): low frequency → long-period features.

### 3.4 Length Extrapolation

RoPE doesn’t extrapolate perfectly out of the box, but it’s far friendlier than learned embeddings. Modern extensions:

- **Position Interpolation (PI)**: scale rotation angles to fit longer contexts.
- **NTK-aware scaling**: scale frequencies non-uniformly (preserve high-frequency).
- **YaRN**: combines NTK-by-parts with attention-temperature correction. Used in LLaMA-3, GPT-OSS.

### 3.5 nanochat Implementation

```python
# Precompute (once)
freqs = 1.0 / (10000 ** (torch.arange(0, d_head, 2) / d_head))
positions = torch.arange(seq_len)
angles = torch.outer(positions, freqs)
cos, sin = angles.cos(), angles.sin()

# Apply to Q and K
def apply_rope(x, cos, sin):
    # x: [..., seq, d_head]; rotate adjacent pairs
    return x * cos + rotate_half(x) * sin

q_rot = apply_rope(q, cos, sin)
k_rot = apply_rope(k, cos, sin)
```

> **Takeaway:** RoPE encodes *relative* position via *absolute* rotations, with **zero learnable parameters**, and supports extrapolation through frequency scaling.

-----

## 4. Deep Dive — Normalization: Pre-norm + RMSNorm + QK-Norm

### 4.1 Pre-norm vs. Post-norm

|                     |Formula                         |Behavior                                                                                      |
|---------------------|--------------------------------|----------------------------------------------------------------------------------------------|
|**Post-norm (GPT-2)**|`x = LayerNorm(x + Sublayer(x))`|Residual stream is normalized after each addition. Requires careful warmup; unstable at scale.|
|**Pre-norm (modern)**|`x = x + Sublayer(LayerNorm(x))`|Only the sublayer input is normalized. Trains stably without warmup.                          |

Trade-off: pre-norm lets the residual stream’s magnitude **grow with depth** — this is exactly the dilution problem AttnRes (§7) addresses.

### 4.2 LayerNorm → RMSNorm

**LayerNorm:**

```
LayerNorm(x) = γ · (x − μ) / σ + β
where  μ = mean(x),  σ = std(x)
```

**RMSNorm:**

```
RMSNorm(x) = γ · x / RMS(x)
where  RMS(x) = sqrt(mean(x²) + ε)
```

Two simplifications:

1. **Drop mean-centering.** Empirically, this term contributes very little — RMS rescaling is doing the real work.
2. **nanochat goes further: drop γ too.** Just `x / RMS(x)`.

Why this matters:

- ~10% throughput improvement (one fewer reduction op).
- Removes mean-stability edge cases.
- Translation-equivariant (small theoretical bonus).

### 4.3 QK-Norm: Stopping Attention Entropy Collapse

**The problem:** at large scale, attention logits `q · k` can drift to large magnitudes, saturating softmax (one token gets ~1.0 weight, the rest ≈0). This kills training.

**The fix:** RMSNorm `q` and `k` *after* RoPE, *before* the attention scores:

```python
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
q, k = rms_norm(q), rms_norm(k)            # ← QK-Norm
scores = (q @ k.transpose(-2, -1)) / sqrt(d_head)
```

Bounds `q` and `k` magnitudes per channel, preventing logit blow-up. Used by Gemma, Chameleon, nanochat.

> QK-Norm and **logit softcapping** (`15 · tanh(logits / 15)` in nanochat) are alternative tools for the same problem. Some models use both.

### 4.4 nanochat Code: RMSNorm Everywhere

RMSNorm fires **5+ times per layer**:

1. After token embedding
2. Before attention (pre-norm)
3. On Q after RoPE (QK-Norm)
4. On K after RoPE (QK-Norm)
5. Before MLP (pre-norm)
6. Before `lm_head`

All without learnable parameters: `F.rms_norm(x, (x.size(-1),))`.

> **Takeaway:** Normalization went from “one LayerNorm per sublayer with γ and β” to “RMSNorm everywhere, no parameters, also applied to Q and K.” Smaller, simpler, more stable.

-----

## 5. Deep Dive — Attention: MQA/GQA + SWA

### 5.1 MHA Dimensional Anatomy (Recap)

Before changing anything, get the shapes precise. For a single attention layer with batch `B`, sequence length `T`, model dimension `d_model`, and `H` heads of dimension `d_head` (so `H · d_head = d_model`):

```
Q projection:  W_Q ∈ ℝ^(d_model × H·d_head)     → Q : [B, T, H, d_head]
K projection:  W_K ∈ ℝ^(d_model × H·d_head)     → K : [B, T, H, d_head]
V projection:  W_V ∈ ℝ^(d_model × H·d_head)     → V : [B, T, H, d_head]
O projection:  W_O ∈ ℝ^(H·d_head × d_model)     → final output
```

Scores per head:  `softmax( Q @ Kᵀ / sqrt(d_head) ) @ V`.

Three observations about MHA that motivate everything in this section:

1. **`W_K` and `W_V` are as large as `W_Q`.** Half of the attention parameter budget goes into K and V projections.
2. **K and V get cached at inference**, but Q does not (we only need the *current* query).
3. **Each query head has its own K and V head.** This is what GQA/MQA changes.

### 5.2 The KV-Cache Problem (with Numbers)

During autoregressive decoding, K and V from every previous token are stored in the KV cache:

```
KV cache (bytes) = 2 · T · n_kv_heads · d_head · n_layers · bytes_per_element
                   ↑
                   K and V
```

**Worked example — LLaMA-3-8B, FP16, 32K context, MHA-equivalent:**

```
2 · 32768 · 32 · 128 · 32 · 2  =  17.2 GB per request
```

Vs. the **8 GB** of model weights. **The cache outweighs the model itself** at long contexts. And it scales linearly with both `T` and concurrent requests, so a server with 80 GB of HBM serving 32K-context requests can handle just 4 simultaneous users with full MHA.

Three knobs in that formula are negotiable: `n_kv_heads`, `d_head`, and effective `T`. GQA tackles the first; MLA tackles the second; SWA tackles the third.

### 5.3 GQA: Sharing K and V Across Query Heads

**The mechanism:** partition the `H` query heads into `G` groups. Each group shares one K head and one V head.

```
n_heads     = H              (unchanged)
n_kv_heads  = G              (= H for MHA, = 1 for MQA)
group_size  = H / G          (queries per shared K/V)
```

**Tensor shapes change for K and V only:**

```
Q : [B, T, H, d_head]              ← unchanged
K : [B, T, G, d_head]              ← was [B, T, H, d_head]
V : [B, T, G, d_head]              ← was [B, T, H, d_head]
```

Inside attention, K and V are *broadcast* (or `repeat_interleave`’d) up to `H` heads to match Q before computing scores:

```python
def gqa_attention(q, k, v, group_size):
    # q: [B, T, H, d_head];  k, v: [B, T, G, d_head]
    k = k.repeat_interleave(group_size, dim=2)  # [B, T, H, d_head]
    v = v.repeat_interleave(group_size, dim=2)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=True
    ).transpose(1, 2)
```

> Real implementations (Flash Attention 2/3, vLLM) skip the explicit `repeat_interleave` and pass `group_size` to the kernel directly — the broadcast is virtual.

### 5.4 The MHA → GQA → MQA Spectrum

|Variant  |`n_kv_heads`  |Group size|KV cache vs. MHA|Quality cost       |
|---------|--------------|----------|----------------|-------------------|
|**MHA**  |`H` (e.g., 32)|1         |1×              |—                  |
|**GQA-8**|8             |4         |1/4×            |~negligible        |
|**GQA-4**|4             |8         |1/8×            |small              |
|**MQA**  |1             |`H`       |1/`H`×          |~1–2% on benchmarks|

**Two savings, not one.** GQA reduces both:

1. **KV-cache memory** — proportional to `n_kv_heads`.
2. **Parameter count of `W_K` and `W_V`** — same proportion.

**Worked example — LLaMA-3-8B (`H=32`, `d_head=128`, `d_model=4096`, 32 layers):**

```
W_K size, MHA:   4096 × 4096 = 16.78M params
W_K size, GQA-8: 4096 × 1024 =  4.19M params       (saves 12.6M)

Per layer (W_K + W_V): 25.2M saved
Across 32 layers:      ≈ 805M params saved
```

That’s ~10% of the model. **GQA is a parameter optimization as much as a memory optimization.**

**Real-world settings:**

- **LLaMA-3-8B / 70B:** GQA-8 (4:1 query:KV ratio).
- **Qwen2.5 series:** GQA with varying ratios (typically 4:1 to 7:1).
- **Mistral 7B / Mixtral:** GQA-8.
- **nanochat:** **MQA** — single KV head. Most aggressive choice; chosen because nanochat optimizes for inference simplicity on a single node, and the ~1–2% quality cost is acceptable at the GPT-2 capability target.
- **Production sweet spot:** GQA with 4–8 KV heads.

**Uptraining trick.** A model trained with MHA can be converted to GQA after the fact by **mean-pooling** groups of K and V heads, then continuing training for ~5% of the original token budget. LLaMA-2 → LLaMA-2-GQA was done this way. Useful when you want to preserve a pretrained checkpoint but cut inference cost.

-----

### 5.5 Sliding Window Attention: the Banded Mask

GQA cuts the `n_kv_heads` factor. SWA cuts the **`T` factor** by limiting each query to a local window of the most recent `W` tokens.

**The attention mask becomes banded:**

```
              key position
              0  1  2  3  4  5  6  7
query  0     [1  .  .  .  .  .  .  .]
       1     [1  1  .  .  .  .  .  .]
       2     [1  1  1  .  .  .  .  .]   W = 4
       3     [1  1  1  1  .  .  .  .]   (causal upper triangle)
       4     [.  1  1  1  1  .  .  .]   ← window slides
       5     [.  .  1  1  1  1  .  .]
       6     [.  .  .  1  1  1  1  .]
       7     [.  .  .  .  1  1  1  1]
```

Mask is the intersection of the standard causal mask with a band of width `W`.

**Why this is fast:** Flash Attention 2/3 supports SWA natively with a `window_size` argument. The kernel skips entire `(query_block, key_block)` pairs that fall outside the band, so compute drops from `O(T²)` to `O(T·W)` — actually realized, not just on paper. Naïve PyTorch masking still computes everything and just zeros it out.

### 5.6 SWA + KV Cache: the Rolling Buffer

This is the part SWA is really for.

With SWA, the KV cache for a layer never needs to hold more than `W` tokens — older tokens fall out of the window and can be evicted. The cache becomes a **circular ring buffer** of size `W`:

```
KV cache (SWA layer) = 2 · min(T, W) · n_kv_heads · d_head · bytes
                            ↑
                            capped at W
```

**Worked example — LLaMA-3-8B-equivalent at 128K context, GQA-8, `W=4096`:**

```
Full attention:  2 · 131072 · 8 · 128 · 32 · 2 = 17.2 GB
SWA (W=4096):    2 ·   4096 · 8 · 128 · 32 · 2 =  0.54 GB    (32× reduction)
```

The KV cache becomes **constant in `T`** for SWA layers. This is the breakthrough behind long-context inference for models like Mistral and Gemma.

### 5.7 Effective Receptive Field and Attention Sinks

**Pure SWA loses long-range information** — a token at position `T` cannot directly attend to the token at position 0. Two facts mitigate this:

**Fact 1 — Stacked SWA propagates through residuals.** With `L` layers of window `W`, information can flow up to `L · W` tokens via the residual stream. For nanochat depth-26 with `W=1024`, that’s ~26K-token effective range, even though no single layer attends beyond 1024.

**Fact 2 — Attention sinks.** *StreamingLLM* (Xiao et al. 2024) observed that pure SWA degrades sharply at long contexts because softmax wants somewhere to “dump” attention mass when nothing in the window is relevant. The fix: always keep the **first 4 tokens** in the cache (the “sinks”), even when they fall outside the sliding window. Restores quality nearly to full attention. Used by Mistral, GPT-OSS.

```
              key position
              [SINKS]    [WINDOW]
              0  1  2  3 ... T-W+1 ... T-1   T
query T  →    1  1  1  1   .  .   1  1  1   1     ← attends to sinks + window
```

### 5.8 Hybrid Patterns: Local + Global

Most production models don’t use *pure* SWA — they alternate local and global layers, getting most of the memory savings while preserving direct long-range attention in some layers:

|Model         |Pattern                                              |
|--------------|-----------------------------------------------------|
|**nanochat**  |`SSSL`: 3 short (`W=1024`) + 1 long (`W=2048`), tiled|
|**GPT-OSS**   |Alternate full attention + `W=128` SWA               |
|**Gemma 2**   |Alternate global + `W=4096` SWA                      |
|**Mistral 7B**|Pure SWA (`W=4096`) + sinks                          |
|**Qwen3**     |Mostly global; some variants use hybrid              |

The “global every Nth layer” pattern usually delivers >95% of the long-context quality of full attention at a fraction of the cache cost.

### 5.9 RoPE Compatibility

One quiet bonus: **RoPE composes cleanly with SWA.** Because RoPE encodes relative position via the dot product (§3.3), and SWA only changes *which* dot products are computed (not how positions are encoded), no special handling is needed. Compare to ALiBi, which would require care to interact correctly with windowing.

This is part of why RoPE + SWA became the dominant combination.

### 5.10 MLA (Briefly)

DeepSeek’s **Multi-Head Latent Attention** takes the third axis: instead of reducing the *number* of KV heads or the effective sequence length, it **compresses the head dimension**.

```
Standard:  cache K, V each with shape [T, H, d_head]
MLA:       cache compressed latent c with shape [T, d_latent]
                                                      ↑
                                                      d_latent ≈ 512 < H·d_head
```

Decompression `K = c · W_UK`, `V = c · W_UV` happens during attention. The trick that makes it work: `W_UK` and the query projection can be **absorbed** into a single matrix at inference time, so MLA pays no extra compute despite the apparent extra projection.

- KV cache ≈ 1/10 of MHA — without GQA’s sharing constraint.
- More complex (RoPE interaction requires partitioned-dimension handling — DeepSeek splits `d_head` into a “rope” portion and a “no-rope” portion).
- Used by DeepSeek-V2/V3, Kimi K2. **Not in nanochat.**

**Takeaway:** KV-cache pressure drives every modern attention variant. The three axes are orthogonal and stackable:

| Axis                      | Knob         | Method      |
| ------------------------- | ------------ | ----------- |
| Number of KV heads        | `n_kv_heads` | GQA / MQA   |
| Effective sequence length | `min(T, W)`  | SWA + sinks |
| Head dimension            | `d_head`     | MLA         |
Production models combine the first two. The frontier (DeepSeek, Kimi) uses MLA *plus* SWA-style local layers.

-----

## 6. Mention — FFN: SwiGLU and MoE

> *Survey only this episode. Full deep-dives reserved for a later episode.*

### 6.1 SwiGLU

GPT-2 uses GeLU in a 2-matrix FFN:

```
FFN(x) = GeLU(x · W_up) · W_down
```

**SwiGLU** adds a gating path:

```
FFN(x) = ( Swish(x · W_gate) ⊙ (x · W_up) ) · W_down
```

Three matrices instead of two. To preserve parameter count, each is sized at `(8/3) · d_model` instead of `4 · d_model`.

**Why it works:** gating provides multiplicative interactions GeLU doesn’t, improving expressivity. Now standard in production.

### 6.2 Mixture of Experts (MoE)

-----

#### The Motivation

Dense scaling is parameter-bound *and* FLOP-bound: doubling parameters means doubling compute per token. MoE breaks this coupling.

**Total parameters scale with `N` experts; active parameters per token scale with `k` (typically 1 or 2).** You get the representational capacity of a huge model at the inference cost of a much smaller one.

-----

#### The Mechanism

Replace the dense FFN with `N` parallel expert FFNs and a router:

```
        ┌─→ Expert_1(x) ─┐
        │                 │
x ──────┼─→ Expert_2(x) ─┼──→  weighted sum ──→ y
        │      ...        │
        └─→ Expert_N(x) ─┘
              ↑
         router(x) picks top-k
```

Forward pass:

```
scores  = softmax( x · W_router )                # [B, T, N]
top_k   = top_k_indices(scores, k=2)             # which experts
weights = scores[top_k]                          # gate values
y       = Σ over selected experts: weights_i · Expert_i(x)
```

`W_router ∈ ℝ^(d_model × N)` is a simple linear layer — that’s the entire routing brain. The expensive components (the experts themselves) are standard SwiGLU FFNs.

**Code skeleton:**

```python
def moe_forward(x, experts, router, k=2):
    # x: [B, T, d_model]
    logits = router(x)                           # [B, T, N]
    scores = F.softmax(logits, dim=-1)
    top_w, top_idx = scores.topk(k, dim=-1)      # [B, T, k]
    top_w = top_w / top_w.sum(-1, keepdim=True)  # renormalize

    y = torch.zeros_like(x)
    for i in range(k):
        idx = top_idx[..., i]                    # [B, T]
        for e in range(len(experts)):
            mask = (idx == e)
            if mask.any():
                y[mask] += top_w[..., i:i+1][mask] * experts[e](x[mask])
    return y
```

Real implementations don’t loop — they use `scatter`/`gather` ops and fused kernels (Megablocks, Tutel) — but the logic is exactly this.

-----

#### Load Balancing: The Central Headache

Routers tend to **collapse**: a few experts get all the tokens, the rest get nothing and stay at initialization. Three standard fixes:

### 1. Auxiliary loss (Switch Transformer, Mixtral)

Add a term that penalizes imbalance:

```
L_aux  =  α · N · Σ over experts: f_i · P_i
```

where `f_i` = fraction of tokens routed to expert `i`, `P_i` = average router probability for expert `i`. Pushes both routing decisions and probabilities toward uniformity.

**Downside:** aux loss fights the main loss. The router is forced to spread tokens for balance, even when a token genuinely belongs to one expert.

### 2. Auxiliary-loss-free (DeepSeek-V3)

Add a learned bias `b_i` per expert to the routing scores. Bias goes **up** for under-utilized experts, **down** for over-utilized ones — adjusted *between* training steps based on observed load, not via gradients. The main loss never gets distorted. Now used by Kimi K2 and several other 2025+ models.

### 3. Expert capacity

A hard cap on how many tokens any single expert can process per batch. Overflow tokens are dropped (skip the FFN, pass through residual only) or rerouted. Capacity factor typically `1.0–1.25 · (T·k/N)`.

-----

##### Architectural Choices in the Design Space

|Choice        |Range                         |Tradeoff                                         |
|--------------|------------------------------|-------------------------------------------------|
|`N` (experts) |8 → 256+                      |More experts = more capacity but harder routing  |
|`k` (active)  |1 or 2 (usually 2)            |`k=2` gives smoother gradients; `k=1` is cheapest|
|Expert size   |Full SwiGLU vs. fine-grained  |Fine-grained = more, smaller experts             |
|Shared experts|0, 1, or 2 always-on          |Captures common knowledge                        |
|Granularity   |All layers MoE vs. some layers|Some models keep early layers dense              |

##### Fine-Grained Experts (DeepSeek MoE)

Instead of `N=8` SwiGLU experts, use `N=64` experts each at 1/8 the size. Same total params, same active params, but more *combinations*: top-2 routing now picks from `C(64,2) = 2016` combinations instead of `C(8,2) = 28`. Empirically a clear win.

##### Shared Experts (DeepSeek-V2/V3)

Reserve 1–2 experts that *always* run for every token, in addition to the top-k routed experts. The shared expert(s) capture common patterns; the routed experts specialize. Reduces the burden on routing to handle every token type.

```
y = Shared(x) + Σ over routed top-k: w_i · Expert_i(x)
```

-----

##### The Real Models

|Model                       |`N` total     |`k` active|Total params|Active params|Notes                                |
|----------------------------|--------------|----------|------------|-------------|-------------------------------------|
|Switch Transformer (2021)   |up to 2048    |1         |1.6T        |~7B/token    |The OG                               |
|**Mixtral 8×7B**            |8             |2         |47B         |13B          |First open competitive MoE           |
|**DeepSeek-V3**             |256 + 1 shared|8 + 1     |671B        |37B          |Fine-grained + shared + aux-loss-free|
|**Qwen3-MoE** (e.g. 30B-A3B)|128           |8         |30B         |3B           |High sparsity ratio                  |
|**Kimi K2**                 |~384          |~8        |1T+         |~32B         |Aux-loss-free; fine-grained          |

**The trend:** more experts, finer granularity, lower active ratios. Mixtral activates 28% of its params per token; DeepSeek-V3 activates 5.5%; Kimi K2 around 3%. The frontier is moving toward extremely sparse activation.

-----

#### Systems Reality

This is where MoE gets nasty for people who think of it as just “swap one block in the architecture.” The router decision means **different tokens in the same batch route to different experts**, often on different GPUs. Three consequences:

1. **All-to-all communication.** Tokens need to be physically moved to the GPU holding their target expert, then results moved back. This is *the* dominant cost in MoE training and inference.
2. **Expert parallelism (EP).** A new parallelism axis on top of TP/PP/DP. DeepSeek-V3 trained with EP=64 across 64 GPUs.
3. **Inference batching is harder.** Different requests’ tokens hit different experts, breaking the clean batching of dense models. Frameworks like vLLM and SGLang have specialized MoE support.

-----
### 6.3 What nanochat Does

**Neither.** ReLU² activation in a 2-matrix dense FFN with `d_ff = 4 · d_model`:

```python
def forward(self, x):
    x = self.c_fc(x)
    x = F.relu(x).square()       # ReLU²
    x = self.c_proj(x)
    return x
```

Deliberately simple. Worth flagging when viewers compare nanochat to LLaMA or Mixtral source — the divergence is principled, not a mistake.

-----

## 7. Mention — Residual Connections: AttnRes

> *Released March 2026 by the Moonshot AI / Kimi Team. Worth knowing — not yet standard.* arXiv:2603.15031.

### 7.1 The Problem

Standard residuals add layer outputs with fixed unit weight:

```
h_l = h_(l−1) + f_l(h_(l−1))
```

Combined with pre-norm, this causes hidden-state magnitudes to grow `O(L)` with depth, **diluting individual layer contributions** — the “PreNorm dilution” pattern.

### 7.2 The Fix: Softmax Attention Over Depth

AttnRes replaces uniform addition with softmax attention over previous layer outputs. Each layer learns a pseudo-query `w_l ∈ ℝ^d` (initialized to **zero**, ensuring uniform initial weights and stable warmup) that selects which earlier layers to draw from.

The same time/depth duality that motivated the 2017 transformer:

- 2017: attention replaced *recurrence across the sequence axis*.
- 2026: AttnRes replaces *additive recurrence across the depth axis*.

### 7.3 Block AttnRes

Full AttnRes is `O(Ld)` memory. **Block AttnRes** partitions layers into ~8 blocks; cross-block attention applies only at block boundaries. Drops memory to `O(Nd)`, with under 4% training overhead and under 2% inference overhead.

### 7.4 Headline Results

- Block AttnRes matches a baseline trained with **1.25× more compute**.
- Tested on Kimi Linear (48B total / 3B active), pre-trained on 1.4T tokens.
- **+7.5 on GPQA-Diamond, +3.1 HumanEval, +3.6 MATH.**

### 7.5 The Pattern

> Every fixed/uniform aggregation in the transformer eventually gets replaced by **learned, input-dependent weighting**.
> 
> - Sequence dimension → attention (2017).
> - Expert routing → MoE (2017–2024).
> - Depth dimension → AttnRes (2026?).

**Repo:** [`MoonshotAI/Attention-Residuals`](https://github.com/MoonshotAI/Attention-Residuals)

-----

## Synthesis

|Component|GPT-2 1.5B (2019)  |Modern (2024–2026)         |Pressure                   |
|---------|-------------------|---------------------------|---------------------------|
|Position |Learned absolute   |RoPE (+ YaRN scaling)      |Length generalization      |
|Norm     |Post-norm LayerNorm|Pre-norm RMSNorm + QK-Norm |Training stability         |
|Attention|MHA                |GQA/MQA + SWA (or MLA)     |KV-cache memory            |
|FFN      |GeLU dense (4×)    |SwiGLU dense (~2.67×) → MoE|Expressivity / sparse scale|
|Residuals|Uniform sum        |AttnRes (frontier)         |Depth dilution             |

Plus reshaped proportions: aspect ratio ≈ 128, `d_head` = 128, vocab 100K+.

### Where nanochat Sits

|            |nanochat           |Production (LLaMA-3 / Qwen / DeepSeek)|
|------------|-------------------|--------------------------------------|
|Position    |RoPE ✓             |RoPE ✓                                |
|Norm        |RMSNorm + QK-Norm ✓|RMSNorm + QK-Norm (varies)            |
|Attention   |MQA + SWA ✓        |GQA + (often) SWA / MLA               |
|FFN         |**ReLU² dense, 4×**|SwiGLU dense or MoE                   |
|Residuals   |Standard           |Standard (AttnRes is brand-new)       |
|Aspect ratio|~64                |~128                                  |
|Vocab       |65K                |128K+                                 |

**nanochat’s design intent:** maximum simplicity per dollar on a single 8×H100 node. Each divergence from production is a deliberate choice, not an oversight.

-----

## References

### Core Papers

- Vaswani et al. 2017 — *Attention Is All You Need*
- Su et al. 2021 — *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- Zhang & Sennrich 2019 — *Root Mean Square Layer Normalization*
- Henry et al. 2020 — *Query-Key Normalization for Transformers*
- Shazeer 2019 — *Fast Transformer Decoding* (MQA)
- Ainslie et al. 2023 — *GQA: Training Generalized Multi-Query Transformer Models*
- Beltagy et al. 2020 — *Longformer* (SWA precursor)
- Shazeer 2020 — *GLU Variants Improve Transformer* (SwiGLU)
- Shazeer et al. 2017 — *Outrageously Large Neural Networks* (MoE)
- DeepSeek-AI 2024 — *DeepSeek-V2* (MLA)
- Kimi Team 2026 — *Attention Residuals* (arXiv:2603.15031)

### Code

- [`karpathy/nanochat`](https://github.com/karpathy/nanochat) — especially `nanochat/gpt.py`
- [`MoonshotAI/Attention-Residuals`](https://github.com/MoonshotAI/Attention-Residuals)

### Further Reading

- Hoffmann et al. 2022 — *Training Compute-Optimal Large Language Models* (Chinchilla)
- Peng et al. 2023 — *YaRN: Efficient Context Window Extension*
- nanochat Discussions #481 — *Beating GPT-2 for <<$100: the nanochat journey*

-----

## Part B Code-Rewrite Modules (Preview)

Each deep-dive in this episode maps to a hands-on rewrite module:

1. **Module 1** — Implement RoPE from scratch, verify relative-position property numerically.
2. **Module 2** — Implement RMSNorm and QK-Norm; ablate each against LayerNorm and no-QK-Norm baselines.
3. **Module 3** — Convert MHA → GQA → MQA; measure KV-cache memory on a fixed sequence; add SWA with the `SSSL` pattern.
4. **Module 4** *(preview, future episode)* — Swap ReLU² → SwiGLU; benchmark.
5. **Module 5** *(preview, future episode)* — Add a tiny MoE block with top-2 routing.
6. **Module 6** *(stretch, future episode)* — Drop in Block AttnRes (8 blocks), measure loss curve vs. baseline.