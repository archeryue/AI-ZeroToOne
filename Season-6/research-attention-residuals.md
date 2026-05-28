# Research Notes: Attention Residuals (AttnRes)

A reference on Kimi Team's *Attention Residuals* — a replacement for fixed-weight residual connections in deep Transformers. Paper: arXiv 2603.15031.

---

## 1. TL;DR

Standard Transformer residual connections sum every layer's output with **fixed unit weights** into a single residual stream. As depth grows, this causes:

- **Hidden-state magnitude blow-up** — magnitude grows roughly `O(L)` with depth.
- **PreNorm dilution** — each layer's contribution becomes a vanishing fraction of the running sum (~1% by layer 100).
- **Loss of selective reuse** — the network can't preferentially pull from a specific earlier layer.

**Attention Residuals (AttnRes)** replace the fixed sum with **softmax attention over previous layer outputs**, letting each layer choose which earlier representations to read from with input-dependent weights. The framing: residual connections over *depth* have the same bottleneck RNNs had over *sequence length* — fixed accumulation. The fix is the same fix: attention.

---

## 2. The Problem in Plain Math

A standard pre-norm Transformer block reads from and writes to a residual stream `h_l`:

```
h_l = h_{l-1} + Block_l( LN( h_{l-1} ) )
```

Unrolling:

```
h_L = h_0  +  Σ_{l=1..L}  Block_l( LN( h_{l-1} ) )
```

Two consequences:

1. **Magnitude grows with depth** — every term adds a roughly unit-norm vector. PreNorm normalizes the *input* to each block, but the *stream itself* keeps growing.
2. **Each layer's contribution is fixed at weight 1** — no mechanism to amplify or suppress a specific layer's output downstream.

So for very deep models, training is hard (gradient pathologies) and the network can't selectively retrieve a useful intermediate representation.

---

## 3. The AttnRes Mechanism

Each layer carries a small **per-layer query/key vector**. When forming the input to layer `l`, instead of `h_{l-1} = h_0 + Σ ...`, the network performs softmax attention along the depth axis:

```
input_l  =  Σ_{j<l}  α_{l,j} · output_j
α_{l,·}  =  Softmax( q_l · K_{<l} )
```

Where:
- `output_j` is the output of block `j`.
- `q_l` is layer `l`'s "depth query" (one `d`-dim vector per layer).
- `K_{<l}` are the depth keys of all preceding layers.
- `α_{l,j}` is a learned, *input-dependent* weight on layer `j`'s contribution.

Cost is small: only one extra `d`-dim vector per layer, and the depth-axis softmax is over `L` items (tiny vs. sequence length).

### Why this fixes the problems
- **Bounded magnitude** — softmax weights sum to 1, so the output is a *convex combination* of past layer outputs, not an unbounded sum.
- **Selective reuse** — a layer can put nearly all its mass on layer 7's output if that's what it needs.
- **Stable gradients** — by replacing addition with weighted averaging, gradient norms across depth are reported 2–3× more stable.

---

## 4. Block AttnRes — The Practical Variant

Naive AttnRes attends over all `L` previous layers per layer; for `L` in the hundreds this is non-trivial. **Block AttnRes** partitions layers into contiguous blocks and attends over **block-level representations** instead of every individual layer.

- Reduces memory footprint substantially.
- Reported **8–10× less compute** than the dense version, with most of the quality gains preserved.
- Adds **cache-based pipeline communication** and a **two-phase computation strategy** so it can drop in to existing training stacks.

This is the variant used in production Kimi models.

---

## 5. Reported Results

- **Scaling-law experiments**: consistent improvements across model sizes; gain widens with depth.
- **Integrated into Kimi Linear** (48B total / 3B activated MoE), trained on **1.4T tokens**.
- **+1.8% to +5.9%** absolute performance gains across evaluation tasks (the upper end at larger scale).
- **2–3× more stable gradients** across depth.
- **More uniform output magnitudes** layer-by-layer (the PreNorm-dilution curve flattens).

---

## 6. Why It Matters

AttnRes is in the same lineage as a small set of recent rethinks of "what the residual stream is for":

- **Anthropic's *A Mathematical Framework for Transformer Circuits*** (2021) — formalized the residual stream as a *communication channel* every block reads from and writes to.
- **DeepCrossAttention** (arXiv 2502.06785) — an earlier attempt to put attention over residual outputs.
- **mHC** (DeepSeek, arXiv 2512.24880) — a sister approach that constrains *multi-stream* residual mixing matrices to the Birkhoff polytope. (See companion doc.)
- **Residual Stream Duality in Modern Transformer Architectures** (2026) — frames depth-wise residual attention as equivalent to a sliding window along the sequence axis.

The common observation: as Transformers get deeper, the *uniform-weight* residual sum becomes the analogue of an RNN's fixed accumulator — fine for short chains, lossy at length. The fix in each case is letting depth-mixing become *learned and input-dependent*.

---

## 7. Practical Engineering Notes

- **Drop-in scope**: AttnRes touches only the residual pathway, not attention or FFN internals — compatible with most existing Transformer codebases.
- **Compute overhead**: dense version is non-trivial at large `L`; Block AttnRes is the version you actually deploy.
- **Memory**: needs to keep prior layer outputs (or block summaries) accessible during forward; with checkpointing this is manageable.
- **Norm interaction**: pairs naturally with PreNorm; PostNorm interactions less explored in the paper.
- **Inference**: the depth-axis attention is per-layer, not per-token in the autoregressive sense, so it doesn't interact badly with KV-caching of the *sequence-axis* attention.

---

## 8. References

- Attention Residuals (arXiv) — https://arxiv.org/abs/2603.15031
- Attention Residuals technical report (PDF) — https://arxiv.org/pdf/2603.15031
- Kimi MoonshotAI Attention-Residuals code — https://github.com/MoonshotAI/Attention-Residuals
- A Mathematical Framework for Transformer Circuits — https://transformer-circuits.pub/2021/framework/index.html
- Exploring the Residual Stream of Transformers — https://arxiv.org/html/2312.12141v1
- DeepCrossAttention — https://arxiv.org/abs/2502.06785
- Residual Stream Duality in Modern Transformer Architectures — https://huggingface.co/papers/2603.16039
- Ziming Liu — *When does Kimi's Attention Residuals work?* — https://kindxiaoming.github.io/blog/2026/attention-residual/
- DataCamp — *Attention Residuals Explained: Rethinking Transformer Depth* — https://www.datacamp.com/blog/attention-residuals-explained
- Renee Jia — *Attention Residuals: A Comprehensive Understanding* — https://renee-jia.github.io/paper%20readings/transformer%20architecture/attention-residuals-comprehensive-understanding/
