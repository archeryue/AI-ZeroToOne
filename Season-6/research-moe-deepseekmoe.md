# Research Notes: Mixture-of-Experts and DeepSeekMoE

A consolidated reference on MoE LLM architectures, with a focus on the DeepSeekMoE family (original paper, V2, V3).

---

## 1. MoE in One Page

### 1.1 What it is
A Mixture-of-Experts layer replaces a Transformer's dense feed-forward network (FFN) with `N` parallel expert FFNs and a small router. Each token is routed to only its top-`K` experts, so total parameters grow `~N×` while FLOPs per token stay close to a dense model's. This is *conditional computation*.

Canonical sparse MoE layer:

```
y = Σ_i  G(x)_i · E_i(x)
G(x)   = Softmax( TopK( x · W_g , k ) )
```

Only `K` of the `N` `E_i` are evaluated; the rest are skipped.

### 1.2 Lineage
| Year | Model | Notable change |
|------|-------|----------------|
| 1991 | Adaptive Mixture of Local Experts | The original idea |
| 2017 | Sparsely-Gated MoE (Shazeer) | 137B-param LSTM, noisy top-K |
| 2020 | GShard | Top-2 routing on 600B+ Transformer |
| 2021 | Switch Transformer | Top-1 routing, 4× pretraining speedup |
| 2021 | GLaM | GPT-3 quality at 1/3 energy |
| 2022 | ST-MoE | Router z-loss for stability |
| 2023 | Mixtral 8×7B | 47B total / ~13B active |
| 2024 | DeepSeekMoE | Fine-grained + shared experts |
| 2024 | DeepSeek-V2 | + Multi-head Latent Attention (MLA) |
| 2024 | DeepSeek-V3 | + Auxiliary-loss-free balancing, MTP |

### 1.3 Standard MoE pain points
- **Routing collapse / load imbalance** — popular experts attract all tokens; usually patched with an auxiliary load-balance loss that itself dents quality.
- **Knowledge hybridity** — with 8–16 coarse experts, each becomes a generalist; specialization is shallow.
- **Knowledge redundancy** — each expert independently relearns common patterns.
- **Capacity & token dropping** — expert-parallel implementations bound per-expert tokens by `(tokens_per_batch / N) × capacity_factor`; overflow is dropped.
- **Fine-tuning** — sparse models overfit small datasets faster than dense ones.
- **Memory** — even though FLOPs are sparse, the *full* parameter set must reside in VRAM.

### 1.4 Routing & load-balance toolbox
- **Top-K selection** of experts (token-choice), or **expert-choice** (experts pick tokens for fixed capacity).
- **Noisy top-K gating** — adds learnable Gaussian noise to logits.
- **Auxiliary balance loss** — penalizes correlation between per-expert frequency and per-expert importance.
- **Router z-loss** — penalizes large logits to keep softmax numerically stable.
- **Capacity factor** — typically `1.0–1.25` in Switch; `<1` saves comm but drops tokens.
- **Block-sparse kernels** (MegaBlocks) — efficiently handle ragged per-expert token counts.

---

## 2. DeepSeekMoE (Jan 2024)

Paper: *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* (arXiv 2401.06066).

The thesis: existing GShard-style MoE designs leave a lot of *expert specialization* on the table because each expert is large and routing is coarse. Two stacking tweaks fix this without changing parameter or FLOP budgets.

### 2.1 Innovation 1 — Fine-grained expert segmentation
Take `N` experts of FFN hidden size `H` and split each into `m` smaller experts of hidden size `H/m`, giving `mN` experts. Activate `mK` instead of `K`.

- **Same total parameters, same FLOPs.**
- The number of distinct top-`mK` *combinations* explodes. E.g. `C(16, 2) = 120` → `C(64, 8) ≈ 4.4 × 10⁹`.
- Each expert now covers a narrower slice of knowledge; the router has finer composition to play with.

### 2.2 Innovation 2 — Shared expert isolation
Carve out `Ks` "shared" experts that **every** token always passes through, in addition to its `mK` top-routed experts.

- Shared experts absorb common patterns (e.g. generic English syntax) so routed experts don't all redundantly relearn them.
- Each routed expert can specialize harder on distinctive features.
- Activated experts per token = `Ks + mK`.

### 2.3 Load balancing in the original paper
Two auxiliary losses:
1. **Expert-level balance loss** — standard frequency × importance penalty.
2. **Device-level balance loss** — groups experts onto devices and balances at the *device* granularity, since fine-grained per-expert imbalance often doesn't hurt utilization once experts are co-located.

### 2.4 Results (from the paper)
| Model | Comparable to | At what cost |
|-------|--------------|--------------|
| DeepSeekMoE 2B | GShard 2.9B | ~1.5× fewer expert params |
| DeepSeekMoE 16B | LLaMA2 7B / DeepSeek 7B | ~40% of FLOPs |
| DeepSeekMoE 145B | DeepSeek 67B | ~28.5% of FLOPs |

---

## 3. DeepSeek-V2 (May 2024)

Builds directly on DeepSeekMoE, adding:

### Multi-head Latent Attention (MLA)
A low-rank joint compression of K and V into a small latent vector of dim `d_c ≪ d_h · n_h`, with a separate RoPE-carrying key component.

- Slashes KV-cache memory at inference (the bottleneck for long-context serving).
- Quality stays comparable to vanilla MHA.

V2 keeps the segmentation + shared experts of the original DeepSeekMoE.

---

## 4. DeepSeek-V3 (Dec 2024)

Paper: *DeepSeek-V3 Technical Report* (arXiv 2412.19437). The current SOTA snapshot of the line.

### 4.1 Scale
- **671B total parameters**
- **37B activated per token**
- **256 routed experts + 8 shared experts**
- **Top-8 routed experts per token**

### 4.2 Auxiliary-loss-free load balancing
The headline algorithmic change. Instead of adding a balance loss to the optimization objective:

- Maintain a per-expert **bias** `b_i` that is added to affinity scores **only for top-K selection** (gating *weights* still come from the un-biased affinities).
- After each step: if expert `i` was overloaded, `b_i ← b_i − γ`; if underloaded, `b_i ← b_i + γ`.
- `γ` is a small "bias update speed" hyperparameter.

Result: balance is enforced as a **routing-time mechanism**, not via a gradient that fights the language-modeling loss. The team reports no token dropping across the entire training run.

A tiny **sequence-wise balance loss** with very small α is kept as a guardrail against pathological within-sequence imbalance.

### 4.3 Multi-Token Prediction (MTP)
`D` additional prediction modules predict the next `D` future tokens, preserving the causal chain at each prediction depth. This:
- Densifies training signal.
- Enables speculative decoding at inference for free.

---

## 5. Why DeepSeekMoE Matters

DeepSeekMoE is the cleanest demonstration that two structural choices — finer experts plus an always-on shared trunk — recover most of the "specialization gap" critics raised against MoE, without growing parameter or FLOP budgets. V3's auxiliary-loss-free routing is rapidly being adopted because it's the first method to decouple balance enforcement from the gradient landscape entirely.

### Practical implications for an engineer evaluating MoE
- **Memory still dominates.** V3's 671B needs the full weight set in VRAM even though you only spend 37B FLOPs/token. Plan for expert parallelism + high-bandwidth interconnect.
- **Fine-grained segmentation increases all-to-all comm.** It only pays off with good expert-parallel kernels (MegaBlocks-style block-sparse).
- **For fine-tuning** small task data: freeze non-expert weights, consider turning off the aux loss for single-task work, instruction-tune *before* downstream fine-tuning.
- **At inference**, MoE wins on throughput (per-token FLOPs are small) but loses on latency-sensitive single-stream workloads (every token still touches the router and triggers all-to-all).

---

## 6. References

- DeepSeekMoE — https://arxiv.org/abs/2401.06066
- DeepSeekMoE code — https://github.com/deepseek-ai/DeepSeek-MoE
- DeepSeek-V2 — https://arxiv.org/pdf/2405.04434
- DeepSeek-V3 Technical Report — https://arxiv.org/html/2412.19437v2
- Auxiliary-Loss-Free Load Balancing for MoE — https://arxiv.org/html/2408.15664v1
- Hugging Face: Mixture of Experts Explained — https://huggingface.co/blog/moe
- Survey: MoE in LLMs — https://arxiv.org/abs/2507.11181
- A Survey on Mixture of Experts — https://arxiv.org/html/2407.06204v2
- Comprehensive Survey of MoE: Algorithms, Theory, Applications — https://arxiv.org/html/2503.07137v1
