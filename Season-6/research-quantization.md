# Research Notes: LLM Weight & Activation Quantization

A reference on the quantization techniques used in modern LLM inference, from naive RTN to llama.cpp's sub-2-bit codebook schemes. Designed to accompany the Quantization deep dive in S06E06.

---

## 1. TL;DR — Why Quantization Exists at All

Decode is HBM-bandwidth-bound (see S06E06 §1.2). Decode latency scales linearly with bytes per weight pulled from HBM. **Quantization is lossy compression of weight tensors that trades model accuracy for HBM bandwidth.** Every technique below is a different way to make the compression less lossy at a fixed bit budget.

Two consequences fall out immediately:

1. The **interesting axis is bit-width**, not algorithm. RTN at 8 bits and codebook quantization at 1.5 bits are answering different questions.
2. The **bottleneck is HBM**, not compute. So in the bandwidth-bound regime, quantizing weights alone (`W4A16`) buys most of the win without touching activations.

---

## 2. RTN: The Baseline Encoding

Round-to-nearest linear quantization. A single scale per tensor maps the dynamic range into the integer range:

```
scale  =  max(|w|) / 127                    (for INT8 symmetric)
w_q    =  round(w / scale)                  (stored, INT8)
w_rec  =  w_q · scale                       (used in matmul)
```

The round operation is the loss source. The scale is the resolution knob — too large wastes precision on small values, too small clips outliers.

**The fundamental problem with per-tensor scale:** one outlier value drags up the scale, demolishing precision for everyone else. Concrete: a weight tensor with values in `[-1, 1]` plus one `5.0` outlier gets `scale = 5.0/127 ≈ 0.039`. Now a typical weight of `0.1` quantizes to `round(0.1/0.039) = 3` and reconstructs as `0.117` — 17% error on a tame value because of an outlier.

Granularity is the answer to this.

---

## 3. The Granularity Ladder

|                 | Scale shape (for `W ∈ [d_in × d_out]`) | Adapts to | Cost |
|-----------------|------------------------------------------|-----------|------|
| per-tensor      | `scalar`                                 | global    | trivial |
| per-channel     | `[d_out]` — one scale per output column | column distribution | tiny |
| **group-wise**  | `[d_in/G, d_out]` — one scale per group of G along input axis | local distribution | ~3% overhead |

### 3.1 Per-channel: scale factors out of the matmul

If each output column `j` has its own `s_j`, the matmul rewrites cleanly:

```
Y[j] = Σ_i X[i] · W_q[i,j] · s_j
     = s_j · (Σ_i X[i] · W_q[i,j])
       ↑
       s_j is constant over i, lifts out of Σ
```

Compute the full INT matmul, then rescale each output column. The matmul kernel is unchanged.

### 3.2 Group-wise: scale folds into matmul tile boundaries

Group-wise quantizes along the **reduction (input) axis**, in fixed-size groups of `G` (typically 128). The scale now depends on both group `g` and output column `j`:

```
Y[j] = Σ_i X[i] · W_q[i,j] · s_{g(i), j}
     = Σ_g s_{g,j} · ( Σ_{i ∈ group g} X[i] · W_q[i,j] )
                       ─────────────┬────────────────
                                    │
                          G-element INT dot product
```

The scale changes at group boundaries but is constant inside each group. Matmul kernels are already tiled along the reduction axis — picking the tile size to match the group size means the scale rescale fits naturally at each tile boundary:

```
for each output position j:
    acc = 0                                       # FP32 accumulator
    for each group g = 0 .. d_in/G - 1:
        partial = Σ_{i in group g} X[i] · W_q[i, j]   # INT dot product
        acc    += s_{g, j} · partial                  # rescale + accumulate
    Y[j] = acc
```

The kernel writers' names for this pattern: **Marlin** (vLLM), **ExLlamaV2 q4**, **TensorRT-LLM weight_only_quant**, **AWQ GEMM**. All have the same inner loop.

### 3.3 Why group-wise wins

Outliers in `W` tend to concentrate in a few input channels (specific reduction-axis positions). Group-wise isolates each outlier's "contamination" to its own group of 128 instead of polluting the entire output column. Empirically: ~3% metadata overhead, 2–5× PPL improvement over per-channel.

Production weight-only quantization (`W4A16`) is essentially universally **group-wise along input, per-channel along output**.

---

## 4. The Weight / Activation Asymmetry

So far we've been talking about weight quantization. Activations are a separate problem with a much harder structure.

### 4.1 What activations look like

LLM activations have **outlier channels**: specific input dimensions that are systematically 50–100× larger than typical channels. Discovered and named by Dettmers' LLM.int8() paper (2022).

The crucial empirical observation: **outlier channels are an architectural property, not a data property.**

- Same model, 1000 different prompts → same 6–8 channels are large.
- The outlier channels are stable across English, Chinese, code, math.
- These are essentially "global bias" channels the model learned.

This stability is what makes static (offline) handling of outliers possible.

### 4.2 Two strategies for the outlier problem

| Strategy | When outliers are handled | Method | Trade |
|----------|---------------------------|--------|-------|
| **Dynamic** (LLM.int8) | At runtime, per batch | Detect outlier channels, route through FP16 | Robust, but breaks pure INT8 GEMM → low throughput |
| **Static** (SmoothQuant) | Offline, once | Migrate outlier magnitude into weights, fold into γ | Fast, but relies on outlier stability |

Industry has converged on SmoothQuant-style static handling because LLM outlier channels are remarkably stable.

---

## 5. SmoothQuant: Offline Migration in Detail

### 5.1 The Mathematical Identity

For `Y = X · W`:

```
Y = X · W
  = (X · diag(s)⁻¹) · (diag(s) · W)
  =       X'        ·       W'
       (smoothed)        (slightly amplified)
```

`s ∈ ℝ^{d_in}` is the **per-input-channel migration vector**. Set `s_i` large when channel `i` of `X` has outliers — that channel of `X'` becomes tame, the corresponding row of `W'` gets larger (but `W` has headroom, since it's tame to start with).

Migration ratio (default α = 0.5):

```
s_i = max(|X[:, i]|)^α  /  max(|W[i, :]|)^(1-α)
```

The α knob balances how much pain to push onto W vs X.

### 5.2 Offline Stage Pseudocode

```python
def smoothquant_offline(model, calibration_data, alpha=0.5):
    # ─── Step 1: collect per-channel activation maxes via calibration forward ───
    activation_max = {}
    for batch in calibration_data:
        for layer_name, X in forward_with_hooks(model, batch):
            # X: [B, T, d_in]
            activation_max[layer_name] = max(
                activation_max.get(layer_name, 0),
                X.abs().amax(dim=(0, 1))      # [d_in] per-channel max
            )

    # ─── Step 2: compute smoothing vector s for each linear layer ───
    for layer in model.linear_layers():
        X_max = activation_max[layer.name]           # [d_in]
        W_max = layer.W.abs().amax(dim=1)            # [d_in] per-row max of W
        s     = (X_max ** alpha) / (W_max ** (1 - alpha))
        s     = s.clamp(min=1e-5)

        # ─── Step 3a: fold 1/s into the preceding norm's gamma ───
        # Critical trick: no explicit X/s op at runtime — the preceding
        # RMSNorm/LayerNorm already outputs smoothed activations.
        prev_norm        = find_previous_norm(layer)
        prev_norm.gamma /= s

        # ─── Step 3b: fold s into the current layer's weights ───
        layer.W *= s.unsqueeze(1)                    # W' = diag(s) · W

        # ─── Step 3c: quantize W' to INT8 per-channel ───
        layer.W_int8, layer.W_scale = quantize_per_channel_int8(layer.W)
        del layer.W                                  # drop FP16 master

    return model
```

**Key observation:** `s` is a calibration-time variable. After offline transformation it does not exist in the model — its effect lives in two constant tensors (`prev_norm.gamma` and `layer.W_int8`). The runtime never sees `s`.

### 5.3 Runtime Stage Pseudocode

```python
def smoothquant_linear_forward(X_residual, layer):
    # X_residual: from residual stream, FP16

    # ─── Step 1: RMSNorm (gamma already absorbed 1/s offline) ───
    X_smoothed = rms_norm(X_residual, layer.prev_norm.gamma)
    # X_smoothed IS X/s — no explicit op was added

    # ─── Step 2: dynamic per-token INT8 quantization ───
    X_scale = X_smoothed.abs().amax(dim=-1, keepdim=True) / 127.0  # [B, T, 1]
    X_int8  = (X_smoothed / X_scale).round().to(int8)

    # ─── Step 3: INT8 matmul (native Hopper / Ampere tensor cores) ───
    Y_int32 = int8_matmul(X_int8, layer.W_int8)      # [B, T, d_out]

    # ─── Step 4: dequantize with X_scale × W_scale ───
    Y_fp16  = Y_int32 * (X_scale * layer.W_scale)

    return Y_fp16
```

Per-step runtime cost vs. FP16 baseline:

| Step | Cost vs FP16 |
|------|--------------|
| RMSNorm | identical (just different γ values) |
| Quantize X | extra amax + div + round (~1% overhead) |
| Matmul | **INT8 replaces FP16** → ~2× theoretical, 1.5–1.8× practical on H100 |
| Dequant | one scalar mul, fuses into downstream |

There is **no "apply s" step** anywhere in the runtime path. That's the whole engineering value.

### 5.4 The Two Scales Never to Confuse

|                | Static smoothing scale `s` | Dynamic quant scale `X_scale` |
|----------------|-----------------------------|-------------------------------|
| What it does   | Suppress outlier channels   | Map smoothed X into [-127, 127] |
| Computed       | **Offline** from calibration | **Runtime**, per token        |
| Shape          | `[d_in]`                    | `[B, T, 1]`                   |
| Where it lives | Folded into γ and W (gone)  | Temporary, regenerated each forward |

Both are necessary. `s` flattens the bumpy distribution; `X_scale` then projects the flattened distribution into INT8 range.

### 5.5 Why the Whole Trick Requires `s` to Be Static

The folding of `s` into `γ` and `W` only works if `s` is a constant:

```
γ_new = γ / s     # if s changes per-step, γ is not a constant anymore
W'    = diag(s) · W   # if s changes, W' has to be re-multiplied each step
```

A dynamic `s` would require an explicit `X / s` operation each step (a per-channel div), plus keeping FP16 W to re-scale. All SmoothQuant's runtime savings would evaporate.

The static assumption is bought by the empirical regularity from §4.1: outlier channels in LLMs are architectural, not input-dependent. A 128-sample calibration generalizes well.

### 5.6 Pre-conditions for the Folding to Work

Two structural assumptions:

1. Every quantized linear is **preceded by a normalization layer with a per-channel γ** (LayerNorm / RMSNorm). True in pre-norm Transformers.
2. The `γ` and `W` input dimension are matched (no shape adapter in between).

When violated (e.g. linear after a pure residual add), the workaround is an explicit per-channel multiply before the linear. Cheap, but slightly less elegant.

---

## 6. Weight-Only Calibration: GPTQ, AWQ

`W4A16` (4-bit weights, FP16 activations) is the production workhorse, but at 4 bits even group-wise RTN has measurable PPL loss. The two main calibration tricks:

### 6.1 GPTQ (Frantar et al. 2023)

Round columns of `W` one at a time, in an order chosen to **minimize the layer's output error**, not the per-weight reconstruction error.

```
For each column j (in Hessian-determined order):
    quantize column j to INT4 (group-wise)
    compute the error introduced
    propagate that error correction to the remaining un-quantized columns
```

Uses the inverse Hessian of `XᵀX` (computed from calibration data) to weight which weights matter more for output fidelity. Slow to calibrate (hours for 70B) but excellent post-calibration accuracy.

### 6.2 AWQ (Lin et al. 2023)

Observes that **~1% of weights are disproportionately important** for output quality, identified by activation magnitude on the corresponding input channel. Strategy: give important channels a higher-precision per-channel scale; quantize the rest normally. Much faster than GPTQ, comparable accuracy.

### 6.3 What changes at runtime

**Nothing.** Both produce a standard group-wise INT4 weight tensor. The runtime kernel is the same as for naive RTN INT4. GPTQ and AWQ are entirely offline calibration strategies. This is the most common misconception about them.

---

## 7. K-quants: Multi-Level Scale Hierarchy (2–6 bit regime)

llama.cpp's `Q2_K`, `Q3_K`, `Q4_K_M`, `Q5_K_M`, `Q6_K` formats.

### 7.1 The Block Structure

Each "super-block" holds 256 weights, divided into 16 sub-blocks of 16 weights:

```
super-block (256 weights):
  ┌─────────────────────────────────────┐
  │  FP16 super-scale                    │
  │  FP16 super-min  (asymmetric only)   │
  │  ┌──────────────────────────────────┐│
  │  │ sub-block 0 (16 weights):        ││
  │  │   6-bit sub-scale                ││
  │  │   6-bit sub-min                  ││
  │  │   16 × N-bit weight codes        ││
  │  ├──────────────────────────────────┤│
  │  │ sub-block 1 ...                  ││
  │  │ ... (16 sub-blocks total)        ││
  │  └──────────────────────────────────┘│
  └─────────────────────────────────────┘
```

### 7.2 The Multi-Level Trick

Sub-block scales are **themselves quantized** (6 bit), reconstructed via the FP16 super-scale. Effective per-weight cost decomposes as:

- N bits for the weight code (e.g. 4 for `Q4_K`).
- ~0.75 bit per weight amortized for the sub-block scale (12 bits / 16 weights).
- Negligible per-weight cost for the super-scale (32 bits / 256 weights).

Compared to flat group-wise INT4 at the same overhead, multi-level scales adapt much better to local distribution shape. `Q4_K_M` lands at ~4.5 bpw with PPL noticeably better than flat W4A16 RTN.

### 7.3 Per-Layer Mixed Precision (the K_M / K_S / K_L Suffix)

`Q4_K_M` is not "all weights at 4 bit". It's a mixed scheme:

```
token embedding         :  Q6_K       (preserve, used widely)
attention output proj   :  Q6_K       (sensitive)
FFN down_proj (reduction direction) :  Q6_K       (sensitive)
everything else         :  Q4_K       (bulk, where the savings live)
```

`_S` (small) is more aggressive, `_L` (large) more conservative. Same idea as AWQ's "protect the important weights" but applied at tensor granularity rather than weight granularity.

---

## 8. IQ-quants: Codebook Quantization (sub-2-bit regime)

llama.cpp's `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, etc. — these are where the linear-quantization paradigm breaks down and gets replaced.

### 8.1 The Information-Theoretic Wall

With INT2 you have 4 possible reconstructed values per weight. Even with group scales, every weight rounds to one of 4 values. For LLM weights this is too coarse — PPL explodes.

This is a fundamental ceiling: under "one weight = N bits", the per-weight entropy is bounded by N. The only escape is to **stop encoding weights individually**.

### 8.2 Vector Quantization Replaces Per-Weight Encoding

The conceptual shift:

> **Encode 8 weights at a time using a shared codebook index, not one weight at a time using a per-weight code.**

A small global codebook is pre-trained (offline, baked into llama.cpp as static tables — see `iq2nl_grid` etc. in `ggml-quants.c`):

```
codebook[0]   = [+1, -1, -1, +1, +1, +1, -1, +1]
codebook[1]   = [-1, -1, +1, -1, +1, -1, -1, +1]
codebook[2]   = [+1, +1, -1, +1, -1, +1, -1, -1]
...
codebook[255] = ...
```

Each entry is an 8-element vector of signed unit values (with a shared magnitude). At quantization time, each consecutive 8 weights in the tensor are matched to the closest codebook entry, and the **index** (8 bits) is stored. Plus a group scale.

Effective rate: 8 bits / 8 weights = **1 bit per weight** for the data, plus scale amortization → 1.5–2.3 bpw in practice.

### 8.3 Why a Small Codebook Suffices

LLM weights aren't IID. After group-wise scale normalization, 8 adjacent weights have structure (sparse-large + many small) that recurs across the model. A few hundred well-chosen 8-element prototypes cover most of the local-pattern distribution.

The codebooks in llama.cpp aren't model-specific — they're built to span the typical LLM-weight local distribution (Gaussian-like with sparse heavy tails). They generalize across LLaMA, Mistral, Qwen.

### 8.4 The Importance Matrix (`imatrix`)

IQ-quants accept an optional **importance matrix** from a calibration run:

```
H_i = how much each weight position affects the model's output
      (similar to GPTQ's Hessian diagonal)

best_codeword = argmin_c  Σ_i  H_i · (w_i − c_i · s)²
                                ↑
                                large H_i forces tight match
```

Important positions must be matched well; unimportant positions can drift. Same `imatrix` trick used by `Q2_K` and below. At sub-2-bit, this is the difference between "usable" and "broken".

### 8.5 Runtime Kernel

Per group during matmul:

```
1. read 1 byte index                        (shared across 8 weights)
2. codebook lookup: c[0..7] = grid[idx]     (table in L1 / shared memory)
3. read group scale s_g (FP16)
4. read corresponding 8 elements of X
5. partial = Σ X[i] · c[i]                  (8-element dot product)
6. acc    += s_g · partial                  (accumulate into FP32)
```

Two changes from group-wise INT4:

- One extra lookup per group (cheap if codebook fits in cache).
- Smaller dot product granularity (8 elements vs 32 or 128).

On CPU, this is cache-friendly and llama.cpp's strong suit. On GPU, IQ-quants are typically slower than `Q4_K` due to fewer parallelizable elements per kernel call. So IQ-quants are mostly used when **memory is the constraint** (e.g. running 70B locally on 16 GB), not when throughput is.

### 8.6 Related Academic Work

The 2024 papers **QuIP#** and **AQLM** independently developed GPU-friendly codebook quantization at 2 bit. Same conceptual move; better tuned kernels. The technique is having a renaissance.

---

## 9. Selection Table by Bit-Width Regime

| Bit budget | Recommended scheme | Calibration | Where it's used |
|------------|---------------------|-------------|-----------------|
| 8 bit | RTN per-channel | None needed | Server inference floor, lossless in practice |
| 6 bit | Group-wise + GPTQ | A few hundred samples | Quality-priority server deployments |
| **4 bit** | **Group-wise W4A16 + GPTQ/AWQ** | A few hundred samples | **Workhorse for production** |
| 4 bit (W4A8) | + SmoothQuant on activations | Calibration for outlier stats | When activation memory also matters |
| 2–3 bit | K-quants (`Q2_K`, `Q3_K`) | Optional imatrix | Local inference, llama.cpp ecosystem |
| ≤ 2 bit | **IQ-quants** (codebook) | Required imatrix | Extreme compression, 70B on 16 GB GPUs |

The progression mirrors the technique stack: as bit budget drops, you need more sophisticated tricks to keep the model alive. RTN works at 8 bit because the headroom is generous. Group-wise + GPTQ holds the line at 4 bit. SmoothQuant adds the activation half. K-quants squeeze further with multi-level scales. IQ-quants abandon the per-weight encoding entirely once linear schemes hit their information-theoretic floor.

---

## 10. Connecting Back to the Inference Story

Quantization is fundamentally a bandwidth optimization for the decode phase:

| Workload | Bottleneck | What quantization buys |
|----------|------------|------------------------|
| Prefill (long prompt) | Compute (FLOPs) | Modest — INT8/FP8 tensor cores have higher throughput |
| Decode (single token) | **HBM bandwidth** | **Major** — fewer bytes per weight directly cuts decode time |
| KV-cache-bound (long context) | HBM (KV cache) | Major — KV quantization (KIVI, KVQuant) targets this separately |

This is why W4A16 is the workhorse: in the decode-dominated production regime, what matters is bytes per weight, not the precision of activations. The 4× weight compression nearly translates directly into 4× decode throughput in the bandwidth-bound regime.

---

## References

### Linear quantization & calibration
- Dettmers et al. 2022 — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*.
- Xiao et al. 2023 — *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*.
- Frantar et al. 2023 — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*.
- Lin et al. 2023 — *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*.

### KV cache quantization
- Liu et al. 2024 — *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*.
- Hooper et al. 2024 — *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*.

### Codebook / sub-2-bit
- Tseng et al. 2024 — *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*.
- Egiazarian et al. 2024 — *Extreme Compression of Large Language Models via Additive Quantization* (AQLM).
- [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp) — `ggml-quants.c` for K-quant and IQ-quant reference impl.

### Code
- [`AutoGPTQ/AutoGPTQ`](https://github.com/AutoGPTQ/AutoGPTQ) — GPTQ implementation.
- [`mit-han-lab/llm-awq`](https://github.com/mit-han-lab/llm-awq) — AWQ implementation.
- [`mit-han-lab/smoothquant`](https://github.com/mit-han-lab/smoothquant) — SmoothQuant reference impl.
- [`IST-DASLab/marlin`](https://github.com/IST-DASLab/marlin) — fast group-wise INT4 GEMM kernels.
