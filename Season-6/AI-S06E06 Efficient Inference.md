## Efficient Inference: Prefill, Decode, and the Systems Around Them

> **Format:** Lecture handout
> **Companion code:** [`karpathy/nanochat`](https://github.com/karpathy/nanochat) (single-node baseline) + [`vllm-project/vllm`](https://github.com/vllm-project/vllm) reference snippets
> **Strategy:** Treat inference as two different workloads in one trench coat. Prefill is compute-bound; decode is memory-bandwidth-bound. Every modern serving system is a different way of acknowledging that split.

---

## Learning Objectives

By the end of this episode you should be able to:

1. Separate **prefill** and **decode** as two distinct workloads with different bottlenecks, and predict which one limits a given workload from first principles.
2. Compute KV-cache memory and decode arithmetic intensity from `B, T, d, L`, and explain why decode is memory-bandwidth-bound on H100.
3. Derive PagedAttention from the OS virtual-memory analogy and explain what fragmentation pre-paging caused.
4. Walk FlashAttention's tile-and-recompute trick and explain why it changes asymptotic memory but not asymptotic FLOPs.
5. Distinguish request-level batching from continuous (iteration-level) batching and quantify the throughput delta.
6. Sketch the speculative-decoding accept/reject loop and explain when it helps vs hurts.
7. Locate quantization, chunked prefill, and prefill/decode disaggregation in the same decode-bottleneck framing.

## Prerequisites

- Episode 6.1 — the roofline model, NVLink/HBM bandwidth, arithmetic-intensity vocabulary.
- Episode 6.2 — KV cache math (§5.2), GQA/MQA/SWA/MLA (we'll lean on the result, not redo the derivation).
- Episode 6.5 — DP/TP/PP for training; we'll contrast how the same axes redeploy for inference.

---

## 0. The Setup: Inference Is Not Training In Reverse

In training, every step processes a batch of `B × T` tokens through a forward + backward pass. Memory pressure comes from activations and optimizer state. Compute and bandwidth are roughly balanced on H100-class hardware.

In **inference**, two things break that symmetry:

1. **Autoregressive generation is sequential.** You emit one token at a time, each conditioning on the last. There is no parallelism over the output sequence — only over batch and (partially) prefill.
2. **The first token is fundamentally different from the rest.** Generating token 1 of a 1024-token prompt requires processing all 1024 prompt tokens. Generating token 2 requires processing just one new token, conditioned on a KV cache that holds the first 1024 (plus the new one).

These two facts produce the **prefill / decode split**, which is the single most important framing in this episode.

```
                       prefill                    decode
                     ┌──────────┐         ┌──────┬──────┬──────┬──...
input            →   │ T_prompt │   →     │ tok₁ │ tok₂ │ tok₃ │
                     └──────────┘         └──────┴──────┴──────┴──...
                                              ↑
                                          KV cache grows here
work per step:    T_prompt × d_model      1 × d_model
arithmetic        compute-bound           memory-bandwidth-bound
intensity:        (FlashAttention, BLAS)  (HBM is the wall)
```

Treat the rest of the episode as a series of optimizations that exist because of this split.

---

## 1. Prefill vs Decode: The Two Workloads

### 1.1 Prefill: Compute-Bound

The prompt is processed in one shot — `T_prompt` tokens through every transformer layer in parallel. For LLaMA-3-8B (`d=4096, L=32, H=32, d_head=128`) at `T_prompt=2048, B=1`:

```
Compute per layer ≈ 2 · T · d²        ≈ 2 · 2048 · 4096²   ≈  6.9 × 10¹⁰ FLOPs
                  + 2 · T² · d  (attn) ≈ 2 · 2048² · 4096  ≈  3.4 × 10¹⁰ FLOPs
Total per layer  ≈ ~10¹¹ FLOPs
× 32 layers      ≈  3.2 × 10¹² FLOPs / forward
```

Bytes moved (weights):

```
Per layer weights ≈ 12 · d² (Q,K,V,O + FFN×3 at SwiGLU)
                  ≈ 12 · 4096² · 2 bytes = 400 MB
× 32 layers       = 12.8 GB
```

Arithmetic intensity ≈ `3.2 × 10¹² / 12.8 × 10⁹ ≈ 250 FLOPs/byte`.

H100's bf16 roofline crosses around **300 FLOPs/byte**. Prefill is **right at the knee** — compute-bound for long prompts, with the matmuls dominating.

### 1.2 Decode: Memory-Bandwidth-Bound

Now generate token N+1 with `T = 1` new token. Same weights, but the work is tiny:

```
Compute per layer ≈ 2 · 1 · d²       ≈ 3.4 × 10⁷ FLOPs
Bytes (weights)   ≈ 400 MB           — unchanged, we still load every weight!
```

Arithmetic intensity ≈ `3.4 × 10⁷ / 4 × 10⁸ ≈ 0.08 FLOPs/byte`. **3000× below the roofline.** Decode is purely HBM-bound: you're paying to drag the full model from HBM once per token.

**This single fact drives every inference optimization in the field.** Everything that follows is a different way to either:

- Hide HBM bandwidth (FlashAttention's SRAM reuse, paging the KV cache).
- Amortize HBM bandwidth across more tokens (batching, speculative decoding).
- Reduce HBM bandwidth (quantization).

### 1.3 The KV Cache Refresher

(Quick re-derivation from E02 §5.2 since it's central this episode.) During decode, K and V for every previous token are cached so each new token only needs *its own* Q computed:

```
KV cache (bytes) = 2 · T · n_kv_heads · d_head · n_layers · bytes_per_element
```

For LLaMA-3-8B at FP16, 32K context, GQA-8:

```
2 · 32768 · 8 · 128 · 32 · 2  ≈  4.3 GB per request
```

Decode bandwidth, per token, per request: ~400 MB weights + the relevant KV slice. With B concurrent requests, weights amortize (read once for B tokens), but KV does not — every request has its own.

**The KV cache is the inference-side memory wall.** The systems below all target it.

---

## 2. Deep Dive — KV Cache and PagedAttention

### 2.1 The Pre-Paging Problem

Standard implementations allocate the KV cache as a **contiguous tensor** of shape `[max_seq_len, n_kv_heads, d_head]` per request. Two flaws emerge under serving load:

1. **Internal fragmentation.** You allocate for `max_seq_len` but most requests stop early. A 32K-cap request that emits 200 tokens wastes 99% of its allocation.
2. **External fragmentation.** Differently-sized live requests leave HBM looking like swiss cheese — there's enough total free memory for a new request, but no contiguous block.

The 2023-era result (measured in the vLLM paper): roughly **60–80% of HBM was wasted** by these two effects on production HuggingFace-Transformers serving.

### 2.2 The PagedAttention Idea

Borrow virtual memory from OS-land. Break the KV cache into fixed-size **blocks** (typically 16 tokens). Each request holds a **block table** — a list of physical block pointers — rather than a contiguous tensor.

```
Logical KV cache (request 7):     [ blk_A | blk_B | blk_C | blk_D ]
                                       ↓     ↓     ↓     ↓
Physical HBM blocks (pool):       … blk_C … blk_A … … blk_D … blk_B …
                                  (non-contiguous, no guarantees)
```

The attention kernel does an extra pointer-chase per block, which is cheap on modern GPUs.

### 2.3 What This Buys You

- **No internal fragmentation.** Blocks are allocated as needed; a 200-token request holds ~13 blocks, full stop.
- **No external fragmentation.** Block size is fixed, so the free list is just a pool.
- **Copy-on-write sharing.** Two requests with the same prompt prefix (system prompt, few-shot examples) can share physical blocks until they diverge. Big win for chat workloads.
- **Beam search / parallel sampling.** Multiple sample branches share the prompt's blocks for free.

vLLM reported ~**2–4× higher serving throughput** vs prior systems from PagedAttention alone, before continuous batching even enters the picture.

### 2.4 Practical Notes

- Block size is a tuning knob. Smaller blocks → less waste, more pointer-chases. 16 tokens is the modern default.
- The block table is small (~hundreds of pointers per request) and lives in HBM next to the kernel.
- FlashAttention 2/3 natively supports paged KV. The kernel walks the block table per query block.

> **Takeaway:** PagedAttention turns KV management from "allocate a worst-case tensor per request" into "page table over a block pool." It is to KV cache what virtual memory is to RAM.

---

## 3. Deep Dive — FlashAttention

### 3.1 The Problem It Solves

Standard attention materializes the full `[B, H, T, T]` scores matrix in HBM. Two costs:

1. **Memory.** Quadratic in `T`. At `T=8192, B=1, H=32, fp16` that's 4 GB just for scores.
2. **Bandwidth.** That tensor is written, then read back for softmax, then read again for the matmul against V. Three HBM round-trips for a tensor that only exists to be reduced away.

The naive attention kernel is *memory-bandwidth-bound on the scores matrix*. Compute units sit idle waiting for HBM.

### 3.2 The Fix: Tile and Recompute

FlashAttention (Dao et al., 2022) splits Q, K, V into blocks that fit in **SRAM** (the GPU's L1/registers, ~100s of KB per SM). For each Q block, stream K and V blocks past it, accumulating the softmax output **without ever materializing the full scores matrix**.

Key trick: **online softmax**. You can compute `softmax(s_1, s_2, ..., s_n)` in a single streaming pass by maintaining a running max `m` and a running normalizer `ℓ`, rescaling as new values arrive:

```
For each new score block s_k:
  m_new = max(m, max(s_k))
  ℓ_new = ℓ · exp(m − m_new) + Σ exp(s_k − m_new)
  O_new = O · (ℓ · exp(m − m_new) / ℓ_new) + (exp(s_k − m_new) / ℓ_new) · V_k
  m, ℓ, O = m_new, ℓ_new, O_new
```

At the end, `O` is exactly the attention output. No full scores matrix ever exists in HBM.

### 3.3 Asymptotic Picture

|              | Standard attention | FlashAttention |
|--------------|-------------------|----------------|
| HBM accesses | `O(T² + Td)`      | `O(Td²/M)` where `M` = SRAM size |
| HBM peak     | `O(T²)`           | `O(T)` (just for input/output) |
| FLOPs        | `O(T²d)`          | `O(T²d)` — identical |

**The FLOPs are unchanged.** FlashAttention does not avoid quadratic compute. It avoids the quadratic *memory traffic*, which is what was actually bottlenecking the kernel.

### 3.4 Backward Pass: Recompute Cheap, Store Costly

Training needs to backprop through attention, which would normally require the scores matrix again. FlashAttention stores only the per-row `(m, ℓ)` scalars and **recomputes** the scores in the backward pass. ~30% more FLOPs in the backward, but ~10× less HBM traffic. Massive net win.

This is also why FlashAttention is used at training time, not just inference: it cuts both peak memory and wall time.

### 3.5 The Generations

- **FlashAttention-1 (2022):** the original. ~2–4× speedup, 5–20× less memory.
- **FlashAttention-2 (2023):** better parallelism (parallelize over `T` instead of just over heads), fewer non-matmul ops. Another ~2× speedup.
- **FlashAttention-3 (2024):** Hopper-specific (wgmma, async copies, FP8 path). Reaches ~75% of H100's peak FLOPs on attention — close to dense GEMM.

Used in: every modern inference and training stack. `torch.nn.functional.scaled_dot_product_attention` will pick FlashAttention 2 or 3 when available.

> **Takeaway:** FlashAttention is a memory-traffic optimization, not a compute optimization. The trick is online softmax; the consequence is that attention is no longer the kernel that starves the compute units.

---

## 4. Deep Dive — Continuous Batching

### 4.1 Request-Level Batching: The Wrong Default

Classic serving systems batch at the **request boundary**: collect N requests, run them through the model together until *all N* finish, then start the next batch.

```
time →
batch 1:  [req A | req B | req C | req D]  ← all generate together
          ──────────────────────────── until all four hit EOS
batch 2:  [req E | req F | req G | req H]
```

This wastes everything. If A finishes at 100 tokens and D needs 1000, D's GPU slot does productive work but A's slot is **padded** for 900 tokens. The GPU sees `B=4` for the whole window when effective B drops to 1 near the end.

### 4.2 Continuous Batching: Iteration-Level Scheduling

vLLM (Kwon et al., 2023) and ORCA (Yu et al., 2022) reorganize around **per-iteration scheduling**. Every decode step, the scheduler:

1. Drops any sequences that hit EOS.
2. Adds any waiting prefill requests it has memory for.
3. Runs one forward pass on whatever set is currently active.

```
time →
step 1:  [A | B | C | D]
step 2:  [A | B | C | D]      ← A finishes
step 3:  [_ | B | C | D | E]  ← E joins (prefill)
step 4:  [_ | B | C | D | E]
step 5:  [_ | _ | C | D | E]  ← B finishes
step 6:  [F | _ | C | D | E]  ← F joins
...
```

Effective batch size stays near max at all times. **No padding, no waiting.**

### 4.3 What This Costs

- The KV cache must support **dynamic insert/evict** at arbitrary points. PagedAttention is what makes this practical — you can grant/revoke individual block lists without copying tensors.
- The scheduler is non-trivial: priority, fairness, memory accounting, optional preemption ("swap out" a sequence's KV to CPU if memory pressure spikes).
- Prefill and decode now mix in the same batch — see §7 (chunked prefill) for the consequences.

### 4.4 The Throughput Story

Measured in the vLLM paper and broadly replicated: **2–4× higher throughput** vs HuggingFace TGI's request-level batching, on top of the 2–4× from PagedAttention. Compounded, **~10× over naive HF Transformers serving**.

This is the dominant story for inference economics in 2024–2026. If your serving stack doesn't do continuous batching + paged KV, you're leaving 10× on the table.

> **Takeaway:** Schedule at the *iteration* level, not the *request* level. Pair with PagedAttention so the KV bookkeeping is cheap. This is most of what "modern inference" means.

---

## 5. Deep Dive — Speculative Decoding

### 5.1 The Asymmetry It Exploits

Decode wastes most of the GPU: arithmetic intensity 0.08, sitting at <1% of peak FLOPs. The model could *process* 100 tokens in roughly the same wall time as 1 — the bottleneck is HBM round-trips, not compute. We just don't have 100 useful future tokens to feed in.

**Unless we can cheaply guess them.**

### 5.2 The Scheme

Run a small, fast **draft model** to propose K tokens. Then run the **target model** on the entire `[prompt + K_draft]` sequence in one forward pass (which is essentially a prefill of length K — compute-bound, cheap relative to K serial decodes). Compare:

```
Draft model (q_d):  proposes  d_1, d_2, d_3, ..., d_K
Target model (q_t): scored in one forward on [..., d_1, d_2, ..., d_K]
                    → produces q_t(token | prefix) for every position
```

Accept token `d_i` with probability `min(1, q_t(d_i | prefix) / q_d(d_i | prefix))`. On the first rejection, sample the corrected token from `max(0, q_t - q_d)` (renormalized) and stop. The math gives a **provably identical distribution** to plain target-model sampling — speculative decoding does not change the output distribution.

### 5.3 What Determines The Speedup

```
expected speedup ≈ (1 + α + α² + ... + α^(K-1)) / (1 + cost_draft / cost_target)
```

where `α` is the average per-token acceptance rate. Practical numbers:

- α ≈ 0.7 with a well-tuned draft model.
- `cost_draft / cost_target` ≈ 0.05 for a 1B-class draft + 70B target.
- → ~2–3× decode speedup, with no quality loss.

### 5.4 Variants

- **Self-speculative / Medusa (2024):** add extra decoder heads to the target model that predict tokens 2, 3, 4 ahead. No separate draft model; same architecture amortizes draft + verify.
- **EAGLE (2024):** a tiny transformer trained to predict the *hidden state* of the target's next token. Higher acceptance rates than vanilla speculative.
- **Lookahead decoding (2024):** generate draft tokens via cached n-gram patterns from the recent KV. No model at all on the draft side — just a sliding-window cache.

### 5.5 When It Doesn't Help

- Large batches. Speculative decoding trades extra compute for fewer HBM round-trips. At `B=32+` you're already compute-bound on most hardware, so the trade dries up.
- Generation tasks with low predictability (creative writing, code with novel APIs). α drops, draft cost dominates.
- Latency-sensitive single requests where the draft model's extra serial step costs more than the saved round-trips.

> **Takeaway:** Speculative decoding spends a small amount of extra compute to claw back HBM-bound wasted cycles. It's a low-batch / latency-sensitive optimization. Identity-preserving, so always safe to enable.

---

## 6. Mention — Quantization

> *Survey only this episode. Full deep-dive reserved for a later episode.*

### 6.1 The Pitch

Decode is HBM-bound. Halve the bytes per weight → halve the HBM traffic → halve the decode time. Modulo accuracy loss.

### 6.2 The Spectrum

| Scheme | Weights | Activations | KV cache | When to use |
|--------|---------|-------------|----------|-------------|
| **bf16/fp16** | 16-bit | 16-bit | 16-bit | Baseline |
| **W8A8 (INT8)** | INT8 | INT8 | FP16 | Training-aware; ~free on H100 with TensorRT |
| **W8A8 (FP8)** | FP8 | FP8 | FP8 | Native on Hopper; nearly lossless |
| **W4A16** | INT4 (group-quantized) | FP16 | FP16 | Memory wall is HBM, not activations — the workhorse |
| **W4A8 / W4A4** | INT4 | INT8 / INT4 | INT8 | Aggressive; accuracy-tax depends on model |

### 6.3 The Workhorse: INT4 Weight-Only (W4A16)

GPTQ, AWQ, and bitsandbytes all converge here. The idea: quantize *only weights* to 4-bit (with group-wise scales every 128 elements), but keep activations and the KV cache in FP16/BF16. At decode, the matmul kernel dequantizes weights on the fly into registers.

- ~4× HBM saved on weights → ~4× decode throughput in the bandwidth-bound regime.
- Activation precision is preserved → minimal accuracy loss.
- Calibration uses a small dataset (a few hundred sequences); no retraining.

This is what makes 70B-class models run on a single 24 GB consumer GPU.

### 6.4 KV Cache Quantization

The KV cache itself can be quantized to INT8 or INT4 (KIVI, KVQuant). Halves the KV memory, modest accuracy cost. Useful when context length is the bottleneck.

> **Takeaway:** In the memory-bandwidth-bound decode regime, every byte saved is a token of latency saved. W4A16 is the modern default; FP8 weights + activations are catching up on Hopper.

---

## 7. Mention — Chunked Prefill and Disaggregated Serving

### 7.1 Chunked Prefill

Once continuous batching mixes prefill and decode in the same forward pass, prefill (long sequence per request, compute-bound) competes with decode (short per-request, bandwidth-bound) for the same kernel call. A 4K-token prefill stalls the decode tokens behind it.

**Fix:** chop each prefill request into chunks of (e.g.) 512 tokens. Each iteration runs one chunk's worth of prefill plus the decode tokens of all live requests. Decode latency stays bounded; prefill takes a few iterations but pipelines cleanly.

Used by vLLM, SGLang, DeepSpeed-MII, TensorRT-LLM. Now table stakes.

### 7.2 Disaggregated Prefill/Decode

The next step: stop mixing them at all. Run prefill on one cluster (tuned for compute, possibly TP + larger batches) and decode on another (tuned for memory bandwidth, possibly different parallelism). Transfer the KV cache over the network between phases.

- **Mooncake (Moonshot, 2024):** disaggregated prefill/decode with a global KV cache pool.
- **DistServe (UCSD, 2024):** academic version with detailed scheduling.
- **DeepSeek-V3 serving:** uses disaggregation to hit dramatic cost-per-token numbers.

The economics: prefill clusters can run at near-100% MFU; decode clusters can run on cheaper memory-rich hardware (think H20 over H100). The KV transfer is the new bottleneck — typically over RDMA between racks.

---

## 8. Mention — Serving Systems

The space has consolidated around three open systems and a few closed ones:

| System | Origin | What it's known for |
|--------|--------|---------------------|
| **vLLM** | UC Berkeley | PagedAttention + continuous batching reference impl |
| **SGLang** | LMSys | Frontend for programmable serving, fast prefix-cache reuse |
| **TensorRT-LLM** | NVIDIA | Closed-source kernels, best raw H100 throughput |
| **TGI** | HuggingFace | Easy deployment; lags vLLM on throughput |
| **DeepSpeed-MII** | Microsoft | Integrated with DeepSpeed training stack |
| **Mooncake / SGLang-disagg** | Moonshot / community | Disaggregated serving frontier |

For most teams in 2026, vLLM or SGLang is the default. TensorRT-LLM wins when you can absorb its build complexity for a few percent more throughput.

---

## 9. Where nanochat Sits

nanochat ships an inference path in `chat_engine.py` (and the `engine.py` core). It is **deliberately minimal** — the goal is the speed-run, not a production serving stack:

| Feature | nanochat | Production (vLLM / SGLang) |
|---------|----------|----------------------------|
| KV cache | Contiguous per-request | Paged |
| Batching | Single-request | Continuous (iteration-level) |
| Attention | FlashAttention via SDPA | FlashAttention-3 + custom paged kernels |
| Quantization | None (bf16) | W4A16 / FP8 standard |
| Speculative decoding | None | Optional (vLLM, SGLang) |
| Prefill / decode split | Implicit (single request) | Often disaggregated |

The story to tell: nanochat lets you *see* the prefill/decode distinction in the simplest possible code, then every system in §2–§8 is a different way of industrializing it.

> **Pedagogical note:** "how would you make nanochat's `engine.py` into vLLM?" is a great mental exercise after this episode. The answer is roughly: replace the KV tensor with a block table, add an iteration-level scheduler, swap the attention call for a paged kernel. That's most of vLLM.

---

## 10. Synthesis

| Optimization | Phase targeted | Mechanism | Typical win |
|--------------|---------------|-----------|-------------|
| **FlashAttention** | Both | Tile + online softmax → no scores matrix in HBM | 2–4× attention speedup, O(T) memory |
| **PagedAttention** | Decode | Block table over fixed-size KV pages | 2–4× throughput from fragmentation reclaim |
| **Continuous batching** | Decode | Iteration-level scheduling, no padding | 2–4× throughput on top of paging |
| **Speculative decoding** | Decode | Cheap draft + parallel verify | 2–3× latency on low batch sizes |
| **Quantization (W4A16)** | Decode | Less HBM traffic per weight | 2–4× decode tokens/s |
| **Chunked prefill** | Both | Bound prefill's impact on decode latency | Tail-latency fix, not throughput |
| **Disaggregation** | Both | Separate clusters per phase | Cost-per-token win, frontier-only |

The unifying picture:

> **Decode is HBM-bandwidth-bound. Every optimization is a different way to either pay less HBM per token or amortize HBM across more tokens.**

Once you see that, every section above lines up:

- FlashAttention pays less HBM per attention call.
- Paging + continuous batching amortize weights across more concurrent requests.
- Quantization pays fewer bytes per weight.
- Speculative decoding amortizes HBM across multiple tokens per forward.
- Disaggregation moves the decode phase to cheaper bandwidth-rich hardware.

### Decision Sketch

```
Goal = max throughput (tokens/s/$)?
  → vLLM or SGLang: paging + continuous batching + W4A16.
Goal = min latency (single-request)?
  → Speculative decoding + smallest batch + FlashAttention-3.
Goal = max context (very long prompts)?
  → SWA / MLA models (architectural) + KV quantization + chunked prefill.
Goal = frontier economics (>10⁹ tokens/day)?
  → Disaggregated prefill/decode + custom kernel stack.
```

---

## References

### Core Papers

- Dao et al. 2022 — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
- Dao 2023 — *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.
- Shah et al. 2024 — *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision*.
- Kwon et al. 2023 — *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM).
- Yu et al. 2022 — *ORCA: A Distributed Serving System for Transformer-Based Generative Models* (continuous batching).
- Leviathan et al. 2023 — *Fast Inference from Transformers via Speculative Decoding*.
- Chen et al. 2023 — *Accelerating Large Language Model Decoding with Speculative Sampling* (DeepMind).
- Cai et al. 2024 — *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*.
- Li et al. 2024 — *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*.
- Frantar et al. 2023 — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*.
- Lin et al. 2023 — *AWQ: Activation-aware Weight Quantization*.
- Liu et al. 2024 — *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*.
- Zhong et al. 2024 — *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving*.
- Qin et al. 2024 — *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*.

### Code

- [`vllm-project/vllm`](https://github.com/vllm-project/vllm) — paging + continuous batching reference.
- [`sgl-project/sglang`](https://github.com/sgl-project/sglang) — programmable serving, fast prefix cache.
- [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) — kernels and Python bindings.
- [`NVIDIA/TensorRT-LLM`](https://github.com/NVIDIA/TensorRT-LLM) — closed-source kernels, integration scaffolding.
- [`karpathy/nanochat`](https://github.com/karpathy/nanochat) — `engine.py` + `chat_engine.py` (single-request baseline).

### Further Reading

- HuggingFace — *LLM Inference Optimization Survey* (running update).
- vLLM blog — *Continuous Batching Explained* and *PagedAttention Deep Dive*.
- DeepSeek-AI — *DeepSeek-V3 Inference Stack* (engineering notes).

---

## Part B Code-Rewrite Modules (Preview)

1. **Module 1** — Trace a single decode step in nanochat. Annotate every HBM read; compute arithmetic intensity.
2. **Module 2** — Replace nanochat's contiguous KV with a 16-token block table. Verify identical outputs.
3. **Module 3** — Implement iteration-level scheduling: a loop that processes a list of in-flight sequences and accepts new requests each step.
4. **Module 4** — Drop in a 70M-param draft model; implement the accept/reject loop for speculative decoding.
5. **Module 5** *(stretch)* — INT4 quantize the FFN weights with a simple group-wise scheme; measure decode throughput.
