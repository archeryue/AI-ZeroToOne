"""
FlashAttention v2 — Triton 实现 (forward pass)
==============================================

参考资料:
  - Tri Dao 的 Triton 官方教程 06-fused-attention.py
  - 论文: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"

本实现展示了 FA v2 相对 v1 的三个核心改进:
  ① 外层循环换成 Q,Q 全程驻留 SRAM/寄存器,O 只 store 一次
  ② 归一化推迟到循环结束,内循环只做 rescale,把非 matmul FLOPs 压到最少
  ③ 在 seq_len 维度加一层并行 (grid = (num_q_blocks, batch*heads))

仅实现 forward。Backward 需要把 logsumexp M 存下来,反向时用 P = exp(QK^T - M)
快速重建 attention 概率,实现见 Tri Dao 的官方版本。

依赖: torch >= 2.0, triton >= 2.1
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------- #
#  autotune 配置: Triton 会在第一次调用时跑一遍每个 config 选最快的
#  不同 SM 架构、不同 head_dim、不同 seqlen 的最优 tile 大小差别很大
# ---------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    ],
    key=['N_CTX', 'HEAD_DIM'],  # 改这俩维度才会重新 autotune
)
@triton.jit
def _flash_attn_v2_fwd_kernel(
    Q, K, V, sm_scale,            # 输入张量 + 1/sqrt(d) 缩放
    L,                            # 输出: logsumexp,反向用 (shape: [B*H, N_CTX])
    Out,                          # 输出: attention 结果 O (shape 同 Q)
    stride_qz, stride_qh, stride_qm, stride_qk,   # Q 的 4 个 stride
    stride_kz, stride_kh, stride_kn, stride_kk,   # K
    stride_vz, stride_vh, stride_vk, stride_vn,   # V
    stride_oz, stride_oh, stride_om, stride_on,   # O
    Z, H, N_CTX,                  # batch、head 数、序列长度
    HEAD_DIM: tl.constexpr,       # 每个 head 的特征维度 (编译期常量)
    BLOCK_M:  tl.constexpr,       # Q tile 大小,autotune 决定
    BLOCK_N:  tl.constexpr,       # KV tile 大小,autotune 决定
    IS_CAUSAL: tl.constexpr,      # 是否 causal mask
):
    # ============================================================
    # 改动 ③:三维 grid。每个 program 负责一个 (Q tile, batch, head)
    # ============================================================
    start_m = tl.program_id(0)               # 当前 Q tile 的索引
    off_hz  = tl.program_id(1)               # 扁平化的 (batch, head)
    off_z   = off_hz // H                    # batch index
    off_h   = off_hz % H                     # head index

    # 当前 (batch, head) 在 HBM 里的起始偏移
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # ============================================================
    # block_ptr: tile-wise 指针视图
    # 让编译器知道访存是 tile-by-tile 连续的,可生成 cp.async (Ampere)
    # 或 TMA (Hopper) 友好的 load
    # ============================================================
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # K 在 kernel 内要做 Q @ K^T,这里直接转置形状,让 tl.dot(Q, K) 就等于 Q @ K^T
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),       # 注意 stride 也跟着转
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # ============================================================
    # online softmax 的三个累加器 (改动 ①: 全程驻留寄存器)
    # ============================================================
    # m_i: 这一行 Q 看过的所有 QK 中的最大值
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l_i: 修正后的 Σ exp(QK - m_i)
    # 初值 1.0 是历史习惯 (第一轮被 alpha=0 清零,不影响数学结果)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc: 未归一化的输出 Σ p · V (改动 ②: 循环里不除 l_i,推迟到循环外)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ============================================================
    # 改动 ①: Q tile 在循环开始前 load 一次,后续不再重读
    # ============================================================
    q = tl.load(Q_block_ptr)
    # sm_scale = 1/sqrt(d)。乘上 log2(e) ≈ 1.4427 后,
    # 下面就可以用更快的 exp2 代替 exp。数学: exp(x*c) = exp2(x * c * log2(e))
    q = (q * (sm_scale * 1.44269504)).to(q.dtype)

    # ============================================================
    # 内层循环: 流式扫所有 K/V tile
    # ============================================================
    # causal 时,只扫到对角线 (j <= i),跨对角线的最末 tile 由 mask 处理
    lo = 0
    if IS_CAUSAL:
        hi = (start_m + 1) * BLOCK_M
    else:
        hi = N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)  # 给编译器对齐提示

        # ---- Step A: S = Q · K^T (Tensor Core) ----
        k  = tl.load(K_block_ptr)
        qk = tl.dot(q, k)                      # [BLOCK_M, BLOCK_N], fp32 累加

        # ---- causal mask: 只对跨对角线的 tile 施加 ----
        if IS_CAUSAL:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            # 保留 q_row >= k_col 的位置 (下三角 + 对角)
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

        # ---- Step B: online softmax 状态更新 ----
        m_ij  = tl.maximum(m_i, tl.max(qk, 1))          # 加入此 tile 后的新 max
        alpha = tl.math.exp2(m_i - m_ij)                # 旧累加器的修正因子
        p     = tl.math.exp2(qk - m_ij[:, None])        # 当前 tile 的 P
        l_ij  = tl.sum(p, 1)                            # 当前 tile 的局部 Σ exp

        # 修正之前的 l 和 acc (V2 关键: 只 rescale,不 normalize)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # ---- Step C: acc += P · V (Tensor Core, 三参数 fused multiply-add) ----
        v   = tl.load(V_block_ptr)
        # tl.dot(p, v, acc) 等价于 acc = p @ v + acc,直接 FMA 进 Tensor Core
        acc = tl.dot(p.to(v.dtype), v, acc)

        # 推进到下一个 KV tile
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # ============================================================
    # 改动 ②: 循环外唯一一次归一化 (省掉 N_CTX/BLOCK_N 次循环内除法)
    # ============================================================
    acc = acc / l_i[:, None]

    # ============================================================
    # 保存 logsumexp M = m + log(l) 给反向传播
    # (我们一直用 exp2,所以存的是 log2 域,反向时换回去即可)
    # ============================================================
    m_i += tl.math.log2(l_i)
    l_ptrs = L + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(l_ptrs, m_i)

    # 输出写回 HBM: 整个 Q tile 只 store 一次 (对比 v1 每个 KV tile 都要写)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# ============================================================================ #
#  高层 Python 接口
# ============================================================================ #
def flash_attention_v2(q, k, v, causal: bool = False, sm_scale: float = None):
    """
    FlashAttention v2 forward pass.

    Args:
        q, k, v: shape [B, H, N, D],dtype fp16 / bf16
        causal:  是否 causal mask
        sm_scale: 缩放系数,默认 1/sqrt(D)

    Returns:
        O: shape [B, H, N, D],dtype 同 q
    """
    assert q.shape == k.shape == v.shape, "QKV shape mismatch"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    B, H, N, D = q.shape
    # HEAD_DIM 在 kernel 里被当作编译期常量,Triton 会针对它生成专门的 PTX
    assert D in (32, 64, 128, 256), f"unsupported head_dim={D}"

    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    O = torch.empty_like(q)
    L = torch.empty((B * H, N), device=q.device, dtype=torch.float32)

    # grid 形状由 BLOCK_M 决定(autotune 时已知)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), B * H)

    _flash_attn_v2_fwd_kernel[grid](
        q, k, v, sm_scale, L, O,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        B, H, N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
    )
    return O


# ============================================================================ #
#  Reference + 正确性 / 性能测试
# ============================================================================ #
def attention_reference(q, k, v, causal=False):
    """朴素 PyTorch attention,用于对比"""
    sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    s = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        N = q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 1024, 64
    dtype = torch.float16

    q = torch.randn(B, H, N, D, device='cuda', dtype=dtype) * 0.5
    k = torch.randn(B, H, N, D, device='cuda', dtype=dtype) * 0.5
    v = torch.randn(B, H, N, D, device='cuda', dtype=dtype) * 0.5

    # 正确性检查
    for causal in (False, True):
        out_ref = attention_reference(q, k, v, causal=causal)
        out_fa  = flash_attention_v2(q, k, v, causal=causal)
        err = (out_ref.float() - out_fa.float()).abs().max().item()
        print(f"[correctness] causal={causal} | max abs diff: {err:.4e}")
        assert err < 1e-2, "FlashAttention v2 result mismatch!"

    # 简单 benchmark
    import time
    torch.cuda.synchronize()
    iters = 50

    t0 = time.time()
    for _ in range(iters):
        flash_attention_v2(q, k, v, causal=True)
    torch.cuda.synchronize()
    fa_ms = (time.time() - t0) / iters * 1000

    t0 = time.time()
    for _ in range(iters):
        attention_reference(q, k, v, causal=True)
    torch.cuda.synchronize()
    ref_ms = (time.time() - t0) / iters * 1000

    print(f"[benchmark] FA v2: {fa_ms:.3f} ms | Reference: {ref_ms:.3f} ms"
          f" | speedup: {ref_ms/fa_ms:.2f}x")
