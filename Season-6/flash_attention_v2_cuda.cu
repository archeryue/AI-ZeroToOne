/*
 * FlashAttention v2 — CUDA 实现 (forward pass)
 * =============================================
 *
 * 参考: "FlashAttention-2: Faster Attention with Better Parallelism and Work
 *       Partitioning", Tri Dao 2023.
 *
 * 本实现的取舍:
 *   - 优先算法清晰度 (one thread per Q row),不用 WMMA / MMA。
 *     工业级实现 (cuDNN, Tri Dao 原版) 会用 Tensor Core,可参考最后的注释。
 *   - HEAD_DIM, BLOCK_M, BLOCK_N 在编译期固定 (模板参数),便于循环 unroll。
 *   - 输入 fp16,累加器 fp32 (防止数值溢出)。
 *   - 仅 forward,无 backward。
 *
 * 已实现的 V2 三大改动 (在代码中用 ① ② ③ 标注):
 *   ① grid = (num_q_blocks, batch*heads):seq_len 维度并行
 *   ② 外层循环 Q,内层循环 KV: Q 全程驻留 SRAM/寄存器,O 只 store 一次
 *   ③ 归一化推迟到循环结束: 循环里 acc 始终是未归一化的加权和
 *
 * 同时也保留了 V1 就有的核心思想:
 *   - 流式 online softmax (Milakov & Gimelshein 2018)
 *   - tile 化避免 N×N attention 矩阵物化
 *
 * 编译:
 *   nvcc -O3 -arch=sm_80 -std=c++17 flash_attention_v2_cuda.cu -o fa_v2
 * 运行:
 *   ./fa_v2
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t e = (call);                                                   \
        if (e != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                          \
                    cudaGetErrorString(e), __FILE__, __LINE__);                   \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)


// ============================================================================
//  FlashAttention v2 forward kernel
//
//  每个 thread block 处理:
//    - 一个 Q tile (BLOCK_M 行)
//    - 一个 (batch, head)
//    - 所有 KV tile (内层循环 streaming)
//
//  Grid:  (ceil(N_CTX / BLOCK_M), batch * num_heads)   ← 改动 ③
//  Block: BLOCK_M threads, 一个线程 = 一个 Q row
//
//  Shared memory 布局:
//    sQ [BLOCK_M, HEAD_DIM]    ← Q tile,改动 ① 中"一次 load,永不重读"
//    sK [BLOCK_N, HEAD_DIM]    ← 当前 KV tile 的 K 部分
//    sV [BLOCK_N, HEAD_DIM]    ← 当前 KV tile 的 V 部分
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attn_v2_fwd_kernel(
    const __half* __restrict__ Q,    // [B, H, N, D]
    const __half* __restrict__ K,    // [B, H, N, D]
    const __half* __restrict__ V,    // [B, H, N, D]
    __half*       __restrict__ O,    // [B, H, N, D]
    float*        __restrict__ L,    // [B, H, N]  logsumexp,反向用
    int           N_CTX,
    int           H,                 // num_heads
    float         sm_scale,          // 1 / sqrt(D)
    bool          causal)
{
    // -------- block / thread 索引 --------
    const int q_tile = blockIdx.x;             // 当前 Q tile
    const int bh    = blockIdx.y;              // 扁平 (batch, head)
    const int tid   = threadIdx.x;             // 本 block 内的线程 id == 当前线程负责的本地 Q 行号

    // 当前 (batch, head) 在 [B*H, N, D] 上的起始偏移
    const long bh_offset = (long)bh * N_CTX * HEAD_DIM;
    const __half* Q_bh = Q + bh_offset;
    const __half* K_bh = K + bh_offset;
    const __half* V_bh = V + bh_offset;
    __half*       O_bh = O + bh_offset;

    // 这个线程负责的全局 Q 行
    const int q_row_global = q_tile * BLOCK_M + tid;
    const bool row_valid   = (q_row_global < N_CTX);

    // ============================================================
    //  Shared memory 申请 (动态)
    // ============================================================
    extern __shared__ __half smem[];
    __half* sQ = smem;
    __half* sK = sQ + BLOCK_M * HEAD_DIM;
    __half* sV = sK + BLOCK_N * HEAD_DIM;

    // ============================================================
    //  改动 ① + ②: Q row、acc、m_i、l_i 全部驻留寄存器
    //
    //  注: BLOCK_N + HEAD_DIM 都不大时可以放寄存器;若 BLOCK_N=128, HEAD_DIM=128,
    //  寄存器压力会爆,需要 spill 或换 WMMA 实现。这里 BLOCK_N=64, HEAD_DIM=64 刚好。
    // ============================================================
    float q_reg[HEAD_DIM];       // 我这一行的 Q,fp32
    float acc[HEAD_DIM];          // 未归一化输出累加器
    float m_i = -INFINITY;        // online softmax: running max
    float l_i = 0.0f;             // online softmax: running Σ exp

    // ============================================================
    //  Step 0: 协作加载 Q tile 到 sQ,然后把自己这一行复制到寄存器
    //  (改动 ①: 整个 kernel 周期里 Q 只读一次 HBM)
    // ============================================================
    if (row_valid) {
        const __half* q_src = Q_bh + q_row_global * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            sQ[tid * HEAD_DIM + d] = q_src[d];
        }
    } else {
        // tail 行 (N 不是 BLOCK_M 倍数时):填 0,避免污染 reduction
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            sQ[tid * HEAD_DIM + d] = __float2half(0.f);
        }
    }
    __syncthreads();

    // 把自己这行 Q 搬到寄存器,顺便乘 sm_scale (1/sqrt(D))
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q_reg[d] = __half2float(sQ[tid * HEAD_DIM + d]) * sm_scale;
        acc[d]   = 0.0f;
    }

    // ============================================================
    //  内层循环: 流式扫所有 KV tile
    //
    //  causal 时: 直接把循环上界压到 (q_tile+1)*BLOCK_M,
    //  跳过对角线之后的整块 tile (这是 V2 对 causal 提速的关键)
    // ============================================================
    const int n_end = causal ? min(N_CTX, (q_tile + 1) * BLOCK_M) : N_CTX;

    for (int kv_start = 0; kv_start < n_end; kv_start += BLOCK_N) {

        // -------- 协作加载 K tile + V tile 到 SRAM --------
        // BLOCK_M 个线程协作搬 BLOCK_N * HEAD_DIM 个 fp16 元素
        // 简化起见用标量 load;生产实现用 128-bit vector load 或 cp.async
        for (int idx = tid; idx < BLOCK_N * HEAD_DIM; idx += BLOCK_M) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            int g_row = kv_start + row;
            if (g_row < N_CTX) {
                sK[idx] = K_bh[g_row * HEAD_DIM + col];
                sV[idx] = V_bh[g_row * HEAD_DIM + col];
            } else {
                sK[idx] = __float2half(0.f);
                sV[idx] = __float2half(0.f);
            }
        }
        __syncthreads();

        // -------- 跳过完全无效的线程,但仍要参与下一轮 __syncthreads --------
        if (row_valid) {
            // ================================================
            //  Step A: 计算 s_j = q_reg · K_j (j in [0, BLOCK_N))
            //  同时找这一 tile 内的局部最大值
            // ================================================
            float s[BLOCK_N];
            float m_ij = m_i;                  // 包含旧 m_i,边算边更新
            for (int j = 0; j < BLOCK_N; ++j) {
                int kv_row = kv_start + j;
                // 越界 (N 尾巴) 或 causal 上三角 → 直接 -inf
                if (kv_row >= N_CTX ||
                    (causal && kv_row > q_row_global)) {
                    s[j] = -INFINITY;
                    continue;
                }
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += q_reg[d] * __half2float(sK[j * HEAD_DIM + d]);
                }
                s[j] = dot;
                if (dot > m_ij) m_ij = dot;
            }

            // ================================================
            //  Step B: online softmax 状态更新
            //    alpha = exp(m_i_old - m_i_new)
            //    l_i_new = l_i_old * alpha + Σ p_j
            //    acc 在 Step C 里 rescale + 加 P·V
            // ================================================
            float alpha = expf(m_i - m_ij);    // 旧累加器修正因子
            float l_new = l_i * alpha;
            float p[BLOCK_N];
            for (int j = 0; j < BLOCK_N; ++j) {
                p[j]   = (s[j] == -INFINITY) ? 0.f : expf(s[j] - m_ij);
                l_new += p[j];
            }
            m_i = m_ij;
            l_i = l_new;

            // ================================================
            //  Step C: acc = acc * alpha + Σ p_j · V_j
            //  改动 ②: 只 rescale + 加权和,不做除法
            //         (除法推迟到循环结束)
            // ================================================
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float new_d = acc[d] * alpha;
                for (int j = 0; j < BLOCK_N; ++j) {
                    new_d += p[j] * __half2float(sV[j * HEAD_DIM + d]);
                }
                acc[d] = new_d;
            }
        }

        // 进入下一轮之前等所有线程做完,避免 sK/sV 被提前覆盖
        __syncthreads();
    }

    // ============================================================
    //  改动 ②: 循环外唯一一次归一化
    // ============================================================
    if (row_valid) {
        __half* o_dst = O_bh + q_row_global * HEAD_DIM;
        float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            o_dst[d] = __float2half(acc[d] * inv_l);
        }
        // 保存 logsumexp = m + log(l) 给反向传播
        L[bh * N_CTX + q_row_global] = m_i + logf(l_i);
    }
}


// ============================================================================
//  Host launcher
// ============================================================================
void flash_attention_v2_forward(
    const __half* d_Q, const __half* d_K, const __half* d_V,
    __half* d_O, float* d_L,
    int B, int H, int N, int D,
    bool causal, cudaStream_t stream = 0)
{
    constexpr int BLOCK_M  = 64;
    constexpr int BLOCK_N  = 64;
    constexpr int HEAD_DIM = 64;
    assert(D == HEAD_DIM && "当前实现固定 head_dim=64,改 D 请同时改模板参数");

    float sm_scale = 1.0f / std::sqrt((float)D);

    dim3 grid((N + BLOCK_M - 1) / BLOCK_M, B * H);
    dim3 block(BLOCK_M);

    size_t smem_bytes = (BLOCK_M + BLOCK_N + BLOCK_N) * HEAD_DIM * sizeof(__half);

    flash_attn_v2_fwd_kernel<BLOCK_M, BLOCK_N, HEAD_DIM>
        <<<grid, block, smem_bytes, stream>>>(
            d_Q, d_K, d_V, d_O, d_L, N, H, sm_scale, causal);
}


// ============================================================================
//  CPU reference (慢速,只用来验证正确性)
// ============================================================================
void attention_reference_cpu(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int D, bool causal)
{
    float sm_scale = 1.0f / std::sqrt((float)D);
    std::vector<float> s(N);

    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
        long bh_off = ((long)b * H + h) * N * D;
        const float* Qbh = Q + bh_off;
        const float* Kbh = K + bh_off;
        const float* Vbh = V + bh_off;
        float* Obh = O + bh_off;

        for (int i = 0; i < N; ++i) {
            int jmax = causal ? (i + 1) : N;

            // s_j = Q_i · K_j * sm_scale
            float row_max = -INFINITY;
            for (int j = 0; j < jmax; ++j) {
                float dot = 0.f;
                for (int d = 0; d < D; ++d) {
                    dot += Qbh[i * D + d] * Kbh[j * D + d];
                }
                s[j] = dot * sm_scale;
                if (s[j] > row_max) row_max = s[j];
            }

            float sum = 0.f;
            for (int j = 0; j < jmax; ++j) {
                s[j] = std::exp(s[j] - row_max);
                sum += s[j];
            }
            float inv = 1.0f / sum;
            for (int j = 0; j < jmax; ++j) s[j] *= inv;

            for (int d = 0; d < D; ++d) {
                float o = 0.f;
                for (int j = 0; j < jmax; ++j) o += s[j] * Vbh[j * D + d];
                Obh[i * D + d] = o;
            }
        }
    }
}


// ============================================================================
//  main: 正确性 + 简单计时
// ============================================================================
int main() {
    const int B = 1, H = 4, N = 512, D = 64;
    const bool causal = true;

    size_t numel   = (size_t)B * H * N * D;
    size_t bytes_h = numel * sizeof(__half);

    // ---- host buffers ----
    std::vector<float>  Q(numel), K(numel), V(numel), O_ref(numel, 0.f);
    std::vector<__half> Q_h(numel), K_h(numel), V_h(numel), O_h(numel);

    std::srand(42);
    for (size_t i = 0; i < numel; ++i) {
        Q[i] = (std::rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        K[i] = (std::rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        V[i] = (std::rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        Q_h[i] = __float2half(Q[i]);
        K_h[i] = __float2half(K[i]);
        V_h[i] = __float2half(V[i]);
    }

    // ---- device buffers ----
    __half *dQ, *dK, *dV, *dO;
    float  *dL;
    CHECK_CUDA(cudaMalloc(&dQ, bytes_h));
    CHECK_CUDA(cudaMalloc(&dK, bytes_h));
    CHECK_CUDA(cudaMalloc(&dV, bytes_h));
    CHECK_CUDA(cudaMalloc(&dO, bytes_h));
    CHECK_CUDA(cudaMalloc(&dL, (size_t)B * H * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, Q_h.data(), bytes_h, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, K_h.data(), bytes_h, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, V_h.data(), bytes_h, cudaMemcpyHostToDevice));

    // ---- warm-up + 计时 ----
    flash_attention_v2_forward(dQ, dK, dV, dO, dL, B, H, N, D, causal);
    CHECK_CUDA(cudaDeviceSynchronize());

    const int iters = 50;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        flash_attention_v2_forward(dQ, dK, dV, dO, dL, B, H, N, D, causal);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // ---- 拉回结果,与 CPU reference 比 ----
    CHECK_CUDA(cudaMemcpy(O_h.data(), dO, bytes_h, cudaMemcpyDeviceToHost));
    attention_reference_cpu(Q.data(), K.data(), V.data(), O_ref.data(), B, H, N, D, causal);

    float max_err = 0.f, mean_err = 0.f;
    for (size_t i = 0; i < numel; ++i) {
        float e = std::fabs(__half2float(O_h[i]) - O_ref[i]);
        if (e > max_err) max_err = e;
        mean_err += e;
    }
    mean_err /= numel;

    printf("=== FlashAttention v2 CUDA forward ===\n");
    printf("  shape       : B=%d H=%d N=%d D=%d  causal=%s\n",
           B, H, N, D, causal ? "true" : "false");
    printf("  vs CPU ref  : max_err=%.4e  mean_err=%.4e\n", max_err, mean_err);
    printf("  kernel time : %.3f ms / iter (avg over %d iters)\n", ms, iters);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
    return 0;
}


/* ============================================================================
 *  生产实现还会做的优化 (本文件未实现,以保持算法清晰):
 *
 *  1. WMMA / MMA: 用 Tensor Core 算 Q·K^T 和 P·V
 *     - 当前每线程标量 dot,每行 BLOCK_N 个 dot,每个 dot HEAD_DIM 次 MAC
 *     - WMMA 一次算 16×16×16 矩阵块,吞吐高数十倍
 *     - 实现见 mma.h / cute / cutlass
 *
 *  2. cp.async (Ampere) / TMA (Hopper) 异步搬运
 *     - 当前 K/V load 是同步的,搬数据时计算单元在等
 *     - cp.async 让搬运后台进行,与上一轮计算重叠 (软件流水线)
 *
 *  3. Warp specialization (FA v3)
 *     - 把 warp 分成 "producer" (搬数据) 和 "consumer" (算 matmul)
 *     - 让 softmax 的 exp/max 与 matmul 完全异步,几乎不互相阻塞
 *
 *  4. 向量化 load/store
 *     - 用 float4 / __half2 一次搬 8/16 字节,带宽利用率才能跑满
 *
 *  5. Shared memory swizzling
 *     - 重排 sK/sV 在 SRAM 里的布局,避免 32-bank conflict
 *
 *  6. Backward pass
 *     - 用本 kernel 存下的 L (logsumexp) 重建 P,反向计算 dQ, dK, dV
 *     - 同样用 online softmax 思路,不物化中间矩阵
 * ============================================================================
 */
