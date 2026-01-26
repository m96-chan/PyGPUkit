/**
 * Flash Attention 3 - Main Header (Simplified Working Version)
 *
 * High-performance attention implementation for SM120 GPUs.
 *
 * Key features:
 * - Online softmax (O(n) memory)
 * - Vectorized loads (float4)
 * - Warp-level softmax with shuffle
 *
 * Reference: FlashAttention-3 (Dao et al., 2024)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

#include "fa3_traits.cuh"
#include "fa3_online_softmax.cuh"
#include "arch/fa3_mma_sm120.cuh"
#include "arch/fa3_mma_sm100.cuh"

namespace pygpukit {
namespace ops {
namespace nn {
namespace fa3 {

// =============================================================================
// Shared Memory Layout
// =============================================================================

template<typename Config>
struct SharedMemoryLayout {
    alignas(128) char smem_q[Config::SMEM_Q_SIZE];
    alignas(128) char smem_k[Config::TILE_KV * Config::HEAD_DIM * sizeof(__nv_bfloat16)];
    alignas(128) char smem_v[Config::TILE_KV * Config::HEAD_DIM * sizeof(__nv_bfloat16)];
    alignas(128) float smem_scores[Config::TILE_Q * Config::TILE_KV];
    alignas(128) __nv_bfloat16 smem_probs_bf16[Config::TILE_Q * Config::TILE_KV];  // For WMMA P@V
    alignas(16) float softmax_max[Config::TILE_Q];
    alignas(16) float softmax_sum[Config::TILE_Q];
};

// =============================================================================
// Vectorized Tile Load (float4 = 8 bf16 elements)
// =============================================================================

template<typename Element, int TILE, int HEAD_DIM>
__device__ __forceinline__ void load_tile_vectorized(
    Element* smem,
    const Element* gmem,
    int tile_start,
    int seq_len,
    int tid,
    int num_threads
) {
    // Each float4 loads 8 bf16 elements (16 bytes)
    constexpr int ELEMS_PER_VEC = 8;  // 8 bf16 = 16 bytes = float4
    constexpr int TOTAL_ELEMS = TILE * HEAD_DIM;
    constexpr int TOTAL_VECS = TOTAL_ELEMS / ELEMS_PER_VEC;

    float4* smem_f4 = reinterpret_cast<float4*>(smem);
    const float4* gmem_f4 = reinterpret_cast<const float4*>(gmem);

    for (int v = tid; v < TOTAL_VECS; v += num_threads) {
        // Calculate position in tile
        int elem_idx = v * ELEMS_PER_VEC;
        int pos = elem_idx / HEAD_DIM;
        int d = elem_idx % HEAD_DIM;

        if (tile_start + pos < seq_len) {
            // Vectorized load from global memory
            smem_f4[v] = gmem_f4[(tile_start + pos) * (HEAD_DIM / ELEMS_PER_VEC) + d / ELEMS_PER_VEC];
        } else {
            // Zero padding for out-of-bounds
            smem_f4[v] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// Simple fallback for non-vectorized loads
template<typename Element, int TILE, int HEAD_DIM>
__device__ __forceinline__ void load_tile_simple(
    Element* smem,
    const Element* gmem,
    int tile_start,
    int seq_len,
    int tid,
    int num_threads
) {
    constexpr int TOTAL_ELEMS = TILE * HEAD_DIM;
    for (int i = tid; i < TOTAL_ELEMS; i += num_threads) {
        int pos = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        if (tile_start + pos < seq_len) {
            smem[i] = gmem[(tile_start + pos) * HEAD_DIM + d];
        } else {
            smem[i] = Element(0);
        }
    }
}

// =============================================================================
// Simple Softmax (All Threads Participate)
// =============================================================================

template<int TILE_Q, int TILE_KV>
__device__ __forceinline__ void simple_row_softmax(
    float* scores,
    float* row_max,
    float* row_sum,
    int row_idx,
    int kv_len,
    bool is_first_tile,
    int tid,
    int num_threads
) {
    float* row = scores + row_idx * TILE_KV;

    // Find max (reduce across threads)
    float local_max = -INFINITY;
    for (int kv = tid; kv < kv_len; kv += num_threads) {
        local_max = fmaxf(local_max, row[kv]);
    }

    // Block-level max reduction via shared memory (use softmax_max as temp)
    __shared__ float temp_max[32];
    int lane = tid % 32;
    int warp = tid / 32;

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    if (lane == 0 && warp < 32) {
        temp_max[warp] = local_max;
    }
    __syncthreads();

    // Final reduction (first warp)
    float new_max;
    if (tid < 12) {  // NUM_WARPS = 12
        local_max = temp_max[tid];
    } else {
        local_max = -INFINITY;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    new_max = local_max;

    // Compute exp and sum
    float old_max = row_max[row_idx];
    float rescale = is_first_tile ? 1.0f : fa3_exp(old_max - new_max);

    float local_sum = 0.0f;
    for (int kv = tid; kv < kv_len; kv += num_threads) {
        float prob = fa3_exp(row[kv] - new_max);
        row[kv] = prob;
        local_sum += prob;
    }

    // Block-level sum reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    __shared__ float temp_sum[32];
    if (lane == 0 && warp < 32) {
        temp_sum[warp] = local_sum;
    }
    __syncthreads();

    if (tid < 12) {
        local_sum = temp_sum[tid];
    } else {
        local_sum = 0.0f;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    // Update state
    if (tid == 0) {
        row_max[row_idx] = new_max;
        if (is_first_tile) {
            row_sum[row_idx] = local_sum;
        } else {
            row_sum[row_idx] = row_sum[row_idx] * rescale + local_sum;
        }
    }
    __syncthreads();
}

// =============================================================================
// Parallel Softmax (Warp-per-Row)
// =============================================================================

template<int TILE_Q, int TILE_KV>
__device__ __forceinline__ void parallel_softmax_all_rows(
    float* scores,          // [TILE_Q, TILE_KV]
    float* row_max,         // [TILE_Q]
    float* row_sum,         // [TILE_Q]
    float* output_acc,      // [TILE_Q, HEAD_DIM] - to rescale
    int HEAD_DIM,
    int kv_len,
    bool is_first_tile,
    int tid,
    int num_threads
) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = num_threads / 32;  // 12 warps

    // Each warp handles ceil(TILE_Q / num_warps) rows
    // With TILE_Q=64 and 12 warps: 6 rows per warp (warp 0-9), 4 leftover for warp 10-11
    for (int q = warp_id; q < TILE_Q; q += num_warps) {
        float* row = scores + q * TILE_KV;

        // 1. Find max across this row (warp-level reduction)
        float local_max = -INFINITY;
        for (int kv = lane_id; kv < kv_len; kv += 32) {
            local_max = fmaxf(local_max, row[kv]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
        }
        float new_max = local_max;  // Now all lanes have the max

        // 2. Compute rescale factor for existing output
        float old_max = row_max[q];
        float rescale = is_first_tile ? 1.0f : fa3_exp(old_max - new_max);

        // 3. Compute exp(scores - new_max) and sum
        float local_sum = 0.0f;
        for (int kv = lane_id; kv < kv_len; kv += 32) {
            float prob = fa3_exp(row[kv] - new_max);
            row[kv] = prob;
            local_sum += prob;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        }

        // 4. Update softmax state (one thread per warp)
        if (lane_id == 0) {
            row_max[q] = new_max;
            if (is_first_tile) {
                row_sum[q] = local_sum;
            } else {
                row_sum[q] = row_sum[q] * rescale + local_sum;
            }
        }

        // 5. Rescale existing output for this row (if max changed)
        if (!is_first_tile && new_max > old_max) {
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                output_acc[q * HEAD_DIM + d] *= rescale;
            }
        }
    }
    __syncthreads();
}

// =============================================================================
// WMMA-based Score Computation (Tensor Core Optimized)
// =============================================================================

template<int TILE_Q, int TILE_KV, int HEAD_DIM>
__device__ __forceinline__ void compute_scores_wmma(
    float* scores,
    const __nv_bfloat16* smem_q,
    const __nv_bfloat16* smem_k,
    float scale,
    int tid,
    int num_threads
) {
    using namespace nvcuda::wmma;

    // WMMA tile size: 16x16x16 for bf16
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Number of tiles
    constexpr int M_TILES = TILE_Q / WMMA_M;    // 64/16 = 4
    constexpr int N_TILES = TILE_KV / WMMA_N;   // 64/16 = 4
    constexpr int K_TILES = HEAD_DIM / WMMA_K;  // 128/16 = 8

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = num_threads / 32;

    // Each warp processes some output tiles
    // Total tiles = M_TILES * N_TILES = 16
    // With 12 warps, some warps do 2 tiles, some do 1
    int tiles_per_warp = (M_TILES * N_TILES + num_warps - 1) / num_warps;

    for (int tile_idx = warp_id; tile_idx < M_TILES * N_TILES; tile_idx += num_warps) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        // Declare fragments
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        fill_fragment(acc_frag, 0.0f);

        // Accumulate over K dimension
        #pragma unroll
        for (int k = 0; k < K_TILES; ++k) {
            // Load Q tile: [m_tile*16 : (m_tile+1)*16, k*16 : (k+1)*16]
            const __nv_bfloat16* q_ptr = smem_q + m_tile * WMMA_M * HEAD_DIM + k * WMMA_K;
            load_matrix_sync(a_frag, q_ptr, HEAD_DIM);

            // Load K tile (transposed): K[n_tile*16 : (n_tile+1)*16, k*16 : (k+1)*16]
            // K is stored as [TILE_KV, HEAD_DIM], we want K^T
            // For col_major B, we load K directly (it's already in the right format for transpose)
            const __nv_bfloat16* k_ptr = smem_k + n_tile * WMMA_N * HEAD_DIM + k * WMMA_K;
            load_matrix_sync(b_frag, k_ptr, HEAD_DIM);

            // MMA: acc += Q * K^T
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        // Apply scale
        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            acc_frag.x[i] *= scale;
        }

        // Store result to scores: [m_tile*16:(m_tile+1)*16, n_tile*16:(n_tile+1)*16]
        float* out_ptr = scores + m_tile * WMMA_M * TILE_KV + n_tile * WMMA_N;
        store_matrix_sync(out_ptr, acc_frag, TILE_KV, mem_row_major);
    }

    __syncwarp();
}

// =============================================================================
// WMMA-based Output Computation (Tensor Core Optimized)
// =============================================================================

template<int TILE_Q, int TILE_KV, int HEAD_DIM>
__device__ __forceinline__ void compute_output_wmma(
    float* output,                     // [TILE_Q, HEAD_DIM] shared memory
    const float* probs,                // [TILE_Q, TILE_KV] softmax probabilities
    const __nv_bfloat16* smem_v,       // [TILE_KV, HEAD_DIM]
    __nv_bfloat16* probs_bf16_smem,    // Temp buffer for converted probs [TILE_Q, TILE_KV]
    int tid,
    int num_threads
) {
    using namespace nvcuda::wmma;

    // WMMA tile size: 16x16x16 for bf16
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Number of tiles for P @ V
    // P: [TILE_Q, TILE_KV] = [64, 64]
    // V: [TILE_KV, HEAD_DIM] = [64, 128]
    // Out: [TILE_Q, HEAD_DIM] = [64, 128]
    constexpr int M_TILES = TILE_Q / WMMA_M;       // 64/16 = 4
    constexpr int N_TILES = HEAD_DIM / WMMA_N;    // 128/16 = 8
    constexpr int K_TILES = TILE_KV / WMMA_K;     // 64/16 = 4

    int warp_id = tid / 32;
    int num_warps = num_threads / 32;

    // First, convert probs from FP32 to BF16 (all threads participate)
    for (int i = tid; i < TILE_Q * TILE_KV; i += num_threads) {
        probs_bf16_smem[i] = __float2bfloat16(probs[i]);
    }
    __syncthreads();

    // Each warp processes some output tiles
    // Total tiles = M_TILES * N_TILES = 4 * 8 = 32
    for (int tile_idx = warp_id; tile_idx < M_TILES * N_TILES; tile_idx += num_warps) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        // Declare fragments
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        // Load existing output accumulator
        float* out_ptr = output + m_tile * WMMA_M * HEAD_DIM + n_tile * WMMA_N;
        load_matrix_sync(acc_frag, out_ptr, HEAD_DIM, mem_row_major);

        // Accumulate over K dimension (TILE_KV)
        #pragma unroll
        for (int k = 0; k < K_TILES; ++k) {
            // Load P tile (probs): [m_tile*16:(m_tile+1)*16, k*16:(k+1)*16]
            const __nv_bfloat16* p_ptr = probs_bf16_smem + m_tile * WMMA_M * TILE_KV + k * WMMA_K;
            load_matrix_sync(a_frag, p_ptr, TILE_KV);

            // Load V tile: [k*16:(k+1)*16, n_tile*16:(n_tile+1)*16]
            const __nv_bfloat16* v_ptr = smem_v + k * WMMA_K * HEAD_DIM + n_tile * WMMA_N;
            load_matrix_sync(b_frag, v_ptr, HEAD_DIM);

            // MMA: acc += P * V
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        // Store result back to output
        store_matrix_sync(out_ptr, acc_frag, HEAD_DIM, mem_row_major);
    }

    __syncwarp();
}

// =============================================================================
// FA3 Forward Kernel - SM120 (Simplified, Scalar Version)
// =============================================================================

template<typename Element, int HEAD_DIM>
__global__ void __launch_bounds__(384, 1)
flash_attention_3_sm120_kernel(
    const Element* __restrict__ Q,
    const Element* __restrict__ K,
    const Element* __restrict__ V,
    Element* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    float scale,
    bool causal
) {
    using Config = TileConfig<Arch::SM120, Element>;

    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<SharedMemoryLayout<Config>*>(smem_raw);

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_tile_idx = blockIdx.x;

    const int q_start = q_tile_idx * Config::TILE_Q;
    if (q_start >= seq_q) return;
    const int q_end = min(q_start + Config::TILE_Q, seq_q);
    const int q_len = q_end - q_start;

    const int64_t q_offset = (int64_t)(batch_idx * num_heads + head_idx) * seq_q * HEAD_DIM;
    const int64_t kv_offset = (int64_t)(batch_idx * num_heads + head_idx) * seq_kv * HEAD_DIM;

    const Element* Q_ptr = Q + q_offset + q_start * HEAD_DIM;
    const Element* K_ptr = K + kv_offset;
    const Element* V_ptr = V + kv_offset;
    Element* O_ptr = output + q_offset + q_start * HEAD_DIM;

    // Use shared memory for output accumulator
    __shared__ float output_acc[Config::TILE_Q * HEAD_DIM];

    // Initialize output accumulator and softmax state
    for (int i = tid; i < Config::TILE_Q * HEAD_DIM; i += num_threads) {
        output_acc[i] = 0.0f;
    }
    if (tid < Config::TILE_Q) {
        smem.softmax_max[tid] = -INFINITY;
        smem.softmax_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Load Q tile (vectorized)
    load_tile_vectorized<Element, Config::TILE_Q, HEAD_DIM>(
        reinterpret_cast<Element*>(smem.smem_q),
        Q_ptr, 0, q_len, tid, num_threads
    );
    __syncthreads();

    // Main loop over KV tiles
    int num_kv_tiles = (seq_kv + Config::TILE_KV - 1) / Config::TILE_KV;
    if (causal) {
        int max_kv_pos = q_start + q_len - 1;
        num_kv_tiles = min(num_kv_tiles, (max_kv_pos + Config::TILE_KV) / Config::TILE_KV);
    }

    Element* smem_k = reinterpret_cast<Element*>(smem.smem_k);
    Element* smem_v = reinterpret_cast<Element*>(smem.smem_v);

    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * Config::TILE_KV;
        const int kv_end = min(kv_start + Config::TILE_KV, seq_kv);
        const int kv_len = kv_end - kv_start;

        // Load K and V (vectorized)
        load_tile_vectorized<Element, Config::TILE_KV, HEAD_DIM>(
            smem_k, K_ptr, kv_start, seq_kv, tid, num_threads);
        load_tile_vectorized<Element, Config::TILE_KV, HEAD_DIM>(
            smem_v, V_ptr, kv_start, seq_kv, tid, num_threads);
        __syncthreads();

        // Compute attention scores S = Q @ K^T (WMMA optimized)
        compute_scores_wmma<Config::TILE_Q, Config::TILE_KV, HEAD_DIM>(
            smem.smem_scores, reinterpret_cast<const Element*>(smem.smem_q),
            smem_k, scale, tid, num_threads
        );
        __syncthreads();

        // Apply causal mask
        if (causal) {
            for (int i = tid; i < Config::TILE_Q * Config::TILE_KV; i += num_threads) {
                int q_idx = i / Config::TILE_KV;
                int kv_idx = i % Config::TILE_KV;
                if (kv_start + kv_idx > q_start + q_idx) {
                    smem.smem_scores[i] = -INFINITY;
                }
            }
            __syncthreads();
        }

        // Online softmax per row (sequential, stable)
        for (int q = 0; q < q_len; ++q) {
            if (kv_tile > 0) {
                float old_max = smem.softmax_max[q];
                simple_row_softmax<Config::TILE_Q, Config::TILE_KV>(
                    smem.smem_scores, smem.softmax_max, smem.softmax_sum,
                    q, kv_len, false, tid, num_threads
                );
                float new_max = smem.softmax_max[q];
                if (new_max > old_max) {
                    float rescale = fa3_exp(old_max - new_max);
                    for (int d = tid; d < HEAD_DIM; d += num_threads) {
                        output_acc[q * HEAD_DIM + d] *= rescale;
                    }
                }
            } else {
                simple_row_softmax<Config::TILE_Q, Config::TILE_KV>(
                    smem.smem_scores, smem.softmax_max, smem.softmax_sum,
                    q, kv_len, true, tid, num_threads
                );
            }
            __syncthreads();
        }

        // Compute P @ V and accumulate (WMMA optimized)
        compute_output_wmma<Config::TILE_Q, Config::TILE_KV, HEAD_DIM>(
            output_acc, smem.smem_scores, smem_v, smem.smem_probs_bf16, tid, num_threads
        );
        __syncthreads();
    }

    // Epilogue: Normalize and write output
    for (int i = tid; i < q_len * HEAD_DIM; i += num_threads) {
        int q_idx = i / HEAD_DIM;
        int d_idx = i % HEAD_DIM;
        float sum = smem.softmax_sum[q_idx];
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        O_ptr[q_idx * HEAD_DIM + d_idx] = Element(output_acc[i] * inv_sum);
    }
}

// =============================================================================
// Kernel Launch Helpers
// =============================================================================

template<typename Element>
inline size_t get_fa3_smem_size(int head_dim) {
    if (head_dim == 128) {
        using Config = TileConfig<Arch::SM120, Element>;
        return sizeof(SharedMemoryLayout<Config>) + Config::TILE_Q * 128 * sizeof(float);
    }
    return 0;
}

template<typename Element>
inline cudaError_t launch_flash_attention_3(
    const Element* Q,
    const Element* K,
    const Element* V,
    Element* output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream
) {
    if (head_dim != 128) {
        return cudaErrorInvalidValue;
    }

    using Config = TileConfig<Arch::SM120, Element>;

    int q_tiles = (seq_q + Config::TILE_Q - 1) / Config::TILE_Q;
    dim3 grid(q_tiles, num_heads, batch_size);
    dim3 block(Config::NUM_WARPS * 32);

    size_t smem_size = sizeof(SharedMemoryLayout<Config>) + Config::TILE_Q * 128 * sizeof(float);

    cudaFuncSetAttribute(
        flash_attention_3_sm120_kernel<Element, 128>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    flash_attention_3_sm120_kernel<Element, 128><<<grid, block, smem_size, stream>>>(
        Q, K, V, output,
        batch_size, num_heads, seq_q, seq_kv,
        scale, causal
    );

    return cudaGetLastError();
}

}  // namespace fa3
}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
