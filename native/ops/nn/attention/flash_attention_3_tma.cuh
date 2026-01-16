/**
 * Flash Attention 3 - TMA Optimized Version
 *
 * Uses TMA (Tensor Memory Accelerator) for async data loading.
 * Requires SM90+ (Hopper/Blackwell).
 *
 * Key features:
 * - TMA async bulk tensor loads
 * - Warp specialization (producer/consumer)
 * - Multi-stage pipeline
 * - mbarrier synchronization
 *
 * Reference: FlashAttention-3 (Dao et al., 2024)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

#include "fa3_traits.cuh"
#include "fa3_online_softmax.cuh"
#include "../../common/tma_utils.cuh"
#include "../../common/warp_scheduler.cuh"
#include "../../common/pipeline.cuh"

namespace pygpukit {
namespace ops {
namespace nn {
namespace fa3 {
namespace tma_kernel {

// =============================================================================
// TMA-Enabled Shared Memory Layout
// =============================================================================

template<typename Element, int TILE_Q, int TILE_KV, int HEAD_DIM, int NUM_STAGES>
struct TmaSharedMemory {
    // Q buffer (single stage - loaded once)
    alignas(1024) Element smem_q[TILE_Q * HEAD_DIM];

    // K/V buffers (multi-stage for pipelining)
    alignas(1024) Element smem_k[NUM_STAGES][TILE_KV * HEAD_DIM];
    alignas(1024) Element smem_v[NUM_STAGES][TILE_KV * HEAD_DIM];

    // Scores and output
    alignas(128) float smem_scores[TILE_Q * TILE_KV];
    alignas(128) Element smem_probs_bf16[TILE_Q * TILE_KV];

    // Softmax state
    alignas(16) float softmax_max[TILE_Q];
    alignas(16) float softmax_sum[TILE_Q];

    // Output accumulator
    alignas(128) float output_acc[TILE_Q * HEAD_DIM];

    // Pipeline barriers (one per stage)
    alignas(8) uint64_t barriers[NUM_STAGES];

    static constexpr size_t size() {
        return sizeof(TmaSharedMemory);
    }
};

// =============================================================================
// TMA Kernel Configuration
// =============================================================================

template<int SM_VERSION>
struct TmaFA3Config {
    // Default configuration for SM120
    static constexpr int TILE_Q = 64;
    static constexpr int TILE_KV = 64;
    static constexpr int HEAD_DIM = 128;
    static constexpr int NUM_STAGES = 4;

    // Warp configuration
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;

    // TMA tile sizes (must align to 128B for swizzle)
    static constexpr int TMA_TILE_D = HEAD_DIM;  // Full head dimension
    static constexpr int TMA_TILE_S = TILE_KV;   // Sequence tile

    using Element = __nv_bfloat16;
    using SharedMemory = TmaSharedMemory<Element, TILE_Q, TILE_KV, HEAD_DIM, NUM_STAGES>;
};

// =============================================================================
// Producer Warp Functions
// =============================================================================

template<typename Config>
__device__ __forceinline__ void producer_load_q_tile(
    typename Config::SharedMemory& smem,
    const CUtensorMap* q_desc,
    int head_idx,
    int q_start
) {
    using namespace pygpukit::ops::tma;

    // Only elected thread issues TMA load
    if (scheduler::elect_one_per_warp()) {
        // Initialize barrier for Q (single load)
        barrier_init(smem.barriers[0], 1);
        barrier_arrive_expect_tx(smem.barriers[0],
            Config::TILE_Q * Config::HEAD_DIM * sizeof(typename Config::Element));

        // Issue TMA load for Q tile
        tma_load_2d(
            q_desc,
            smem.smem_q,
            &smem.barriers[0],
            q_start,       // Sequence coordinate
            0              // Head dimension coordinate (start at 0)
        );
    }
}

template<typename Config>
__device__ __forceinline__ void producer_load_kv_tile(
    typename Config::SharedMemory& smem,
    const CUtensorMap* k_desc,
    const CUtensorMap* v_desc,
    int stage,
    int kv_start
) {
    using namespace pygpukit::ops::tma;

    int producer_warp = scheduler::get_producer_warp_idx(Config::NUM_PRODUCER_WARPS);
    if (producer_warp < 0) return;  // Not a producer

    // Only elected thread per warp issues loads
    if (scheduler::elect_one_per_warp()) {
        uint32_t tx_bytes = Config::TILE_KV * Config::HEAD_DIM * sizeof(typename Config::Element);

        // Initialize barrier for this stage
        if (producer_warp == 0) {
            barrier_arrive_expect_tx(smem.barriers[stage], tx_bytes * 2);  // K + V
        }

        // Divide work among producer warps
        // Warp 0-1: Load K, Warp 2-3: Load V
        if (producer_warp < 2) {
            tma_load_2d(
                k_desc,
                smem.smem_k[stage],
                &smem.barriers[stage],
                kv_start,
                0
            );
        } else {
            tma_load_2d(
                v_desc,
                smem.smem_v[stage],
                &smem.barriers[stage],
                kv_start,
                0
            );
        }
    }
}

// =============================================================================
// Consumer Warp Functions
// =============================================================================

template<typename Config>
__device__ __forceinline__ void consumer_compute_scores(
    typename Config::SharedMemory& smem,
    int stage,
    float scale,
    int tid,
    int num_threads
) {
    using namespace nvcuda::wmma;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int M_TILES = Config::TILE_Q / WMMA_M;
    constexpr int N_TILES = Config::TILE_KV / WMMA_N;
    constexpr int K_TILES = Config::HEAD_DIM / WMMA_K;

    int warp_id = tid / 32;
    int num_warps = num_threads / 32;

    // Each warp handles some score tiles
    for (int tile_idx = warp_id; tile_idx < M_TILES * N_TILES; tile_idx += num_warps) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> q_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> k_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        fill_fragment(acc_frag, 0.0f);

        #pragma unroll
        for (int k = 0; k < K_TILES; ++k) {
            const __nv_bfloat16* q_ptr = smem.smem_q +
                m_tile * WMMA_M * Config::HEAD_DIM + k * WMMA_K;
            const __nv_bfloat16* k_ptr = smem.smem_k[stage] +
                n_tile * WMMA_N * Config::HEAD_DIM + k * WMMA_K;

            load_matrix_sync(q_frag, q_ptr, Config::HEAD_DIM);
            load_matrix_sync(k_frag, k_ptr, Config::HEAD_DIM);
            mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        }

        // Apply scale and store
        float* score_ptr = smem.smem_scores + m_tile * WMMA_M * Config::TILE_KV + n_tile * WMMA_N;
        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            acc_frag.x[i] *= scale;
        }
        store_matrix_sync(score_ptr, acc_frag, Config::TILE_KV, mem_row_major);
    }
}

template<typename Config>
__device__ __forceinline__ void consumer_compute_output(
    typename Config::SharedMemory& smem,
    int stage,
    int tid,
    int num_threads
) {
    using namespace nvcuda::wmma;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int M_TILES = Config::TILE_Q / WMMA_M;
    constexpr int N_TILES = Config::HEAD_DIM / WMMA_N;
    constexpr int K_TILES = Config::TILE_KV / WMMA_K;

    int warp_id = tid / 32;
    int num_warps = num_threads / 32;

    // Convert probs to BF16
    for (int i = tid; i < Config::TILE_Q * Config::TILE_KV; i += num_threads) {
        smem.smem_probs_bf16[i] = __float2bfloat16(smem.smem_scores[i]);
    }
    __syncthreads();

    // Each warp handles some output tiles
    for (int tile_idx = warp_id; tile_idx < M_TILES * N_TILES; tile_idx += num_warps) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> p_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> v_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        // Load existing accumulator
        float* out_ptr = smem.output_acc + m_tile * WMMA_M * Config::HEAD_DIM + n_tile * WMMA_N;
        load_matrix_sync(acc_frag, out_ptr, Config::HEAD_DIM, mem_row_major);

        #pragma unroll
        for (int k = 0; k < K_TILES; ++k) {
            const __nv_bfloat16* p_ptr = smem.smem_probs_bf16 +
                m_tile * WMMA_M * Config::TILE_KV + k * WMMA_K;
            const __nv_bfloat16* v_ptr = smem.smem_v[stage] +
                k * WMMA_K * Config::HEAD_DIM + n_tile * WMMA_N;

            load_matrix_sync(p_frag, p_ptr, Config::TILE_KV);
            load_matrix_sync(v_frag, v_ptr, Config::HEAD_DIM);
            mma_sync(acc_frag, p_frag, v_frag, acc_frag);
        }

        store_matrix_sync(out_ptr, acc_frag, Config::HEAD_DIM, mem_row_major);
    }
}

// =============================================================================
// TMA-Enabled FA3 Kernel
// =============================================================================

template<typename Config>
__global__ void __launch_bounds__(Config::NUM_THREADS, 1)
flash_attention_3_tma_kernel(
    const __grid_constant__ CUtensorMap q_desc,
    const __grid_constant__ CUtensorMap k_desc,
    const __grid_constant__ CUtensorMap v_desc,
    typename Config::Element* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    float scale,
    bool causal
) {
    using namespace pygpukit::ops::tma;
    using namespace pygpukit::ops::scheduler;
    using Element = typename Config::Element;

    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<typename Config::SharedMemory*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_tile_idx = blockIdx.x;

    const int q_start = q_tile_idx * Config::TILE_Q;
    if (q_start >= seq_q) return;
    const int q_len = min(Config::TILE_Q, seq_q - q_start);

    // Initialize shared memory
    if (tid == 0) {
        for (int s = 0; s < Config::NUM_STAGES; ++s) {
            barrier_init(smem.barriers[s], 1);
        }
    }
    for (int i = tid; i < Config::TILE_Q * Config::HEAD_DIM; i += blockDim.x) {
        smem.output_acc[i] = 0.0f;
    }
    if (tid < Config::TILE_Q) {
        smem.softmax_max[tid] = -INFINITY;
        smem.softmax_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Determine warp role
    bool is_producer = is_producer_warp(Config::NUM_PRODUCER_WARPS);
    bool is_consumer = !is_producer;

    // Calculate number of KV tiles
    int num_kv_tiles = (seq_kv + Config::TILE_KV - 1) / Config::TILE_KV;
    if (causal) {
        int max_kv_pos = q_start + q_len - 1;
        num_kv_tiles = min(num_kv_tiles, (max_kv_pos + Config::TILE_KV) / Config::TILE_KV);
    }

    // === Producer: Load Q tile (all producer warps) ===
    if (is_producer && elect_one_per_warp()) {
        if (warp_id == 0) {
            barrier_arrive_expect_tx(smem.barriers[0],
                Config::TILE_Q * Config::HEAD_DIM * sizeof(Element));
            // 3D coordinates: (dim0=0, dim1=q_start, dim2=head_idx)
            tma_load_3d(&q_desc, smem.smem_q, &smem.barriers[0], 0, q_start, head_idx);
        }
    }

    // Wait for Q to be ready
    barrier_wait(smem.barriers[0], 0);

    // === Main loop: Pipeline K/V loading with computation ===
    int read_stage = 0;
    int write_stage = 0;
    int phase = 0;

    // Prefill pipeline
    int prefill_tiles = min(Config::NUM_STAGES - 1, num_kv_tiles);
    for (int t = 0; t < prefill_tiles; ++t) {
        if (is_producer && elect_one_per_warp()) {
            int kv_start = t * Config::TILE_KV;
            uint32_t tx_bytes = Config::TILE_KV * Config::HEAD_DIM * sizeof(Element) * 2;

            if (warp_id == 0) {
                barrier_arrive_expect_tx(smem.barriers[write_stage], tx_bytes);
            }

            // Producer warp 0-1: K, warp 2-3: V
            // 3D coordinates: (dim0=0, dim1=kv_start, dim2=head_idx)
            if (warp_id < 2) {
                tma_load_3d(&k_desc, smem.smem_k[write_stage], &smem.barriers[write_stage], 0, kv_start, head_idx);
            } else if (warp_id < 4) {
                tma_load_3d(&v_desc, smem.smem_v[write_stage], &smem.barriers[write_stage], 0, kv_start, head_idx);
            }
        }
        write_stage = (write_stage + 1) % Config::NUM_STAGES;
    }

    // Main loop
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        // Wait for current KV tile
        barrier_wait(smem.barriers[read_stage], phase);
        __syncthreads();

        int kv_start = kv_tile * Config::TILE_KV;
        int kv_len = min(Config::TILE_KV, seq_kv - kv_start);

        // === Consumer: Compute attention ===
        if (is_consumer) {
            // Compute scores: Q @ K^T
            consumer_compute_scores<Config>(smem, read_stage, scale, tid, Config::NUM_THREADS);
            __syncthreads();

            // Apply causal mask
            if (causal) {
                for (int i = tid; i < Config::TILE_Q * Config::TILE_KV; i += blockDim.x) {
                    int q_idx = i / Config::TILE_KV;
                    int kv_idx = i % Config::TILE_KV;
                    if (kv_start + kv_idx > q_start + q_idx) {
                        smem.smem_scores[i] = -INFINITY;
                    }
                }
                __syncthreads();
            }

            // Online softmax (simplified - all threads)
            for (int q = 0; q < q_len; ++q) {
                float* row = smem.smem_scores + q * Config::TILE_KV;

                // Find max
                float local_max = -INFINITY;
                for (int kv = lane_id; kv < kv_len; kv += 32) {
                    local_max = fmaxf(local_max, row[kv]);
                }
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
                }

                float old_max = smem.softmax_max[q];
                float new_max = fmaxf(old_max, local_max);
                float rescale = (kv_tile > 0) ? expf(old_max - new_max) : 1.0f;

                // Compute exp and sum
                float local_sum = 0.0f;
                for (int kv = lane_id; kv < kv_len; kv += 32) {
                    float prob = expf(row[kv] - new_max);
                    row[kv] = prob;
                    local_sum += prob;
                }
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
                }

                // Update state (lane 0 only)
                if (lane_id == 0) {
                    smem.softmax_max[q] = new_max;
                    smem.softmax_sum[q] = smem.softmax_sum[q] * rescale + local_sum;
                }

                // Rescale output accumulator
                if (kv_tile > 0 && rescale != 1.0f) {
                    for (int d = lane_id; d < Config::HEAD_DIM; d += 32) {
                        smem.output_acc[q * Config::HEAD_DIM + d] *= rescale;
                    }
                }
            }
            __syncthreads();

            // Compute output: P @ V
            consumer_compute_output<Config>(smem, read_stage, tid, Config::NUM_THREADS);
        }

        // === Producer: Prefetch next KV tile ===
        int next_tile = kv_tile + prefill_tiles;
        if (next_tile < num_kv_tiles && is_producer && elect_one_per_warp()) {
            int next_kv_start = next_tile * Config::TILE_KV;
            uint32_t tx_bytes = Config::TILE_KV * Config::HEAD_DIM * sizeof(Element) * 2;

            if (warp_id == 0) {
                barrier_arrive_expect_tx(smem.barriers[write_stage], tx_bytes);
            }

            // 3D coordinates: (dim0=0, dim1=next_kv_start, dim2=head_idx)
            if (warp_id < 2) {
                tma_load_3d(&k_desc, smem.smem_k[write_stage], &smem.barriers[write_stage], 0, next_kv_start, head_idx);
            } else if (warp_id < 4) {
                tma_load_3d(&v_desc, smem.smem_v[write_stage], &smem.barriers[write_stage], 0, next_kv_start, head_idx);
            }

            write_stage = (write_stage + 1) % Config::NUM_STAGES;
        }

        // Advance read stage
        read_stage = (read_stage + 1) % Config::NUM_STAGES;
        if (read_stage == 0) phase ^= 1;

        __syncthreads();
    }

    // === Finalize: Normalize and write output ===
    __syncthreads();

    const int64_t out_offset = (int64_t)(batch_idx * num_heads + head_idx) * seq_q * Config::HEAD_DIM;
    Element* O_ptr = output + out_offset + q_start * Config::HEAD_DIM;

    for (int i = tid; i < q_len * Config::HEAD_DIM; i += blockDim.x) {
        int q = i / Config::HEAD_DIM;
        int d = i % Config::HEAD_DIM;
        float val = smem.output_acc[i] / smem.softmax_sum[q];
        O_ptr[q * Config::HEAD_DIM + d] = __float2bfloat16(val);
    }
}

// =============================================================================
// Host-Side Launch Helper
// =============================================================================

template<typename Config>
inline cudaError_t launch_flash_attention_3_tma(
    const CUtensorMap& q_desc,
    const CUtensorMap& k_desc,
    const CUtensorMap& v_desc,
    typename Config::Element* output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    float scale,
    bool causal,
    cudaStream_t stream
) {
    int num_q_tiles = (seq_q + Config::TILE_Q - 1) / Config::TILE_Q;
    dim3 grid(num_q_tiles, num_heads, batch_size);
    dim3 block(Config::NUM_THREADS);

    size_t smem_size = Config::SharedMemory::size();

    // Set shared memory configuration
    cudaFuncSetAttribute(
        flash_attention_3_tma_kernel<Config>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    flash_attention_3_tma_kernel<Config><<<grid, block, smem_size, stream>>>(
        q_desc, k_desc, v_desc, output,
        batch_size, num_heads, seq_q, seq_kv,
        scale, causal
    );

    return cudaGetLastError();
}

}  // namespace tma_kernel
}  // namespace fa3
}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
