/**
 * Flash Attention 4 - SM120 (RTX 5090 Blackwell GeForce)
 *
 * Key differences from FA3:
 * - Phase 1: BF16 baseline (same as FA3 TMA)
 * - Phase 2: NVFP4 Q@K^T with block_scale MMA
 * - Phase 3: Full NVFP4 pipeline (Q, K, V)
 *
 * SM120-specific features:
 * - mma.sync.aligned.block_scale.m64n64k64.f32.nvf4.nvf4
 * - No TMEM (use shared memory)
 * - ClusterShape 1x1x1 only
 * - 99KB shared memory limit
 *
 * Reference: PyGPUkit Issue #192
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>

#include "fa3_traits.cuh"
#include "fa3_online_softmax.cuh"
#include "../../common/tma_utils.cuh"
#include "../../common/warp_scheduler.cuh"
#include "../../common/pipeline.cuh"

// Only compile for SM120+
#if __CUDA_ARCH__ >= 1200 || !defined(__CUDA_ARCH__)

namespace pygpukit {
namespace ops {
namespace nn {
namespace fa4 {

// =============================================================================
// FA4 Configuration for SM120
// =============================================================================

struct FA4Config {
    // Tile sizes optimized for SM120 (99KB smem limit)
    static constexpr int TILE_Q = 64;     // Q tile
    static constexpr int TILE_KV = 64;    // KV tile (matches block_scale MMA)
    static constexpr int HEAD_DIM = 128;  // Standard head dimension
    static constexpr int NUM_STAGES = 2;  // Pipeline stages

    // Smem calculation for BF16 Phase 1:
    //   smem_q: 64 * 128 * 2 = 16KB
    //   smem_k: 2 * 64 * 128 * 2 = 32KB
    //   smem_v: 2 * 64 * 128 * 2 = 32KB
    //   smem_scores: 64 * 64 * 4 = 16KB
    //   output_acc: 64 * 128 * 4 = 32KB
    //   Total: ~128KB > 99KB limit!
    //
    // Reduce to fit:
    //   TILE_Q = 32 for Phase 1 (same as FA3)
    //   TILE_Q = 64 possible with NVFP4 (4-bit = 1/4 memory)

    // Warp configuration (same as FA3)
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;  // 384 threads

    // Element types
    using Element = __nv_bfloat16;
    using AccumType = float;
};

// Phase 1 config: BF16 baseline (same tiles as FA3)
struct FA4Phase1Config : FA4Config {
    static constexpr int TILE_Q = 32;  // Fit within 99KB
};

// Phase 2/3 config: NVFP4 (can use larger tiles due to 4-bit compression)
struct FA4Phase2Config : FA4Config {
    static constexpr int TILE_Q = 64;  // Larger tile possible with NVF4
    // NVFP4 smem calculation:
    //   smem_q: 64 * 128 * 0.5 = 4KB (NVF4)
    //   smem_k: 2 * 64 * 128 * 0.5 = 8KB (NVF4)
    //   smem_v: 2 * 64 * 128 * 0.5 = 8KB (NVF4)
    //   smem_scores: 64 * 64 * 4 = 16KB (FP32)
    //   output_acc: 64 * 128 * 4 = 32KB (FP32)
    //   scale_q: 64/32 * 4 = 8B
    //   scale_k: 2 * 64/32 * 4 = 16B
    //   Total: ~68KB < 99KB
};

// =============================================================================
// Shared Memory Layout
// =============================================================================

template<typename Element, int TILE_Q, int TILE_KV, int HEAD_DIM, int NUM_STAGES>
struct FA4SharedMemory {
    // Q buffer (single stage)
    alignas(1024) Element smem_q[TILE_Q * HEAD_DIM];

    // K/V buffers (multi-stage pipeline)
    alignas(1024) Element smem_k[NUM_STAGES][TILE_KV * HEAD_DIM];
    alignas(1024) Element smem_v[NUM_STAGES][TILE_KV * HEAD_DIM];

    // Scores/Probs union (saves memory)
    union alignas(128) {
        float smem_scores[TILE_Q * TILE_KV];
        Element smem_probs[TILE_Q * TILE_KV * 2];
    };

    // Softmax state
    alignas(16) float softmax_max[TILE_Q];
    alignas(16) float softmax_sum[TILE_Q];

    // Output accumulator
    alignas(128) float output_acc[TILE_Q * HEAD_DIM];

    // Pipeline barriers
    alignas(64) uint64_t barriers[NUM_STAGES];

    static constexpr size_t size() {
        return sizeof(FA4SharedMemory);
    }
};

// =============================================================================
// Phase 1: BF16 Baseline (Reuse FA3 Logic)
// =============================================================================

namespace phase1 {

using Config = FA4Phase1Config;
using SharedMemory = FA4SharedMemory<
    Config::Element,
    Config::TILE_Q,
    Config::TILE_KV,
    Config::HEAD_DIM,
    Config::NUM_STAGES
>;

// Consumer: Compute Q @ K^T scores using WMMA BF16
template<typename Cfg>
__device__ __forceinline__ void compute_scores_bf16(
    FA4SharedMemory<typename Cfg::Element, Cfg::TILE_Q, Cfg::TILE_KV, Cfg::HEAD_DIM, Cfg::NUM_STAGES>& smem,
    int stage,
    float scale,
    int tid
) {
    using namespace nvcuda::wmma;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int M_TILES = Cfg::TILE_Q / WMMA_M;
    constexpr int N_TILES = Cfg::TILE_KV / WMMA_N;
    constexpr int K_TILES = Cfg::HEAD_DIM / WMMA_K;

    int global_warp_id = tid / 32;
    int consumer_warp_idx = global_warp_id - Cfg::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) return;

    constexpr int num_consumer_warps = Cfg::NUM_CONSUMER_WARPS;

    for (int tile_idx = consumer_warp_idx; tile_idx < M_TILES * N_TILES; tile_idx += num_consumer_warps) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> q_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> k_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        fill_fragment(acc_frag, 0.0f);

        #pragma unroll
        for (int k = 0; k < K_TILES; ++k) {
            const __nv_bfloat16* q_ptr = smem.smem_q +
                m_tile * WMMA_M * Cfg::HEAD_DIM + k * WMMA_K;
            const __nv_bfloat16* k_ptr = smem.smem_k[stage] +
                n_tile * WMMA_N * Cfg::HEAD_DIM + k * WMMA_K;

            load_matrix_sync(q_frag, q_ptr, Cfg::HEAD_DIM);
            load_matrix_sync(k_frag, k_ptr, Cfg::HEAD_DIM);
            mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        }

        // Apply scale and store
        float* score_ptr = smem.smem_scores + m_tile * WMMA_M * Cfg::TILE_KV + n_tile * WMMA_N;
        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            acc_frag.x[i] *= scale;
        }
        store_matrix_sync(score_ptr, acc_frag, Cfg::TILE_KV, mem_row_major);
    }
}

// Consumer: Compute P @ V output using WMMA BF16
template<typename Cfg>
__device__ __forceinline__ void compute_output_bf16(
    FA4SharedMemory<typename Cfg::Element, Cfg::TILE_Q, Cfg::TILE_KV, Cfg::HEAD_DIM, Cfg::NUM_STAGES>& smem,
    int stage,
    int tid
) {
    using namespace nvcuda::wmma;
    using Element = typename Cfg::Element;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int M_TILES = Cfg::TILE_Q / WMMA_M;
    constexpr int N_TILES = Cfg::HEAD_DIM / WMMA_N;
    constexpr int K_TILES = Cfg::TILE_KV / WMMA_K;

    int global_warp_id = tid / 32;
    int consumer_warp_idx = global_warp_id - Cfg::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) return;

    constexpr int num_consumer_warps = Cfg::NUM_CONSUMER_WARPS;

    for (int tile_idx = consumer_warp_idx; tile_idx < M_TILES * N_TILES; tile_idx += num_consumer_warps) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> p_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> v_frag;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        float* out_ptr = smem.output_acc + m_tile * WMMA_M * Cfg::HEAD_DIM + n_tile * WMMA_N;
        load_matrix_sync(acc_frag, out_ptr, Cfg::HEAD_DIM, mem_row_major);

        #pragma unroll
        for (int k = 0; k < K_TILES; ++k) {
            const Element* p_ptr = smem.smem_probs +
                m_tile * WMMA_M * Cfg::TILE_KV + k * WMMA_K;
            const Element* v_ptr = smem.smem_v[stage] +
                k * WMMA_K * Cfg::HEAD_DIM + n_tile * WMMA_N;

            load_matrix_sync(p_frag, p_ptr, Cfg::TILE_KV);
            load_matrix_sync(v_frag, v_ptr, Cfg::HEAD_DIM);
            mma_sync(acc_frag, p_frag, v_frag, acc_frag);
        }

        store_matrix_sync(out_ptr, acc_frag, Cfg::HEAD_DIM, mem_row_major);
    }
}

// Two-phase softmax (same as FA3 to avoid union race)
template<typename Cfg>
__device__ __forceinline__ void softmax_phase1_read(
    FA4SharedMemory<typename Cfg::Element, Cfg::TILE_Q, Cfg::TILE_KV, Cfg::HEAD_DIM, Cfg::NUM_STAGES>& smem,
    int kv_tile,
    int kv_len,
    int q_len,
    int warp_id,
    int lane_id,
    float* reg_probs,
    float* reg_rescales,
    int* reg_q_indices,
    int& num_rows_handled
) {
    const int consumer_warp_idx = warp_id - Cfg::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) {
        num_rows_handled = 0;
        return;
    }

    const int num_consumer_warps = Cfg::NUM_CONSUMER_WARPS;
    constexpr int ELEMS_PER_LANE = (Cfg::TILE_KV + 31) / 32;

    num_rows_handled = 0;

    for (int q = consumer_warp_idx; q < q_len; q += num_consumer_warps) {
        float* row = smem.smem_scores + q * Cfg::TILE_KV;

        // Find row maximum
        float local_max = -INFINITY;
        #pragma unroll
        for (int kv = lane_id; kv < kv_len; kv += 32) {
            local_max = fmaxf(local_max, row[kv]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
        }

        reg_q_indices[num_rows_handled] = q;

        if (local_max == -INFINITY) {
            reg_rescales[num_rows_handled] = -INFINITY;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; ++e) {
                reg_probs[num_rows_handled * ELEMS_PER_LANE + e] = 0.0f;
            }
            num_rows_handled++;
            continue;
        }

        float old_max = smem.softmax_max[q];
        float new_max = fmaxf(old_max, local_max);
        float rescale = (kv_tile > 0) ? expf(old_max - new_max) : 1.0f;
        reg_rescales[num_rows_handled] = rescale;

        float local_sum = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int kv = lane_id + e * 32;
            float prob = 0.0f;
            if (kv < kv_len) {
                prob = expf(row[kv] - new_max);
                local_sum += prob;
            }
            reg_probs[num_rows_handled * ELEMS_PER_LANE + e] = prob;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        }

        if (lane_id == 0) {
            smem.softmax_max[q] = new_max;
            smem.softmax_sum[q] = smem.softmax_sum[q] * rescale + local_sum;
        }

        if (kv_tile > 0 && rescale != 1.0f) {
            #pragma unroll
            for (int d = lane_id; d < Cfg::HEAD_DIM; d += 32) {
                smem.output_acc[q * Cfg::HEAD_DIM + d] *= rescale;
            }
        }

        num_rows_handled++;
    }
}

template<typename Cfg>
__device__ __forceinline__ void softmax_phase2_write(
    FA4SharedMemory<typename Cfg::Element, Cfg::TILE_Q, Cfg::TILE_KV, Cfg::HEAD_DIM, Cfg::NUM_STAGES>& smem,
    int warp_id,
    int lane_id,
    const float* reg_probs,
    const int* reg_q_indices,
    int num_rows_handled
) {
    const int consumer_warp_idx = warp_id - Cfg::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) return;

    using Element = typename Cfg::Element;
    constexpr int ELEMS_PER_LANE = (Cfg::TILE_KV + 31) / 32;

    for (int r = 0; r < num_rows_handled; ++r) {
        int q = reg_q_indices[r];
        Element* prob_row = smem.smem_probs + q * Cfg::TILE_KV;

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int kv = lane_id + e * 32;
            if (kv < Cfg::TILE_KV) {
                prob_row[kv] = __float2bfloat16(reg_probs[r * ELEMS_PER_LANE + e]);
            }
        }
    }
}

}  // namespace phase1

// =============================================================================
// FA4 Phase 1 Kernel (BF16 Baseline)
// =============================================================================

template<typename Config>
__global__ void __launch_bounds__(Config::NUM_THREADS, 1)
fa4_kernel_phase1(
    const CUtensorMap* __restrict__ q_desc_ptr,
    const CUtensorMap* __restrict__ k_desc_ptr,
    const CUtensorMap* __restrict__ v_desc_ptr,
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
    auto& smem = *reinterpret_cast<phase1::SharedMemory*>(smem_raw);

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
    __threadfence_block();
    for (int i = tid; i < Config::TILE_Q * Config::HEAD_DIM; i += blockDim.x) {
        smem.output_acc[i] = 0.0f;
    }
    if (tid < Config::TILE_Q) {
        smem.softmax_max[tid] = -INFINITY;
        smem.softmax_sum[tid] = 0.0f;
    }
    __syncthreads();

    bool is_producer = is_producer_warp(Config::NUM_PRODUCER_WARPS);
    bool is_consumer = !is_producer;

    int num_kv_tiles = (seq_kv + Config::TILE_KV - 1) / Config::TILE_KV;
    if (causal) {
        int max_kv_pos = q_start + q_len - 1;
        num_kv_tiles = min(num_kv_tiles, (max_kv_pos + Config::TILE_KV) / Config::TILE_KV);
    }

    // Load Q tile
    if (is_producer && elect_one_per_warp()) {
        if (warp_id == 0) {
            barrier_arrive_expect_tx(smem.barriers[0],
                Config::TILE_Q * Config::HEAD_DIM * sizeof(Element));
            tma_load_3d(q_desc_ptr, smem.smem_q, &smem.barriers[0], 0, q_start, head_idx);
        }
    }
    __syncthreads();
    barrier_wait(smem.barriers[0], 0);

    // Reinitialize barriers for KV pipeline
    __syncthreads();
    if (tid == 0) {
        for (int s = 0; s < Config::NUM_STAGES; ++s) {
            barrier_invalidate(smem.barriers[s]);
            barrier_init(smem.barriers[s], 1);
        }
    }
    __threadfence_block();
    __syncthreads();

    // Pipeline state
    int read_stage = 0;
    int write_stage = 0;
    int phase = 0;

    // Prefill pipeline
    int prefill_tiles = min(Config::NUM_STAGES - 1, num_kv_tiles);
    for (int t = 0; t < prefill_tiles; ++t) {
        if (is_producer && warp_id == 0 && lane_id == 0) {
            int kv_start = t * Config::TILE_KV;
            uint32_t tx_bytes = Config::TILE_KV * Config::HEAD_DIM * sizeof(Element) * 2;
            barrier_arrive_expect_tx(smem.barriers[write_stage], tx_bytes);
            tma_load_3d(k_desc_ptr, smem.smem_k[write_stage], &smem.barriers[write_stage], 0, kv_start, head_idx);
            tma_load_3d(v_desc_ptr, smem.smem_v[write_stage], &smem.barriers[write_stage], 0, kv_start, head_idx);
        }
        write_stage = (write_stage + 1) % Config::NUM_STAGES;
    }

    // Main loop
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        barrier_wait(smem.barriers[read_stage], phase);
        __syncthreads();

        int kv_start = kv_tile * Config::TILE_KV;
        int kv_len = min(Config::TILE_KV, seq_kv - kv_start);

        // Compute Q @ K^T
        if (is_consumer) {
            phase1::compute_scores_bf16<Config>(smem, read_stage, scale, tid);
        }
        __syncthreads();

        // Causal mask
        if (causal) {
            for (int i = tid; i < Config::TILE_Q * Config::TILE_KV; i += blockDim.x) {
                int q_idx = i / Config::TILE_KV;
                int kv_idx = i % Config::TILE_KV;
                if (kv_start + kv_idx > q_start + q_idx) {
                    smem.smem_scores[i] = -INFINITY;
                }
            }
        }
        __syncthreads();

        // Two-phase softmax
        constexpr int MAX_ROWS_PER_WARP = (Config::TILE_Q + Config::NUM_CONSUMER_WARPS - 1) / Config::NUM_CONSUMER_WARPS;
        constexpr int ELEMS_PER_LANE = (Config::TILE_KV + 31) / 32;
        float reg_probs[MAX_ROWS_PER_WARP * ELEMS_PER_LANE];
        float reg_rescales[MAX_ROWS_PER_WARP];
        int reg_q_indices[MAX_ROWS_PER_WARP];
        int num_rows_handled = 0;

        phase1::softmax_phase1_read<Config>(
            smem, kv_tile, kv_len, q_len, warp_id, lane_id,
            reg_probs, reg_rescales, reg_q_indices, num_rows_handled);

        __syncthreads();

        phase1::softmax_phase2_write<Config>(
            smem, warp_id, lane_id,
            reg_probs, reg_q_indices, num_rows_handled);

        __syncthreads();

        // Compute P @ V
        if (is_consumer) {
            phase1::compute_output_bf16<Config>(smem, read_stage, tid);
        }

        // Prefetch next KV tile
        int next_tile = kv_tile + prefill_tiles;
        if (next_tile < num_kv_tiles && is_producer && warp_id == 0 && lane_id == 0) {
            int next_kv_start = next_tile * Config::TILE_KV;
            uint32_t tx_bytes = Config::TILE_KV * Config::HEAD_DIM * sizeof(Element) * 2;
            barrier_arrive_expect_tx(smem.barriers[write_stage], tx_bytes);
            tma_load_3d(k_desc_ptr, smem.smem_k[write_stage], &smem.barriers[write_stage], 0, next_kv_start, head_idx);
            tma_load_3d(v_desc_ptr, smem.smem_v[write_stage], &smem.barriers[write_stage], 0, next_kv_start, head_idx);
            write_stage = (write_stage + 1) % Config::NUM_STAGES;
        }

        read_stage = (read_stage + 1) % Config::NUM_STAGES;
        if (read_stage == 0) phase ^= 1;
        __syncthreads();
    }

    // Finalize: normalize and write output
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
// Host Launch Functions
// =============================================================================

inline cudaError_t launch_fa4_phase1(
    CUtensorMap* d_q_desc,
    CUtensorMap* d_k_desc,
    CUtensorMap* d_v_desc,
    __nv_bfloat16* output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    float scale,
    bool causal,
    cudaStream_t stream
) {
    using Config = FA4Phase1Config;

    int num_q_tiles = (seq_q + Config::TILE_Q - 1) / Config::TILE_Q;
    dim3 grid(num_q_tiles, num_heads, batch_size);
    dim3 block(Config::NUM_THREADS);

    size_t smem_size = phase1::SharedMemory::size();

    static bool smem_configured = false;
    if (!smem_configured) {
        cudaError_t attr_err = cudaFuncSetAttribute(
            fa4_kernel_phase1<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        if (attr_err != cudaSuccess) return attr_err;
        smem_configured = true;
    }

    fa4_kernel_phase1<Config><<<grid, block, smem_size, stream>>>(
        d_q_desc, d_k_desc, d_v_desc, output,
        batch_size, num_heads, seq_q, seq_kv,
        scale, causal
    );

    return cudaGetLastError();
}

// Timed version for benchmarking
inline cudaError_t launch_fa4_phase1_timed(
    CUtensorMap* d_q_desc,
    CUtensorMap* d_k_desc,
    CUtensorMap* d_v_desc,
    __nv_bfloat16* output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    float scale,
    bool causal,
    cudaStream_t stream,
    float* kernel_time_us
) {
    using Config = FA4Phase1Config;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_q_tiles = (seq_q + Config::TILE_Q - 1) / Config::TILE_Q;
    dim3 grid(num_q_tiles, num_heads, batch_size);
    dim3 block(Config::NUM_THREADS);
    size_t smem_size = phase1::SharedMemory::size();

    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(
            fa4_kernel_phase1<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        smem_configured = true;
    }

    cudaEventRecord(start, stream);

    fa4_kernel_phase1<Config><<<grid, block, smem_size, stream>>>(
        d_q_desc, d_k_desc, d_v_desc, output,
        batch_size, num_heads, seq_q, seq_kv,
        scale, causal
    );

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    *kernel_time_us = ms * 1000.0f;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaGetLastError();
}

// =============================================================================
// Version Info
// =============================================================================

inline const char* get_fa4_version() {
    return "FA4 SM120 v0.1.0 (Phase 1: BF16 Baseline)";
}

inline bool is_fa4_available() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return (props.major == 12);  // SM120/SM121
}

}  // namespace fa4
}  // namespace nn
}  // namespace ops
}  // namespace pygpukit

#endif  // __CUDA_ARCH__ >= 1200

