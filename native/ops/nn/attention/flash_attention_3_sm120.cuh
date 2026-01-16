/**
 * Flash Attention 3 - SM120 (RTX 5090 Blackwell) Tuned Version
 *
 * SM120-specific optimizations:
 * - 128KB shared memory (vs generic 99KB limit)
 * - Larger tile sizes for better compute utilization
 * - Swizzled shared memory layout for bank conflict avoidance
 * - Tuned warp specialization for SM120 scheduler
 *
 * Baseline: FA3 TMA at 51.97 TFLOPS
 * Target: 60+ TFLOPS
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
namespace fa3_sm120 {

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

    // Scores/Probs union - saves 8KB by reusing same memory
    // smem_scores used during softmax computation (float precision)
    // smem_probs used during P@V matmul (BF16 for WMMA)
    // These are NEVER used simultaneously - conversion happens between phases
    union alignas(128) {
        float smem_scores[TILE_Q * TILE_KV];     // 16KB - softmax phase
        Element smem_probs[TILE_Q * TILE_KV * 2]; // Padded to same size for union
    };

    // Softmax state
    alignas(16) float softmax_max[TILE_Q];
    alignas(16) float softmax_sum[TILE_Q];

    // Output accumulator
    alignas(128) float output_acc[TILE_Q * HEAD_DIM];

    // Pipeline barriers (one per stage)
    // mbarrier must be 64-byte aligned for optimal performance
    alignas(64) uint64_t barriers[NUM_STAGES];

    static constexpr size_t size() {
        return sizeof(TmaSharedMemory);
    }
};

// =============================================================================
// SM120 Tuning Configurations
// =============================================================================

// Version 0: Baseline (TILE_Q=32, TILE_KV=64, 2-stage, 4+8 warps) - ~96KB
// Version 1: Smaller tiles (TILE_KV=32) for better occupancy - ~60KB
// Version 2: 3-stage pipeline with smaller tiles (TILE_KV=32) - ~76KB
// Version 3: More consumer warps (2+10) - same ~96KB
// Version 4: Even more consumer warps (4+12) - same ~96KB

template<int VERSION = 0>
struct SM120Config;

// Version 0: Baseline configuration (reference)
template<>
struct SM120Config<0> {
    static constexpr int TILE_Q = 32;
    static constexpr int TILE_KV = 64;
    static constexpr int HEAD_DIM = 128;
    static constexpr int NUM_STAGES = 2;
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int TMA_TILE_D = HEAD_DIM;
    static constexpr int TMA_TILE_S = TILE_KV;
    // Smem: ~96KB
    using Element = __nv_bfloat16;
    using SharedMemory = TmaSharedMemory<Element, TILE_Q, TILE_KV, HEAD_DIM, NUM_STAGES>;
};

// Version 1: Smaller K/V tiles for better occupancy
// TILE_KV=32 reduces smem, allows more concurrent blocks
template<>
struct SM120Config<1> {
    static constexpr int TILE_Q = 32;
    static constexpr int TILE_KV = 32;   // Halved from 64
    static constexpr int HEAD_DIM = 128;
    static constexpr int NUM_STAGES = 2;
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int TMA_TILE_D = HEAD_DIM;
    static constexpr int TMA_TILE_S = TILE_KV;
    // Smem: smem_q=8KB, smem_k/v=16KB each, scores=4KB, output=16KB = ~60KB
    using Element = __nv_bfloat16;
    using SharedMemory = TmaSharedMemory<Element, TILE_Q, TILE_KV, HEAD_DIM, NUM_STAGES>;
};

// Version 2: 3-stage pipeline with smaller tiles
// 3-stage requires smaller TILE_KV to stay within smem limit
template<>
struct SM120Config<2> {
    static constexpr int TILE_Q = 32;
    static constexpr int TILE_KV = 32;   // Reduced to fit 3-stage
    static constexpr int HEAD_DIM = 128;
    static constexpr int NUM_STAGES = 3;  // 3-stage for better latency hiding
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int TMA_TILE_D = HEAD_DIM;
    static constexpr int TMA_TILE_S = TILE_KV;
    // Smem: smem_q=8KB, smem_k/v=24KB each (3*32*128*2), scores=4KB, output=16KB = ~76KB
    using Element = __nv_bfloat16;
    using SharedMemory = TmaSharedMemory<Element, TILE_Q, TILE_KV, HEAD_DIM, NUM_STAGES>;
};

// Version 3: More consumer warps (2 producer, 10 consumer)
template<>
struct SM120Config<3> {
    static constexpr int TILE_Q = 32;
    static constexpr int TILE_KV = 64;
    static constexpr int HEAD_DIM = 128;
    static constexpr int NUM_STAGES = 2;
    static constexpr int NUM_PRODUCER_WARPS = 2;   // Reduced from 4
    static constexpr int NUM_CONSUMER_WARPS = 10;  // Increased from 8
    static constexpr int NUM_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int TMA_TILE_D = HEAD_DIM;
    static constexpr int TMA_TILE_S = TILE_KV;
    using Element = __nv_bfloat16;
    using SharedMemory = TmaSharedMemory<Element, TILE_Q, TILE_KV, HEAD_DIM, NUM_STAGES>;
};

// Version 4: More warps (16 total) for better compute throughput
// More consumer warps to maximize MMA throughput
template<>
struct SM120Config<4> {
    static constexpr int TILE_Q = 32;
    static constexpr int TILE_KV = 64;
    static constexpr int HEAD_DIM = 128;
    static constexpr int NUM_STAGES = 2;
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int NUM_CONSUMER_WARPS = 12;  // Increased from 8
    static constexpr int NUM_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int TMA_TILE_D = HEAD_DIM;
    static constexpr int TMA_TILE_S = TILE_KV;
    // Smem: same as V0 ~96KB, just more compute warps
    using Element = __nv_bfloat16;
    using SharedMemory = TmaSharedMemory<Element, TILE_Q, TILE_KV, HEAD_DIM, NUM_STAGES>;
};

// Alias for backward compatibility
template<int SM_VERSION>
using TmaFA3Config = SM120Config<0>;

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

    // Use consumer-relative warp index (0-7) instead of global warp_id (4-11)
    // This ensures all tiles 0 to M_TILES*N_TILES-1 are covered
    int global_warp_id = tid / 32;
    int consumer_warp_idx = global_warp_id - Config::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) return;  // Producer warps should not call this

    constexpr int num_consumer_warps = Config::NUM_CONSUMER_WARPS;

    // Each consumer warp handles tiles in round-robin fashion
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

// =============================================================================
// Warp-Parallel Online Softmax
// =============================================================================
// Each consumer warp handles DIFFERENT q rows in parallel.
// NO __syncthreads() inside - purely warp-synchronous.
// This is the key optimization: 8 consumer warps process 8 rows simultaneously.

// =============================================================================
// Two-Phase Softmax to Avoid Union Race Condition
// =============================================================================
// CRITICAL: smem_scores (float) and smem_probs (bf16) share memory via union.
// When multiple warps process different Q rows in parallel:
// - Warp A reads smem_scores[row_A]
// - Warp B writes smem_probs[row_B]
// These can alias! E.g., smem_probs[row_B] bytes overlap smem_scores[row_A] bytes.
//
// FIX: Split into two phases:
// Phase 1: ALL warps read scores, compute probs, store to REGISTERS
// Phase 2: After sync, ALL warps write probs from registers to smem
//
// Register budget: 4 rows/warp * 2 elements/lane = 8 floats/lane = 32 bytes

template<typename Config>
__device__ __forceinline__ void consumer_softmax_phase1_read(
    typename Config::SharedMemory& smem,
    int kv_tile,
    int kv_len,
    int q_len,
    int warp_id,
    int lane_id,
    // Output: per-lane register storage for probs (max 4 rows * 2 elements = 8)
    float* reg_probs,      // [MAX_ROWS_PER_WARP * ELEMS_PER_LANE]
    float* reg_rescales,   // [MAX_ROWS_PER_WARP] - rescale factors per row
    int* reg_q_indices,    // [MAX_ROWS_PER_WARP] - which q rows this warp handles
    int& num_rows_handled
) {
    // Consumer warp index: warps 0-3 are producers, 4-11 are consumers
    const int consumer_warp_idx = warp_id - Config::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) {
        num_rows_handled = 0;
        return;
    }

    const int num_consumer_warps = Config::NUM_CONSUMER_WARPS;
    constexpr int ELEMS_PER_LANE = (Config::TILE_KV + 31) / 32;  // 2 for TILE_KV=64

    num_rows_handled = 0;

    // Each consumer warp handles different q rows in round-robin fashion
    for (int q = consumer_warp_idx; q < q_len; q += num_consumer_warps) {
        float* row = smem.smem_scores + q * Config::TILE_KV;

        // === Step 1: Find row maximum (warp-level reduction) ===
        float local_max = -INFINITY;
        #pragma unroll
        for (int kv = lane_id; kv < kv_len; kv += 32) {
            local_max = fmaxf(local_max, row[kv]);
        }
        // Warp-level max reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
        }

        // Store which q row we're handling
        reg_q_indices[num_rows_handled] = q;

        // === Handle fully masked rows ===
        if (local_max == -INFINITY) {
            // Mark with special rescale value to indicate zero-fill in phase 2
            reg_rescales[num_rows_handled] = -INFINITY;
            // Store zeros to registers
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; ++e) {
                reg_probs[num_rows_handled * ELEMS_PER_LANE + e] = 0.0f;
            }
            num_rows_handled++;
            continue;
        }

        // === Step 2: Online softmax update ===
        float old_max = smem.softmax_max[q];
        float new_max = fmaxf(old_max, local_max);
        float rescale = (kv_tile > 0) ? expf(old_max - new_max) : 1.0f;
        reg_rescales[num_rows_handled] = rescale;

        // === Step 3: Compute exp(x - new_max) and sum, store probs to registers ===
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

        // Warp-level sum reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        }

        // === Step 4: Update softmax state (lane 0 only) ===
        if (lane_id == 0) {
            smem.softmax_max[q] = new_max;
            smem.softmax_sum[q] = smem.softmax_sum[q] * rescale + local_sum;
        }

        // === Step 5: Rescale output accumulator if needed ===
        if (kv_tile > 0 && rescale != 1.0f) {
            #pragma unroll
            for (int d = lane_id; d < Config::HEAD_DIM; d += 32) {
                smem.output_acc[q * Config::HEAD_DIM + d] *= rescale;
            }
        }

        num_rows_handled++;
    }
}

template<typename Config>
__device__ __forceinline__ void consumer_softmax_phase2_write(
    typename Config::SharedMemory& smem,
    int warp_id,
    int lane_id,
    const float* reg_probs,
    const int* reg_q_indices,
    int num_rows_handled
) {
    const int consumer_warp_idx = warp_id - Config::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) return;

    using Element = typename Config::Element;
    constexpr int ELEMS_PER_LANE = (Config::TILE_KV + 31) / 32;

    // Write probs from registers to smem_probs
    for (int r = 0; r < num_rows_handled; ++r) {
        int q = reg_q_indices[r];
        Element* prob_row = smem.smem_probs + q * Config::TILE_KV;

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int kv = lane_id + e * 32;
            if (kv < Config::TILE_KV) {
                prob_row[kv] = __float2bfloat16(reg_probs[r * ELEMS_PER_LANE + e]);
            }
        }
    }
}

// NOTE: This function is split into multiple parts to avoid __syncthreads() divergence
// The conversion phase uses ALL threads (not just consumers) to avoid sync issues

template<typename Config>
__device__ __forceinline__ void convert_scores_to_probs(
    typename Config::SharedMemory& smem,
    int tid,
    int num_threads
) {
    using Element = typename Config::Element;
    constexpr int SCORE_SIZE = Config::TILE_Q * Config::TILE_KV;
    constexpr int ELEMS_PER_THREAD = (SCORE_SIZE + Config::NUM_THREADS - 1) / Config::NUM_THREADS;

    Element local_probs[ELEMS_PER_THREAD];

    // Pass 1: Read all float values into registers (ALL threads participate)
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int i = tid + e * num_threads;
        if (i < SCORE_SIZE) {
            local_probs[e] = __float2bfloat16(smem.smem_scores[i]);
        }
    }
    __syncthreads();  // ALL threads sync here

    // Pass 2: Write BF16 values to shared memory (ALL threads participate)
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int i = tid + e * num_threads;
        if (i < SCORE_SIZE) {
            smem.smem_probs[i] = local_probs[e];
        }
    }
    __syncthreads();  // ALL threads sync here
}

template<typename Config>
__device__ __forceinline__ void consumer_compute_output_matmul(
    typename Config::SharedMemory& smem,
    int stage,
    int tid,
    int num_threads
) {
    using namespace nvcuda::wmma;
    using Element = typename Config::Element;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int M_TILES = Config::TILE_Q / WMMA_M;
    constexpr int N_TILES = Config::HEAD_DIM / WMMA_N;
    constexpr int K_TILES = Config::TILE_KV / WMMA_K;

    // Use consumer-relative warp index (0-7) instead of global warp_id (4-11)
    // This ensures all tiles 0 to M_TILES*N_TILES-1 are covered
    int global_warp_id = tid / 32;
    int consumer_warp_idx = global_warp_id - Config::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) return;  // Producer warps should not call this

    constexpr int num_consumer_warps = Config::NUM_CONSUMER_WARPS;

    // Each consumer warp handles output tiles in round-robin fashion (NO __syncthreads)
    for (int tile_idx = consumer_warp_idx; tile_idx < M_TILES * N_TILES; tile_idx += num_consumer_warps) {
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
            const Element* p_ptr = smem.smem_probs +
                m_tile * WMMA_M * Config::TILE_KV + k * WMMA_K;
            const Element* v_ptr = smem.smem_v[stage] +
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
    __threadfence_block();  // Ensure barrier init is visible to all threads
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

    // === Producer: Load Q tile ===
    if (is_producer && elect_one_per_warp()) {
        if (warp_id == 0) {
            barrier_arrive_expect_tx(smem.barriers[0],
                Config::TILE_Q * Config::HEAD_DIM * sizeof(Element));
            // 3D coordinates: (dim0=0, dim1=q_start, dim2=head_idx)
            tma_load_3d(q_desc_ptr, smem.smem_q, &smem.barriers[0], 0, q_start, head_idx);
        }
    }
    __syncthreads();  // Ensure all threads see the barrier state

    // Wait for Q to be ready
    barrier_wait(smem.barriers[0], 0);

    // Reinitialize barriers for KV pipeline (Q used barriers[0], need to reset for reuse)
    // This is needed because mbarrier state persists after completion
    __syncthreads();
    if (tid == 0) {
        // Invalidate old barriers and reinit for KV pipeline
        for (int s = 0; s < Config::NUM_STAGES; ++s) {
            barrier_invalidate(smem.barriers[s]);
            barrier_init(smem.barriers[s], 1);
        }
    }
    __threadfence_block();
    __syncthreads();

    // === Main loop: Pipeline K/V loading with computation ===
    int read_stage = 0;
    int write_stage = 0;
    int phase = 0;

    // Prefill pipeline
    // Single warp (warp 0 lane 0) does ALL prefetch work: barrier setup + K load + V load
    // This avoids race conditions between barrier setup and TMA loads
    int prefill_tiles = min(Config::NUM_STAGES - 1, num_kv_tiles);
    for (int t = 0; t < prefill_tiles; ++t) {
        // Only warp 0 lane 0 does all the work
        if (is_producer && warp_id == 0 && lane_id == 0) {
            int kv_start = t * Config::TILE_KV;
            uint32_t tx_bytes = Config::TILE_KV * Config::HEAD_DIM * sizeof(Element) * 2;

            // Set up expected bytes FIRST
            barrier_arrive_expect_tx(smem.barriers[write_stage], tx_bytes);

            // Then issue both TMA loads (they complete asynchronously)
            tma_load_3d(k_desc_ptr, smem.smem_k[write_stage], &smem.barriers[write_stage], 0, kv_start, head_idx);
            tma_load_3d(v_desc_ptr, smem.smem_v[write_stage], &smem.barriers[write_stage], 0, kv_start, head_idx);
        }
        write_stage = (write_stage + 1) % Config::NUM_STAGES;
    }

    // Main loop: process KV tiles
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        // Wait for current KV tile
        barrier_wait(smem.barriers[read_stage], phase);
        __syncthreads();

        int kv_start = kv_tile * Config::TILE_KV;
        int kv_len = min(Config::TILE_KV, seq_kv - kv_start);

        // === Consumer: Compute attention ===
        // Compute scores: Q @ K^T (only consumer warps)
        if (is_consumer) {
            consumer_compute_scores<Config>(smem, read_stage, scale, tid, Config::NUM_THREADS);
        }
        __syncthreads();

        // Apply causal mask (all threads participate for even work distribution)
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

        // === Two-Phase Softmax to Avoid Union Race Condition ===
        // smem_scores (float) and smem_probs (bf16) share memory via union.
        // Phase 1: ALL warps read scores, compute probs to REGISTERS
        // Phase 2: After sync, ALL warps write probs from registers to smem
        //
        // Register storage: max 4 rows/warp * 2 elements/lane = 8 floats
        constexpr int MAX_ROWS_PER_WARP = (Config::TILE_Q + Config::NUM_CONSUMER_WARPS - 1) / Config::NUM_CONSUMER_WARPS;
        constexpr int ELEMS_PER_LANE = (Config::TILE_KV + 31) / 32;
        float reg_probs[MAX_ROWS_PER_WARP * ELEMS_PER_LANE];
        float reg_rescales[MAX_ROWS_PER_WARP];
        int reg_q_indices[MAX_ROWS_PER_WARP];
        int num_rows_handled = 0;

        // Phase 1: Read scores and compute probs to registers
        consumer_softmax_phase1_read<Config>(
            smem, kv_tile, kv_len, q_len, warp_id, lane_id,
            reg_probs, reg_rescales, reg_q_indices, num_rows_handled);

        // CRITICAL SYNC: Ensure ALL score reads complete before ANY prob writes
        // This prevents the union race condition between smem_scores and smem_probs
        __syncthreads();

        // Phase 2: Write probs from registers to smem_probs
        consumer_softmax_phase2_write<Config>(
            smem, warp_id, lane_id,
            reg_probs, reg_q_indices, num_rows_handled);

        // Sync needed: probs written, P@V matmul reads them
        __syncthreads();

        // Compute output: P @ V (only consumer warps do the matmul)
        // BF16 probs already in smem_probs from softmax above
        if (is_consumer) {
            consumer_compute_output_matmul<Config>(smem, read_stage, tid, Config::NUM_THREADS);
        }

        // === Producer: Prefetch next KV tile ===
        // Single warp (warp 0 lane 0) does all prefetch to avoid races
        int next_tile = kv_tile + prefill_tiles;
        if (next_tile < num_kv_tiles && is_producer && warp_id == 0 && lane_id == 0) {
            int next_kv_start = next_tile * Config::TILE_KV;
            uint32_t tx_bytes = Config::TILE_KV * Config::HEAD_DIM * sizeof(Element) * 2;

            barrier_arrive_expect_tx(smem.barriers[write_stage], tx_bytes);
            tma_load_3d(k_desc_ptr, smem.smem_k[write_stage], &smem.barriers[write_stage], 0, next_kv_start, head_idx);
            tma_load_3d(v_desc_ptr, smem.smem_v[write_stage], &smem.barriers[write_stage], 0, next_kv_start, head_idx);

            write_stage = (write_stage + 1) % Config::NUM_STAGES;
        }

        // Advance read stage and phase
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
    CUtensorMap q_desc,
    CUtensorMap k_desc,
    CUtensorMap v_desc,
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

    fprintf(stderr, "[DEBUG TMA LAUNCH] grid=(%d,%d,%d) block=%d smem=%zu bytes\n",
            grid.x, grid.y, grid.z, block.x, smem_size);

    // Query device shared memory limit
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    fprintf(stderr, "[DEBUG TMA LAUNCH] Device max smem per block: %zu bytes\n",
            props.sharedMemPerBlockOptin);

    // Query kernel attributes before setting
    cudaFuncAttributes func_attrs;
    cudaError_t query_err = cudaFuncGetAttributes(&func_attrs, flash_attention_3_tma_kernel<Config>);
    if (query_err != cudaSuccess) {
        fprintf(stderr, "[DEBUG TMA LAUNCH] cudaFuncGetAttributes FAILED: %s\n",
                cudaGetErrorString(query_err));
        return query_err;
    }
    fprintf(stderr, "[DEBUG TMA LAUNCH] Kernel static smem: %zu, max threads: %d\n",
            func_attrs.sharedSizeBytes, func_attrs.maxThreadsPerBlock);

    // Set shared memory configuration
    cudaError_t attr_err = cudaFuncSetAttribute(
        flash_attention_3_tma_kernel<Config>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    if (attr_err != cudaSuccess) {
        fprintf(stderr, "[DEBUG TMA LAUNCH] cudaFuncSetAttribute FAILED: %s\n",
                cudaGetErrorString(attr_err));
        return attr_err;
    }

    // Allocate device memory for tensor maps (TMA requires them in device-accessible memory)
    CUtensorMap* d_q_desc;
    CUtensorMap* d_k_desc;
    CUtensorMap* d_v_desc;

    cudaError_t alloc_err;
    alloc_err = cudaMalloc(&d_q_desc, sizeof(CUtensorMap));
    if (alloc_err != cudaSuccess) return alloc_err;
    alloc_err = cudaMalloc(&d_k_desc, sizeof(CUtensorMap));
    if (alloc_err != cudaSuccess) { cudaFree(d_q_desc); return alloc_err; }
    alloc_err = cudaMalloc(&d_v_desc, sizeof(CUtensorMap));
    if (alloc_err != cudaSuccess) { cudaFree(d_q_desc); cudaFree(d_k_desc); return alloc_err; }

    // Copy tensor maps to device
    cudaMemcpyAsync(d_q_desc, &q_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_k_desc, &k_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v_desc, &v_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    fprintf(stderr, "[DEBUG TMA LAUNCH] Tensor maps copied to device: q=%p k=%p v=%p\n",
            (void*)d_q_desc, (void*)d_k_desc, (void*)d_v_desc);

    flash_attention_3_tma_kernel<Config><<<grid, block, smem_size, stream>>>(
        d_q_desc, d_k_desc, d_v_desc, output,
        batch_size, num_heads, seq_q, seq_kv,
        scale, causal
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[DEBUG TMA LAUNCH] Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
        cudaFree(d_q_desc);
        cudaFree(d_k_desc);
        cudaFree(d_v_desc);
        return launch_err;
    }

    // Synchronize to wait for kernel completion and flush printf buffer
    cudaStreamSynchronize(stream);

    // Check for kernel execution errors AFTER sync
    cudaError_t exec_err = cudaGetLastError();
    if (exec_err != cudaSuccess) {
        fprintf(stderr, "[DEBUG TMA LAUNCH] Kernel execution failed: %s\n", cudaGetErrorString(exec_err));
    }

    cudaFree(d_q_desc);
    cudaFree(d_k_desc);
    cudaFree(d_v_desc);

    return exec_err;
}

// Explicit template instantiations for all SM120 config versions
template cudaError_t launch_flash_attention_3_tma<SM120Config<0>>(
    CUtensorMap, CUtensorMap, CUtensorMap,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t);
template cudaError_t launch_flash_attention_3_tma<SM120Config<1>>(
    CUtensorMap, CUtensorMap, CUtensorMap,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t);
template cudaError_t launch_flash_attention_3_tma<SM120Config<2>>(
    CUtensorMap, CUtensorMap, CUtensorMap,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t);
template cudaError_t launch_flash_attention_3_tma<SM120Config<3>>(
    CUtensorMap, CUtensorMap, CUtensorMap,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t);
template cudaError_t launch_flash_attention_3_tma<SM120Config<4>>(
    CUtensorMap, CUtensorMap, CUtensorMap,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t);

// =============================================================================
// Optimized Launch (Cached Descriptors, No Per-Call Overhead)
// =============================================================================

/**
 * Launch FA3 TMA kernel with pre-cached device descriptors.
 * - No cudaMalloc/cudaFree per call
 * - No cudaMemcpy per call
 * - No cudaStreamSynchronize (caller decides when to sync)
 *
 * This is the fast path for repeated calls with same tensor shapes.
 */
template<typename Config>
inline cudaError_t launch_flash_attention_3_tma_cached(
    CUtensorMap* d_q_desc,   // Device pointer (cached)
    CUtensorMap* d_k_desc,   // Device pointer (cached)
    CUtensorMap* d_v_desc,   // Device pointer (cached)
    typename Config::Element* output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    float scale,
    bool causal,
    cudaStream_t stream,
    bool verbose = false
) {
    int num_q_tiles = (seq_q + Config::TILE_Q - 1) / Config::TILE_Q;
    dim3 grid(num_q_tiles, num_heads, batch_size);
    dim3 block(Config::NUM_THREADS);

    size_t smem_size = Config::SharedMemory::size();

    if (verbose) {
        fprintf(stderr, "[TMA CACHED] grid=(%d,%d,%d) block=%d smem=%zu\n",
                grid.x, grid.y, grid.z, block.x, smem_size);
    }

    // Set shared memory configuration (cached after first call by CUDA runtime)
    static bool smem_configured = false;
    if (!smem_configured) {
        cudaError_t attr_err = cudaFuncSetAttribute(
            flash_attention_3_tma_kernel<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        if (attr_err != cudaSuccess) {
            fprintf(stderr, "[TMA CACHED] cudaFuncSetAttribute FAILED: %s\n",
                    cudaGetErrorString(attr_err));
            return attr_err;
        }
        smem_configured = true;
    }

    // Launch kernel (no sync, no malloc, no memcpy)
    flash_attention_3_tma_kernel<Config><<<grid, block, smem_size, stream>>>(
        d_q_desc, d_k_desc, d_v_desc, output,
        batch_size, num_heads, seq_q, seq_kv,
        scale, causal
    );

    return cudaGetLastError();
}

// Explicit template instantiations for all SM120 config versions
template cudaError_t launch_flash_attention_3_tma_cached<SM120Config<0>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, bool);
template cudaError_t launch_flash_attention_3_tma_cached<SM120Config<1>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, bool);
template cudaError_t launch_flash_attention_3_tma_cached<SM120Config<2>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, bool);
template cudaError_t launch_flash_attention_3_tma_cached<SM120Config<3>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, bool);
template cudaError_t launch_flash_attention_3_tma_cached<SM120Config<4>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, bool);

// =============================================================================
// Kernel Timing with CUDA Events
// =============================================================================

/**
 * Launch FA3 TMA kernel with CUDA event timing.
 * Returns kernel execution time in microseconds.
 *
 * @param kernel_time_us  Output: kernel execution time in microseconds
 * @return                cudaSuccess on success
 */
template<typename Config>
inline cudaError_t launch_flash_attention_3_tma_timed(
    CUtensorMap* d_q_desc,
    CUtensorMap* d_k_desc,
    CUtensorMap* d_v_desc,
    typename Config::Element* output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    float scale,
    bool causal,
    cudaStream_t stream,
    float* kernel_time_us
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_q_tiles = (seq_q + Config::TILE_Q - 1) / Config::TILE_Q;
    dim3 grid(num_q_tiles, num_heads, batch_size);
    dim3 block(Config::NUM_THREADS);
    size_t smem_size = Config::SharedMemory::size();

    // Set shared memory (cached after first call)
    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(
            flash_attention_3_tma_kernel<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        smem_configured = true;
    }

    // Record start event
    cudaEventRecord(start, stream);

    // Launch kernel
    flash_attention_3_tma_kernel<Config><<<grid, block, smem_size, stream>>>(
        d_q_desc, d_k_desc, d_v_desc, output,
        batch_size, num_heads, seq_q, seq_kv,
        scale, causal
    );

    // Record stop event
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    *kernel_time_us = ms * 1000.0f;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaGetLastError();
}

// Explicit template instantiations for timed launch
template cudaError_t launch_flash_attention_3_tma_timed<SM120Config<0>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, float*);
template cudaError_t launch_flash_attention_3_tma_timed<SM120Config<1>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, float*);
template cudaError_t launch_flash_attention_3_tma_timed<SM120Config<2>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, float*);
template cudaError_t launch_flash_attention_3_tma_timed<SM120Config<3>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, float*);
template cudaError_t launch_flash_attention_3_tma_timed<SM120Config<4>>(
    CUtensorMap*, CUtensorMap*, CUtensorMap*,
    __nv_bfloat16*, int, int, int, int, float, bool, cudaStream_t, float*);

}  // namespace fa3_sm120
}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
