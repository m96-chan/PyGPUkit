/**
 * Flash Attention 3 - FP8 Extension for SM120 (RTX 5090 Blackwell)
 *
 * Key optimizations:
 * - FP8 E4M3 block-scale MMA for Q@K^T (16x8x32 tile, FP32 accumulator)
 * - BF16 WMMA for P@V (V stays BF16 for precision)
 * - ~50% memory bandwidth reduction for Q, K
 * - Expected error: ~0.25% (validated via CPU simulation)
 *
 * Architecture:
 *   Q, K: FP8 E4M3 + UE8M0 scale factors (per 32 elements)
 *   V: BF16 (unchanged - FP8 V causes ~18% error)
 *   Q@K^T: mma.sync.aligned.block_scale.m16n8k32.f32.e4m3.e4m3
 *   P@V: wmma 16x16x16 BF16
 *
 * Reference: tma_fa3_optimization_status memory
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include "fa3_traits.cuh"
#include "fa3_online_softmax.cuh"
#include "../../common/tma_utils.cuh"
#include "../../common/warp_scheduler.cuh"
#include "../../common/pipeline.cuh"
#include "../../matmul/gemm/fp8_block_scale/fp8_block_scale_mma_sm120.cuh"

namespace pygpukit {
namespace ops {
namespace nn {
namespace fa3_fp8_sm120 {

// =============================================================================
// FP8 Shared Memory Layout
// =============================================================================
// Q, K stored as FP8 E4M3 (1 byte per element) with per-32-element scale factors
// V stays BF16 for precision (FP8 V causes ~18% error)

template<int TILE_Q, int TILE_KV, int HEAD_DIM, int NUM_STAGES>
struct FP8SharedMemory {
    // Q buffer: FP8 E4M3 (single stage - loaded once)
    alignas(128) uint8_t smem_q_fp8[TILE_Q * HEAD_DIM];

    // Per-head global scale for Q (single UE8M0 value for entire head)
    // This is required for block-scale MMA which expects uniform scales
    static constexpr int SCALES_PER_ROW = HEAD_DIM / 32;  // Keep for compatibility
    alignas(16) uint8_t smem_q_scale;  // Single per-head scale (UE8M0)

    // K buffers: FP8 E4M3 (multi-stage for pipelining)
    alignas(128) uint8_t smem_k_fp8[NUM_STAGES][TILE_KV * HEAD_DIM];

    // Per-head global scale for K (single UE8M0 value per stage)
    alignas(16) uint8_t smem_k_scale[NUM_STAGES];  // Single per-head scale per stage

    // V buffers: BF16 (multi-stage) - kept as BF16 for precision
    alignas(1024) __nv_bfloat16 smem_v[NUM_STAGES][TILE_KV * HEAD_DIM];

    // Scores/Probs union - same as BF16 FA3
    union alignas(128) {
        float smem_scores[TILE_Q * TILE_KV];
        __nv_bfloat16 smem_probs[TILE_Q * TILE_KV * 2];
    };

    // Softmax state
    alignas(16) float softmax_max[TILE_Q];
    alignas(16) float softmax_sum[TILE_Q];

    // Output accumulator
    alignas(128) float output_acc[TILE_Q * HEAD_DIM];

    // Pipeline barriers
    alignas(64) uint64_t barriers[NUM_STAGES];

    static constexpr size_t size() {
        return sizeof(FP8SharedMemory);
    }
};

// =============================================================================
// FP8 Configuration
// =============================================================================

template<int VERSION = 0>
struct FP8Config;

// Version 0: Baseline FP8 configuration
// Same tile sizes as BF16, but Q/K use FP8 with block scaling
template<>
struct FP8Config<0> {
    static constexpr int TILE_Q = 32;
    static constexpr int TILE_KV = 64;
    static constexpr int HEAD_DIM = 128;
    static constexpr int NUM_STAGES = 2;
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;

    // MMA tile sizes for FP8 block-scale (16x8x32)
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 32;

    // Scale factor granularity
    static constexpr int SCALE_BLOCK_SIZE = 32;  // One scale per 32 elements

    using Element = __nv_bfloat16;  // Output element type
    using SharedMemory = FP8SharedMemory<TILE_Q, TILE_KV, HEAD_DIM, NUM_STAGES>;
};

// =============================================================================
// UE8M0 Scale Factor Decoding
// =============================================================================

__device__ __forceinline__ float decode_ue8m0_scale(uint8_t ue8m0) {
    // UE8M0: 8-bit unsigned exponent, no mantissa
    // Value = 2^(ue8m0 - 127)
    // 127 = 1.0, 128 = 2.0, 126 = 0.5, etc.
    int exp = static_cast<int>(ue8m0) - 127;
    return exp2f(static_cast<float>(exp));
}

// =============================================================================
// FP8 Q@K^T Computation using Block-Scale MMA
// =============================================================================
// Uses mma.sync.aligned.block_scale.m16n8k32.f32.e4m3.e4m3
// Each MMA computes a 16x8 output tile from 16x32 A and 32x8 B

template<typename Config>
__device__ __forceinline__ void consumer_compute_scores_fp8(
    typename Config::SharedMemory& smem,
    int stage,
    float attn_scale,  // 1/sqrt(head_dim)
    int tid,
    int num_threads
) {
    using namespace pygpukit::ops::matmul::fp8_mma_sm120;

    constexpr int MMA_M = Config::MMA_M;  // 16
    constexpr int MMA_N = Config::MMA_N;  // 8
    constexpr int MMA_K = Config::MMA_K;  // 32

    constexpr int M_TILES = Config::TILE_Q / MMA_M;    // 32/16 = 2
    constexpr int N_TILES = Config::TILE_KV / MMA_N;   // 64/8 = 8
    constexpr int K_TILES = Config::HEAD_DIM / MMA_K;  // 128/32 = 4

    // Consumer warp index
    int global_warp_id = tid / 32;
    int consumer_warp_idx = global_warp_id - Config::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) return;

    int lane_id = tid % 32;
    constexpr int num_consumer_warps = Config::NUM_CONSUMER_WARPS;

    // Per-head global scales - single scale for all Q, single scale for all K
    // This is the key fix: uniform scales ensure block-scale MMA produces correct results
    uint8_t scale_a_ue8m0 = smem.smem_q_scale;
    uint8_t scale_b_ue8m0 = smem.smem_k_scale[stage];

    // Each consumer warp handles tiles in round-robin fashion
    // Total tiles: M_TILES * N_TILES = 2 * 8 = 16 tiles
    for (int tile_idx = consumer_warp_idx; tile_idx < M_TILES * N_TILES; tile_idx += num_consumer_warps) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        // Accumulator for this tile (16x8 = 128 elements, 4 per thread)
        float d_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Loop over K dimension
        #pragma unroll
        for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
            // Load A (Q) fragment: 16x32 FP8
            // A is row-major: [TILE_Q, HEAD_DIM]
            const uint8_t* A_ptr = smem.smem_q_fp8 +
                m_tile * MMA_M * Config::HEAD_DIM + k_tile * MMA_K;

            // Load B (K) fragment: 32x8 FP8, but K is [TILE_KV, HEAD_DIM]
            // We need K^T, so access K[n, k] = K[n_tile*8 + n_idx, k_tile*32 + k_idx]
            const uint8_t* B_ptr = smem.smem_k_fp8[stage] +
                n_tile * MMA_N * Config::HEAD_DIM + k_tile * MMA_K;

            // Load A fragment into registers
            uint32_t a_frag[4];
            int t0 = lane_id / 8;
            int t1 = lane_id % 8;

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int v1 = r % 2;
                int v2 = r / 2;
                uint8_t bytes[4];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int flat = 64 * t0 + t1 + 16 * b + 8 * v1 + 256 * v2;
                    int row = flat / MMA_K;  // 0-15
                    int col = flat % MMA_K;  // 0-31
                    bytes[b] = A_ptr[row * Config::HEAD_DIM + col];
                }
                a_frag[r] = bytes[0] | (uint32_t(bytes[1]) << 8) |
                            (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
            }

            // Load B fragment (K^T) - CORRECT formula from Test 4 routing discovery
            // Key insight: n_idx = lane_id / 4 (NOT lane_id / 8 as CuTe suggests)
            // Each group of 4 lanes loads one B column with all 32 k values
            uint32_t b_frag[2];
            int n_idx = lane_id / 4;            // B column: 0-7 (one column per 4 lanes)
            int k_base = (lane_id % 4) * 8;     // Base k: each of 4 lanes handles 8 consecutive k values
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                uint8_t bytes[4];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int k_idx = k_base + r * 4 + b;  // k = 0-7 for lane%4=0, 8-15 for lane%4=1, etc.
                    bytes[b] = B_ptr[n_idx * Config::HEAD_DIM + k_idx];
                }
                b_frag[r] = bytes[0] | (uint32_t(bytes[1]) << 8) |
                            (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
            }

            // Execute MMA with accumulation: D = A @ B * scales + D
            // Pass d_frag as both output and accumulator input
            mma_fp8_block_scale_16x8x32(
                d_frag[0], d_frag[1], d_frag[2], d_frag[3],
                a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                b_frag[0], b_frag[1],
                d_frag[0], d_frag[1], d_frag[2], d_frag[3],  // accumulate into d_frag
                scale_a_ue8m0, scale_b_ue8m0
            );
        }

        // Apply attention scale and store to smem_scores
        // CORRECT C/D Fragment Layout (SM80_16x8_Row from CUTLASS):
        //   row0 = lane_id / 4        (0-7)
        //   row1 = lane_id / 4 + 8    (8-15)
        //   col0 = (lane_id % 4) * 2  (0, 2, 4, 6)
        //   col1 = (lane_id % 4) * 2 + 1 (1, 3, 5, 7)
        //   d[0] = C[row0, col0], d[1] = C[row0, col1]
        //   d[2] = C[row1, col0], d[3] = C[row1, col1]
        int row0 = m_tile * MMA_M + lane_id / 4;
        int row1 = m_tile * MMA_M + lane_id / 4 + 8;
        int col0 = n_tile * MMA_N + (lane_id % 4) * 2;
        int col1 = n_tile * MMA_N + (lane_id % 4) * 2 + 1;

        if (row0 < Config::TILE_Q && col0 < Config::TILE_KV) {
            smem.smem_scores[row0 * Config::TILE_KV + col0] = d_frag[0] * attn_scale;
        }
        if (row0 < Config::TILE_Q && col1 < Config::TILE_KV) {
            smem.smem_scores[row0 * Config::TILE_KV + col1] = d_frag[1] * attn_scale;
        }
        if (row1 < Config::TILE_Q && col0 < Config::TILE_KV) {
            smem.smem_scores[row1 * Config::TILE_KV + col0] = d_frag[2] * attn_scale;
        }
        if (row1 < Config::TILE_Q && col1 < Config::TILE_KV) {
            smem.smem_scores[row1 * Config::TILE_KV + col1] = d_frag[3] * attn_scale;
        }
    }
}

// =============================================================================
// P@V Computation - Reuse BF16 WMMA from FA3
// =============================================================================
// V stays as BF16, probs are converted to BF16 after softmax

template<typename Config>
__device__ __forceinline__ void consumer_compute_output_fp8(
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

    int global_warp_id = tid / 32;
    int consumer_warp_idx = global_warp_id - Config::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) return;

    constexpr int num_consumer_warps = Config::NUM_CONSUMER_WARPS;

    // Each consumer warp handles tiles in round-robin fashion
    for (int tile_idx = consumer_warp_idx; tile_idx < M_TILES * N_TILES; tile_idx += num_consumer_warps) {
        int m_tile = tile_idx / N_TILES;
        int n_tile = tile_idx % N_TILES;

        // Load existing accumulator
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        float* acc_ptr = smem.output_acc + m_tile * WMMA_M * Config::HEAD_DIM + n_tile * WMMA_N;
        load_matrix_sync(acc_frag, acc_ptr, Config::HEAD_DIM, mem_row_major);

        // P @ V
        #pragma unroll
        for (int k = 0; k < K_TILES; ++k) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> p_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> v_frag;

            // Load P (probs) - BF16
            const __nv_bfloat16* p_ptr = smem.smem_probs +
                m_tile * WMMA_M * Config::TILE_KV + k * WMMA_K;
            // Load V - BF16
            const __nv_bfloat16* v_ptr = smem.smem_v[stage] +
                k * WMMA_K * Config::HEAD_DIM + n_tile * WMMA_N;

            load_matrix_sync(p_frag, p_ptr, Config::TILE_KV);
            load_matrix_sync(v_frag, v_ptr, Config::HEAD_DIM);
            mma_sync(acc_frag, p_frag, v_frag, acc_frag);
        }

        // Store back
        store_matrix_sync(acc_ptr, acc_frag, Config::HEAD_DIM, mem_row_major);
    }
}

// =============================================================================
// Two-Phase Softmax - CRITICAL: Avoids smem_scores/smem_probs Race Condition
// =============================================================================
// The union of smem_scores (float) and smem_probs (BF16) causes memory overlap:
// - smem_probs[q*64..q*64+63] (bytes q*128..q*128+127) overlaps with
//   smem_scores[q*32..q*32+31] (bytes q*128..q*128+127)
//
// When multiple warps process different Q rows concurrently, writing probs for
// row q+1 can corrupt scores for row q that another warp is still reading.
//
// FIX: Split into two phases with __syncthreads() between:
// Phase 1: ALL warps read scores → compute probs → store to REGISTERS
// Phase 2: After sync, ALL warps write probs from registers to smem_probs
//
// Register budget: TILE_KV/32 = 2 elements per lane per row

template<typename Config>
__device__ __forceinline__ void consumer_softmax_phase1_read_fp8(
    typename Config::SharedMemory& smem,
    int kv_tile,
    int kv_len,
    int q_len,
    int warp_id,
    int lane_id,
    // Output: per-lane register storage for probs
    float* reg_probs,       // [MAX_ROWS_PER_WARP * ELEMS_PER_LANE]
    int* reg_q_indices,     // [MAX_ROWS_PER_WARP] - which q rows this warp handles
    int& num_rows_handled
) {
    int consumer_warp_idx = warp_id - Config::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) {
        num_rows_handled = 0;
        return;
    }

    constexpr int num_consumer_warps = Config::NUM_CONSUMER_WARPS;
    constexpr int ELEMS_PER_LANE = (Config::TILE_KV + 31) / 32;  // 2 for TILE_KV=64

    num_rows_handled = 0;

    // Each consumer warp handles different Q rows in round-robin fashion
    for (int q = consumer_warp_idx; q < q_len; q += num_consumer_warps) {
        // Store which q row we're handling
        reg_q_indices[num_rows_handled] = q;

        // === Step 1: Find row maximum (warp-level reduction) ===
        float row_max = -INFINITY;
        for (int kv = lane_id; kv < kv_len; kv += 32) {
            float score = smem.smem_scores[q * Config::TILE_KV + kv];
            row_max = fmaxf(row_max, score);
        }

        // Warp reduce max
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, offset));
        }

        // === Step 2: Online softmax update ===
        float old_max = smem.softmax_max[q];
        float new_max = fmaxf(old_max, row_max);
        float rescale = (kv_tile > 0) ? exp2f((old_max - new_max) * 1.4426950408889634f) : 1.0f;

        if (lane_id == 0) {
            smem.softmax_max[q] = new_max;
            smem.softmax_sum[q] *= rescale;
        }
        __syncwarp();

        // === Step 3: Rescale existing output accumulator ===
        float rescale_bcast = __shfl_sync(0xffffffff, rescale, 0);
        if (kv_tile > 0 && rescale_bcast != 1.0f) {
            for (int d = lane_id; d < Config::HEAD_DIM; d += 32) {
                smem.output_acc[q * Config::HEAD_DIM + d] *= rescale_bcast;
            }
        }

        // === Step 4: Compute exp and sum, store probs to REGISTERS (not smem!) ===
        float row_sum = 0.0f;
        new_max = smem.softmax_max[q];

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int kv = lane_id + e * 32;
            float prob = 0.0f;
            if (kv < kv_len) {
                float score = smem.smem_scores[q * Config::TILE_KV + kv];
                prob = exp2f((score - new_max) * 1.4426950408889634f);
                row_sum += prob;
            }
            // Store to registers, NOT to shared memory
            reg_probs[num_rows_handled * ELEMS_PER_LANE + e] = prob;
        }

        // Warp reduce sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
        }

        if (lane_id == 0) {
            smem.softmax_sum[q] += row_sum;
        }

        num_rows_handled++;
    }
}

template<typename Config>
__device__ __forceinline__ void consumer_softmax_phase2_write_fp8(
    typename Config::SharedMemory& smem,
    int kv_len,
    int warp_id,
    int lane_id,
    const float* reg_probs,
    const int* reg_q_indices,
    int num_rows_handled
) {
    int consumer_warp_idx = warp_id - Config::NUM_PRODUCER_WARPS;
    if (consumer_warp_idx < 0) return;

    constexpr int ELEMS_PER_LANE = (Config::TILE_KV + 31) / 32;

    // Write probs from registers to smem_probs
    for (int r = 0; r < num_rows_handled; ++r) {
        int q = reg_q_indices[r];

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            int kv = lane_id + e * 32;
            if (kv < Config::TILE_KV) {
                float prob = reg_probs[r * ELEMS_PER_LANE + e];
                smem.smem_probs[q * Config::TILE_KV + kv] = __float2bfloat16(prob);
            }
        }
    }
}

// =============================================================================
// Host-side FP8 Quantization Helper - Per-Head Global Scale
// =============================================================================
// Quantize BF16 Q/K to FP8 E4M3 with ONE global scale per head.
// This is required for block-scale MMA which expects uniform scales.
//
// Block-scale MMA: mma.sync.aligned.block_scale.m16n8k32.f32.e4m3.e4m3
// expects scaleA = one scale for 16×32 A block, scaleB = one scale for 32×8 B block.
// Using per-row scales causes ~50% error due to scale mismatch.
// Using per-head global scales ensures all MMA tiles use consistent scales.

__global__ void quantize_to_fp8_e4m3_per_head_kernel(
    const __nv_bfloat16* __restrict__ input,  // [batch, num_heads, seq, head_dim]
    uint8_t* __restrict__ output_fp8,         // [batch, num_heads, seq, head_dim]
    uint8_t* __restrict__ output_scale,       // [batch * num_heads] - ONE scale per head
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // One block per (batch, head) pair
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    int64_t head_offset = (int64_t)(batch_idx * num_heads + head_idx);
    int64_t head_size = (int64_t)seq_len * head_dim;
    const __nv_bfloat16* head_in = input + head_offset * head_size;
    uint8_t* head_out = output_fp8 + head_offset * head_size;

    // Shared memory for reduction
    __shared__ float s_absmax[256];

    // Phase 1: Find global absmax across all elements in this head
    float local_absmax = 0.0f;
    for (int64_t i = tid; i < head_size; i += num_threads) {
        float val = __bfloat162float(head_in[i]);
        local_absmax = fmaxf(local_absmax, fabsf(val));
    }

    // Store to shared memory
    s_absmax[tid] = local_absmax;
    __syncthreads();

    // Block reduction to find global absmax
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_absmax[tid] = fmaxf(s_absmax[tid], s_absmax[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 computes and stores the global scale
    float global_absmax = s_absmax[0];
    constexpr float FP8_E4M3_MAX = 448.0f;
    float scale = (global_absmax > 0.0f) ? (global_absmax / FP8_E4M3_MAX) : 1.0f;
    int exp = static_cast<int>(ceilf(log2f(scale))) + 127;
    exp = max(0, min(255, exp));
    uint8_t global_scale_ue8m0 = static_cast<uint8_t>(exp);

    // Write the single per-head scale
    if (tid == 0) {
        output_scale[head_offset] = global_scale_ue8m0;
    }

    // Broadcast scale to all threads via shared memory
    __shared__ float s_inv_scale;
    if (tid == 0) {
        s_inv_scale = 1.0f / exp2f(static_cast<float>(exp - 127));
    }
    __syncthreads();
    float inv_scale = s_inv_scale;

    // Phase 2: Quantize all elements using the global scale
    for (int64_t i = tid; i < head_size; i += num_threads) {
        float val = __bfloat162float(head_in[i]) * inv_scale;
        // Clamp and convert to FP8
        val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        head_out[i] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3));
    }
}

// =============================================================================
// Main FA3 FP8 Kernel
// =============================================================================
// Uses FP8 Q@K^T with block-scale MMA, BF16 P@V with WMMA

template<typename Config>
__global__ void __launch_bounds__(Config::NUM_THREADS, 1)
flash_attention_3_fp8_kernel(
    const uint8_t* __restrict__ Q_fp8,      // [batch, num_heads, seq_q, head_dim] FP8
    const uint8_t* __restrict__ K_fp8,      // [batch, num_heads, seq_kv, head_dim] FP8
    const uint8_t* __restrict__ Q_scale,    // [batch * num_heads] - ONE scale per head
    const uint8_t* __restrict__ K_scale,    // [batch * num_heads] - ONE scale per head
    const __nv_bfloat16* __restrict__ V,    // [batch, num_heads, seq_kv, head_dim] BF16
    __nv_bfloat16* __restrict__ output,     // [batch, num_heads, seq_q, head_dim] BF16
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    float attn_scale,
    bool causal
) {
    using namespace pygpukit::ops::tma;

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

    // Calculate offsets
    const int64_t head_offset = (int64_t)(batch_idx * num_heads + head_idx);
    const int64_t q_base = head_offset * seq_q * Config::HEAD_DIM;
    const int64_t kv_base = head_offset * seq_kv * Config::HEAD_DIM;

    // Initialize shared memory
    if (tid == 0) {
        for (int s = 0; s < Config::NUM_STAGES; ++s) {
            barrier_init(smem.barriers[s], 1);
        }
        // Load single per-head scales to shared memory
        smem.smem_q_scale = Q_scale[head_offset];
        // K scale will be loaded per-stage (same value for all stages in this head)
        for (int s = 0; s < Config::NUM_STAGES; ++s) {
            smem.smem_k_scale[s] = K_scale[head_offset];
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

    // Load Q tile (FP8) to shared memory - simple copy for now
    // Scale is already loaded (single per-head value)
    for (int i = tid; i < q_len * Config::HEAD_DIM; i += blockDim.x) {
        int q_idx = i / Config::HEAD_DIM;
        int d_idx = i % Config::HEAD_DIM;
        smem.smem_q_fp8[q_idx * Config::HEAD_DIM + d_idx] =
            Q_fp8[q_base + (q_start + q_idx) * Config::HEAD_DIM + d_idx];
    }
    // Zero-init unused Q rows for partial Q tiles
    if (q_len < Config::TILE_Q) {
        for (int i = tid; i < (Config::TILE_Q - q_len) * Config::HEAD_DIM; i += blockDim.x) {
            int q_idx = q_len + i / Config::HEAD_DIM;
            int d_idx = i % Config::HEAD_DIM;
            smem.smem_q_fp8[q_idx * Config::HEAD_DIM + d_idx] = 0;  // FP8 zero
        }
    }
    __syncthreads();

    // Warp role
    bool is_producer = (warp_id < Config::NUM_PRODUCER_WARPS);
    bool is_consumer = !is_producer;

    // Calculate number of KV tiles
    int num_kv_tiles = (seq_kv + Config::TILE_KV - 1) / Config::TILE_KV;
    if (causal) {
        int max_kv_pos = q_start + q_len - 1;
        num_kv_tiles = min(num_kv_tiles, (max_kv_pos + Config::TILE_KV) / Config::TILE_KV);
    }

    // Main loop: process KV tiles
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        int kv_start = kv_tile * Config::TILE_KV;
        int kv_len = min(Config::TILE_KV, seq_kv - kv_start);
        int stage = kv_tile % Config::NUM_STAGES;

        // Load K tile (FP8) and V tile (BF16)
        // K scale is already loaded (single per-head value)
        // Simple copy - TODO: Use TMA
        __syncthreads();

        // For partial tiles: zero-initialize positions >= kv_len to avoid NaN from garbage in MMA
        if (kv_len < Config::TILE_KV) {
            for (int i = tid; i < (Config::TILE_KV - kv_len) * Config::HEAD_DIM; i += blockDim.x) {
                int kv_idx = kv_len + i / Config::HEAD_DIM;
                int d_idx = i % Config::HEAD_DIM;
                smem.smem_k_fp8[stage][kv_idx * Config::HEAD_DIM + d_idx] = 0;  // FP8 zero
                smem.smem_v[stage][kv_idx * Config::HEAD_DIM + d_idx] = __float2bfloat16(0.0f);
            }
            // No K scale zero-init needed - we use single per-head scale
        }

        // Load valid K and V data
        for (int i = tid; i < kv_len * Config::HEAD_DIM; i += blockDim.x) {
            int kv_idx = i / Config::HEAD_DIM;
            int d_idx = i % Config::HEAD_DIM;
            smem.smem_k_fp8[stage][kv_idx * Config::HEAD_DIM + d_idx] =
                K_fp8[kv_base + (kv_start + kv_idx) * Config::HEAD_DIM + d_idx];
            smem.smem_v[stage][kv_idx * Config::HEAD_DIM + d_idx] =
                V[kv_base + (kv_start + kv_idx) * Config::HEAD_DIM + d_idx];
        }
        // K scale is already loaded at kernel start (single per-head value)
        __syncthreads();

        // Compute Q @ K^T using FP8 block-scale MMA
        if (is_consumer) {
            consumer_compute_scores_fp8<Config>(smem, stage, attn_scale, tid, Config::NUM_THREADS);
        }
        __syncthreads();

        // Apply masks: partial tile mask (kv_idx >= kv_len) AND causal mask
        // Note: Even with zero-initialized K, we mask to -INFINITY for correct softmax normalization
        for (int i = tid; i < Config::TILE_Q * Config::TILE_KV; i += blockDim.x) {
            int q_idx = i / Config::TILE_KV;
            int kv_idx = i % Config::TILE_KV;
            // Partial tile mask: mask positions beyond valid kv_len
            if (kv_idx >= kv_len) {
                smem.smem_scores[i] = -INFINITY;
            }
            // Causal mask: mask future positions
            else if (causal && kv_start + kv_idx > q_start + q_idx) {
                smem.smem_scores[i] = -INFINITY;
            }
        }
        __syncthreads();

        // Two-phase softmax to avoid smem_scores/smem_probs race condition
        // (The union overlap causes concurrent warps to corrupt each other's data)
        {
            int warp_id = tid / 32;
            int lane_id = tid % 32;

            // Register storage for probs - max rows per warp and 2 elements per lane
            constexpr int MAX_ROWS_PER_WARP = (Config::TILE_Q + Config::NUM_CONSUMER_WARPS - 1) / Config::NUM_CONSUMER_WARPS;
            constexpr int ELEMS_PER_LANE = (Config::TILE_KV + 31) / 32;
            float reg_probs[MAX_ROWS_PER_WARP * ELEMS_PER_LANE];
            int reg_q_indices[MAX_ROWS_PER_WARP];
            int num_rows_handled;

            // Phase 1: Read scores, compute softmax, store to REGISTERS (not smem)
            consumer_softmax_phase1_read_fp8<Config>(
                smem, kv_tile, kv_len, q_len, warp_id, lane_id,
                reg_probs, reg_q_indices, num_rows_handled);
            __syncthreads();  // CRITICAL: Ensure all warps finish reading scores

            // Phase 2: Write probs from registers to smem_probs
            consumer_softmax_phase2_write_fp8<Config>(
                smem, kv_len, warp_id, lane_id,
                reg_probs, reg_q_indices, num_rows_handled);
        }
        __syncthreads();

        // Compute P @ V using BF16 WMMA
        if (is_consumer) {
            consumer_compute_output_fp8<Config>(smem, stage, tid, Config::NUM_THREADS);
        }
        __syncthreads();
    }

    // Finalize: normalize and write output
    __syncthreads();
    const int64_t out_offset = head_offset * seq_q * Config::HEAD_DIM;
    __nv_bfloat16* O_ptr = output + out_offset + q_start * Config::HEAD_DIM;

    for (int i = tid; i < q_len * Config::HEAD_DIM; i += blockDim.x) {
        int q = i / Config::HEAD_DIM;
        int d = i % Config::HEAD_DIM;
        float val = smem.output_acc[i] / smem.softmax_sum[q];
        O_ptr[q * Config::HEAD_DIM + d] = __float2bfloat16(val);
    }
}

// =============================================================================
// Launch Wrapper
// =============================================================================

template<typename Config = FP8Config<0>>
cudaError_t flash_attention_3_fp8_sm120(
    const __nv_bfloat16* Q,   // [batch, num_heads, seq_q, head_dim] BF16
    const __nv_bfloat16* K,   // [batch, num_heads, seq_kv, head_dim] BF16
    const __nv_bfloat16* V,   // [batch, num_heads, seq_kv, head_dim] BF16
    __nv_bfloat16* output,    // [batch, num_heads, seq_q, head_dim] BF16
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    float scale,
    bool causal,
    cudaStream_t stream = nullptr
) {
    // Check SM version
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    if (props.major < 12) {
        fprintf(stderr, "[FA3 FP8] Error: SM %d.%d not supported (requires SM120+)\n",
                props.major, props.minor);
        return cudaErrorNotSupported;
    }

    // Calculate sizes
    const size_t head_dim = Config::HEAD_DIM;
    const size_t q_fp8_size = (size_t)batch_size * num_heads * seq_q * head_dim;
    const size_t k_fp8_size = (size_t)batch_size * num_heads * seq_kv * head_dim;
    // Per-head scales: ONE scale per head (not per row)
    const size_t num_total_heads = (size_t)batch_size * num_heads;
    const size_t q_scale_size = num_total_heads;  // [batch * num_heads]
    const size_t k_scale_size = num_total_heads;  // [batch * num_heads]

    // Allocate temporary FP8 buffers
    uint8_t *d_Q_fp8, *d_K_fp8, *d_Q_scale, *d_K_scale;
    cudaError_t err;

    err = cudaMalloc(&d_Q_fp8, q_fp8_size);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&d_K_fp8, k_fp8_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp8); return err; }

    err = cudaMalloc(&d_Q_scale, q_scale_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp8); cudaFree(d_K_fp8); return err; }

    err = cudaMalloc(&d_K_scale, k_scale_size);
    if (err != cudaSuccess) { cudaFree(d_Q_fp8); cudaFree(d_K_fp8); cudaFree(d_Q_scale); return err; }

    // Quantize Q and K to FP8 with per-head global scales
    // One CUDA block per (batch, head) pair
    {
        dim3 block(256);
        dim3 grid_q(num_heads, batch_size);  // [num_heads, batch_size]
        dim3 grid_k(num_heads, batch_size);

        quantize_to_fp8_e4m3_per_head_kernel<<<grid_q, block, 0, stream>>>(
            Q, d_Q_fp8, d_Q_scale,
            batch_size, num_heads, seq_q, head_dim);

        quantize_to_fp8_e4m3_per_head_kernel<<<grid_k, block, 0, stream>>>(
            K, d_K_fp8, d_K_scale,
            batch_size, num_heads, seq_kv, head_dim);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_Q_fp8); cudaFree(d_K_fp8);
            cudaFree(d_Q_scale); cudaFree(d_K_scale);
            return err;
        }
    }

    // Launch main FA3 FP8 kernel
    {
        int num_q_tiles = (seq_q + Config::TILE_Q - 1) / Config::TILE_Q;
        dim3 grid(num_q_tiles, num_heads, batch_size);
        dim3 block(Config::NUM_THREADS);
        size_t smem_size = Config::SharedMemory::size();

        // Set dynamic shared memory size
        err = cudaFuncSetAttribute(
            flash_attention_3_fp8_kernel<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "[FA3 FP8] Error: Failed to set smem size (%zu bytes): %s\n",
                    smem_size, cudaGetErrorString(err));
            cudaFree(d_Q_fp8); cudaFree(d_K_fp8);
            cudaFree(d_Q_scale); cudaFree(d_K_scale);
            return err;
        }

        flash_attention_3_fp8_kernel<Config><<<grid, block, smem_size, stream>>>(
            d_Q_fp8, d_K_fp8, d_Q_scale, d_K_scale, V, output,
            batch_size, num_heads, seq_q, seq_kv, scale, causal
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[FA3 FP8] Error: Kernel launch failed: %s\n",
                    cudaGetErrorString(err));
        }
    }

    // Free temporary buffers
    cudaFree(d_Q_fp8);
    cudaFree(d_K_fp8);
    cudaFree(d_Q_scale);
    cudaFree(d_K_scale);

    return err;
}

}  // namespace fa3_fp8_sm120
}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
