/**
 * Flash Attention 3 - SM120 MMA Operations
 *
 * MMA wrappers for NVIDIA Blackwell GeForce (SM120).
 * Uses mma.sync.aligned instructions (not tcgen05).
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace pygpukit {
namespace ops {
namespace nn {
namespace fa3 {
namespace sm120 {

// =============================================================================
// MMA Fragment Types
// =============================================================================

/**
 * BF16 MMA fragment for m16n8k16.
 * Each thread holds part of the matrix.
 */
struct MmaFragmentBF16 {
    // A fragment: 4 x uint32 (8 x bf16)
    uint32_t a[4];
    // B fragment: 2 x uint32 (4 x bf16)
    uint32_t b[2];
    // C/D fragment: 4 x float
    float c[4];

    __device__ __forceinline__ void clear_accumulator() {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            c[i] = 0.0f;
        }
    }
};

/**
 * FP16 MMA fragment for m16n8k16.
 */
struct MmaFragmentFP16 {
    uint32_t a[4];
    uint32_t b[2];
    float c[4];

    __device__ __forceinline__ void clear_accumulator() {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            c[i] = 0.0f;
        }
    }
};

// =============================================================================
// MMA PTX Instructions
// =============================================================================

/**
 * BF16 MMA: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 *
 * Computes D = A * B + C where:
 * - A: 16x16 (row-major)
 * - B: 16x8 (col-major)
 * - C/D: 16x8 (row-major)
 */
__device__ __forceinline__ void mma_sync_m16n8k16_bf16(
    float* d,           // Output [4]
    const uint32_t* a,  // A fragment [4]
    const uint32_t* b,  // B fragment [2]
    const float* c      // Accumulator [4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

/**
 * FP16 MMA: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 */
__device__ __forceinline__ void mma_sync_m16n8k16_fp16(
    float* d,
    const uint32_t* a,
    const uint32_t* b,
    const float* c
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// =============================================================================
// Fragment Load Operations
// =============================================================================

/**
 * Load A fragment from shared memory (row-major).
 *
 * For m16n8k16, each warp loads 16x16 elements.
 * Lane mapping: lane_id -> (row, col) in the 16x16 tile
 */
__device__ __forceinline__ void load_a_fragment_bf16(
    uint32_t* a_frag,           // Output fragment [4]
    const __nv_bfloat16* smem,  // Shared memory base
    int row_offset,             // Row offset in smem
    int col_offset,             // Col offset in smem
    int stride                  // Row stride in smem
) {
    int lane_id = threadIdx.x % 32;

    // A fragment layout for m16n8k16:
    // Each thread holds 8 elements across 4 registers
    // a[0]: rows 0-7, specific cols based on lane
    // a[1]: rows 8-15, specific cols
    // a[2]: rows 0-7, different cols
    // a[3]: rows 8-15, different cols

    int row_group = (lane_id / 4);      // 0-7
    int col_group = (lane_id % 4) * 2;  // 0,2,4,6

    const __nv_bfloat16* base = smem + row_offset * stride + col_offset;

    // Load 2 bf16 values per register (packed as uint32)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = (i < 2) ? row_group : (row_group + 8);
        int col = (i % 2 == 0) ? col_group : (col_group + 8);

        const __nv_bfloat16* ptr = base + row * stride + col;
        a_frag[i] = *reinterpret_cast<const uint32_t*>(ptr);
    }
}

/**
 * Load B fragment from shared memory (col-major for transpose).
 *
 * For m16n8k16, B is 16x8 (K x N).
 */
__device__ __forceinline__ void load_b_fragment_bf16(
    uint32_t* b_frag,           // Output fragment [2]
    const __nv_bfloat16* smem,  // Shared memory base
    int row_offset,             // Row offset (K dimension)
    int col_offset,             // Col offset (N dimension)
    int stride                  // Row stride
) {
    int lane_id = threadIdx.x % 32;

    // B fragment layout for m16n8k16 (col-major):
    // Each thread holds 4 elements across 2 registers
    int k_idx = (lane_id % 4) * 2;       // K position: 0,2,4,6
    int n_idx = lane_id / 4;             // N position: 0-7

    const __nv_bfloat16* base = smem + row_offset * stride + col_offset;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int k = k_idx + i * 8;
        const __nv_bfloat16* ptr = base + k * stride + n_idx;
        // Pack 2 bf16 values (but from different K positions)
        __nv_bfloat16 v0 = ptr[0];
        __nv_bfloat16 v1 = ptr[stride];
        // Pack two bf16 values into uint32 using bit manipulation
        uint16_t u0 = *reinterpret_cast<uint16_t*>(&v0);
        uint16_t u1 = *reinterpret_cast<uint16_t*>(&v1);
        b_frag[i] = (static_cast<uint32_t>(u1) << 16) | static_cast<uint32_t>(u0);
    }
}

// =============================================================================
// Fragment Store Operations
// =============================================================================

/**
 * Store C fragment to shared memory.
 */
__device__ __forceinline__ void store_c_fragment(
    float* smem,            // Shared memory base
    const float* c_frag,    // Fragment [4]
    int row_offset,
    int col_offset,
    int stride
) {
    int lane_id = threadIdx.x % 32;

    // C fragment layout for m16n8k16:
    // c[0]: row (lane/4), col (lane%4)*2
    // c[1]: row (lane/4), col (lane%4)*2 + 1
    // c[2]: row (lane/4)+8, col (lane%4)*2
    // c[3]: row (lane/4)+8, col (lane%4)*2 + 1

    int row_base = lane_id / 4;
    int col_base = (lane_id % 4) * 2;

    float* base = smem + row_offset * stride + col_offset;

    base[(row_base) * stride + col_base]     = c_frag[0];
    base[(row_base) * stride + col_base + 1] = c_frag[1];
    base[(row_base + 8) * stride + col_base]     = c_frag[2];
    base[(row_base + 8) * stride + col_base + 1] = c_frag[3];
}

// =============================================================================
// Attention Score Computation (Q * K^T)
// =============================================================================

/**
 * Compute attention scores for one tile.
 *
 * Q: [TILE_Q, HEAD_DIM]
 * K: [TILE_KV, HEAD_DIM]
 * S: [TILE_Q, TILE_KV] = Q @ K^T
 */
template<int TILE_Q, int TILE_KV, int HEAD_DIM>
__device__ __forceinline__ void compute_attention_scores_bf16(
    float* scores,                          // Output [TILE_Q, TILE_KV]
    const __nv_bfloat16* smem_q,           // Q in smem [TILE_Q, HEAD_DIM]
    const __nv_bfloat16* smem_k,           // K in smem [TILE_KV, HEAD_DIM]
    int q_stride,
    int k_stride,
    float scale
) {
    // Tile the computation using m16n8k16 MMAs
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    constexpr int M_TILES = TILE_Q / MMA_M;
    constexpr int N_TILES = TILE_KV / MMA_N;
    constexpr int K_TILES = HEAD_DIM / MMA_K;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp computes a subset of output tiles
    // TODO: Proper work distribution across warps

    MmaFragmentBF16 frag;

    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            frag.clear_accumulator();

            // Accumulate over K dimension
            #pragma unroll
            for (int k = 0; k < K_TILES; ++k) {
                load_a_fragment_bf16(frag.a, smem_q, m * MMA_M, k * MMA_K, q_stride);
                load_b_fragment_bf16(frag.b, smem_k, k * MMA_K, n * MMA_N, k_stride);
                mma_sync_m16n8k16_bf16(frag.c, frag.a, frag.b, frag.c);
            }

            // Apply scale and store
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                frag.c[i] *= scale;
            }

            store_c_fragment(scores, frag.c, m * MMA_M, n * MMA_N, TILE_KV);
        }
    }
}

// =============================================================================
// FP32 to BF16 Conversion Helpers
// =============================================================================

/**
 * Convert FP32 to BF16 (truncation, fast)
 */
__device__ __forceinline__ __nv_bfloat16 fp32_to_bf16_fast(float f) {
    return __float2bfloat16_rn(f);
}

/**
 * Pack two BF16 values into uint32 for MMA fragment
 */
__device__ __forceinline__ uint32_t pack_bf16x2(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*(uint16_t*)&a), "h"(*(uint16_t*)&b));
    return result;
}

/**
 * Load A fragment from FP32 source (converts to BF16 on-the-fly).
 * Used for P matrix in P @ V computation.
 */
__device__ __forceinline__ void load_a_fragment_fp32_to_bf16(
    uint32_t* a_frag,           // Output fragment [4]
    const float* smem,          // Shared memory base (FP32)
    int row_offset,             // Row offset in smem
    int col_offset,             // Col offset in smem
    int stride                  // Row stride in smem
) {
    int lane_id = threadIdx.x % 32;

    // A fragment layout for m16n8k16 (same as BF16, but source is FP32)
    int row_group = (lane_id / 4);      // 0-7
    int col_group = (lane_id % 4) * 2;  // 0,2,4,6

    const float* base = smem + row_offset * stride + col_offset;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = (i < 2) ? row_group : (row_group + 8);
        int col = (i % 2 == 0) ? col_group : (col_group + 8);

        const float* ptr = base + row * stride + col;
        // Load 2 FP32, convert to BF16, pack
        __nv_bfloat16 v0 = fp32_to_bf16_fast(ptr[0]);
        __nv_bfloat16 v1 = fp32_to_bf16_fast(ptr[1]);
        a_frag[i] = pack_bf16x2(v0, v1);
    }
}

// =============================================================================
// Attention Output Computation (P * V)
// =============================================================================

/**
 * Compute attention output for one tile.
 *
 * P: [TILE_Q, TILE_KV] (softmax probabilities, FP32 in smem)
 * V: [TILE_KV, HEAD_DIM] (BF16 in smem)
 * O: per-thread accumulator [M_TILES][N_TILES][4]
 *
 * This computes O += P @ V where P is converted to BF16 on-the-fly.
 * The output is stored in per-thread registers, NOT a shared array.
 *
 * output_acc layout: [M_TILES][N_TILES][4] where:
 * - M_TILES = TILE_Q / 16
 * - N_TILES = HEAD_DIM / 8
 * - 4 = elements per thread per MMA tile
 */
template<int TILE_Q, int TILE_KV, int HEAD_DIM>
__device__ __forceinline__ void compute_attention_output_bf16(
    float* output_acc,                     // Per-thread accumulator [M_TILES][N_TILES][4]
    const float* smem_probs,               // P in smem [TILE_Q, TILE_KV] (FP32)
    const __nv_bfloat16* smem_v,           // V in smem [TILE_KV, HEAD_DIM]
    int p_stride,                          // Stride for P (= TILE_KV)
    int v_stride                           // Stride for V (= HEAD_DIM)
) {
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    constexpr int M_TILES = TILE_Q / MMA_M;
    constexpr int N_TILES = HEAD_DIM / MMA_N;
    constexpr int K_TILES = TILE_KV / MMA_K;
    constexpr int THREAD_ELEMS = 4;

    MmaFragmentBF16 frag;

    // Each warp processes all M_TILES x N_TILES output tiles
    #pragma unroll
    for (int m = 0; m < M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < N_TILES; ++n) {
            // Load existing accumulator from per-thread register array
            // Layout: output_acc[m][n][0..3]
            int acc_idx = (m * N_TILES + n) * THREAD_ELEMS;
            frag.c[0] = output_acc[acc_idx + 0];
            frag.c[1] = output_acc[acc_idx + 1];
            frag.c[2] = output_acc[acc_idx + 2];
            frag.c[3] = output_acc[acc_idx + 3];

            // Accumulate over K dimension (TILE_KV)
            #pragma unroll
            for (int k = 0; k < K_TILES; ++k) {
                // Load P fragment (FP32 -> BF16)
                load_a_fragment_fp32_to_bf16(frag.a, smem_probs, m * MMA_M, k * MMA_K, p_stride);
                // Load V fragment (BF16)
                load_b_fragment_bf16(frag.b, smem_v, k * MMA_K, n * MMA_N, v_stride);
                // MMA: output += P * V
                mma_sync_m16n8k16_bf16(frag.c, frag.a, frag.b, frag.c);
            }

            // Store back to per-thread accumulator
            output_acc[acc_idx + 0] = frag.c[0];
            output_acc[acc_idx + 1] = frag.c[1];
            output_acc[acc_idx + 2] = frag.c[2];
            output_acc[acc_idx + 3] = frag.c[3];
        }
    }
}

// =============================================================================
// Online Softmax + Output Accumulation (Fused)
// =============================================================================

/**
 * Apply online softmax rescaling to output accumulator.
 *
 * When the max value changes during online softmax, we need to rescale
 * the existing output accumulator: O *= exp(old_max - new_max)
 */
__device__ __forceinline__ void rescale_output_accumulator(
    float* output,          // Per-thread output values
    int num_elements,       // Number of elements this thread owns
    float rescale_factor    // exp(old_max - new_max)
) {
    #pragma unroll
    for (int i = 0; i < num_elements; ++i) {
        output[i] *= rescale_factor;
    }
}

}  // namespace sm120
}  // namespace fa3
}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
