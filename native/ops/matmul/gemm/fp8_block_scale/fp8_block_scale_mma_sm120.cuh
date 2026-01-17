/**
 * FP8 Block-Scale MMA Native PTX Implementation for SM120 (Blackwell GeForce)
 *
 * Based on CUTLASS reference: cute/arch/mma_sm120.hpp, cute/atom/mma_traits_sm80.hpp
 *
 * Key instruction:
 *   mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
 *
 * Tile dimensions: M=16, N=8, K=32
 * Scale format: UE8M0 (8-bit unsigned exponent, no mantissa)
 * Block size for scaling: 32 elements (matches K dimension, MXFP8 standard)
 *
 * Register layout per warp (32 threads):
 *   D[4]: Output FP32 (16x8 matrix)
 *   A[4]: Input FP8 E4M3 as uint32_t (16x32 matrix, 4 FP8 per register = 16 FP8 per thread)
 *   B[2]: Input FP8 E4M3 as uint32_t (32x8 matrix, 4 FP8 per register = 8 FP8 per thread)
 *   C[4]: Accumulator FP32 (16x8 matrix)
 *   SFA[1]: Scale factor for A (UE8M0)
 *   SFB[1]: Scale factor for B (UE8M0)
 *
 * C/D Fragment Layout (SM80_16x8_Row from CUTLASS):
 *   For lane_id in [0, 31]:
 *     row0 = lane_id / 4        (0-7)
 *     row1 = lane_id / 4 + 8    (8-15)
 *     col0 = (lane_id % 4) * 2  (0, 2, 4, 6)
 *     col1 = (lane_id % 4) * 2 + 1 (1, 3, 5, 7)
 *
 *     d[0] = C[row0, col0]
 *     d[1] = C[row0, col1]
 *     d[2] = C[row1, col0]
 *     d[3] = C[row1, col1]
 *
 * A Fragment Layout (16x32, row-major):
 *   ALayout = Layout<Shape<Shape<_4,_8>, Shape<_4,_2,_2>>,
 *                    Stride<Stride<_64,_1>, Stride<_16,_8,_256>>>
 *   Each thread loads 16 consecutive FP8 values, packed into 4 x uint32_t
 *
 * B Fragment Layout (32x8, col-major for TN):
 *   BLayout = Layout<Shape<Shape<_4,_8>, Shape<_4,_2>>,
 *                    Stride<Stride<_32,_1>, Stride<_8,_128>>>
 *   Each thread loads 8 consecutive FP8 values, packed into 2 x uint32_t
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

// Require CUDA 13.x for SM120 FP8 block-scale MMA support
// Note: __CUDACC_VER_MAJOR__ is defined at compile time (host and device)
//       __CUDA_ARCH__ is only defined during device compilation
#if __CUDACC_VER_MAJOR__ >= 13

namespace pygpukit {
namespace ops {
namespace matmul {
namespace fp8_mma_sm120 {

// =============================================================================
// MMA Tile Configuration
// =============================================================================

struct MMA_16x8x32_Config {
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 32;

    // Scale block size (MXFP8 standard)
    static constexpr int SCALE_BLOCK_SIZE = 32;

    // Register counts per thread
    static constexpr int D_REGS = 4;  // FP32 output
    static constexpr int A_REGS = 4;  // FP8 input (packed 4 per uint32)
    static constexpr int B_REGS = 2;  // FP8 input (packed 4 per uint32)
    static constexpr int C_REGS = 4;  // FP32 accumulator
};

// =============================================================================
// Native PTX FP8 Block-Scale MMA
// =============================================================================

/**
 * Execute FP8 E4M3 x E4M3 -> FP32 MMA with block scaling.
 *
 * PTX instruction: mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
 *                  .m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
 *
 * @param d0-d3: Output FP32 registers (will be written)
 * @param a0-a3: A matrix FP8 registers (uint32_t, 4 FP8 values each)
 * @param b0-b1: B matrix FP8 registers (uint32_t, 4 FP8 values each)
 * @param c0-c3: Accumulator FP32 registers (input)
 * @param sfa: Scale factor for A (UE8M0 format)
 * @param sfb: Scale factor for B (UE8M0 format)
 */
__device__ __forceinline__ void mma_fp8_block_scale_16x8x32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint8_t sfa,
    uint8_t sfb
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // Static block/thread IDs for simple case (all 0)
    // These would be non-zero for more complex tensor layouts
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidB = 0;
    static constexpr uint16_t tidB = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"     // D registers (output)
        "{%4,  %5,  %6,  %7},"     // A registers (input)
        "{%8,  %9},"               // B registers (input)
        "{%10, %11, %12, %13},"    // C registers (accumulator)
        "{%14},"                   // Scale factor A (ue8m0)
        "{%15, %16},"              // Block ID A, Thread ID A
        "{%17},"                   // Scale factor B (ue8m0)
        "{%18, %19};\n"            // Block ID B, Thread ID B
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(uint32_t(sfa)), "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb)), "h"(bidB), "h"(tidB)
    );
#else
    // Fallback for non-SM120 compilation (should not be reached at runtime)
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

// =============================================================================
// Helper: Convert float scale to UE8M0
// =============================================================================

/**
 * Convert a floating-point scale factor to UE8M0 format.
 *
 * UE8M0: 8-bit unsigned exponent, no mantissa
 * Represents powers of 2: value = 2^(exp - 127)
 * Range: 2^-127 to 2^127
 */
__device__ __forceinline__ uint8_t float_to_ue8m0(float scale) {
    if (scale == 0.0f) return 0;

    // Extract exponent from IEEE 754 float
    uint32_t bits = __float_as_uint(scale);
    uint8_t exp = (bits >> 23) & 0xFF;

    return exp;
}

/**
 * Convert UE8M0 to float scale factor.
 */
__device__ __forceinline__ float ue8m0_to_float(uint8_t ue8m0) {
    if (ue8m0 == 0) return 0.0f;

    // Reconstruct float with just exponent (mantissa = 0)
    uint32_t bits = uint32_t(ue8m0) << 23;
    return __uint_as_float(bits);
}

// =============================================================================
// FP8 Block-Scale GEMM Tile (single 16x8x32 MMA)
// =============================================================================

/**
 * Compute a single 16x8 output tile using FP8 block-scale MMA.
 *
 * A: [16, 32] row-major FP8 E4M3
 * B: [32, 8] col-major FP8 E4M3 (for A @ B pattern)
 * C: [16, 8] row-major FP32
 *
 * This function handles:
 * 1. Loading A/B fragments into registers (NON-CONTIGUOUS layout!)
 * 2. Computing scale factors
 * 3. Executing MMA instruction
 * 4. Storing output
 *
 * =============================================================================
 * FRAGMENT LAYOUTS (from CUTLASS CuTe mma_traits_sm80.hpp):
 * =============================================================================
 *
 * ALayout = Layout<Shape<(4,8), (4,2,2)>, Stride<(64,1), (16,8,256)>>
 *   Thread coord: t0 = lane_id/8, t1 = lane_id%8
 *   Value coord: v0 in [0,4), v1 in [0,2), v2 in [0,2)
 *   flat_index = 64*t0 + t1 + 16*v0 + 8*v1 + 256*v2
 *   For A[16,32] row-major: (row, col) = (flat_index/32, flat_index%32)
 *
 * BLayout = Layout<Shape<(4,8), (4,2)>, Stride<(32,1), (8,128)>>
 *   flat_index = 32*t0 + t1 + 8*v0 + 128*v1
 *   For B[32,8] col-major: (k, n) = (flat_index%32, flat_index/32)
 *
 * CLayout = SM80_16x8_Row (CORRECT - use this, not CuTe layout!)
 *   row0 = lane_id / 4        (0-7)
 *   row1 = lane_id / 4 + 8    (8-15)
 *   col0 = (lane_id % 4) * 2  (0, 2, 4, 6)
 *   col1 = (lane_id % 4) * 2 + 1 (1, 3, 5, 7)
 *   d[0] = C[row0, col0], d[1] = C[row0, col1]
 *   d[2] = C[row1, col0], d[3] = C[row1, col1]
 * =============================================================================
 */
__device__ __forceinline__ void gemm_tile_fp8_block_scale_16x8x32(
    const __nv_fp8_e4m3* __restrict__ A,  // [16, 32] row-major
    const __nv_fp8_e4m3* __restrict__ B,  // [32, 8] col-major
    float* __restrict__ C,                 // [16, 8] row-major
    float scale_a,                         // Pre-computed scale for A block
    float scale_b,                         // Pre-computed scale for B block
    int lda,                               // Leading dimension of A (usually K=32)
    int ldb,                               // Leading dimension of B (usually K=32 for col-major)
    int ldc                                // Leading dimension of C (usually N=8)
) {
    int lane_id = threadIdx.x % 32;

    // Thread coordinates for CuTe layout
    int t0 = lane_id / 8;   // 0-3
    int t1 = lane_id % 8;   // 0-7

    // Convert float scales to UE8M0
    uint8_t sfa = float_to_ue8m0(scale_a);
    uint8_t sfb = float_to_ue8m0(scale_b);

    // ==========================================================================
    // Load A fragment (16x32 matrix, row-major)
    // NON-CONTIGUOUS layout per CuTe ALayout
    //
    // For register r (0-3): r = v1 + 2*v2
    //   r=0: v1=0, v2=0
    //   r=1: v1=1, v2=0
    //   r=2: v1=0, v2=1
    //   r=3: v1=1, v2=1
    //
    // For byte b (0-3) within register: b = v0
    // flat_index = 64*t0 + t1 + 16*v0 + 8*v1 + 256*v2
    //            = 64*t0 + t1 + 16*b + 8*(r%2) + 256*(r/2)
    // ==========================================================================

    uint32_t a_frag[4];
    const uint8_t* A_bytes = reinterpret_cast<const uint8_t*>(A);

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int v1 = r % 2;
        int v2 = r / 2;
        uint8_t bytes[4];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            int flat = 64 * t0 + t1 + 16 * b + 8 * v1 + 256 * v2;
            int row = flat / 32;
            int col = flat % 32;
            bytes[b] = A_bytes[row * lda + col];
        }

        // Pack 4 bytes into uint32_t (little-endian)
        a_frag[r] = bytes[0] | (uint32_t(bytes[1]) << 8) |
                    (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    }

    // ==========================================================================
    // Load B fragment (32x8 matrix, col-major)
    // NON-CONTIGUOUS layout per CuTe BLayout
    //
    // For register r (0-1): r = v1
    // For byte b (0-3) within register: b = v0
    // flat_index = 32*t0 + t1 + 8*v0 + 128*v1
    //            = 32*t0 + t1 + 8*b + 128*r
    //
    // For col-major B[K=32, N=8]: B[k,n] stored at index n*32 + k
    //   k = flat_index % 32
    //   n = flat_index / 32
    // ==========================================================================

    uint32_t b_frag[2];
    const uint8_t* B_bytes = reinterpret_cast<const uint8_t*>(B);

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        uint8_t bytes[4];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            int flat = 32 * t0 + t1 + 8 * b + 128 * r;
            int k = flat % 32;
            int n = flat / 32;
            // Col-major storage: B[k,n] at B_bytes[n * ldb + k]
            bytes[b] = B_bytes[n * ldb + k];
        }

        // Pack 4 bytes into uint32_t (little-endian)
        b_frag[r] = bytes[0] | (uint32_t(bytes[1]) << 8) |
                    (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    }

    // ==========================================================================
    // Initialize accumulator (load existing C or zero)
    // ==========================================================================

    float c_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // ==========================================================================
    // Execute MMA
    // ==========================================================================

    float d_frag[4];

    mma_fp8_block_scale_16x8x32(
        d_frag[0], d_frag[1], d_frag[2], d_frag[3],
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        b_frag[0], b_frag[1],
        c_frag[0], c_frag[1], c_frag[2], c_frag[3],
        sfa, sfb
    );

    // ==========================================================================
    // Store D fragment back to C
    //
    // CORRECT C/D Fragment Layout (SM80_16x8_Row from CUTLASS):
    //   row0 = lane_id / 4        (0-7)
    //   row1 = lane_id / 4 + 8    (8-15)
    //   col0 = (lane_id % 4) * 2  (0, 2, 4, 6)
    //   col1 = (lane_id % 4) * 2 + 1 (1, 3, 5, 7)
    //
    //   d[0] = C[row0, col0]
    //   d[1] = C[row0, col1]
    //   d[2] = C[row1, col0]
    //   d[3] = C[row1, col1]
    // ==========================================================================

    {
        int row0 = lane_id / 4;
        int row1 = lane_id / 4 + 8;
        int col0 = (lane_id % 4) * 2;
        int col1 = (lane_id % 4) * 2 + 1;

        C[row0 * ldc + col0] = d_frag[0];
        C[row0 * ldc + col1] = d_frag[1];
        C[row1 * ldc + col0] = d_frag[2];
        C[row1 * ldc + col1] = d_frag[3];
    }
}

// =============================================================================
// Test Kernel: Validate FP8 Block-Scale MMA Correctness
// =============================================================================

static __global__ void test_fp8_block_scale_mma_kernel(
    const __nv_fp8_e4m3* __restrict__ A,  // [16, 32] row-major FP8
    const __nv_fp8_e4m3* __restrict__ B,  // [32, 8] col-major FP8
    float* __restrict__ C,                 // [16, 8] output
    float scale_a,                         // Scale for A
    float scale_b                          // Scale for B
) {
    // Single warp executes one 16x8x32 MMA
    if (threadIdx.x >= 32) return;

    gemm_tile_fp8_block_scale_16x8x32(
        A, B, C,
        scale_a, scale_b,
        32,  // lda = K
        32,  // ldb = K (for col-major B)
        8    // ldc = N
    );
}

// Reference kernel: compute same result using scalar FP32 math
static __global__ void reference_fp8_matmul_kernel(
    const __nv_fp8_e4m3* __restrict__ A,  // [16, 32] row-major FP8
    const __nv_fp8_e4m3* __restrict__ B,  // [32, 8] col-major FP8
    float* __restrict__ C,                 // [16, 8] output
    float scale_a,                         // Scale for A
    float scale_b                          // Scale for B
) {
    int tid = threadIdx.x;
    int m = tid / 8;   // 0-15
    int n = tid % 8;   // 0-7

    if (m >= 16 || n >= 8) return;

    float acc = 0.0f;

    for (int k = 0; k < 32; ++k) {
        // A[m, k] - row major
        float a_val = float(A[m * 32 + k]) * scale_a;

        // B[k, n] - col major: B stored as B[col][row] = B[n * 32 + k]
        float b_val = float(B[n * 32 + k]) * scale_b;

        acc += a_val * b_val;
    }

    C[m * 8 + n] = acc;
}

}  // namespace fp8_mma_sm120
}  // namespace matmul
}  // namespace ops
}  // namespace pygpukit

#endif  // __CUDACC_VER_MAJOR__ >= 13
