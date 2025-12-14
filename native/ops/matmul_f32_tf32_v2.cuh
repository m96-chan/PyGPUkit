/**
 * TF32 TensorCore GEMM v2 - CUTLASS-inspired Optimizations
 *
 * Target: 90%+ of cuBLAS performance (37.6+ TFLOPS on RTX 3090 Ti)
 *
 * Key optimizations:
 * 1. 3-stage software pipeline with cp.async
 * 2. Optimized warp tile configuration
 * 3. Vectorized memory loads
 */

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace pygpukit {
namespace ops {
namespace tf32_v2 {

// ============================================================================
// Configuration - Tuned for RTX 3090 Ti (SM 8.6)
// ============================================================================

// CTA tile dimensions
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

// MMA tile dimensions (m16n8k8 instruction)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 8;

// Warp configuration: 4x2 warps = 8 warps = 256 threads
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * 32;

// Each warp computes WARP_TILES_M x WARP_TILES_N MMA tiles
constexpr int WARP_TILES_M = 2;  // 2 * 16 = 32 rows per warp
constexpr int WARP_TILES_N = 8;  // 8 * 8 = 64 cols per warp

// Pipeline stages
// smA: 2 * 128 * 16 * 4 = 16384 bytes
// smB: 2 * 16 * 128 * 4 = 16384 bytes
// Total: 32768 bytes = 32KB (easily fits)
constexpr int STAGES = 2;

// Padding to avoid bank conflicts
constexpr int A_PAD = 4;
constexpr int B_PAD = 4;

// ============================================================================
// cp.async Helpers
// ============================================================================

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 smem64; "
        "  cvta.to.shared.u64 smem64, %1; "
        "  cvt.u32.u64 %0, smem64; }"
        : "=r"(addr) : "l"(ptr)
    );
    return addr;
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t addr = smem_u32(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(addr), "l"(gmem)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_0() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ void cp_async_wait_1() {
    asm volatile("cp.async.wait_group 1;");
}

// ============================================================================
// Main Kernel - Correct Implementation with Double Buffering
// ============================================================================

__global__ void __launch_bounds__(256, 2)
sgemm_tf32_v2_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int cta_m = blockIdx.y * BM;
    const int cta_n = blockIdx.x * BN;

    const int warp_row = warp_id / WARPS_N;  // 0-3
    const int warp_col = warp_id % WARPS_N;  // 0-1

    // Warp's starting position within CTA tile
    const int warp_m = warp_row * (WARP_TILES_M * WMMA_M);  // 0, 32, 64, 96
    const int warp_n = warp_col * (WARP_TILES_N * WMMA_N);  // 0, 64

    // Shared memory with padding
    __shared__ float smA[STAGES][BM][BK + A_PAD];
    __shared__ float smB[STAGES][BK][BN + B_PAD];

    // Accumulators
    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = K / BK;

    // Fragment index mappings (verified)
    const int a_row_base = lane / 4;   // 0-7
    const int a_col_base = lane % 4;   // 0-3
    const int b_row_base = lane % 4;   // 0-3
    const int b_col = lane / 4;        // 0-7
    const int c_row_base = lane / 4;
    const int c_col_base = (lane % 4) * 2;

    // ====== Load helpers ======
    auto load_A_async = [&](int stage, int kt) {
        // Each thread loads: tid / 4 determines which row, tid % 4 * 4 determines col
        const int a_row = tid / 4;  // 0-63
        const int a_col = (tid % 4) * 4;  // 0, 4, 8, 12

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int row = a_row + i * 64;  // 0-63, then 64-127
            int gm = cta_m + row;
            int gk = kt * BK + a_col;

            if (gm < M && gk < K) {
                cp_async_16(&smA[stage][row][a_col], &A[gm * K + gk]);
            }
        }
    };

    auto load_B_async = [&](int stage, int kt) {
        const int b_row_ld = tid / 32;  // 0-7
        const int b_col_ld = (tid % 32) * 4;  // 0, 4, 8, ..., 124

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int k = b_row_ld + i * 8;  // 0-7, then 8-15
            int gk = kt * BK + k;
            int gn = cta_n + b_col_ld;

            if (gk < K && gn < N) {
                cp_async_16(&smB[stage][k][b_col_ld], &B[gk * N + gn]);
            }
        }
    };

    // ====== Prologue: load first tile ======
    load_A_async(0, 0);
    load_B_async(0, 0);
    cp_async_commit();
    cp_async_wait_0();
    __syncthreads();

    // ====== Main loop with double buffering ======
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        // Prefetch next tile
        if (kt + 1 < num_k_tiles) {
            load_A_async(next, kt + 1);
            load_B_async(next, kt + 1);
        }
        cp_async_commit();

        // Process current tile
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                int tile_m = warp_m + wm * WMMA_M;

                // Load A fragment
                float a0 = smA[curr][tile_m + a_row_base][kk + a_col_base];
                float a1 = smA[curr][tile_m + a_row_base + 8][kk + a_col_base];
                float a2 = smA[curr][tile_m + a_row_base][kk + a_col_base + 4];
                float a3 = smA[curr][tile_m + a_row_base + 8][kk + a_col_base + 4];

                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                    int tile_n = warp_n + wn * WMMA_N;

                    // Load B fragment
                    float b0 = smB[curr][kk + b_row_base][tile_n + b_col];
                    float b1 = smB[curr][kk + b_row_base + 4][tile_n + b_col];

                    // MMA instruction
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(acc[wm][wn][0]), "+f"(acc[wm][wn][1]),
                          "+f"(acc[wm][wn][2]), "+f"(acc[wm][wn][3])
                        : "r"(__float_as_uint(a0)), "r"(__float_as_uint(a1)),
                          "r"(__float_as_uint(a2)), "r"(__float_as_uint(a3)),
                          "r"(__float_as_uint(b0)), "r"(__float_as_uint(b1))
                    );
                }
            }
        }

        // Wait for prefetch
        cp_async_wait_0();
        __syncthreads();
    }

    // ====== Epilogue: Write results ======
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; ++wn) {
            int tile_m = cta_m + warp_m + wm * WMMA_M;
            int tile_n = cta_n + warp_n + wn * WMMA_N;

            int out_row0 = tile_m + c_row_base;
            int out_row1 = tile_m + c_row_base + 8;
            int out_col0 = tile_n + c_col_base;
            int out_col1 = tile_n + c_col_base + 1;

            if (out_row0 < M && out_col0 < N) C[out_row0 * N + out_col0] = acc[wm][wn][0];
            if (out_row0 < M && out_col1 < N) C[out_row0 * N + out_col1] = acc[wm][wn][1];
            if (out_row1 < M && out_col0 < N) C[out_row1 * N + out_col0] = acc[wm][wn][2];
            if (out_row1 < M && out_col1 < N) C[out_row1 * N + out_col1] = acc[wm][wn][3];
        }
    }
}

// ============================================================================
// Launch Helper
// ============================================================================

inline cudaError_t launch_sgemm_tf32_v2(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tf32_v2_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

} // namespace tf32_v2
} // namespace ops
} // namespace pygpukit
