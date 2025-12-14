/**
 * TF32 TensorCore GEMM v2 - Optimized PTX implementation
 *
 * Target: 90%+ of cuBLAS performance (37.6+ TFLOPS on RTX 3090 Ti)
 *
 * Configuration:
 * - 128x128 CTA tile, BK=16
 * - 4x2 warps (256 threads)
 * - Each warp: 32x64 output (2x4 WMMA 16x16 tiles)
 * - 2-stage double buffering with cp.async
 */

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

namespace pygpukit {
namespace ops {
namespace tf32_v2 {

// ============================================================================
// Configuration
// ============================================================================

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

// WMMA dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

// Warp configuration: 4x2 warps
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;

// Each warp computes 2x4 WMMA tiles (32x64 output)
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;

constexpr int STAGES = 2;
constexpr int SMEM_PAD = 8;  // Padding for bank conflict avoidance

// ============================================================================
// cp.async helpers
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
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_0() {
    asm volatile("cp.async.wait_group 0;");
}

// ============================================================================
// Main Kernel using WMMA API (simpler and often equally fast)
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

    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    const int warp_m = warp_row * (WARP_TILES_M * WMMA_M);  // 0, 32, 64, 96
    const int warp_n = warp_col * (WARP_TILES_N * WMMA_N);  // 0, 64

    // Shared memory: row-major with padding
    extern __shared__ float smem[];
    float* smA = smem;  // [STAGES][BM][BK + SMEM_PAD]
    float* smB = smA + STAGES * BM * (BK + SMEM_PAD);  // [STAGES][BK][BN + SMEM_PAD]

    const int A_stride = BK + SMEM_PAD;
    const int B_stride = BN + SMEM_PAD;

    // WMMA fragments for accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; ++wn) {
            wmma::fill_fragment(acc[wm][wn], 0.0f);
        }
    }

    const int num_k_tiles = K / BK;

    // Load helpers
    // A: 128x16, each thread loads 8 floats (2 float4)
    // B: 16x128, each thread loads 8 floats (2 float4)
    auto load_A = [&](int stage, int kt) {
        float* dst = smA + stage * BM * A_stride;

        // Thread mapping: 256 threads load 128x16 = 2048 elements
        // Each thread loads 8 elements (2 float4)
        const int row0 = tid / 2;          // 0-127
        const int col0 = (tid & 1) * 8;    // 0 or 8

        int gm = cta_m + row0;
        int gk = kt * BK + col0;

        if (gm < M && gk + 3 < K) {
            cp_async_16(&dst[row0 * A_stride + col0], &A[gm * K + gk]);
        }
        if (gm < M && gk + 7 < K) {
            cp_async_16(&dst[row0 * A_stride + col0 + 4], &A[gm * K + gk + 4]);
        }
    };

    auto load_B = [&](int stage, int kt) {
        float* dst = smB + stage * BK * B_stride;

        // Thread mapping: 256 threads load 16x128 = 2048 elements
        // Each thread loads 8 elements (2 float4)
        const int row0 = tid / 16;         // 0-15
        const int col0 = (tid & 15) * 8;   // 0, 8, 16, ..., 120

        int gk = kt * BK + row0;
        int gn = cta_n + col0;

        if (gk < K && gn + 3 < N) {
            cp_async_16(&dst[row0 * B_stride + col0], &B[gk * N + gn]);
        }
        if (gk < K && gn + 7 < N) {
            cp_async_16(&dst[row0 * B_stride + col0 + 4], &B[gk * N + gn + 4]);
        }
    };

    // Prologue
    load_A(0, 0);
    load_B(0, 0);
    cp_async_commit();
    cp_async_wait_0();
    __syncthreads();

    // Main loop
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        if (kt + 1 < num_k_tiles) {
            load_A(next, kt + 1);
            load_B(next, kt + 1);
        }
        cp_async_commit();

        float* A_tile = smA + curr * BM * A_stride;
        float* B_tile = smB + curr * BK * B_stride;

        // Process K dimension in chunks of WMMA_K (8)
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            // Load A fragments
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag[WARP_TILES_M];

            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                int tile_m = warp_m + wm * WMMA_M;
                wmma::load_matrix_sync(a_frag[wm], &A_tile[tile_m * A_stride + kk], A_stride);
            }

            // Load B fragments and compute
            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                int tile_n = warp_n + wn * WMMA_N;

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &B_tile[kk * B_stride + tile_n], B_stride);

                #pragma unroll
                for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                    wmma::mma_sync(acc[wm][wn], a_frag[wm], b_frag, acc[wm][wn]);
                }
            }
        }

        cp_async_wait_0();
        __syncthreads();
    }

    // Epilogue: write results
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; ++wn) {
            int tile_m = cta_m + warp_m + wm * WMMA_M;
            int tile_n = cta_n + warp_n + wn * WMMA_N;

            if (tile_m < M && tile_n < N) {
                wmma::store_matrix_sync(&C[tile_m * N + tile_n], acc[wm][wn], N, wmma::mem_row_major);
            }
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

    // Shared memory size
    size_t smem_size = STAGES * BM * (BK + SMEM_PAD) * sizeof(float) +
                       STAGES * BK * (BN + SMEM_PAD) * sizeof(float);

    sgemm_tf32_v2_kernel<<<grid, block, smem_size, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

} // namespace tf32_v2
} // namespace ops
} // namespace pygpukit
