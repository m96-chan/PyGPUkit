/**
 * TF32 TensorCore GEMM v2 - Using WMMA API
 *
 * Target: 90%+ of cuBLAS performance (37.6+ TFLOPS on RTX 3090 Ti)
 *
 * This version uses the nvcuda::wmma API for cleaner fragment handling.
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

// Each warp computes multiple WMMA tiles
constexpr int WARP_TILES_M = 2;  // 32 rows per warp
constexpr int WARP_TILES_N = 4;  // 64 cols per warp

constexpr int STAGES = 2;
constexpr int A_PAD = 4;
constexpr int B_PAD = 4;

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
// Main Kernel using WMMA API
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

    const int warp_m = warp_row * (WARP_TILES_M * WMMA_M);
    const int warp_n = warp_col * (WARP_TILES_N * WMMA_N);

    __shared__ float smA[STAGES][BM][BK + A_PAD];
    __shared__ float smB[STAGES][BK][BN + B_PAD];

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
    auto load_A_async = [&](int stage, int kt) {
        const int a_row = tid / 4;
        const int a_col = (tid % 4) * 4;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int row = a_row + i * 64;
            int gm = cta_m + row;
            int gk = kt * BK + a_col;

            if (gm < M && gk < K) {
                cp_async_16(&smA[stage][row][a_col], &A[gm * K + gk]);
            }
        }
    };

    auto load_B_async = [&](int stage, int kt) {
        const int b_row_ld = tid / 32;
        const int b_col_ld = (tid % 32) * 4;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int k = b_row_ld + i * 8;
            int gk = kt * BK + k;
            int gn = cta_n + b_col_ld;

            if (gk < K && gn < N) {
                cp_async_16(&smB[stage][k][b_col_ld], &B[gk * N + gn]);
            }
        }
    };

    // Prologue
    load_A_async(0, 0);
    load_B_async(0, 0);
    cp_async_commit();
    cp_async_wait_0();
    __syncthreads();

    // Main loop
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        if (kt + 1 < num_k_tiles) {
            load_A_async(next, kt + 1);
            load_B_async(next, kt + 1);
        }
        cp_async_commit();

        // Process current tile using WMMA
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                int tile_m = warp_m + wm * WMMA_M;

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &smA[curr][tile_m][kk], BK + A_PAD);

                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                    int tile_n = warp_n + wn * WMMA_N;

                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, &smB[curr][kk][tile_n], BN + B_PAD);

                    wmma::mma_sync(acc[wm][wn], a_frag, b_frag, acc[wm][wn]);
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
    sgemm_tf32_v2_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

} // namespace tf32_v2
} // namespace ops
} // namespace pygpukit
