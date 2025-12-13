#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

/*
 * PyGPUkit TF32 TensorCore GEMM
 * High-performance CUTLASS-style kernel
 *
 * Target (RTX 3090 Ti):
 *   - 26〜29 TFLOPS (TF32 TensorCore)
 *   - Beats PyTorch/cuBLAS FP32
 *
 * Tile:
 *   - BM = 128, BN = 128, BK = 32
 *   - 256 threads = 8 warps (16×16)
 *   - 2-stage cp.async pipeline
 */

namespace pygpukit {
namespace ops {
namespace tf32 {

using namespace nvcuda::wmma;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

// 4x2 warp grid
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;

// warp computes:
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;

// Shared memory padding avoids bank conflicts
constexpr int A_PAD = 4;
constexpr int B_PAD = 4;

// ==========================================================================
// cp.async utilities
// ==========================================================================
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr;
    asm volatile(
        "{ .reg .u64 smem64;\n"
        "  cvta.to.shared.u64 smem64, %1;\n"
        "  cvt.u32.u64 %0, smem64; }\n"
        : "=r"(smem_addr) : "l"(smem_ptr)
    );
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_1() {
    asm volatile("cp.async.wait_group 1;");
}

// ==========================================================================
// Kernel
// ==========================================================================
__global__ void __launch_bounds__(256, 2)
sgemm_tf32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    const int warp_m = (warp_id / WARPS_N) * WARP_TILES_M * WMMA_M;
    const int warp_n = (warp_id % WARPS_N) * WARP_TILES_N * WMMA_N;

    // Shared memory layout
    __shared__ float smA[2][BM][BK + A_PAD];
    __shared__ float smB[2][BK][BN + B_PAD];

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag[WARP_TILES_M];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> b_frag[WARP_TILES_N];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ni++)
            fill_fragment(c_frag[mi][ni], 0.0f);

    int k_tiles = K / BK;

    // -------------------------------
    // Load tile helper
    // -------------------------------
    auto load_A = [&](int stage, int kt) {
        int k0 = kt * BK;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 256;
            int m = idx / (BK / 4);
            int k = (idx % (BK / 4)) * 4;
            if (m < BM && k0 + k < K) {
                cp_async_cg_16(&smA[stage][m][k], &A[(block_m + m) * K + (k0 + k)]);
            }
        }
    };

    auto load_B = [&](int stage, int kt) {
        int k0 = kt * BK;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 256;
            int k = idx / (BN / 4);
            int n = (idx % (BN / 4)) * 4;
            if (n < BN && k0 + k < K) {
                float4 v = *reinterpret_cast<const float4*>(
                    &B[(k0 + k) * N + (block_n + n)]
                );
                smB[stage][k][n + 0] = v.x;
                smB[stage][k][n + 1] = v.y;
                smB[stage][k][n + 2] = v.z;
                smB[stage][k][n + 3] = v.w;
            }
        }
    };

    // -------------------------------
    // Prologue
    // -------------------------------
    load_A(0, 0);
    load_B(0, 0);
    cp_async_commit();

    if (k_tiles > 1) {
        load_A(1, 1);
        load_B(1, 1);
        cp_async_commit();
    }

    cp_async_wait_1();
    __syncthreads();

    // -------------------------------
    // Main loop
    // -------------------------------
    for (int kt = 0; kt < k_tiles; kt++) {

        int curr = kt & 1;
        int next = 1 - curr;

        // Prefetch tile kt+2
        if (kt + 2 < k_tiles) {
            load_A(next, kt + 2);
            load_B(next, kt + 2);
            cp_async_commit();
        }

        // Compute on curr
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {

            #pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; mi++) {
                int m0 = warp_m + mi * WMMA_M;
                load_matrix_sync(
                    a_frag[mi],
                    &smA[curr][m0][kk],
                    BK + A_PAD
                );
            }

            #pragma unroll
            for (int ni = 0; ni < WARP_TILES_N; ni++) {
                int n0 = warp_n + ni * WMMA_N;
                load_matrix_sync(
                    b_frag[ni],
                    &smB[curr][kk][n0],
                    BN + B_PAD
                );
            }

            #pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; mi++)
                #pragma unroll
                for (int ni = 0; ni < WARP_TILES_N; ni++)
                    mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }

        cp_async_wait_1();
        __syncthreads();
    }

    // -------------------------------
    // Epilogue
    // -------------------------------
    const bool aligned = (N % 8 == 0);

    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; mi++) {
        for (int ni = 0; ni < WARP_TILES_N; ni++) {

            int m0 = block_m + warp_m + mi * WMMA_M;
            int n0 = block_n + warp_n + ni * WMMA_N;

            int valid_m = min(WMMA_M, M - m0);
            int valid_n = min(WMMA_N, N - n0);

            if (aligned && valid_m == WMMA_M && valid_n == WMMA_N) {
                store_matrix_sync(
                    &C[m0 * N + n0],
                    c_frag[mi][ni],
                    (unsigned int)N,
                    mem_row_major
                );
            } else {
                float tmp[WMMA_M][WMMA_N];
                store_matrix_sync(&tmp[0][0], c_frag[mi][ni], WMMA_N, mem_row_major);

                if (lane_id < WMMA_N) {
                    for (int r = 0; r < valid_m; r++) {
                        if (n0 + lane_id < N)
                            C[(m0 + r) * N + (n0 + lane_id)] = tmp[r][lane_id];
                    }
                }
            }
        }
    }
}

// ==========================================================================
// Launcher
// ==========================================================================
inline cudaError_t launch_sgemm_tf32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tf32_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

} // namespace tf32
} // namespace ops
} // namespace pygpukit
