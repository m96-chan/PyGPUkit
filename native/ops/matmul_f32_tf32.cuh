/**
 * TF32 TensorCore GEMM Kernel for Ampere+ GPUs (SM 80+)
 *
 * Target: 25 TFLOPS on RTX 3090 Ti
 *
 * Architecture:
 * - BM=128, BN=128, BK=16
 * - 256 threads (16x16), 8 warps
 * - 2-stage cp.async pipeline with wait_group(1)
 * - ~40KB shared memory -> 2 blocks/SM
 *
 * Warp mapping: 4x2 grid (4 rows, 2 cols)
 * Each warp computes 2x4 WMMA tiles = 32x64 output
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace pygpukit {
namespace ops {
namespace tf32 {

using namespace nvcuda::wmma;

// ============================================================================
// Tile Configuration
// ============================================================================
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;

constexpr int A_PAD = 4;
constexpr int B_PAD = 4;

// ============================================================================
// cp.async Intrinsics
// ============================================================================

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
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group_1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// ============================================================================
// TF32 WMMA Kernel with 2-stage Pipeline
// ============================================================================

__global__ void __launch_bounds__(256, 2)
sgemm_tf32_wmma_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int cta_m = by * BM;
    const int cta_n = bx * BN;

    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int warp_m = warp_row * (WARP_TILES_M * WMMA_M);
    const int warp_n = warp_col * (WARP_TILES_N * WMMA_N);

    // A_smem[stage][m][k] - row-major
    // B_smem[stage][n][k] - transposed for col_major WMMA
    __shared__ float A_smem[2][BM][BK + A_PAD];
    __shared__ float B_smem[2][BN][BK + B_PAD];

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag[WARP_TILES_M];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> b_frag[WARP_TILES_N];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
            fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    const int num_k_tiles = K / BK;

    auto load_A_tile = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * 256;
            const int m = idx / 4;
            const int k = (idx % 4) * 4;
            cp_async_cg_16(&A_smem[stage][m][k], &A[(cta_m + m) * K + k_base + k]);
        }
    };

    auto load_B_tile = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * 256;
            const int k = idx / 32;
            const int n = (idx % 32) * 4;
            float4 tmp = *reinterpret_cast<const float4*>(&B[(k_base + k) * N + cta_n + n]);
            B_smem[stage][n + 0][k] = tmp.x;
            B_smem[stage][n + 1][k] = tmp.y;
            B_smem[stage][n + 2][k] = tmp.z;
            B_smem[stage][n + 3][k] = tmp.w;
        }
    };

    // PROLOGUE
    load_A_tile(0, 0);
    load_B_tile(0, 0);
    cp_async_commit();

    if (num_k_tiles > 1) {
        load_A_tile(1, 1);
        load_B_tile(1, 1);
        cp_async_commit();
    }

    cp_async_wait_group_1();
    __syncthreads();

    // MAIN LOOP
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int curr = k_tile & 1;
        const int next = 1 - curr;

        if (k_tile + 2 < num_k_tiles) {
            load_A_tile(next, k_tile + 2);
            load_B_tile(next, k_tile + 2);
        }
        cp_async_commit();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                const int m_off = warp_m + mi * WMMA_M;
                load_matrix_sync(a_frag[mi], &A_smem[curr][m_off][kk], BK + A_PAD);
            }

            #pragma unroll
            for (int ni = 0; ni < WARP_TILES_N; ++ni) {
                const int n_off = warp_n + ni * WMMA_N;
                load_matrix_sync(b_frag[ni], &B_smem[curr][n_off][kk], BK + B_PAD);
            }

            #pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < WARP_TILES_N; ++ni) {
                    mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
                }
            }
        }

        cp_async_wait_group_1();
        __syncthreads();
    }

    // EPILOGUE
    __shared__ float C_smem[8][WMMA_M][WMMA_N + 4];
    const bool aligned = (N % 8 == 0);

    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
            const int m_off = cta_m + warp_m + mi * WMMA_M;
            const int n_off = cta_n + warp_n + ni * WMMA_N;

            if (m_off < M && n_off < N) {
                const int valid_m = min(WMMA_M, M - m_off);
                const int valid_n = min(WMMA_N, N - n_off);

                if (aligned && valid_m == WMMA_M && valid_n == WMMA_N) {
                    store_matrix_sync(&C[m_off * N + n_off], c_frag[mi][ni], N, mem_row_major);
                } else {
                    store_matrix_sync(&C_smem[warp_id][0][0], c_frag[mi][ni], WMMA_N + 4, mem_row_major);
                    __syncwarp();
                    if (lane_id < 16) {
                        for (int r = 0; r < valid_m; ++r) {
                            if (n_off + lane_id < N) {
                                C[(m_off + r) * N + n_off + lane_id] = C_smem[warp_id][r][lane_id];
                            }
                        }
                    }
                    __syncwarp();
                }
            }
        }
    }
}

inline cudaError_t launch_sgemm_tf32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tf32_wmma_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

}  // namespace tf32
}  // namespace ops
}  // namespace pygpukit
