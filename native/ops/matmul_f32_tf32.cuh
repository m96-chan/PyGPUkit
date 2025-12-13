#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace pygpukit {
namespace ops {
namespace tf32 {

using namespace nvcuda;

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

// ----------------------------------------------------------------------------
// cp.async wrapper (16 bytes)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr;
    asm volatile(
        "{ .reg .u64 smem64;\n"
        "  cvta.to.shared.u64 smem64, %1;\n"
        "  cvt.u32.u64 %0, smem64; }\n"
        : "=r"(smem_addr)
        : "l"(smem_ptr)
    );
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

// ============================================================================
// TF32 TensorCore GEMM Kernel (2-stage pipeline)
// ============================================================================
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

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int cta_m = by * BM;
    const int cta_n = bx * BN;

    const int warp_m = (warp_id / WARPS_N) * (WARP_TILES_M * WMMA_M);
    const int warp_n = (warp_id % WARPS_N) * (WARP_TILES_N * WMMA_N);

    // Shared memory (2 stages)
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BN][BK];   // IMPORTANT: [n][k] = col-major

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag[WARP_TILES_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag[WARP_TILES_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    const int num_tiles = K / BK;

    // ========================================================================
    // LOAD TILE 0 & 1  (Pipeline Prologue)
    // ========================================================================
    auto load_A = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 256;
            int m = idx / 4;
            int k4 = (idx % 4) * 4;
            cp_async_16(&As[stage][m][k4], &A[(cta_m + m) * K + k_base + k4]);
        }
    };

    auto load_B = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * 256;
            int k = idx / 32;
            int n = (idx % 32) * 4;

            // global → col-major Bs[n][k]
            float4 v = *reinterpret_cast<const float4*>(&B[(k_base + k) * N + (cta_n + n)]);
            Bs[stage][n + 0][k] = v.x;
            Bs[stage][n + 1][k] = v.y;
            Bs[stage][n + 2][k] = v.z;
            Bs[stage][n + 3][k] = v.w;
        }
    };

    // Prologue load
    load_A(0, 0);
    load_B(0, 0);
    cp_async_commit();

    if (num_tiles > 1) {
        load_A(1, 1);
        load_B(1, 1);
        cp_async_commit();
    }

    cp_async_wait_1();
    __syncthreads();

    // ========================================================================
    // MAIN LOOP
    // ========================================================================
    for (int t = 0; t < num_tiles; t++) {
        int curr = t & 1;
        int next = (t + 2) & 1;

        // Prefetch tile t+2
        if (t + 2 < num_tiles) {
            load_A(next, t + 2);
            load_B(next, t + 2);
            cp_async_commit();
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; i++) {
                int m_off = warp_m + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[curr][m_off][kk], BK);
            }

            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++) {
                int n_off = warp_n + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[curr][n_off][kk], BK);
            }

            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; i++)
                for (int j = 0; j < WARP_TILES_N; j++)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }

        cp_async_wait_1();
        __syncthreads();
    }

    // ========================================================================
    // STORE (fast path only for aligned tile)
    // ========================================================================
    const bool aligned = (N % 8 == 0);

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++) {
        for (int j = 0; j < WARP_TILES_N; j++) {
            int m_off = cta_m + warp_m + i * WMMA_M;
            int n_off = cta_n + warp_n + j * WMMA_N;

            if (m_off < M && n_off < N) {
                int valid_m = min(WMMA_M, M - m_off);
                int valid_n = min(WMMA_N, N - n_off);

                if (aligned && valid_m == 16 && valid_n == 16) {
                    wmma::store_matrix_sync(&C[m_off * N + n_off], c_frag[i][j], N, wmma::mem_row_major);
                } else {
                    float tmp[16 * 16];
                    wmma::store_matrix_sync(tmp, c_frag[i][j], 16, wmma::mem_row_major);
                    for (int r = 0; r < valid_m; r++)
                        for (int c = 0; c < valid_n; c++)
                            C[(m_off + r) * N + (n_off + c)] = tmp[r * 16 + c];
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------

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

}  // namespace tf32
}  // namespace ops
}  // namespace pygpukit
