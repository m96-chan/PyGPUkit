/**
 * TF32 TensorCore GEMM Kernel - High Performance Version
 * Target: 25+ TFLOPS on RTX 3090 Ti
 *
 * Key Design Principles:
 * 1. BOTH A and B use cp.async (no synchronous scatter-stores)
 * 2. B stored row-major K×N (not transposed)
 * 3. row_major fragments for both A and B
 * 4. True 2-stage cp.async pipeline
 * 5. ~37KB shared memory → 2 blocks/SM
 */

#pragma once

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
constexpr int BK = 16;       // BK=16 for 2-stage pipeline under 40KB

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

constexpr int WARPS_M = 4;   // 4 warps vertically
constexpr int WARPS_N = 2;   // 2 warps horizontally
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;

// Padding for bank conflict avoidance
constexpr int A_PAD = 4;     // A stride = BK + 4 = 20
constexpr int B_PAD = 4;     // B stride = BN + 4 = 132

// Shared memory sizes:
// A_smem: 2 × 128 × 20 × 4 = 20,480 bytes
// B_smem: 2 × 16 × 132 × 4 = 16,896 bytes
// Total: 37,376 bytes ≈ 37KB (allows 2 blocks/SM)

// ============================================================================
// cp.async Intrinsics
// ============================================================================

__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* gmem) {
    uint32_t smem_addr;
    asm volatile(
        "{ .reg .u64 s64;\n"
        "  cvta.to.shared.u64 s64, %1;\n"
        "  cvt.u32.u64 %0, s64; }\n"
        : "=r"(smem_addr) : "l"(smem)
    );
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem)
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
// Main Kernel
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

    // Warp position in 4×2 grid
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int warp_m = warp_row * (WARP_TILES_M * WMMA_M);  // 0, 32, 64, 96
    const int warp_n = warp_col * (WARP_TILES_N * WMMA_N);  // 0, 64

    // Shared memory: row-major for both A and B
    // A: [stage][m][k] - M×K row-major
    // B: [stage][k][n] - K×N row-major (NOT transposed!)
    __shared__ float A_smem[2][BM][BK + A_PAD];
    __shared__ float B_smem[2][BK][BN + B_PAD];

    // WMMA fragments - both row_major since both are stored row-major
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag[WARP_TILES_M];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag[WARP_TILES_N];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];

    // Initialize accumulators
    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
            fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    const int num_k_tiles = K / BK;

    // ========================================================================
    // Load A tile: 128×16 = 2048 floats = 512 float4
    // 256 threads × 2 iterations = 512 cp.async loads
    // ========================================================================
    auto load_A_tile = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * 256;
            const int m = idx / 4;            // 0-127 (BM rows)
            const int k = (idx % 4) * 4;      // 0,4,8,12 (BK/4 groups)
            cp_async_cg_16(&A_smem[stage][m][k], &A[(cta_m + m) * K + k_base + k]);
        }
    };

    // ========================================================================
    // Load B tile: 16×128 = 2048 floats = 512 float4
    // 256 threads × 2 iterations = 512 cp.async loads
    // B is loaded directly into K×N row-major layout (NO transpose!)
    // ========================================================================
    auto load_B_tile = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * 256;
            const int k = idx / 32;           // 0-15 (BK rows)
            const int n = (idx % 32) * 4;     // 0,4,8,...,124 (BN/4 groups)
            cp_async_cg_16(&B_smem[stage][k][n], &B[(k_base + k) * N + cta_n + n]);
        }
    };

    // ========================================================================
    // PROLOGUE: Load first 2 tiles
    // ========================================================================
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

    // ========================================================================
    // MAIN LOOP: 2-stage pipeline
    // ========================================================================
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int curr = k_tile & 1;
        const int next = 1 - curr;

        // 1. Prefetch tile (k+2) into next stage
        if (k_tile + 2 < num_k_tiles) {
            load_A_tile(next, k_tile + 2);
            load_B_tile(next, k_tile + 2);
        }

        // 2. Compute MMA on current tile
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            // Load A fragments
            #pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                const int m_off = warp_m + mi * WMMA_M;
                load_matrix_sync(a_frag[mi], &A_smem[curr][m_off][kk], BK + A_PAD);
            }

            // Load B fragments (row_major from K×N layout)
            // ldm = BN + B_PAD = 132 (stride between rows)
            #pragma unroll
            for (int ni = 0; ni < WARP_TILES_N; ++ni) {
                const int n_off = warp_n + ni * WMMA_N;
                load_matrix_sync(b_frag[ni], &B_smem[curr][kk][n_off], BN + B_PAD);
            }

            // Matrix multiply-accumulate
            #pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < WARP_TILES_N; ++ni) {
                    mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
                }
            }
        }

        // 3. Commit prefetch group
        cp_async_commit();

        // 4. Wait for previous prefetch
        cp_async_wait_group_1();

        // 5. Synchronize
        __syncthreads();
    }

    // ========================================================================
    // EPILOGUE: Store results
    // ========================================================================
    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
            const int m_off = cta_m + warp_m + mi * WMMA_M;
            const int n_off = cta_n + warp_n + ni * WMMA_N;

            if (m_off < M && n_off < N) {
                // Direct store for aligned full tiles
                if (m_off + WMMA_M <= M && n_off + WMMA_N <= N) {
                    store_matrix_sync(&C[m_off * N + n_off], c_frag[mi][ni], N, mem_row_major);
                } else {
                    // Partial tile: use shared memory staging
                    __shared__ float C_tile[WMMA_M][WMMA_N];
                    store_matrix_sync(&C_tile[0][0], c_frag[mi][ni], WMMA_N, mem_row_major);
                    __syncwarp();

                    const int valid_m = min(WMMA_M, M - m_off);
                    const int valid_n = min(WMMA_N, N - n_off);

                    // Cooperative store
                    for (int r = lane_id; r < valid_m * valid_n; r += 32) {
                        const int row = r / valid_n;
                        const int col = r % valid_n;
                        C[(m_off + row) * N + n_off + col] = C_tile[row][col];
                    }
                    __syncwarp();
                }
            }
        }
    }
}

// ============================================================================
// Launch Function
// ============================================================================

inline cudaError_t launch_sgemm_tf32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);  // 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tf32_wmma_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

}  // namespace tf32
}  // namespace ops
}  // namespace pygpukit
