/**
 * TF32 TensorCore GEMM v2 - Double nested pipeline (CUTLASS-style)
 *
 * Target: 90%+ of cuBLAS performance (37.6+ TFLOPS on RTX 3090 Ti)
 *
 * Key insight from CUTLASS:
 * - Outer pipeline: GMEM -> SMEM (cp.async)
 * - Inner pipeline: SMEM -> RMEM (software pipelined)
 */

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../../../../../core/cuda_graph.hpp"

namespace pygpukit {
namespace ops {
namespace tf32_v2 {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 8;

constexpr int A_PAD = 4;
constexpr int B_PAD = 4;

// ============================================================
// cp.async helpers
// ============================================================
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

// ============================================================
// Main kernel with double nested pipeline
// ============================================================
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

    const int warp_m = warp_row * (WARP_TILES_M * MMA_M);
    const int warp_n = warp_col * (WARP_TILES_N * MMA_N);

    __shared__ float smA[2][BM][BK + A_PAD];
    __shared__ float smB[2][BK][BN + B_PAD];

    // Accumulators
    float acc[WARP_TILES_M][WARP_TILES_N][4];
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; ++wn) {
            acc[wm][wn][0] = 0.0f;
            acc[wm][wn][1] = 0.0f;
            acc[wm][wn][2] = 0.0f;
            acc[wm][wn][3] = 0.0f;
        }
    }

    const int num_k_tiles = K / BK;

    // Fragment index mappings
    const int a_row_base = lane / 4;
    const int a_col_base = lane % 4;
    const int b_row_base = lane % 4;
    const int b_col = lane / 4;
    const int c_row_base = lane / 4;
    const int c_col_base = (lane % 4) * 2;

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
        const int b_row = tid / 32;
        const int b_col_ld = (tid % 32) * 4;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int k = b_row + i * 8;
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

    // Main loop with outer pipeline (GMEM->SMEM)
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        // Prefetch next tile
        load_A_async(next, kt + 1);
        load_B_async(next, kt + 1);
        cp_async_commit();

        // Inner pipeline: process all 4 kk iterations with preloaded A fragments
        #pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            // Load A fragments for this kk
            float a_frag[WARP_TILES_M][4];
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                int tile_m = warp_m + wm * MMA_M;
                a_frag[wm][0] = smA[curr][tile_m + a_row_base][kk + a_col_base];
                a_frag[wm][1] = smA[curr][tile_m + a_row_base + 8][kk + a_col_base];
                a_frag[wm][2] = smA[curr][tile_m + a_row_base][kk + a_col_base + 4];
                a_frag[wm][3] = smA[curr][tile_m + a_row_base + 8][kk + a_col_base + 4];
            }

            // Process all B tiles with preloaded A
            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                int tile_n = warp_n + wn * MMA_N;
                float b0 = smB[curr][kk + b_row_base][tile_n + b_col];
                float b1 = smB[curr][kk + b_row_base + 4][tile_n + b_col];

                #pragma unroll
                for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(acc[wm][wn][0]), "+f"(acc[wm][wn][1]),
                          "+f"(acc[wm][wn][2]), "+f"(acc[wm][wn][3])
                        : "r"(__float_as_uint(a_frag[wm][0])), "r"(__float_as_uint(a_frag[wm][1])),
                          "r"(__float_as_uint(a_frag[wm][2])), "r"(__float_as_uint(a_frag[wm][3])),
                          "r"(__float_as_uint(b0)), "r"(__float_as_uint(b1))
                    );
                }
            }
        }

        cp_async_wait_0();
        __syncthreads();
    }

    // Epilogue - vectorized stores
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; ++wn) {
            int tile_m = cta_m + warp_m + wm * MMA_M;
            int tile_n = cta_n + warp_n + wn * MMA_N;

            int out_row0 = tile_m + c_row_base;
            int out_row1 = tile_m + c_row_base + 8;
            int out_col0 = tile_n + c_col_base;

            if (out_row0 < M && out_col0 + 1 < N) {
                float2 v = make_float2(acc[wm][wn][0], acc[wm][wn][1]);
                *reinterpret_cast<float2*>(&C[out_row0 * N + out_col0]) = v;
            }
            if (out_row1 < M && out_col0 + 1 < N) {
                float2 v = make_float2(acc[wm][wn][2], acc[wm][wn][3]);
                *reinterpret_cast<float2*>(&C[out_row1 * N + out_col0]) = v;
            }
        }
    }
}

inline cudaError_t launch_sgemm_tf32_v2(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    cudaStream_t stream = internal::get_capture_stream();
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tf32_v2_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

} // namespace tf32_v2
} // namespace ops
} // namespace pygpukit
