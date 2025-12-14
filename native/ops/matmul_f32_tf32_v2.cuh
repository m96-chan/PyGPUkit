/**
 * TF32 TensorCore GEMM v2 - 3-stage pipeline with BK=8
 *
 * Target: 90%+ of cuBLAS performance (37.6+ TFLOPS on RTX 3090 Ti)
 *
 * Key insight: BK=8 uses less shared memory, enabling 3-stage pipelining
 * without occupancy loss.
 *
 * Shared memory: 2 * 128 * 12 * 4 = 12KB for A per stage
 *                2 * 8 * 132 * 4 = 8KB for B per stage
 *                Total: 3 * 20KB = 60KB (fits in 100KB limit)
 */

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace pygpukit {
namespace ops {
namespace tf32_v2 {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;  // Reduced from 16 to enable 3-stage

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 8;

constexpr int STAGES = 3;

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

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// ============================================================
// Main kernel with 3-stage pipeline
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

    __shared__ float smA[STAGES][BM][BK + A_PAD];
    __shared__ float smB[STAGES][BK][BN + B_PAD];

    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = K / BK;

    // Fragment index mappings
    const int a_row_base = lane / 4;
    const int a_col_base = lane % 4;
    const int b_row_base = lane % 4;
    const int b_col = lane / 4;
    const int c_row_base = lane / 4;
    const int c_col_base = (lane % 4) * 2;

    // Load A: 128x8 = 1024 elements, 256 threads, 4 elements per thread
    auto load_A_async = [&](int stage, int kt) {
        const int a_row = tid / 2;       // 0-127
        const int a_col = (tid % 2) * 4; // 0 or 4

        int gm = cta_m + a_row;
        int gk = kt * BK + a_col;
        if (gm < M && gk < K) {
            cp_async_16(&smA[stage][a_row][a_col], &A[gm * K + gk]);
        }
    };

    // Load B: 8x128 = 1024 elements, 256 threads, 4 elements per thread
    auto load_B_async = [&](int stage, int kt) {
        const int b_row = tid / 32;      // 0-7
        const int b_col = (tid % 32) * 4; // 0, 4, ..., 124

        int gk = kt * BK + b_row;
        int gn = cta_n + b_col;
        if (gk < K && gn < N) {
            cp_async_16(&smB[stage][b_row][b_col], &B[gk * N + gn]);
        }
    };

    // Prologue: fill stages 0, 1
    load_A_async(0, 0);
    load_B_async(0, 0);
    cp_async_commit();

    if (num_k_tiles > 1) {
        load_A_async(1, 1);
        load_B_async(1, 1);
    }
    cp_async_commit();

    // Wait for stage 0 to be ready
    cp_async_wait<1>();
    __syncthreads();

    // Main loop
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt % STAGES;

        // Prefetch stage kt+2
        if (kt + 2 < num_k_tiles) {
            int prefetch_stage = (kt + 2) % STAGES;
            load_A_async(prefetch_stage, kt + 2);
            load_B_async(prefetch_stage, kt + 2);
        }
        cp_async_commit();

        // Process current tile (BK=8 means only 1 MMA_K iteration)
        #pragma unroll
        for (int wm = 0; wm < WARP_TILES_M; ++wm) {
            int tile_m = warp_m + wm * MMA_M;
            float a0 = smA[curr][tile_m + a_row_base][a_col_base];
            float a1 = smA[curr][tile_m + a_row_base + 8][a_col_base];
            float a2 = smA[curr][tile_m + a_row_base][a_col_base + 4];
            float a3 = smA[curr][tile_m + a_row_base + 8][a_col_base + 4];

            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                int tile_n = warp_n + wn * MMA_N;
                float b0 = smB[curr][b_row_base][tile_n + b_col];
                float b1 = smB[curr][b_row_base + 4][tile_n + b_col];

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

        // Wait for next stage
        if (kt + 1 < num_k_tiles) {
            cp_async_wait<1>();
            __syncthreads();
        }
    }

    // Epilogue
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
