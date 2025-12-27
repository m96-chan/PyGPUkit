/**
 * FP16/BF16 TensorCore Matrix Multiplication
 *
 * Uses mma.sync.aligned.m16n8k16 for TensorCore acceleration
 * - FP16: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 * - BF16: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 *
 * Both use FP32 accumulation for numerical stability.
 *
 * Performance target: 50+ TFLOPS on RTX 3090 Ti
 */

#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../../../../core/cuda_graph.hpp"

namespace pygpukit {
namespace ops {
namespace fp16_bf16_tc {

// Block tile dimensions
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;  // K=16 per MMA, 2 MMAs per BK iteration

// MMA tile dimensions (m16n8k16)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// Warp configuration
constexpr int WARPS_M = 4;  // 4 warps along M
constexpr int WARPS_N = 2;  // 2 warps along N
constexpr int WARP_TILES_M = 2;  // 2 MMA tiles per warp along M
constexpr int WARP_TILES_N = 8;  // 8 MMA tiles per warp along N

// Padding to avoid bank conflicts
constexpr int A_PAD = 8;
constexpr int B_PAD = 8;

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
// FP16 TensorCore GEMM Kernel (FP32 accumulation)
// ============================================================
__global__ void __launch_bounds__(256, 2)
sgemm_f16_tc_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
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

    // Shared memory: store as half
    __shared__ __half smA[2][BM][BK + A_PAD];
    __shared__ __half smB[2][BK][BN + B_PAD];

    // Accumulators (FP32)
    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = K / BK;

    // Fragment index mappings for m16n8k16
    // A fragment: 8 half elements per thread = 4 packed uint32
    // groupID = lane >> 2, threadID_in_group = lane % 4
    const int groupID = lane >> 2;           // 0-7
    const int tid_in_group = lane & 3;       // 0-3

    // C fragment mapping (same as TF32 m16n8k8 output)
    const int c_row_base = groupID;
    const int c_col_base = tid_in_group * 2;

    // ====== cp.async load helpers ======
    auto load_A_async = [&](int stage, int kt) {
        // 256 threads, load BM*BK = 128*32 = 4096 halves
        // Each thread loads 16 halves = 32 bytes = 2x cp.async_16
        const int elems_per_thread = (BM * BK) / 256;  // 16
        const int half_per_load = 8;  // cp.async_16 loads 8 halves

        #pragma unroll
        for (int i = 0; i < elems_per_thread / half_per_load; ++i) {
            int elem_idx = tid * (elems_per_thread / half_per_load) + i;
            int row = (elem_idx * half_per_load) / BK;
            int col = (elem_idx * half_per_load) % BK;
            int gm = cta_m + row;
            int gk = kt * BK + col;
            if (gm < M && gk + 7 < K) {
                cp_async_16(&smA[stage][row][col], &A[gm * K + gk]);
            }
        }
    };

    auto load_B_async = [&](int stage, int kt) {
        // 256 threads, load BK*BN = 32*128 = 4096 halves
        const int elems_per_thread = (BK * BN) / 256;  // 16
        const int half_per_load = 8;

        #pragma unroll
        for (int i = 0; i < elems_per_thread / half_per_load; ++i) {
            int elem_idx = tid * (elems_per_thread / half_per_load) + i;
            int row = (elem_idx * half_per_load) / BN;
            int col = (elem_idx * half_per_load) % BN;
            int gk = kt * BK + row;
            int gn = cta_n + col;
            if (gk < K && gn + 7 < N) {
                cp_async_16(&smB[stage][row][col], &B[gk * N + gn]);
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

        // Process current tile: 2 MMA iterations per BK (BK=32, MMA_K=16)
        #pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                int tile_m = warp_m + wm * MMA_M;

                // Load A fragment (8 halves = 4 packed uint32)
                // A is 16x16, row-major
                // For mma.m16n8k16:
                // a[i] where i=0..7 maps to:
                //   row = groupID + 8 * ((i/2) % 2)
                //   col = tid_in_group * 2 + (i % 2) + 8 * (i / 4)
                uint32_t a_frag[4];

                // Pack halves into uint32
                // a_frag[0] = (a[1] << 16) | a[0]
                // a_frag[1] = (a[3] << 16) | a[2]
                // a_frag[2] = (a[5] << 16) | a[4]
                // a_frag[3] = (a[7] << 16) | a[6]

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    int i0 = p * 2;
                    int i1 = p * 2 + 1;

                    int row0 = groupID + 8 * ((i0 / 2) % 2);
                    int col0 = tid_in_group * 2 + (i0 % 2) + 8 * (i0 / 4);
                    int row1 = groupID + 8 * ((i1 / 2) % 2);
                    int col1 = tid_in_group * 2 + (i1 % 2) + 8 * (i1 / 4);

                    __half h0 = smA[curr][tile_m + row0][kk + col0];
                    __half h1 = smA[curr][tile_m + row1][kk + col1];

                    // Pack two halves into uint32
                    a_frag[p] = __half_as_ushort(h0) | (uint32_t(__half_as_ushort(h1)) << 16);
                }

                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                    int tile_n = warp_n + wn * MMA_N;

                    // Load B fragment (4 halves = 2 packed uint32)
                    // B is 16x8, col-major storage in row-major layout
                    // For mma.m16n8k16:
                    // b[i] where i=0..3 maps to:
                    //   row = tid_in_group * 2 + (i % 2) + 8 * (i / 2)
                    //   col = groupID
                    uint32_t b_frag[2];

                    #pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        int i0 = p * 2;
                        int i1 = p * 2 + 1;

                        int row0 = tid_in_group * 2 + (i0 % 2) + 8 * (i0 / 2);
                        int col0 = groupID;
                        int row1 = tid_in_group * 2 + (i1 % 2) + 8 * (i1 / 2);
                        int col1 = groupID;

                        __half h0 = smB[curr][kk + row0][tile_n + col0];
                        __half h1 = smB[curr][kk + row1][tile_n + col1];

                        b_frag[p] = __half_as_ushort(h0) | (uint32_t(__half_as_ushort(h1)) << 16);
                    }

                    // Execute MMA: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(acc[wm][wn][0]), "+f"(acc[wm][wn][1]),
                          "+f"(acc[wm][wn][2]), "+f"(acc[wm][wn][3])
                        : "r"(a_frag[0]), "r"(a_frag[1]),
                          "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_frag[0]), "r"(b_frag[1])
                    );
                }
            }
        }

        cp_async_wait_0();
        __syncthreads();
    }

    // ====== Epilogue: Store results ======
    // C fragment mapping (16x8):
    // c[i] where i=0..3:
    //   row = groupID + 8 * (i / 2)
    //   col = tid_in_group * 2 + (i % 2)
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; ++wn) {
            int tile_m = cta_m + warp_m + wm * MMA_M;
            int tile_n = cta_n + warp_n + wn * MMA_N;

            int out_row0 = tile_m + c_row_base;
            int out_row1 = tile_m + c_row_base + 8;
            int out_col0 = tile_n + c_col_base;
            int out_col1 = tile_n + c_col_base + 1;

            if (out_row0 < M && out_col0 < N) C[out_row0 * N + out_col0] = __float2half(acc[wm][wn][0]);
            if (out_row0 < M && out_col1 < N) C[out_row0 * N + out_col1] = __float2half(acc[wm][wn][1]);
            if (out_row1 < M && out_col0 < N) C[out_row1 * N + out_col0] = __float2half(acc[wm][wn][2]);
            if (out_row1 < M && out_col1 < N) C[out_row1 * N + out_col1] = __float2half(acc[wm][wn][3]);
        }
    }
}

// ============================================================
// BF16 TensorCore GEMM Kernel (FP32 accumulation)
// ============================================================
__global__ void __launch_bounds__(256, 2)
sgemm_bf16_tc_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
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

    __shared__ __nv_bfloat16 smA[2][BM][BK + A_PAD];
    __shared__ __nv_bfloat16 smB[2][BK][BN + B_PAD];

    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = K / BK;

    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;
    const int c_row_base = groupID;
    const int c_col_base = tid_in_group * 2;

    auto load_A_async = [&](int stage, int kt) {
        const int elems_per_thread = (BM * BK) / 256;
        const int half_per_load = 8;

        #pragma unroll
        for (int i = 0; i < elems_per_thread / half_per_load; ++i) {
            int elem_idx = tid * (elems_per_thread / half_per_load) + i;
            int row = (elem_idx * half_per_load) / BK;
            int col = (elem_idx * half_per_load) % BK;
            int gm = cta_m + row;
            int gk = kt * BK + col;
            if (gm < M && gk + 7 < K) {
                cp_async_16(&smA[stage][row][col], &A[gm * K + gk]);
            }
        }
    };

    auto load_B_async = [&](int stage, int kt) {
        const int elems_per_thread = (BK * BN) / 256;
        const int half_per_load = 8;

        #pragma unroll
        for (int i = 0; i < elems_per_thread / half_per_load; ++i) {
            int elem_idx = tid * (elems_per_thread / half_per_load) + i;
            int row = (elem_idx * half_per_load) / BN;
            int col = (elem_idx * half_per_load) % BN;
            int gk = kt * BK + row;
            int gn = cta_n + col;
            if (gk < K && gn + 7 < N) {
                cp_async_16(&smB[stage][row][col], &B[gk * N + gn]);
            }
        }
    };

    load_A_async(0, 0);
    load_B_async(0, 0);
    cp_async_commit();
    cp_async_wait_0();
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        if (kt + 1 < num_k_tiles) {
            load_A_async(next, kt + 1);
            load_B_async(next, kt + 1);
        }
        cp_async_commit();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                int tile_m = warp_m + wm * MMA_M;

                uint32_t a_frag[4];

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    int i0 = p * 2;
                    int i1 = p * 2 + 1;

                    int row0 = groupID + 8 * ((i0 / 2) % 2);
                    int col0 = tid_in_group * 2 + (i0 % 2) + 8 * (i0 / 4);
                    int row1 = groupID + 8 * ((i1 / 2) % 2);
                    int col1 = tid_in_group * 2 + (i1 % 2) + 8 * (i1 / 4);

                    __nv_bfloat16 h0 = smA[curr][tile_m + row0][kk + col0];
                    __nv_bfloat16 h1 = smA[curr][tile_m + row1][kk + col1];

                    a_frag[p] = __bfloat16_as_ushort(h0) | (uint32_t(__bfloat16_as_ushort(h1)) << 16);
                }

                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                    int tile_n = warp_n + wn * MMA_N;

                    uint32_t b_frag[2];

                    #pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        int i0 = p * 2;
                        int i1 = p * 2 + 1;

                        int row0 = tid_in_group * 2 + (i0 % 2) + 8 * (i0 / 2);
                        int col0 = groupID;
                        int row1 = tid_in_group * 2 + (i1 % 2) + 8 * (i1 / 2);
                        int col1 = groupID;

                        __nv_bfloat16 h0 = smB[curr][kk + row0][tile_n + col0];
                        __nv_bfloat16 h1 = smB[curr][kk + row1][tile_n + col1];

                        b_frag[p] = __bfloat16_as_ushort(h0) | (uint32_t(__bfloat16_as_ushort(h1)) << 16);
                    }

                    // Execute MMA: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(acc[wm][wn][0]), "+f"(acc[wm][wn][1]),
                          "+f"(acc[wm][wn][2]), "+f"(acc[wm][wn][3])
                        : "r"(a_frag[0]), "r"(a_frag[1]),
                          "r"(a_frag[2]), "r"(a_frag[3]),
                          "r"(b_frag[0]), "r"(b_frag[1])
                    );
                }
            }
        }

        cp_async_wait_0();
        __syncthreads();
    }

    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; ++wn) {
            int tile_m = cta_m + warp_m + wm * MMA_M;
            int tile_n = cta_n + warp_n + wn * MMA_N;

            int out_row0 = tile_m + c_row_base;
            int out_row1 = tile_m + c_row_base + 8;
            int out_col0 = tile_n + c_col_base;
            int out_col1 = tile_n + c_col_base + 1;

            if (out_row0 < M && out_col0 < N) C[out_row0 * N + out_col0] = __float2bfloat16_rn(acc[wm][wn][0]);
            if (out_row0 < M && out_col1 < N) C[out_row0 * N + out_col1] = __float2bfloat16_rn(acc[wm][wn][1]);
            if (out_row1 < M && out_col0 < N) C[out_row1 * N + out_col0] = __float2bfloat16_rn(acc[wm][wn][2]);
            if (out_row1 < M && out_col1 < N) C[out_row1 * N + out_col1] = __float2bfloat16_rn(acc[wm][wn][3]);
        }
    }
}

// ============================================================
// Launch functions
// ============================================================
inline cudaError_t launch_sgemm_f16_tc(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K
) {
    cudaStream_t stream = internal::get_capture_stream();
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_f16_tc_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

inline cudaError_t launch_sgemm_bf16_tc(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K
) {
    cudaStream_t stream = internal::get_capture_stream();
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_bf16_tc_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

} // namespace fp16_bf16_tc
} // namespace ops
} // namespace pygpukit
