#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace pygpukit {
namespace ops {
namespace tf32 {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 8;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 8;

constexpr int A_PAD = 4;
constexpr int B_PAD = 4;

// ============================================================
// cp.async helpers (for optimized kernel)
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

__device__ __forceinline__ void cp_async_wait_1() {
    asm volatile("cp.async.wait_group 1;");
}

// ============================================================
// 単一タイル検証用カーネル（実測マッピング使用）
// ============================================================
__global__ void sgemm_tf32_single_tile_verified(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    // 実測マッピング
    int a_row_base = lane / 4;    // 0-7
    int a_col_base = lane % 4;    // 0-3
    
    int b_row_base = lane % 4;    // 0-3
    int b_col = lane / 4;         // 0-7

    for (int k = 0; k < K; k += WMMA_K) {
        // A fragment (16×8)
        // a[0] = A[a_row][a_col]
        // a[1] = A[a_row + 8][a_col]
        // a[2] = A[a_row][a_col + 4]
        // a[3] = A[a_row + 8][a_col + 4]
        float a0 = A[(a_row_base) * K + k + a_col_base];
        float a1 = A[(a_row_base + 8) * K + k + a_col_base];
        float a2 = A[(a_row_base) * K + k + a_col_base + 4];
        float a3 = A[(a_row_base + 8) * K + k + a_col_base + 4];
        
        // B fragment (8×8)
        // b[0] = B[b_row][b_col]
        // b[1] = B[b_row + 4][b_col]
        float b0 = B[(k + b_row_base) * N + b_col];
        float b1 = B[(k + b_row_base + 4) * N + b_col];
        
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)
            : "r"(__float_as_uint(a0)), "r"(__float_as_uint(a1)),
              "r"(__float_as_uint(a2)), "r"(__float_as_uint(a3)),
              "r"(__float_as_uint(b0)), "r"(__float_as_uint(b1))
        );
    }

    // C fragment (16×8) - 実測マッピング (dump_c_fragment.cu で確認)
    // c[0] = C[t/4][(t%4)*2]
    // c[1] = C[t/4][(t%4)*2 + 1]
    // c[2] = C[t/4 + 8][(t%4)*2]
    // c[3] = C[t/4 + 8][(t%4)*2 + 1]
    int c_row_base = lane / 4;           // 0-7
    int c_col_base = (lane % 4) * 2;     // 0, 2, 4, 6

    if (c_row_base < M && c_col_base < N)
        C[c_row_base * N + c_col_base] = acc0;
    if (c_row_base < M && c_col_base + 1 < N)
        C[c_row_base * N + c_col_base + 1] = acc1;
    if (c_row_base + 8 < M && c_col_base < N)
        C[(c_row_base + 8) * N + c_col_base] = acc2;
    if (c_row_base + 8 < M && c_col_base + 1 < N)
        C[(c_row_base + 8) * N + c_col_base + 1] = acc3;
}

inline cudaError_t launch_single_tile_verified(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    sgemm_tf32_single_tile_verified<<<1, 32, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

// ============================================================
// フルカーネル（cp.async + 2-stage pipeline + 正確なフラグメントマッピング）
// ============================================================
__global__ void __launch_bounds__(256, 2)
sgemm_tf32_ampere_kernel(
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

    __shared__ float smA[2][BM][BK + A_PAD];
    __shared__ float smB[2][BK][BN + B_PAD];

    // Note: Zero-init removed for performance. Requires aligned sizes (multiple of BM, BN, BK).
    // For non-aligned sizes, add zero-init back.

    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = K / BK;

    // 実測マッピング用のインデックス (dump_c_fragment.cu で確認)
    // A fragment: a[0] = A[row][col], a[1] = A[row+8][col], a[2] = A[row][col+4], a[3] = A[row+8][col+4]
    const int a_row_base = lane / 4;   // 0-7
    const int a_col_base = lane % 4;   // 0-3
    // B fragment: b[0] = B[row][col], b[1] = B[row+4][col]
    const int b_row_base = lane % 4;   // 0-3
    const int b_col = lane / 4;        // 0-7
    // C fragment: c[0] = C[row][col*2], c[1] = C[row][col*2+1], c[2] = C[row+8][col*2], c[3] = C[row+8][col*2+1]
    const int c_row_base = lane / 4;
    const int c_col_base = (lane % 4) * 2;

    // ====== cp.async load helpers ======
    // A: 128x16 = 2048 floats, 256 threads → 8 floats/thread (2 x float4)
    auto load_A_async = [&](int stage, int kt) {
        const int a_row = tid / 4;         // 0..63
        const int a_col = (tid % 4) * 4;   // 0, 4, 8, 12

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

    // B: 16x128 = 2048 floats, 256 threads → 8 floats/thread (2 x float4)
    auto load_B_async = [&](int stage, int kt) {
        const int b_row = tid / 32;           // 0..7 (k dimension)
        const int b_col_ld = (tid % 32) * 4;  // 0..124 (n dimension)

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

    // ====== Prologue: load first tile ======
    load_A_async(0, 0);
    load_B_async(0, 0);
    cp_async_commit();
    cp_async_wait_0();
    __syncthreads();

    // ====== Main loop with simple double buffering ======
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        // Prefetch next tile into the OTHER buffer (if exists)
        if (kt + 1 < num_k_tiles) {
            load_A_async(next, kt + 1);
            load_B_async(next, kt + 1);
            cp_async_commit();
        }

        // Process current tile (BK=16 with 2 WMMA_K=8 iterations)
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {

            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {

                    int tile_m = warp_m + wm * WMMA_M;
                    int tile_n = warp_n + wn * WMMA_N;

                    // A fragment (実測マッピング - 正確!)
                    // a[0] = A[row][col], a[1] = A[row+8][col], a[2] = A[row][col+4], a[3] = A[row+8][col+4]
                    float a0 = smA[curr][tile_m + a_row_base][kk + a_col_base];
                    float a1 = smA[curr][tile_m + a_row_base + 8][kk + a_col_base];
                    float a2 = smA[curr][tile_m + a_row_base][kk + a_col_base + 4];
                    float a3 = smA[curr][tile_m + a_row_base + 8][kk + a_col_base + 4];

                    // B fragment (実測マッピング - 正確!)
                    // b[0] = B[row][col], b[1] = B[row+4][col]
                    float b0 = smB[curr][kk + b_row_base][tile_n + b_col];
                    float b1 = smB[curr][kk + b_row_base + 4][tile_n + b_col];

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

        // Wait for prefetch to complete before next iteration
        if (kt + 1 < num_k_tiles) {
            cp_async_wait_0();
        }
        __syncthreads();
    }

    // ====== Epilogue (実測マッピング) ======
    // c[0] → C[row][col], c[1] → C[row][col+1]
    // c[2] → C[row+8][col], c[3] → C[row+8][col+1]
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

inline cudaError_t launch_sgemm_tf32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tf32_ampere_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

} // namespace tf32
} // namespace ops
} // namespace pygpukit