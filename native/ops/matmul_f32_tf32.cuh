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
// フルカーネル（実測マッピング使用）
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

    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = K / BK;

    // 実測マッピング用のインデックス (dump_c_fragment.cu で確認)
    int a_row_base = lane / 4;
    int a_col_base = lane % 4;
    int b_row_base = lane % 4;
    int b_col = lane / 4;
    int c_row_base = lane / 4;
    int c_col_base = (lane % 4) * 2;  // C fragment は 2列ずつ

    auto load_A = [&](int stage, int kt) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 256;
            int row = idx / BK;
            int col = idx % BK;
            if (row < BM) {
                int gm = cta_m + row;
                int gk = kt * BK + col;
                smA[stage][row][col] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
            }
        }
    };

    auto load_B = [&](int stage, int kt) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int gk = kt * BK + row;
                int gn = cta_n + col;
                smB[stage][row][col] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
            }
        }
    };

    load_A(0, 0);
    load_B(0, 0);
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        if (kt + 1 < num_k_tiles) {
            load_A(next, kt + 1);
            load_B(next, kt + 1);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                    
                    int tile_m = warp_m + wm * WMMA_M;
                    int tile_n = warp_n + wn * WMMA_N;
                    
                    // A fragment (実測マッピング)
                    float a0 = smA[curr][tile_m + a_row_base][kk + a_col_base];
                    float a1 = smA[curr][tile_m + a_row_base + 8][kk + a_col_base];
                    float a2 = smA[curr][tile_m + a_row_base][kk + a_col_base + 4];
                    float a3 = smA[curr][tile_m + a_row_base + 8][kk + a_col_base + 4];
                    
                    // B fragment (実測マッピング)
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

        __syncthreads();
    }

    // Epilogue (実測マッピング - dump_c_fragment.cu で確認)
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