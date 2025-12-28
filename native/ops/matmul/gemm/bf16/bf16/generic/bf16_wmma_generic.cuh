/**
 * FP16/BF16 TensorCore Generic GEMM (with boundary handling)
 *
 * Supports arbitrary matrix sizes with M,N >= 16 and K % 8 == 0
 * Uses mma.sync.aligned.m16n8k8 for flexibility
 *
 * Trade-off: Slightly slower than TC_FAST due to boundary checks,
 *            but supports many more matrix sizes.
 */

#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../../../../core/cuda_graph.hpp"

namespace pygpukit {
namespace ops {
namespace fp16_bf16_tc_generic {

// Smaller tile for better flexibility
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;  // Match MMA K dimension

// MMA tile dimensions (m16n8k8 for FP16)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;

// Warp configuration (4 warps = 128 threads)
constexpr int WARPS_M = 2;  // 2 warps along M
constexpr int WARPS_N = 2;  // 2 warps along N
constexpr int WARP_TILES_M = 2;  // 2 MMA tiles per warp along M (32 rows)
constexpr int WARP_TILES_N = 4;  // 4 MMA tiles per warp along N (32 cols)

// Padding for bank conflict avoidance
constexpr int A_PAD = 8;
constexpr int B_PAD = 8;

// ============================================================
// Helpers
// ============================================================
__device__ __forceinline__ uint32_t smem_u32_generic(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 smem64; "
        "  cvta.to.shared.u64 smem64, %1; "
        "  cvt.u32.u64 %0, smem64; }"
        : "=r"(addr) : "l"(ptr)
    );
    return addr;
}

__device__ __forceinline__ void cp_async_16_generic(void* smem, const void* gmem) {
    uint32_t addr = smem_u32_generic(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(addr), "l"(gmem)
    );
}

__device__ __forceinline__ void cp_async_commit_generic() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_0_generic() {
    asm volatile("cp.async.wait_group 0;");
}

// ============================================================
// FP16 TensorCore Generic GEMM Kernel
// ============================================================
__global__ void __launch_bounds__(128, 4)
sgemm_f16_tc_generic_kernel(
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

    // Shared memory
    __shared__ __half smA[BM][BK + A_PAD];
    __shared__ __half smB[BK][BN + B_PAD];

    // Accumulators (FP32)
    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = (K + BK - 1) / BK;

    // Fragment index mappings for m16n8k8
    const int groupID = lane >> 2;           // 0-7
    const int tid_in_group = lane & 3;       // 0-3

    // C fragment mapping
    const int c_row_base = groupID;
    const int c_col_base = tid_in_group * 2;

    // ====== Main loop ======
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Load A tile with boundary check
        // 128 threads, BM*BK = 64*8 = 512 halves = 4 per thread
        {
            const int elems_per_thread = 4;
            #pragma unroll
            for (int i = 0; i < elems_per_thread; ++i) {
                int idx = tid * elems_per_thread + i;
                int row = idx / BK;
                int col = idx % BK;
                int gm = cta_m + row;
                int gk = kt * BK + col;

                __half val = __float2half(0.0f);
                if (gm < M && gk < K) {
                    val = A[gm * K + gk];
                }
                smA[row][col] = val;
            }
        }

        // Load B tile with boundary check
        // BK*BN = 8*64 = 512 halves = 4 per thread
        {
            const int elems_per_thread = 4;
            #pragma unroll
            for (int i = 0; i < elems_per_thread; ++i) {
                int idx = tid * elems_per_thread + i;
                int row = idx / BN;
                int col = idx % BN;
                int gk = kt * BK + row;
                int gn = cta_n + col;

                __half val = __float2half(0.0f);
                if (gk < K && gn < N) {
                    val = B[gk * N + gn];
                }
                smB[row][col] = val;
            }
        }

        __syncthreads();

        // Compute MMA for this K tile (single k iteration since BK == MMA_K)
        #pragma unroll
        for (int wm = 0; wm < WARP_TILES_M; ++wm) {
            int tile_m = warp_m + wm * MMA_M;

            // Load A fragment (4 halves = 2 packed uint32 for m16n8k8)
            // m16n8k8 A fragment: 4 registers
            // a[i] for i=0..3:
            //   row = groupID + 8 * (i / 2)
            //   col = tid_in_group * 2 + (i % 2)
            uint32_t a_frag[2];

            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                int i0 = p * 2;
                int i1 = p * 2 + 1;

                int row0 = groupID + 8 * (i0 / 2);
                int col0 = tid_in_group * 2 + (i0 % 2);
                int row1 = groupID + 8 * (i1 / 2);
                int col1 = tid_in_group * 2 + (i1 % 2);

                __half h0 = (tile_m + row0 < BM) ? smA[tile_m + row0][col0] : __float2half(0.0f);
                __half h1 = (tile_m + row1 < BM) ? smA[tile_m + row1][col1] : __float2half(0.0f);

                a_frag[p] = __half_as_ushort(h0) | (uint32_t(__half_as_ushort(h1)) << 16);
            }

            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                int tile_n = warp_n + wn * MMA_N;

                // Load B fragment (2 halves = 1 packed uint32 for m16n8k8)
                // m16n8k8 B fragment: 2 registers
                // b[i] for i=0..1:
                //   row = tid_in_group * 2 + (i % 2)
                //   col = groupID
                uint32_t b_frag;

                {
                    int row0 = tid_in_group * 2;
                    int row1 = tid_in_group * 2 + 1;
                    int col = groupID;

                    __half h0 = (row0 < BK && tile_n + col < BN) ? smB[row0][tile_n + col] : __float2half(0.0f);
                    __half h1 = (row1 < BK && tile_n + col < BN) ? smB[row1][tile_n + col] : __float2half(0.0f);

                    b_frag = __half_as_ushort(h0) | (uint32_t(__half_as_ushort(h1)) << 16);
                }

                // Execute MMA: mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5}, "
                    "{%6}, "
                    "{%0, %1, %2, %3};"
                    : "+f"(acc[wm][wn][0]), "+f"(acc[wm][wn][1]),
                      "+f"(acc[wm][wn][2]), "+f"(acc[wm][wn][3])
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(b_frag)
                );
            }
        }

        __syncthreads();
    }

    // ====== Epilogue: Store results with boundary check ======
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
// BF16 TensorCore Generic GEMM Kernel
// ============================================================
__global__ void __launch_bounds__(128, 4)
sgemm_bf16_tc_generic_kernel(
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

    __shared__ __nv_bfloat16 smA[BM][BK + A_PAD];
    __shared__ __nv_bfloat16 smB[BK][BN + B_PAD];

    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};

    const int num_k_tiles = (K + BK - 1) / BK;

    const int groupID = lane >> 2;
    const int tid_in_group = lane & 3;
    const int c_row_base = groupID;
    const int c_col_base = tid_in_group * 2;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Load A tile
        {
            const int elems_per_thread = 4;
            #pragma unroll
            for (int i = 0; i < elems_per_thread; ++i) {
                int idx = tid * elems_per_thread + i;
                int row = idx / BK;
                int col = idx % BK;
                int gm = cta_m + row;
                int gk = kt * BK + col;

                __nv_bfloat16 val = __float2bfloat16_rn(0.0f);
                if (gm < M && gk < K) {
                    val = A[gm * K + gk];
                }
                smA[row][col] = val;
            }
        }

        // Load B tile
        {
            const int elems_per_thread = 4;
            #pragma unroll
            for (int i = 0; i < elems_per_thread; ++i) {
                int idx = tid * elems_per_thread + i;
                int row = idx / BN;
                int col = idx % BN;
                int gk = kt * BK + row;
                int gn = cta_n + col;

                __nv_bfloat16 val = __float2bfloat16_rn(0.0f);
                if (gk < K && gn < N) {
                    val = B[gk * N + gn];
                }
                smB[row][col] = val;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int wm = 0; wm < WARP_TILES_M; ++wm) {
            int tile_m = warp_m + wm * MMA_M;

            uint32_t a_frag[2];

            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                int i0 = p * 2;
                int i1 = p * 2 + 1;

                int row0 = groupID + 8 * (i0 / 2);
                int col0 = tid_in_group * 2 + (i0 % 2);
                int row1 = groupID + 8 * (i1 / 2);
                int col1 = tid_in_group * 2 + (i1 % 2);

                __nv_bfloat16 h0 = (tile_m + row0 < BM) ? smA[tile_m + row0][col0] : __float2bfloat16_rn(0.0f);
                __nv_bfloat16 h1 = (tile_m + row1 < BM) ? smA[tile_m + row1][col1] : __float2bfloat16_rn(0.0f);

                a_frag[p] = __bfloat16_as_ushort(h0) | (uint32_t(__bfloat16_as_ushort(h1)) << 16);
            }

            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                int tile_n = warp_n + wn * MMA_N;

                uint32_t b_frag;

                {
                    int row0 = tid_in_group * 2;
                    int row1 = tid_in_group * 2 + 1;
                    int col = groupID;

                    __nv_bfloat16 h0 = (row0 < BK && tile_n + col < BN) ? smB[row0][tile_n + col] : __float2bfloat16_rn(0.0f);
                    __nv_bfloat16 h1 = (row1 < BK && tile_n + col < BN) ? smB[row1][tile_n + col] : __float2bfloat16_rn(0.0f);

                    b_frag = __bfloat16_as_ushort(h0) | (uint32_t(__bfloat16_as_ushort(h1)) << 16);
                }

                // Execute MMA: mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5}, "
                    "{%6}, "
                    "{%0, %1, %2, %3};"
                    : "+f"(acc[wm][wn][0]), "+f"(acc[wm][wn][1]),
                      "+f"(acc[wm][wn][2]), "+f"(acc[wm][wn][3])
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(b_frag)
                );
            }
        }

        __syncthreads();
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
inline cudaError_t launch_sgemm_f16_tc_generic(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K
) {
    cudaStream_t stream = internal::get_capture_stream();
    dim3 block(128);  // 4 warps
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_f16_tc_generic_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

inline cudaError_t launch_sgemm_bf16_tc_generic(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K
) {
    cudaStream_t stream = internal::get_capture_stream();
    dim3 block(128);  // 4 warps
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_bf16_tc_generic_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

} // namespace fp16_bf16_tc_generic
} // namespace ops
} // namespace pygpukit
