#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace pygpukit {
namespace ops {
namespace tf32 {

// ================================================================
// CTA Tile configuration (Ampere TF32, mma.sync path)
// ================================================================
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;  // TF32: k=8 per mma, 2 mma per BK

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 8;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 4;

constexpr int A_PAD = 4;
constexpr int B_PAD = 4;

// ================================================================
// shared memory address helper
// ================================================================
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

// ================================================================
// cp.async helpers
// ================================================================
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

// ================================================================
// Kernel
// ================================================================
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

    const int warp_row = warp_id / WARPS_N;  // 0..3
    const int warp_col = warp_id % WARPS_N;  // 0..1

    const int warp_m = warp_row * (WARP_TILES_M * WMMA_M);  // 0, 32, 64, 96
    const int warp_n = warp_col * (WARP_TILES_N * WMMA_N);  // 0, 32

    // A: row-major [BM][BK] 
    // B: col-major [BK][BN] -> stored as [BN][BK] for coalesced access
    __shared__ float smA[2][BM][BK + A_PAD];
    __shared__ float smB[2][BK][BN + B_PAD];  // K x N layout (col-major for B)

    // Accumulators: 2x4 tiles of m16n8k8, each produces 2 floats per thread
    float acc[WARP_TILES_M][WARP_TILES_N][4] = {};  // 4 floats per m16n8k8

    const int num_k_tiles = K / BK;

    // ------------------------------------------------------------
    // Load helpers
    // ------------------------------------------------------------
    // A: 128x16 = 2048 floats, 256 threads, 8 floats/thread = 2 x float4
    auto load_A = [&](int stage, int kt) {
        const int a_row = tid / 4;           // 0..63
        const int a_col = (tid % 4) * 4;     // 0,4,8,12
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int row = a_row + i * 64;
            if (cta_m + row < M && kt * BK + a_col < K) {
                cp_async_16(
                    &smA[stage][row][a_col],
                    &A[(cta_m + row) * K + kt * BK + a_col]
                );
            }
        }
    };

    // B: need col-major, B is row-major [K][N]
    // Load B[k][n] into smB[k][n]
    // 16x128 = 2048 floats
    auto load_B = [&](int stage, int kt) {
        const int b_row = tid / 32;          // 0..7 (k dimension)
        const int b_col = (tid % 32) * 4;    // 0..124 (n dimension)
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int k = b_row + i * 8;
            if (kt * BK + k < K && cta_n + b_col < N) {
                cp_async_16(
                    &smB[stage][k][b_col],
                    &B[(kt * BK + k) * N + cta_n + b_col]
                );
            }
        }
    };

    // ------------------------------------------------------------
    // Prologue: load first tile
    // ------------------------------------------------------------
    load_A(0, 0);
    load_B(0, 0);
    cp_async_commit();

    if (num_k_tiles > 1) {
        load_A(1, 1);
        load_B(1, 1);
        cp_async_commit();
    }

    cp_async_wait_1();
    __syncthreads();

    // ------------------------------------------------------------
    // TF32 mma.sync register layout for m16n8k8:
    // A: 4 registers (a0,a1,a2,a3) - each thread holds 4 TF32 values
    // B: 2 registers (b0,b1) - each thread holds 2 TF32 values  
    // C: 4 registers (c0,c1,c2,c3) - 4 FP32 outputs
    //
    // Thread mapping in warp (32 threads):
    // For A (16x8, row-major):
    //   row = (lane % 16), but grouped: lane/4 gives row group
    //   Thread lane maps to: rows [lane%16][k] where k from registers
    //
    // For B (8x8, col-major):
    //   Thread lane maps to columns
    // ------------------------------------------------------------

    // ------------------------------------------------------------
    // Main loop
    // ------------------------------------------------------------
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr = kt & 1;
        int next = curr ^ 1;

        // Prefetch next tile
        if (kt + 2 < num_k_tiles) {
            load_A(next, kt + 2);
            load_B(next, kt + 2);
            cp_async_commit();
        }

        // Process current tile: BK=16, WMMA_K=8, so 2 k-iterations
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            
            // Load A fragments for this warp's tiles
            // Each warp processes WARP_TILES_M (2) x WARP_TILES_N (4) output tiles
            
            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; ++wm) {
                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                    
                    int tile_m = warp_m + wm * WMMA_M;
                    int tile_n = warp_n + wn * WMMA_N;
                    
                    // ============================================
                    // Load A fragment (m16n8k8 needs 4 TF32 values per thread)
                    // A is 16x8, row-major
                    // Thread mapping: 
                    //   group_id = lane / 4  (0..7)
                    //   thread_in_group = lane % 4 (0..3)
                    //   Each group of 4 threads handles 2 rows
                    //   row0 = group_id * 2
                    //   row1 = group_id * 2 + 1
                    // ============================================
                    int a_group = lane / 4;
                    int a_tid = lane % 4;
                    
                    int a_row0 = tile_m + a_group * 2;
                    int a_row1 = tile_m + a_group * 2 + 1;
                    int a_col0 = kk + a_tid * 2;
                    int a_col1 = kk + a_tid * 2 + 1;
                    
                    float a0 = smA[curr][a_row0][a_col0];
                    float a1 = smA[curr][a_row0][a_col1];
                    float a2 = smA[curr][a_row1][a_col0];
                    float a3 = smA[curr][a_row1][a_col1];
                    
                    // ============================================
                    // Load B fragment (m16n8k8 needs 2 TF32 values per thread)
                    // B is 8x8 (k x n), col-major for mma
                    // smB is stored as [k][n]
                    // Thread mapping:
                    //   Each thread loads from specific k,n position
                    //   b_k = lane % 4 * 2  -> k positions 0,2,4,6
                    //   b_n = lane / 4      -> n positions 0..7
                    // ============================================
                    int b_k = (lane % 4) * 2;
                    int b_n = lane / 4;
                    
                    float b0 = smB[curr][kk + b_k][tile_n + b_n];
                    float b1 = smB[curr][kk + b_k + 1][tile_n + b_n];
                    
                    // ============================================
                    // Execute mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                    // ============================================
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

        if (kt + 2 < num_k_tiles) {
            cp_async_wait_1();
        }
        __syncthreads();
    }

    // ------------------------------------------------------------
    // Epilogue: Store results
    // m16n8k8 output layout:
    //   4 floats per thread: (row0,col0), (row0,col1), (row8,col0), (row8,col1)
    //   where:
    //     row_base = (lane / 4) * 2 for lanes 0-15
    //     row_base = (lane / 4) * 2 - 8 for lanes 16-31? 
    //   Actually for m16n8k8:
    //     c[0],c[1] -> rows 0-7 (lane/4), cols (lane%4)*2, (lane%4)*2+1
    //     c[2],c[3] -> rows 8-15 (lane/4 + 8), same cols
    // ------------------------------------------------------------
    
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; ++wn) {
            int tile_m = cta_m + warp_m + wm * WMMA_M;
            int tile_n = cta_n + warp_n + wn * WMMA_N;
            
            // Output mapping for m16n8k8:
            // Thread lane -> (row, col) for each of 4 output elements
            int out_row0 = tile_m + (lane / 4);
            int out_row1 = tile_m + (lane / 4) + 8;
            int out_col = tile_n + (lane % 4) * 2;
            
            if (out_row0 < M && out_col + 1 < N) {
                C[out_row0 * N + out_col]     = acc[wm][wn][0];
                C[out_row0 * N + out_col + 1] = acc[wm][wn][1];
            }
            if (out_row1 < M && out_col + 1 < N) {
                C[out_row1 * N + out_col]     = acc[wm][wn][2];
                C[out_row1 * N + out_col + 1] = acc[wm][wn][3];
            }
        }
    }
}

// ================================================================
// Launcher
// ================================================================
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
} // namespace pygpukitS