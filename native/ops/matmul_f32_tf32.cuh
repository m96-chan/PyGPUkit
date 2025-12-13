/**
 * TF32 TensorCore GEMM Kernel for Ampere+ GPUs (SM 80+)
 *
 * Target: 22-30 TFLOPS on RTX 3090 Ti (vs 156 TFLOPS theoretical TF32)
 *
 * Key features:
 * - WMMA API for TF32 TensorCore operations
 * - Double-buffered shared memory
 * - Proper memory layout for WMMA fragments
 *
 * TF32 Precision:
 * - Input: TF32 (19-bit: 1 sign + 8 exp + 10 mantissa)
 * - Accumulator: FP32
 * - Expected error: ~1e-3 relative (vs FP32's ~1e-6)
 *
 * Architecture: SM 80+ (Ampere, RTX 30XX / A100 / H100)
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
// Configuration Constants
// ============================================================================

constexpr int BM = 128;           // Tile rows per block
constexpr int BN = 128;           // Tile cols per block
constexpr int BK = 32;            // Tile depth

// WMMA tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

// Padding for shared memory to avoid bank conflicts
constexpr int A_PAD = 8;
constexpr int B_PAD = 8;

// ============================================================================
// TF32 TensorCore GEMM Kernel using WMMA API
// ============================================================================

// Limit registers to prevent spilling (255 regs causes crashes on large grids)
#pragma nv_diag_suppress 20236  // Suppress "controlling expression is constant" warning
__global__ void __launch_bounds__(256, 2)  // 256 threads, 2 min blocks = 128 regs max
sgemm_tf32_wmma_128x128x32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int cta_row = by * BM;
    const int cta_col = bx * BN;

    // Thread indices
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp position in 4x2 grid (4 rows, 2 cols of warps)
    // Each warp handles 32x64 output (2x4 WMMA tiles)
    const int warp_row = warp_id / 2;  // 0-3
    const int warp_col = warp_id % 2;  // 0-1

    // WMMA tiles per warp
    constexpr int WARP_TILES_M = 2;  // 2 * 16 = 32 rows per warp
    constexpr int WARP_TILES_N = 4;  // 4 * 16 = 64 cols per warp

    // Shared memory for double buffering
    // A: [2][BM][BK + pad] = [2][128][40] - row-major for row_major WMMA
    // B: [2][BN][BK + pad] = [2][128][40] - transposed for col_major WMMA
    __shared__ float As[2][BM][BK + A_PAD];
    __shared__ float Bs[2][BN][BK + B_PAD];  // Transposed: B[k][n] stored as Bs[n][k]

    // Declare WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag[WARP_TILES_M];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> b_frag[WARP_TILES_N];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];

    // Initialize accumulators
    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
            fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    const int num_k_tiles = (K + BK - 1) / BK;

    // Load tile from global to shared memory
    auto load_tile = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;

        // Load A tile: BM x BK = 128 x 32 = 4096 elements
        // 256 threads, each loads 16 elements
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = tid + i * 256;
            const int m = idx / BK;       // 0-127
            const int k = idx % BK;       // 0-31

            const int global_m = cta_row + m;
            const int global_k = k_base + k;

            if (global_m < M && global_k < K) {
                As[stage][m][k] = A[global_m * K + global_k];
            } else {
                As[stage][m][k] = 0.0f;
            }
        }

        // Load B tile: BK x BN = 32 x 128 = 4096 elements
        // Store TRANSPOSED: B[k][n] -> Bs[n][k]
        // This makes consecutive k values contiguous for col_major WMMA
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = tid + i * 256;
            const int k = idx / BN;       // 0-31
            const int n = idx % BN;       // 0-127

            const int global_k = k_base + k;
            const int global_n = cta_col + n;

            if (global_k < K && global_n < N) {
                Bs[stage][n][k] = B[global_k * N + global_n];  // Transposed storage
            } else {
                Bs[stage][n][k] = 0.0f;
            }
        }
    };

    // Load first tile
    load_tile(0, 0);
    __syncthreads();

    // Main loop with double buffering
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int curr_stage = k_tile % 2;
        const int next_stage = 1 - curr_stage;

        // Prefetch next tile
        if (k_tile + 1 < num_k_tiles) {
            load_tile(next_stage, k_tile + 1);
        }

        // Compute current tile
        #pragma unroll
        for (int k = 0; k < BK; k += WMMA_K) {
            // Load A fragments
            // A is stored row-major: As[m][k]
            // WMMA row_major expects consecutive k in memory
            // As[m][k] has consecutive k, so stride = BK + A_PAD
            #pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                const int m_offset = warp_row * 32 + mi * WMMA_M;
                load_matrix_sync(a_frag[mi], &As[curr_stage][m_offset][k], BK + A_PAD);
            }

            // Load B fragments
            // B is stored transposed: Bs[n][k]
            // WMMA col_major expects consecutive k in memory (column of B)
            // Bs[n][k] has consecutive k, so stride = BK + B_PAD
            #pragma unroll
            for (int ni = 0; ni < WARP_TILES_N; ++ni) {
                const int n_offset = warp_col * 64 + ni * WMMA_N;
                load_matrix_sync(b_frag[ni], &Bs[curr_stage][n_offset][k], BK + B_PAD);
            }

            // Perform WMMA operations
            #pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < WARP_TILES_N; ++ni) {
                    mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    // For partial tile handling, we need shared memory for store_matrix_sync
    // Each warp needs 16x16 floats with proper stride = 256 floats = 1KB
    __shared__ float partial_tile[8][WMMA_M][WMMA_N];  // 8 warps, 16x16 each

    // WMMA store_matrix_sync with mem_row_major requires leading dimension (N) % 8 == 0
    const bool n_aligned = (N % 8 == 0);

    // Store results to global memory with proper boundary handling
    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
            const int m_offset = cta_row + warp_row * 32 + mi * WMMA_M;
            const int n_offset = cta_col + warp_col * 64 + ni * WMMA_N;

            // Skip tiles completely outside bounds
            if (m_offset >= M || n_offset >= N) continue;

            // Compute valid rows and cols for this tile
            const int valid_rows = min(WMMA_M, M - m_offset);
            const int valid_cols = min(WMMA_N, N - n_offset);

            // Fast path: full 16x16 tile AND N aligned for WMMA store
            // WMMA store_matrix_sync with mem_row_major requires leading dimension % 8 == 0
            if (valid_rows == WMMA_M && valid_cols == WMMA_N && n_aligned) {
                store_matrix_sync(&C[m_offset * N + n_offset], c_frag[mi][ni], N, mem_row_major);
            } else {
                // Tail path: partial tile OR unaligned N
                // Store to shared memory with stride WMMA_N (16), then copy to global
                float* tile_ptr = &partial_tile[warp_id][0][0];
                store_matrix_sync(tile_ptr, c_frag[mi][ni], WMMA_N, mem_row_major);
                __syncwarp();  // Ensure all lanes have written

                // Lane 0 copies valid elements to global memory
                const int lane = tid % 32;
                if (lane == 0) {
                    for (int r = 0; r < valid_rows; ++r) {
                        for (int c = 0; c < valid_cols; ++c) {
                            C[(m_offset + r) * N + (n_offset + c)] = partial_tile[warp_id][r][c];
                        }
                    }
                }
                __syncwarp();  // Ensure store complete before next iteration
            }
        }
    }
}

// ============================================================================
// Optimized TF32 Kernel with cp.async (for higher performance)
// ============================================================================

// cp.async helper functions
__device__ __forceinline__ unsigned int cvta_to_shared(const void* ptr) {
    unsigned int smem_addr;
    asm volatile(
        "{ .reg .u64 smem_ptr64;\n"
        "  cvta.to.shared.u64 smem_ptr64, %1;\n"
        "  cvt.u32.u64 %0, smem_ptr64; }\n"
        : "=r"(smem_addr) : "l"(ptr)
    );
    return smem_addr;
}

__device__ __forceinline__ void cp_async_cg_16(void* dst, const void* src) {
    unsigned int dst_smem = cvta_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_smem), "l"(src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Pipeline stages for optimized kernel
// 2 stages to fit within 100KB shared memory limit
// (3 stages would need 122KB which exceeds SM 86 limit)
constexpr int STAGES = 2;

__global__ void __launch_bounds__(256, 2)
sgemm_tf32_wmma_pipelined(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int cta_row = by * BM;
    const int cta_col = bx * BN;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;

    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    constexpr int WARP_TILES_M = 2;
    constexpr int WARP_TILES_N = 4;

    // Multi-stage shared memory
    __shared__ float As[STAGES][BM][BK + A_PAD];
    __shared__ float Bs[STAGES][BN][BK + B_PAD];

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

    const int num_k_tiles = (K + BK - 1) / BK;

    // Synchronous load function (more reliable for boundary handling)
    auto load_tile = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;

        // Load A tile: BM x BK = 128 x 32 = 4096 elements
        // 256 threads, each loads 16 elements
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = tid + i * 256;
            const int m = idx / BK;       // 0-127
            const int k = idx % BK;       // 0-31

            const int global_m = cta_row + m;
            const int global_k = k_base + k;

            if (global_m < M && global_k < K) {
                As[stage][m][k] = A[global_m * K + global_k];
            } else {
                As[stage][m][k] = 0.0f;
            }
        }

        // Load B tile: BK x BN = 32 x 128 = 4096 elements
        // Store TRANSPOSED: B[k][n] -> Bs[n][k]
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = tid + i * 256;
            const int k = idx / BN;       // 0-31
            const int n = idx % BN;       // 0-127

            const int global_k = k_base + k;
            const int global_n = cta_col + n;

            if (global_k < K && global_n < N) {
                Bs[stage][n][k] = B[global_k * N + global_n];
            } else {
                Bs[stage][n][k] = 0.0f;
            }
        }
    };

    // Load first tile
    load_tile(0, 0);
    __syncthreads();

    // Main loop with double buffering
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int curr_stage = k_tile % STAGES;
        const int next_stage = 1 - curr_stage;

        // Prefetch next tile
        if (k_tile + 1 < num_k_tiles) {
            load_tile(next_stage, k_tile + 1);
        }

        // Compute current tile
        // Skip computation entirely if this warp's output region is completely out of bounds
        const int warp_m_start = cta_row + warp_row * 32;
        const int warp_n_start = cta_col + warp_col * 64;

        if (warp_m_start < M && warp_n_start < N) {
            #pragma unroll
            for (int k = 0; k < BK; k += WMMA_K) {
                #pragma unroll
                for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                    const int m_offset = warp_row * 32 + mi * WMMA_M;
                    load_matrix_sync(a_frag[mi], &As[curr_stage][m_offset][k], BK + A_PAD);
                }

                #pragma unroll
                for (int ni = 0; ni < WARP_TILES_N; ++ni) {
                    const int n_offset = warp_col * 64 + ni * WMMA_N;
                    load_matrix_sync(b_frag[ni], &Bs[curr_stage][n_offset][k], BK + B_PAD);
                }

                #pragma unroll
                for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                    #pragma unroll
                    for (int ni = 0; ni < WARP_TILES_N; ++ni) {
                        mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
                    }
                }
            }
        }

        __syncthreads();
    }

    // For partial tile handling, we need shared memory for store_matrix_sync
    // Each warp needs 16x16 floats with proper stride = 256 floats = 1KB
    __shared__ float partial_tile_p[8][WMMA_M][WMMA_N];  // 8 warps, 16x16 each

    // WMMA store_matrix_sync with mem_row_major requires leading dimension (N) % 8 == 0
    const bool n_aligned = (N % 8 == 0);

    // Epilogue: store results with proper boundary handling
    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
            const int m_offset = cta_row + warp_row * 32 + mi * WMMA_M;
            const int n_offset = cta_col + warp_col * 64 + ni * WMMA_N;

            // Skip tiles completely outside bounds
            if (m_offset >= M || n_offset >= N) continue;

            // Compute valid rows and cols for this tile
            const int valid_rows = min(WMMA_M, M - m_offset);
            const int valid_cols = min(WMMA_N, N - n_offset);

            // Fast path: full 16x16 tile AND N aligned for WMMA store
            // WMMA store_matrix_sync with mem_row_major requires leading dimension % 8 == 0
            if (valid_rows == WMMA_M && valid_cols == WMMA_N && n_aligned) {
                store_matrix_sync(&C[m_offset * N + n_offset], c_frag[mi][ni], N, mem_row_major);
            } else {
                // Tail path: partial tile OR unaligned N
                // Store to shared memory with stride WMMA_N (16), then copy to global
                float* tile_ptr = &partial_tile_p[warp_id][0][0];
                store_matrix_sync(tile_ptr, c_frag[mi][ni], WMMA_N, mem_row_major);
                __syncwarp();  // Ensure all lanes have written

                // Lane 0 copies valid elements to global memory
                const int lane = tid % 32;
                if (lane == 0) {
                    for (int r = 0; r < valid_rows; ++r) {
                        for (int c = 0; c < valid_cols; ++c) {
                            C[(m_offset + r) * N + (n_offset + c)] = partial_tile_p[warp_id][r][c];
                        }
                    }
                }
                __syncwarp();  // Ensure store complete before next iteration
            }
        }
    }
}

// ============================================================================
// Kernel Launch Helper
// ============================================================================

// Alignment constant for K dimension (must be multiple of BK for proper WMMA operation)
constexpr int K_ALIGNMENT = 32;

inline cudaError_t launch_sgemm_tf32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);  // 256 threads = 8 warps
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Use double-buffered kernel (has boundary handling in store)
    sgemm_tf32_wmma_128x128x32<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();

#if 0  // Double-buffered kernel disabled - uses 255 regs even with launch_bounds
    // Check if K needs padding for WMMA alignment
    const int K_rem = K % K_ALIGNMENT;

    if (K_rem == 0) {
        // K is already aligned, use direct kernel
        sgemm_tf32_wmma_128x128x32<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        return cudaGetLastError();
    }

    // K needs padding - create padded copies of A and B
    const int K_padded = K + (K_ALIGNMENT - K_rem);

    // Allocate padded matrices
    float* A_padded = nullptr;
    float* B_padded = nullptr;

    cudaError_t err = cudaMalloc(&A_padded, (size_t)M * K_padded * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&B_padded, (size_t)K_padded * N * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(A_padded);
        return err;
    }

    // Zero-initialize padded matrices
    err = cudaMemset(A_padded, 0, (size_t)M * K_padded * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(A_padded);
        cudaFree(B_padded);
        return err;
    }

    err = cudaMemset(B_padded, 0, (size_t)K_padded * N * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(A_padded);
        cudaFree(B_padded);
        return err;
    }

    // Copy A with padding (row by row)
    err = cudaMemcpy2D(
        A_padded, K_padded * sizeof(float),  // dst, dst pitch
        A, K * sizeof(float),                 // src, src pitch
        K * sizeof(float),                    // width to copy
        M,                                    // height
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        cudaFree(A_padded);
        cudaFree(B_padded);
        return err;
    }

    // Copy B with padding (row by row)
    err = cudaMemcpy2D(
        B_padded, N * sizeof(float),          // dst, dst pitch (N stays same)
        B, N * sizeof(float),                 // src, src pitch
        N * sizeof(float),                    // width to copy
        K,                                    // height (original K rows)
        cudaMemcpyDeviceToDevice
    );
    if (err != cudaSuccess) {
        cudaFree(A_padded);
        cudaFree(B_padded);
        return err;
    }

    // Launch kernel with padded dimensions
    sgemm_tf32_wmma_128x128x32<<<grid, block, 0, stream>>>(
        A_padded, B_padded, C, M, N, K_padded);

    err = cudaGetLastError();

    // Synchronize and free padded matrices
    cudaDeviceSynchronize();
    cudaFree(A_padded);
    cudaFree(B_padded);

    return err;
#endif  // Disabled for debugging
}

}  // namespace tf32
}  // namespace ops
}  // namespace pygpukit
