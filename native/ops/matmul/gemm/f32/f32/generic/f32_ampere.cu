/**
 * Ampere-Optimized FP32 GEMM Kernel Implementation
 */

#include "f32_ampere.cuh"

namespace pygpukit {
namespace ops {
namespace ampere {

// ============================================================================
// High-Performance SGEMM Kernel with TRUE 3-Stage Pipeline
// ============================================================================

/**
 * C = A x B
 * A: M x K (row-major)
 * B: K x N (row-major)
 * C: M x N (row-major)
 *
 * Grid:  ((N + BN - 1) / BN, (M + BM - 1) / BM)
 * Block: (16, 16) = 256 threads
 *
 * Shared memory layout:
 *   As[stage][k][m] - A tile stored transposed for coalesced smem reads
 *   Bs[stage][k][n] - B tile stored normally
 */
__global__ void __launch_bounds__(256, 2)
sgemm_128x128x32_3stage(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Thread indices
    const int tx = threadIdx.x;  // 0-15 (column within thread block)
    const int ty = threadIdx.y;  // 0-15 (row within thread block)
    const int tid = ty * BLOCK_DIM_X + tx;  // 0-255

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Global starting positions for this CTA
    const int cta_row = by * BM;
    const int cta_col = bx * BN;

    // ========================================================================
    // Shared Memory Allocation (4-stage pipeline)
    // ========================================================================
    extern __shared__ float smem[];

    // Am[stage][m][k] - ROW-MAJOR storage for float4 async loads
    // Bs[stage][k][n] - row-major storage
    float* Am = smem;
    float* Bs = smem + STAGES * A_STAGE_SIZE;

    // Indexing macros for shared memory
    // Am[m][k] with stride=20: row-major, float4 aligned for k={0,4,8,12}
    // Position = stage * 2560 + m * 20 + k
    #define AM(stage, m, k) Am[(stage) * A_STAGE_SIZE + (m) * A_SMEM_STRIDE + (k)]
    #define BS(stage, k, n) Bs[(stage) * B_STAGE_SIZE + (k) * B_SMEM_STRIDE + (n)]

    // ========================================================================
    // Register Allocation
    // ========================================================================
    // Accumulators: 8x8 = 64 floats per thread
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Fragments for register tiling
    float a_frag[TM];
    float b_frag[TN];

    // ========================================================================
    // Load Configuration
    // ========================================================================
    // Each thread loads multiple elements per tile
    // A tile: BM x BK = 128 x 32 = 4096 elements, 256 threads -> 16 per thread
    // B tile: BK x BN = 32 x 128 = 4096 elements, 256 threads -> 16 per thread

    // Number of K tiles
    const int num_k_tiles = (K + BK - 1) / BK;

    // ========================================================================
    // Async Load Functions (WARP-COALESCED patterns)
    // ========================================================================
    //
    // CRITICAL for performance:
    // - A is row-major (MxK): A[m,k] = A[m*K + k]
    //   Consecutive K values are contiguous in memory
    //   -> Organize so consecutive THREADS load consecutive K values
    //
    // - B is row-major (KxN): B[k,n] = B[k*N + n]
    //   Consecutive N values are contiguous in memory
    //   -> Already using float4 for consecutive N values (good)
    //
    // Pattern: elem_idx = tid + i * NUM_THREADS ensures consecutive threads
    //          load consecutive elements in each iteration.

    // Load A tile with ROW-MAJOR storage using float4 cp.async
    // A is row-major in global memory: A[m][k] = A[m*K + k]
    // Store to shared memory ROW-MAJOR: AM[m][k] with stride=20
    //
    // Tile: 128 x 16 = 2048 elements = 512 float4s
    // 256 threads x 2 float4s/thread = 512 float4s
    //
    // CRITICAL: cp.async.cg.16 requires 16-byte aligned source address
    // When K % 4 != 0, row stride is not 16-byte aligned, must use scalar loads
    const bool a_aligned = (K % 4 == 0);

    auto load_A_tile = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;

        // Each thread loads 2 float4s (8 floats total)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int float4_idx = tid + i * NUM_THREADS;

            // Each row has BK/4 = 4 float4s (k = 0,4,8,12)
            const int a_m = float4_idx / (BK / 4);      // 0-127
            const int a_k = (float4_idx % (BK / 4)) * 4;  // 0, 4, 8, 12

            const int global_m = cta_row + a_m;
            const int global_k = k_base + a_k;

            // Destination in shared memory: AM[stage][m][k]
            float* dst = &AM(stage, a_m, a_k);

            // Only use float4 cp.async when K is 16-byte aligned AND within bounds
            if (a_aligned && global_m < M && global_k + 3 < K) {
                // float4 cp.async - both src and dst are 16-byte aligned
                const float* src = &A[global_m * K + global_k];
                cp_async_cg_16(dst, src);
            } else {
                // Fallback: scalar loads (handles misaligned K and boundaries)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (global_m < M && global_k + j < K) {
                        cp_async_cg_4(&dst[j], &A[global_m * K + global_k + j]);
                    } else {
                        dst[j] = 0.0f;
                    }
                }
            }
        }
    };

    // Load B tile with COALESCED float4 access
    // 16 x 128 = 2048 elements = 512 float4s (BK=16)
    // 256 threads x 2 float4s/thread = 512 float4s
    //
    // CRITICAL: cp.async.cg.16 requires 16-byte aligned source address
    // When N % 4 != 0, row stride is not 16-byte aligned, must use scalar loads
    const bool b_aligned = (N % 4 == 0);

    auto load_B_tile = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            // Consecutive threads load consecutive float4s
            const int float4_idx = tid + i * NUM_THREADS;

            // Each row has BN/4 = 32 float4s
            const int b_k = float4_idx / (BN / 4);   // 0-15
            const int b_n = (float4_idx % (BN / 4)) * 4;  // 0, 4, 8, ..., 124

            const int global_k = k_base + b_k;
            const int global_n = cta_col + b_n;

            float* dst = &BS(stage, b_k, b_n);

            // Only use float4 cp.async when N is 16-byte aligned AND within bounds
            if (b_aligned && global_k < K && global_n + 3 < N) {
                const float* src = &B[global_k * N + global_n];
                cp_async_cg_16(dst, src);  // float4 = 16 bytes, coalesced!
            } else {
                // Fallback: scalar loads (handles misaligned N and boundaries)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (global_k < K && global_n + j < N) {
                        const float* src = &B[global_k * N + global_n + j];
                        cp_async_cg_4(&dst[j], src);
                    } else {
                        dst[j] = 0.0f;
                    }
                }
            }
        }
    };

    // ========================================================================
    // Pipeline Prologue: Fill STAGES-1 tiles
    // ========================================================================
    // Issue async loads for first STAGES-1 tiles
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            load_A_tile(s, s);
            load_B_tile(s, s);
        }
        cp_async_commit();
    }

    // ========================================================================
    // Main Pipeline Loop
    // ========================================================================
    // CRITICAL: The correct pipeline pattern is:
    //   1. Issue load for tile k+STAGES-1
    //   2. Commit
    //   3. Wait for tile k to be ready
    //   4. Sync
    //   5. Compute tile k

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // Current stage for compute
        const int compute_stage = k_tile % STAGES;

        // Stage for loading (STAGES-1 tiles ahead)
        const int load_stage = (k_tile + STAGES - 1) % STAGES;
        const int load_k_tile = k_tile + STAGES - 1;

        // Step 1: Issue async loads for future tile
        if (load_k_tile < num_k_tiles) {
            load_A_tile(load_stage, load_k_tile);
            load_B_tile(load_stage, load_k_tile);
        }

        // Step 2: Commit this group
        cp_async_commit();

        // Step 3: Wait for compute stage to be ready
        // wait_group(STAGES-2) means wait until <=(STAGES-2) groups outstanding
        // With STAGES=3, wait_group(1) ensures tile k_tile is ready
        cp_async_wait_group(STAGES - 2);

        // Step 4: Single sync point
        __syncthreads();

        // Step 5: Compute - load fragments from shared memory and accumulate
        // ROW-MAJOR A storage: AM[m][k]
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load A fragment: 8 values from AM[ty*8..ty*8+7][k]
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                a_frag[m] = AM(compute_stage, ty * TM + m, k);
            }

            // Load B fragment: 8 consecutive N values for this thread
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                b_frag[n] = BS(compute_stage, k, tx * TN + n);
            }

            // Outer product with FMA
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] = fmaf(a_frag[m], b_frag[n], acc[m][n]);
                }
            }
        }

        // NO second __syncthreads() here - single barrier per iteration
    }

    // ========================================================================
    // Epilogue: Write results to global memory
    // ========================================================================
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int global_m = cta_row + ty * TM + m;

        if (global_m < M) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                const int global_n = cta_col + tx * TN + n;

                if (global_n < N) {
                    C[global_m * N + global_n] = acc[m][n];
                }
            }
        }
    }

    #undef AM
    #undef BS
}

// ============================================================================
// Alternative: 4-Stage Pipeline with BK=16 (fits in default 48KB smem)
// ============================================================================

/**
 * 4-stage pipeline variant with BK=16
 * Slightly less compute per load, but more stages for latency hiding
 */
__global__ void __launch_bounds__(256, 2)
sgemm_128x128x16_4stage(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_DIM_X + tx;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int cta_row = by * BM;
    const int cta_col = bx * BN;

    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = smem + STAGES_4 * A_STAGE_SIZE_SMALL;

    #define AS4(stage, k, m) As[(stage) * A_STAGE_SIZE_SMALL + (k) * A_SMEM_STRIDE + (m)]
    #define BS4(stage, k, n) Bs[(stage) * B_STAGE_SIZE_SMALL + (k) * B_SMEM_STRIDE + (n)]

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    float a_frag[TM];
    float b_frag[TN];

    const int num_k_tiles = (K + BK_SMALL - 1) / BK_SMALL;

    // Load functions for BK=16 with FULLY ASYNC cp.async
    auto load_A = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK_SMALL;
        // 128 x 16 = 2048 elements
        // 256 threads x 8 elements/thread = 2048 elements

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int elem_idx = tid + i * NUM_THREADS;

            const int a_k = elem_idx % BK_SMALL;  // 0-15
            const int a_m = elem_idx / BK_SMALL;  // 0-127

            const int global_m = cta_row + a_m;
            const int global_k = k_base + a_k;

            float* dst = &AS4(stage, a_k, a_m);

            if (global_m < M && global_k < K) {
                const float* src = &A[global_m * K + global_k];
                cp_async_cg_4(dst, src);  // Fully async
            } else {
                *dst = 0.0f;
            }
        }
    };

    // CRITICAL: cp.async.cg.16 requires 16-byte aligned source address
    // When N % 4 != 0, row stride is not 16-byte aligned, must use scalar loads
    const bool b_aligned_4 = (N % 4 == 0);

    auto load_B = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK_SMALL;
        // 16 x 128 = 2048 elements = 512 float4s, 256 threads -> 2 float4 per thread
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int float4_idx = tid + i * NUM_THREADS;
            const int b_k = float4_idx / (BN / 4);
            const int b_n = (float4_idx % (BN / 4)) * 4;

            const int global_k = k_base + b_k;
            const int global_n = cta_col + b_n;

            float* dst = &BS4(stage, b_k, b_n);

            // Only use float4 cp.async when N is 16-byte aligned AND within bounds
            if (b_aligned_4 && global_k < K && global_n + 3 < N) {
                cp_async_cg_16(dst, &B[global_k * N + global_n]);
            } else {
                // Fallback: scalar loads (handles misaligned N and boundaries)
                for (int j = 0; j < 4; ++j) {
                    if (global_k < K && global_n + j < N) {
                        cp_async_cg_4(&dst[j], &B[global_k * N + global_n + j]);
                    } else {
                        dst[j] = 0.0f;
                    }
                }
            }
        }
    };

    // Prologue: fill STAGES_4-1 = 3 tiles
    #pragma unroll
    for (int s = 0; s < STAGES_4 - 1; ++s) {
        if (s < num_k_tiles) {
            load_A(s, s);
            load_B(s, s);
        }
        cp_async_commit();
    }

    // Main loop
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int compute_stage = k_tile % STAGES_4;
        const int load_stage = (k_tile + STAGES_4 - 1) % STAGES_4;
        const int load_k_tile = k_tile + STAGES_4 - 1;

        if (load_k_tile < num_k_tiles) {
            load_A(load_stage, load_k_tile);
            load_B(load_stage, load_k_tile);
        }
        cp_async_commit();

        // wait_group(STAGES_4 - 2) = wait_group(2)
        cp_async_wait_group(STAGES_4 - 2);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK_SMALL; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                a_frag[m] = AS4(compute_stage, k, ty * TM + m);
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                b_frag[n] = BS4(compute_stage, k, tx * TN + n);
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] = fmaf(a_frag[m], b_frag[n], acc[m][n]);
                }
            }
        }
    }

    // Epilogue
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int global_m = cta_row + ty * TM + m;
        if (global_m < M) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                const int global_n = cta_col + tx * TN + n;
                if (global_n < N) {
                    C[global_m * N + global_n] = acc[m][n];
                }
            }
        }
    }

    #undef AS4
    #undef BS4
}

// ============================================================================
// Kernel Launch Helper with Dynamic Shared Memory Configuration
// ============================================================================

cudaError_t launch_sgemm_ampere(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    cudaStream_t stream = internal::get_capture_stream();
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);  // 16x16 = 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Calculate shared memory sizes
    const size_t smem_3stage = (STAGES * A_STAGE_SIZE + STAGES * B_STAGE_SIZE) * sizeof(float);
    const size_t smem_4stage = (STAGES_4 * A_STAGE_SIZE_SMALL + STAGES_4 * B_STAGE_SIZE_SMALL) * sizeof(float);

    // Try 3-stage kernel first (needs ~104KB smem)
    cudaError_t err;

    // Configure extended shared memory for 3-stage kernel
    err = cudaFuncSetAttribute(
        sgemm_128x128x32_3stage,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_3stage
    );

    if (err == cudaSuccess) {
        // Use 3-stage kernel with BK=32
        sgemm_128x128x32_3stage<<<grid, block, smem_3stage, stream>>>(A, B, C, M, N, K);
        return cudaGetLastError();
    }

    // Fallback to 4-stage kernel with BK=16
    err = cudaFuncSetAttribute(
        sgemm_128x128x16_4stage,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_4stage
    );

    if (err == cudaSuccess) {
        sgemm_128x128x16_4stage<<<grid, block, smem_4stage, stream>>>(A, B, C, M, N, K);
        return cudaGetLastError();
    }

    return err;
}

}  // namespace ampere
}  // namespace ops
}  // namespace pygpukit
