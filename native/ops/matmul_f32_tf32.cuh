/**
 * TF32 TensorCore GEMM Kernel for Ampere+ GPUs (SM 80+)
 *
 * Target: 22-30 TFLOPS on RTX 3090 Ti (vs 156 TFLOPS theoretical TF32)
 *
 * Key features:
 * - mma.sync.aligned.m16n8k8.row.col.tf32.tf32.f32 PTX instruction
 * - ldmatrix.sync for efficient fragment loading
 * - 4-stage cp.async software pipeline
 * - Shared memory swizzling for conflict-free access
 *
 * TF32 Precision:
 * - Input: TF32 (19-bit: 1 sign + 8 exp + 10 mantissa)
 * - Accumulator: FP32
 * - Expected error: ~1e-2 relative (vs FP32's ~1e-5)
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

// ============================================================================
// Configuration Constants - Tuned for TF32 TensorCore
// ============================================================================

// CTA tile dimensions
constexpr int BM = 128;           // Tile rows per block
constexpr int BN = 128;           // Tile cols per block
constexpr int BK = 32;            // Tile depth - multiple of 8 for mma.m16n8k8

// Warp tile dimensions (output per warp)
constexpr int WM = 64;            // Rows per warp
constexpr int WN = 64;            // Cols per warp

// MMA tile dimensions (single mma.sync operation)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;

// Block dimensions: 4 warps (128 threads)
// Each warp handles WM×WN = 64×64 output tile
// Block handles BM×BN = 128×128 with 2×2 warp arrangement
constexpr int WARPS_M = BM / WM;  // 2
constexpr int WARPS_N = BN / WN;  // 2
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 4
constexpr int NUM_THREADS = NUM_WARPS * 32;   // 128

// Pipeline stages
constexpr int STAGES = 4;

// Shared memory padding for bank conflict avoidance
// Using swizzle pattern: XOR with (row/4) to distribute banks
constexpr int SMEM_PAD_A = 8;     // A stride = BK + 8 = 40
constexpr int SMEM_PAD_B = 8;     // B stride = BN + 8 = 136

constexpr int A_SMEM_STRIDE = BK + SMEM_PAD_A;   // 40
constexpr int B_SMEM_STRIDE = BN + SMEM_PAD_B;   // 136

// Shared memory sizes per stage
constexpr int A_STAGE_SIZE = BM * A_SMEM_STRIDE;  // 128 * 40 = 5120 floats
constexpr int B_STAGE_SIZE = BK * B_SMEM_STRIDE;  // 32 * 136 = 4352 floats

// Total shared memory: 4 stages * (5120 + 4352) * 4 = 151,552 bytes = 148 KB
// Note: May need to reduce stages or BK for GPUs with less shared memory

// ============================================================================
// Helper Functions
// ============================================================================

// Convert generic pointer to shared memory address for PTX
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

// cp.async 16-byte copy
__device__ __forceinline__ void cp_async_cg_16(void* dst, const void* src) {
    unsigned int dst_smem = cvta_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_smem), "l"(src)
    );
}

// cp.async 4-byte copy
__device__ __forceinline__ void cp_async_ca_4(void* dst, const void* src) {
    unsigned int dst_smem = cvta_to_shared(dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
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

// ============================================================================
// TF32 MMA Fragment Types
// ============================================================================

// Fragment for A matrix (m16k8): 4 floats per thread
// Fragment for B matrix (k8n8): 2 floats per thread
// Fragment for C/D matrix (m16n8): 4 floats per thread

struct FragmentA {
    float x[4];  // 4 TF32 values per thread for m16k8
};

struct FragmentB {
    float x[2];  // 2 TF32 values per thread for k8n8
};

struct FragmentC {
    float x[4];  // 4 FP32 values per thread for m16n8 accumulator
};

// ============================================================================
// ldmatrix.sync helpers - Load fragments from shared memory
// ============================================================================

// ldmatrix.sync.aligned.x4.m8n8.shared.b16 loads 4 8x8 matrices
// For TF32 mma.m16n8k8, we need specific fragment layouts

__device__ __forceinline__ void ldmatrix_a(FragmentA& frag, const float* smem_ptr) {
    unsigned int smem_addr = cvta_to_shared(smem_ptr);
    unsigned int* dst = reinterpret_cast<unsigned int*>(frag.x);

    // ldmatrix.sync.aligned.x4.m8n8.shared.b16
    // Loads 4 x (8x8) matrices = 16 rows x 8 cols = m16k8 fragment
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_b(FragmentB& frag, const float* smem_ptr) {
    unsigned int smem_addr = cvta_to_shared(smem_ptr);
    unsigned int* dst = reinterpret_cast<unsigned int*>(frag.x);

    // ldmatrix.sync.aligned.x2.m8n8.shared.b16
    // Loads 2 x (8x8) matrices transposed = k8n8 fragment
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(smem_addr)
    );
}

// ============================================================================
// TF32 mma.sync instruction
// ============================================================================

// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
// D = A * B + C where A is m16k8, B is k8n8, C/D are m16n8
__device__ __forceinline__ void mma_sync_tf32(
    FragmentC& d,
    const FragmentA& a,
    const FragmentB& b,
    const FragmentC& c
) {
    const unsigned int* ua = reinterpret_cast<const unsigned int*>(a.x);
    const unsigned int* ub = reinterpret_cast<const unsigned int*>(b.x);

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
        : "r"(ua[0]), "r"(ua[1]), "r"(ua[2]), "r"(ua[3]),
          "r"(ub[0]), "r"(ub[1]),
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3])
    );
}

// ============================================================================
// Swizzle function for bank conflict-free access
// ============================================================================

// XOR-based swizzle: XOR the column index with (row / 4) to distribute banks
__device__ __forceinline__ int swizzle_offset(int row, int col, int stride) {
    // Swizzle pattern: XOR lower bits of col with bits from row
    int swizzled_col = col ^ ((row >> 2) & 0x7);
    return row * stride + swizzled_col;
}

// ============================================================================
// TF32 TensorCore GEMM Kernel
// ============================================================================

__global__ void __launch_bounds__(128, 2)
sgemm_tf32_128x128x32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Thread/warp indices
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp position in 2x2 grid
    const int warp_m = warp_id / WARPS_N;  // 0 or 1
    const int warp_n = warp_id % WARPS_N;  // 0 or 1

    // Block position
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int cta_row = by * BM;
    const int cta_col = bx * BN;

    // ========================================================================
    // Shared Memory
    // ========================================================================
    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = smem + STAGES * A_STAGE_SIZE;

    #define AS(stage, m, k) As[(stage) * A_STAGE_SIZE + (m) * A_SMEM_STRIDE + (k)]
    #define BS(stage, k, n) Bs[(stage) * B_STAGE_SIZE + (k) * B_SMEM_STRIDE + (n)]

    // ========================================================================
    // Accumulators - each warp computes 64x64 output
    // 64x64 = (4*16) x (8*8) = 4x8 mma tiles = 32 mma.sync per warp
    // Each mma.sync produces 16x8 output with 4 floats per thread
    // Total per warp: 32 * 4 = 128 floats per thread... but overlapping
    // Actually: 4x8 mma tiles, each with 4 floats = 128 floats per thread
    // ========================================================================

    // Warp tile: 64x64 output = (4 mma_m) x (8 mma_n) = 4x8 = 32 mma tiles
    constexpr int WARP_MMA_M = WM / MMA_M;  // 64/16 = 4
    constexpr int WARP_MMA_N = WN / MMA_N;  // 64/8 = 8

    FragmentC acc[WARP_MMA_M][WARP_MMA_N];

    // Initialize accumulators to zero
    #pragma unroll
    for (int i = 0; i < WARP_MMA_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_MMA_N; ++j) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                acc[i][j].x[k] = 0.0f;
            }
        }
    }

    const int num_k_tiles = (K + BK - 1) / BK;

    // ========================================================================
    // Load functions with cp.async
    // ========================================================================

    // Load A tile: BM x BK = 128 x 32 = 4096 floats
    // 128 threads, each loads 32 floats
    auto load_A = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;

        // Each thread loads 32 floats = 8 float4s
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int float4_idx = tid + i * NUM_THREADS;
            const int a_m = float4_idx / (BK / 4);       // 0-127
            const int a_k = (float4_idx % (BK / 4)) * 4; // 0, 4, 8, ..., 28

            const int global_m = cta_row + a_m;
            const int global_k = k_base + a_k;

            float* dst = &AS(stage, a_m, a_k);

            if (global_m < M && global_k + 3 < K) {
                const float* src = &A[global_m * K + global_k];
                cp_async_cg_16(dst, src);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (global_m < M && global_k + j < K) {
                        cp_async_ca_4(&dst[j], &A[global_m * K + global_k + j]);
                    } else {
                        dst[j] = 0.0f;
                    }
                }
            }
        }
    };

    // Load B tile: BK x BN = 32 x 128 = 4096 floats
    auto load_B = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int float4_idx = tid + i * NUM_THREADS;
            const int b_k = float4_idx / (BN / 4);
            const int b_n = (float4_idx % (BN / 4)) * 4;

            const int global_k = k_base + b_k;
            const int global_n = cta_col + b_n;

            float* dst = &BS(stage, b_k, b_n);

            if (global_k < K && global_n + 3 < N) {
                const float* src = &B[global_k * N + global_n];
                cp_async_cg_16(dst, src);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (global_k < K && global_n + j < N) {
                        cp_async_ca_4(&dst[j], &B[global_k * N + global_n + j]);
                    } else {
                        dst[j] = 0.0f;
                    }
                }
            }
        }
    };

    // ========================================================================
    // Pipeline Prologue
    // ========================================================================
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < num_k_tiles) {
            load_A(s, s);
            load_B(s, s);
        }
        cp_async_commit();
    }

    // ========================================================================
    // Main Loop
    // ========================================================================
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int compute_stage = k_tile % STAGES;
        const int load_stage = (k_tile + STAGES - 1) % STAGES;
        const int load_k_tile = k_tile + STAGES - 1;

        // Issue loads for future tile
        if (load_k_tile < num_k_tiles) {
            load_A(load_stage, load_k_tile);
            load_B(load_stage, load_k_tile);
        }
        cp_async_commit();

        // Wait for compute tile
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();

        // Compute: iterate over K dimension in chunks of MMA_K=8
        #pragma unroll
        for (int k = 0; k < BK; k += MMA_K) {
            // Load A and B fragments for all mma tiles in warp
            FragmentA a_frag[WARP_MMA_M];
            FragmentB b_frag[WARP_MMA_N];

            // Load A fragments: 4 x m16k8 fragments
            // Each warp row loads from warp_m * WM = 0 or 64
            #pragma unroll
            for (int mi = 0; mi < WARP_MMA_M; ++mi) {
                const int m_offset = warp_m * WM + mi * MMA_M;
                const int lane_row = lane_id % 16;
                const int lane_group = lane_id / 16;
                const float* a_ptr = &AS(compute_stage, m_offset + lane_row, k + lane_group * 4);
                ldmatrix_a(a_frag[mi], a_ptr);
            }

            // Load B fragments: 8 x k8n8 fragments
            #pragma unroll
            for (int ni = 0; ni < WARP_MMA_N; ++ni) {
                const int n_offset = warp_n * WN + ni * MMA_N;
                const int lane_row = lane_id % 8;
                const int lane_col = (lane_id / 8) % 4;
                const float* b_ptr = &BS(compute_stage, k + lane_row, n_offset + lane_col * 2);
                ldmatrix_b(b_frag[ni], b_ptr);
            }

            // Execute mma.sync for all tile combinations
            #pragma unroll
            for (int mi = 0; mi < WARP_MMA_M; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < WARP_MMA_N; ++ni) {
                    mma_sync_tf32(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }
    }

    // ========================================================================
    // Epilogue: Store results
    // ========================================================================
    // Each thread stores its portion of the accumulator
    // FragmentC layout for m16n8: 4 floats per thread
    // Thread mapping in warp for m16n8:
    //   lane 0-15: rows 0-7 (lane%8), cols 0-1 ((lane/8)*2)
    //   lane 16-31: rows 8-15 ((lane-16)%8+8), cols 0-1

    #pragma unroll
    for (int mi = 0; mi < WARP_MMA_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_MMA_N; ++ni) {
            // Calculate output position for this mma tile
            const int tile_m = cta_row + warp_m * WM + mi * MMA_M;
            const int tile_n = cta_col + warp_n * WN + ni * MMA_N;

            // Thread's position within the 16x8 output tile
            // Accumulator layout: lane maps to specific (row, col) pairs
            const int lane_row_base = (lane_id % 8) + (lane_id / 16) * 8;
            const int lane_col_base = ((lane_id / 8) % 2) * 2;

            // Each thread has 4 elements: 2 consecutive cols at 2 row positions
            #pragma unroll
            for (int elem = 0; elem < 4; ++elem) {
                const int row_offset = (elem / 2) * 8;  // 0 or 8
                const int col_offset = elem % 2;       // 0 or 1

                const int global_m = tile_m + lane_row_base + row_offset;
                const int global_n = tile_n + lane_col_base + col_offset;

                if (global_m < M && global_n < N) {
                    C[global_m * N + global_n] = acc[mi][ni].x[elem];
                }
            }
        }
    }

    #undef AS
    #undef BS
}

// ============================================================================
// Simplified TF32 Kernel using WMMA API (for correctness baseline)
// ============================================================================

using namespace nvcuda::wmma;

__global__ void __launch_bounds__(256, 2)
sgemm_tf32_wmma_128x128x32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // WMMA dimensions
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 8;

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Starting positions
    const int cta_row = by * BM;
    const int cta_col = bx * BN;

    // Warp position: 8 warps in 4x2 arrangement
    const int warp_m = warp_id / 2;  // 0-3
    const int warp_n = warp_id % 2;  // 0-1

    // Each warp handles 32x64 output (2 WMMA_M x 4 WMMA_N)
    constexpr int WARP_TILES_M = 2;
    constexpr int WARP_TILES_N = 4;

    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag[WARP_TILES_M];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> b_frag[WARP_TILES_N];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];

    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j) {
            fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // Shared memory for double buffering
    __shared__ float As[2][BM][BK + 8];  // +8 for padding
    __shared__ float Bs[2][BK][BN + 8];

    const int num_k_tiles = (K + BK - 1) / BK;

    // Load first tile
    auto load_tile = [&](int stage, int k_tile) {
        const int k_base = k_tile * BK;

        // Load A: 128x32, 256 threads -> 16 elements per thread
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = tid + i * 256;
            const int m = idx / BK;
            const int k = idx % BK;
            const int global_m = cta_row + m;
            const int global_k = k_base + k;

            if (global_m < M && global_k < K) {
                As[stage][m][k] = A[global_m * K + global_k];
            } else {
                As[stage][m][k] = 0.0f;
            }
        }

        // Load B: 32x128, 256 threads -> 16 elements per thread
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = tid + i * 256;
            const int k = idx / BN;
            const int n = idx % BN;
            const int global_k = k_base + k;
            const int global_n = cta_col + n;

            if (global_k < K && global_n < N) {
                Bs[stage][k][n] = B[global_k * N + global_n];
            } else {
                Bs[stage][k][n] = 0.0f;
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
            #pragma unroll
            for (int mi = 0; mi < WARP_TILES_M; ++mi) {
                const int m_offset = warp_m * 32 + mi * WMMA_M;
                load_matrix_sync(a_frag[mi], &As[curr_stage][m_offset][k], BK + 8);
            }

            // Load B fragments
            #pragma unroll
            for (int ni = 0; ni < WARP_TILES_N; ++ni) {
                const int n_offset = warp_n * 64 + ni * WMMA_N;
                load_matrix_sync(b_frag[ni], &Bs[curr_stage][k][n_offset], BN + 8);
            }

            // Compute
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

    // Store results
    #pragma unroll
    for (int mi = 0; mi < WARP_TILES_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WARP_TILES_N; ++ni) {
            const int m_offset = cta_row + warp_m * 32 + mi * WMMA_M;
            const int n_offset = cta_col + warp_n * 64 + ni * WMMA_N;

            if (m_offset < M && n_offset < N) {
                store_matrix_sync(&C[m_offset * N + n_offset], c_frag[mi][ni], N, mem_row_major);
            }
        }
    }
}

// ============================================================================
// Kernel Launch Helper
// ============================================================================

inline cudaError_t launch_sgemm_tf32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    // Use WMMA kernel for now (more reliable)
    dim3 block(16, 16);  // 256 threads = 8 warps
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Calculate shared memory size
    const size_t smem_size = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(float);

    // Check if we can use extended shared memory
    cudaError_t err = cudaFuncSetAttribute(
        sgemm_tf32_wmma_128x128x32,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        0  // Using static shared memory
    );

    sgemm_tf32_wmma_128x128x32<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    return cudaGetLastError();
}

}  // namespace tf32
}  // namespace ops
}  // namespace pygpukit
