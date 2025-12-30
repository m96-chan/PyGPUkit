/**
 * Native Int8 GEMM using CUDA cores (SM120)
 *
 * SM120 (RTX 5090) does NOT have native Int8 TensorCore MMA instructions.
 * This kernel uses CUDA cores with vectorized dp4a (dot product of 4 Int8 values).
 *
 * dp4a: Dot Product and Accumulate (4 elements)
 * D = A.x*B.x + A.y*B.y + A.z*B.z + A.w*B.w + C
 * where A, B are int8x4 packed in uint32, C and D are int32
 *
 * Layout:
 * - A: [M, K] Int8, row-major
 * - B: [N, K] Int8, row-major (transposed B, col-major in terms of original B)
 * - D: [M, N] Int32
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace gemm {

// ============================================================================
// Configuration
// ============================================================================

struct Int8GemmConfig {
    static constexpr int BLOCK_M = 64;
    static constexpr int BLOCK_N = 64;
    static constexpr int BLOCK_K = 32;  // Must be multiple of 4 for dp4a
    static constexpr int THREAD_M = 4;
    static constexpr int THREAD_N = 4;
    static constexpr int THREADS_PER_BLOCK = 256;
};

// ============================================================================
// dp4a intrinsic wrapper
// ============================================================================

__device__ __forceinline__ int32_t dp4a(uint32_t a, uint32_t b, int32_t c) {
    int32_t result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

// ============================================================================
// Native Int8 GEMM Kernel with dp4a
// ============================================================================

/**
 * Each thread block computes a BLOCK_M x BLOCK_N tile of C.
 * Each thread computes a THREAD_M x THREAD_N sub-tile.
 *
 * Uses shared memory for A and B tiles to reduce global memory bandwidth.
 */
template<typename Config = Int8GemmConfig>
__global__ void int8_gemm_native_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ D,
    int M, int N, int K
) {
    // Block and thread indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    // Thread position within block
    const int thread_row = tx / (Config::BLOCK_N / Config::THREAD_N);
    const int thread_col = tx % (Config::BLOCK_N / Config::THREAD_N);

    // Starting position for this block
    const int block_row_start = by * Config::BLOCK_M;
    const int block_col_start = bx * Config::BLOCK_N;

    // Shared memory for tiles (K rounded up to multiple of 4)
    __shared__ int8_t smem_A[Config::BLOCK_M][Config::BLOCK_K];
    __shared__ int8_t smem_B[Config::BLOCK_N][Config::BLOCK_K];

    // Register accumulators for THREAD_M x THREAD_N output
    int32_t acc[Config::THREAD_M][Config::THREAD_N] = {0};

    // K-dimension tiles
    const int num_k_tiles = (K + Config::BLOCK_K - 1) / Config::BLOCK_K;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_start = kt * Config::BLOCK_K;

        // Cooperative load of A tile into shared memory
        // Each thread loads multiple elements
        for (int i = tx; i < Config::BLOCK_M * Config::BLOCK_K; i += Config::THREADS_PER_BLOCK) {
            int row = i / Config::BLOCK_K;
            int col = i % Config::BLOCK_K;
            int global_row = block_row_start + row;
            int global_col = k_start + col;

            if (global_row < M && global_col < K) {
                smem_A[row][col] = A[global_row * K + global_col];
            } else {
                smem_A[row][col] = 0;
            }
        }

        // Cooperative load of B tile into shared memory
        // B is [N, K], row-major (transposed)
        for (int i = tx; i < Config::BLOCK_N * Config::BLOCK_K; i += Config::THREADS_PER_BLOCK) {
            int row = i / Config::BLOCK_K;
            int col = i % Config::BLOCK_K;
            int global_row = block_col_start + row;
            int global_col = k_start + col;

            if (global_row < N && global_col < K) {
                smem_B[row][col] = B[global_row * K + global_col];
            } else {
                smem_B[row][col] = 0;
            }
        }

        __syncthreads();

        // Compute using dp4a (4 Int8 values at a time)
        #pragma unroll
        for (int kk = 0; kk < Config::BLOCK_K; kk += 4) {
            // Load THREAD_M rows of A as uint32 (4 Int8 values)
            uint32_t a_vals[Config::THREAD_M];
            #pragma unroll
            for (int m = 0; m < Config::THREAD_M; ++m) {
                int row = thread_row * Config::THREAD_M + m;
                a_vals[m] = *reinterpret_cast<const uint32_t*>(&smem_A[row][kk]);
            }

            // Load THREAD_N rows of B as uint32 (4 Int8 values)
            uint32_t b_vals[Config::THREAD_N];
            #pragma unroll
            for (int n = 0; n < Config::THREAD_N; ++n) {
                int row = thread_col * Config::THREAD_N + n;
                b_vals[n] = *reinterpret_cast<const uint32_t*>(&smem_B[row][kk]);
            }

            // Accumulate using dp4a
            #pragma unroll
            for (int m = 0; m < Config::THREAD_M; ++m) {
                #pragma unroll
                for (int n = 0; n < Config::THREAD_N; ++n) {
                    acc[m][n] = dp4a(a_vals[m], b_vals[n], acc[m][n]);
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int m = 0; m < Config::THREAD_M; ++m) {
        #pragma unroll
        for (int n = 0; n < Config::THREAD_N; ++n) {
            int global_row = block_row_start + thread_row * Config::THREAD_M + m;
            int global_col = block_col_start + thread_col * Config::THREAD_N + n;

            if (global_row < M && global_col < N) {
                D[global_row * N + global_col] = acc[m][n];
            }
        }
    }
}

// ============================================================================
// Launch Function
// ============================================================================

cudaError_t launch_int8_gemm_native(
    const int8_t* A,
    const int8_t* B,
    int32_t* D,
    int M, int N, int K,
    cudaStream_t stream
) {
    using Config = Int8GemmConfig;

    dim3 block(Config::THREADS_PER_BLOCK);
    dim3 grid(
        (N + Config::BLOCK_N - 1) / Config::BLOCK_N,
        (M + Config::BLOCK_M - 1) / Config::BLOCK_M
    );

    int8_gemm_native_kernel<Config><<<grid, block, 0, stream>>>(
        A, B, D, M, N, K
    );

    return cudaGetLastError();
}

}  // namespace gemm
}  // namespace ops
}  // namespace pygpukit

// ============================================================================
// Extern C Interface
// ============================================================================

extern "C" {

/**
 * Native Int8 GEMM using dp4a CUDA cores
 *
 * @param A     [M, K] Int8 input matrix (row-major)
 * @param B     [N, K] Int8 weight matrix (row-major, transposed)
 * @param D     [M, N] Int32 output matrix
 * @param M     Number of rows in A and D
 * @param N     Number of columns in D (rows in B)
 * @param K     Inner dimension
 * @param stream CUDA stream
 */
cudaError_t pygpukit_gemm_int8_native_sm120(
    const int8_t* A,
    const int8_t* B,
    int32_t* D,
    int M, int N, int K,
    cudaStream_t stream
) {
    return pygpukit::ops::gemm::launch_int8_gemm_native(A, B, D, M, N, K, stream);
}

/**
 * Check if native Int8 GEMM is available
 * Always available on any GPU with dp4a support (SM61+)
 */
bool pygpukit_int8_native_gemm_available() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return false;

    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    int sm = major * 10 + minor;
    return sm >= 61;  // dp4a available from SM61 (Pascal)
}

}  // extern "C"
