/**
 * FP32/FP64 Matrix Multiplication Kernels
 *
 * Three implementations:
 * 1. L2-optimized kernel: For small matrices (<128), uses __ldg() cache
 * 2. Tiled kernel: For medium matrices, uses shared memory double buffering
 * 3. Optimized kernel: For large matrices (>=128), high-performance SGEMM
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "../../../../../../core/cuda_graph.hpp"

namespace pygpukit {
namespace ops {
namespace matmul_fp32 {

// Block size for L2-optimized kernel
constexpr int BLOCK_SIZE = 16;

// Tiled matmul configuration
constexpr int TILE_M = 64;     // Output tile height
constexpr int TILE_N = 64;     // Output tile width
constexpr int TILE_K = 16;     // Reduction tile depth
constexpr int THREAD_M = 4;    // Elements per thread in M dimension
constexpr int THREAD_N = 4;    // Elements per thread in N dimension

// ============================================================================
// L2-Optimized Kernels (Small Matrices)
// ============================================================================

__global__ void matmul_f32_l2opt_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        #pragma unroll 4
        for (size_t k = 0; k < K; ++k) {
            sum += __ldg(&A[row * K + k]) * __ldg(&B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_f64_l2opt_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        #pragma unroll 4
        for (size_t k = 0; k < K; ++k) {
            sum += __ldg(&A[row * K + k]) * __ldg(&B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Tiled Kernel with Double Buffering (FP32)
// ============================================================================

__global__ void matmul_f32_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    __shared__ float As[2][TILE_K][TILE_M + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];

    float accum[THREAD_M][THREAD_N] = {{0.0f}};

    const size_t block_row_start = by * TILE_M;
    const size_t block_col_start = bx * TILE_N;

    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    int curr_buf = 0;

    // Prefetch first tile
    {
        const int a_loads_per_thread = (TILE_M * TILE_K + num_threads - 1) / num_threads;
        for (int i = 0; i < a_loads_per_thread; ++i) {
            int load_idx = tid + i * num_threads;
            if (load_idx < TILE_M * TILE_K) {
                int a_row = load_idx / TILE_K;
                int a_col = load_idx % TILE_K;
                size_t global_row = block_row_start + a_row;
                size_t global_col = a_col;
                if (global_row < M && global_col < K) {
                    As[0][a_col][a_row] = A[global_row * K + global_col];
                } else {
                    As[0][a_col][a_row] = 0.0f;
                }
            }
        }

        const int b_loads_per_thread = (TILE_K * TILE_N + num_threads - 1) / num_threads;
        for (int i = 0; i < b_loads_per_thread; ++i) {
            int load_idx = tid + i * num_threads;
            if (load_idx < TILE_K * TILE_N) {
                int b_row = load_idx / TILE_N;
                int b_col = load_idx % TILE_N;
                size_t global_row = b_row;
                size_t global_col = block_col_start + b_col;
                if (global_row < K && global_col < N) {
                    Bs[0][b_row][b_col] = B[global_row * N + global_col];
                } else {
                    Bs[0][b_row][b_col] = 0.0f;
                }
            }
        }
    }
    __syncthreads();

    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        int next_buf = 1 - curr_buf;

        if (tile_k + 1 < num_k_tiles) {
            size_t k_offset = (tile_k + 1) * TILE_K;

            const int a_loads_per_thread = (TILE_M * TILE_K + num_threads - 1) / num_threads;
            for (int i = 0; i < a_loads_per_thread; ++i) {
                int load_idx = tid + i * num_threads;
                if (load_idx < TILE_M * TILE_K) {
                    int a_row = load_idx / TILE_K;
                    int a_col = load_idx % TILE_K;
                    size_t global_row = block_row_start + a_row;
                    size_t global_col = k_offset + a_col;
                    if (global_row < M && global_col < K) {
                        As[next_buf][a_col][a_row] = A[global_row * K + global_col];
                    } else {
                        As[next_buf][a_col][a_row] = 0.0f;
                    }
                }
            }

            const int b_loads_per_thread = (TILE_K * TILE_N + num_threads - 1) / num_threads;
            for (int i = 0; i < b_loads_per_thread; ++i) {
                int load_idx = tid + i * num_threads;
                if (load_idx < TILE_K * TILE_N) {
                    int b_row = load_idx / TILE_N;
                    int b_col = load_idx % TILE_N;
                    size_t global_row = k_offset + b_row;
                    size_t global_col = block_col_start + b_col;
                    if (global_row < K && global_col < N) {
                        Bs[next_buf][b_row][b_col] = B[global_row * N + global_col];
                    } else {
                        Bs[next_buf][b_row][b_col] = 0.0f;
                    }
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_frag[THREAD_M];
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                a_frag[m] = As[curr_buf][k][ty * THREAD_M + m];
            }

            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                float b_val = Bs[curr_buf][k][tx * THREAD_N + n];
                #pragma unroll
                for (int m = 0; m < THREAD_M; ++m) {
                    accum[m][n] += a_frag[m] * b_val;
                }
            }
        }

        __syncthreads();
        curr_buf = next_buf;
    }

    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        size_t out_row = block_row_start + ty * THREAD_M + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                size_t out_col = block_col_start + tx * THREAD_N + n;
                if (out_col < N) {
                    C[out_row * N + out_col] = accum[m][n];
                }
            }
        }
    }
}

// ============================================================================
// Tiled Kernel with Double Buffering (FP64)
// ============================================================================

__global__ void matmul_f64_tiled_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    constexpr int TILE_K_F64 = 8;
    __shared__ double As[2][TILE_K_F64][TILE_M + 1];
    __shared__ double Bs[2][TILE_K_F64][TILE_N + 1];

    double accum[THREAD_M][THREAD_N] = {{0.0}};

    const size_t block_row_start = by * TILE_M;
    const size_t block_col_start = bx * TILE_N;

    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;
    const int num_k_tiles = (K + TILE_K_F64 - 1) / TILE_K_F64;

    int curr_buf = 0;

    // Prefetch first tile
    {
        const int a_loads_per_thread = (TILE_M * TILE_K_F64 + num_threads - 1) / num_threads;
        for (int i = 0; i < a_loads_per_thread; ++i) {
            int load_idx = tid + i * num_threads;
            if (load_idx < TILE_M * TILE_K_F64) {
                int a_row = load_idx / TILE_K_F64;
                int a_col = load_idx % TILE_K_F64;
                size_t global_row = block_row_start + a_row;
                size_t global_col = a_col;
                if (global_row < M && global_col < K) {
                    As[0][a_col][a_row] = A[global_row * K + global_col];
                } else {
                    As[0][a_col][a_row] = 0.0;
                }
            }
        }

        const int b_loads_per_thread = (TILE_K_F64 * TILE_N + num_threads - 1) / num_threads;
        for (int i = 0; i < b_loads_per_thread; ++i) {
            int load_idx = tid + i * num_threads;
            if (load_idx < TILE_K_F64 * TILE_N) {
                int b_row = load_idx / TILE_N;
                int b_col = load_idx % TILE_N;
                size_t global_row = b_row;
                size_t global_col = block_col_start + b_col;
                if (global_row < K && global_col < N) {
                    Bs[0][b_row][b_col] = B[global_row * N + global_col];
                } else {
                    Bs[0][b_row][b_col] = 0.0;
                }
            }
        }
    }
    __syncthreads();

    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        int next_buf = 1 - curr_buf;

        if (tile_k + 1 < num_k_tiles) {
            size_t k_offset = (tile_k + 1) * TILE_K_F64;

            const int a_loads_per_thread = (TILE_M * TILE_K_F64 + num_threads - 1) / num_threads;
            for (int i = 0; i < a_loads_per_thread; ++i) {
                int load_idx = tid + i * num_threads;
                if (load_idx < TILE_M * TILE_K_F64) {
                    int a_row = load_idx / TILE_K_F64;
                    int a_col = load_idx % TILE_K_F64;
                    size_t global_row = block_row_start + a_row;
                    size_t global_col = k_offset + a_col;
                    if (global_row < M && global_col < K) {
                        As[next_buf][a_col][a_row] = A[global_row * K + global_col];
                    } else {
                        As[next_buf][a_col][a_row] = 0.0;
                    }
                }
            }

            const int b_loads_per_thread = (TILE_K_F64 * TILE_N + num_threads - 1) / num_threads;
            for (int i = 0; i < b_loads_per_thread; ++i) {
                int load_idx = tid + i * num_threads;
                if (load_idx < TILE_K_F64 * TILE_N) {
                    int b_row = load_idx / TILE_N;
                    int b_col = load_idx % TILE_N;
                    size_t global_row = k_offset + b_row;
                    size_t global_col = block_col_start + b_col;
                    if (global_row < K && global_col < N) {
                        Bs[next_buf][b_row][b_col] = B[global_row * N + global_col];
                    } else {
                        Bs[next_buf][b_row][b_col] = 0.0;
                    }
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < TILE_K_F64; ++k) {
            double a_frag[THREAD_M];
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                a_frag[m] = As[curr_buf][k][ty * THREAD_M + m];
            }

            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                double b_val = Bs[curr_buf][k][tx * THREAD_N + n];
                #pragma unroll
                for (int m = 0; m < THREAD_M; ++m) {
                    accum[m][n] += a_frag[m] * b_val;
                }
            }
        }

        __syncthreads();
        curr_buf = next_buf;
    }

    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        size_t out_row = block_row_start + ty * THREAD_M + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                size_t out_col = block_col_start + tx * THREAD_N + n;
                if (out_col < N) {
                    C[out_row * N + out_col] = accum[m][n];
                }
            }
        }
    }
}

// ============================================================================
// Launch Helpers
// ============================================================================

inline void launch_l2opt_f32(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = internal::get_capture_stream();
    matmul_f32_l2opt_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

inline void launch_l2opt_f64(const double* A, const double* B, double* C, size_t M, size_t N, size_t K) {
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = internal::get_capture_stream();
    matmul_f64_l2opt_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

inline void launch_tiled_f32(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    dim3 block_size(TILE_N / THREAD_N, TILE_M / THREAD_M);
    dim3 grid_size((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    cudaStream_t stream = internal::get_capture_stream();
    matmul_f32_tiled_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

inline void launch_tiled_f64(const double* A, const double* B, double* C, size_t M, size_t N, size_t K) {
    dim3 block_size(TILE_N / THREAD_N, TILE_M / THREAD_M);
    dim3 grid_size((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    cudaStream_t stream = internal::get_capture_stream();
    matmul_f64_tiled_kernel<<<grid_size, block_size, 0, stream>>>(A, B, C, M, N, K);
}

} // namespace matmul_fp32
} // namespace ops
} // namespace pygpukit
