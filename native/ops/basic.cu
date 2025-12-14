#include "basic.cuh"
#include "matmul_f32_ampere.cuh"
#include "matmul_f32_tf32.cuh"
#include <stdexcept>
#include <cstdlib>

#ifdef PYGPUKIT_DRIVER_ONLY
#include "../core/driver_context.hpp"
#include <cuda.h>

namespace pygpukit {
namespace ops {

namespace {

void check_driver_error(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* error_str = nullptr;
        cuGetErrorString(result, &error_str);
        throw CudaError(std::string(msg) + ": " + (error_str ? error_str : "unknown error"));
    }
}

void sync_and_check(const char* msg) {
    check_driver_error(cuCtxSynchronize(), msg);
}

#else
#include <cuda_runtime.h>

namespace pygpukit {
namespace ops {

namespace {

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw CudaError(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

void sync_and_check(const char* msg) {
    check_cuda_error(cudaGetLastError(), msg);
    check_cuda_error(cudaDeviceSynchronize(), msg);
}

#endif // PYGPUKIT_DRIVER_ONLY

void validate_same_shape(const GPUArray& a, const GPUArray& b, const char* op_name) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error(std::string(op_name) + " requires arrays of same shape");
    }
}

void validate_same_dtype(const GPUArray& a, const GPUArray& b, const char* op_name) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error(std::string(op_name) + " requires arrays of same dtype");
    }
}

void validate_matmul_shapes(const GPUArray& a, const GPUArray& b, const char* op_name) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " requires 2D arrays");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error(std::string(op_name) + " dimension mismatch");
    }
}

} // anonymous namespace

// ============================================================================
// Add kernels
// ============================================================================

__global__ void add_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_f64_kernel(const double* a, const double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_i32_kernel(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_i64_kernel(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void add(const GPUArray& a, const GPUArray& b, GPUArray& c) {
    validate_same_shape(a, b, "add");
    validate_same_dtype(a, b, "add");
    validate_same_shape(a, c, "add");
    validate_same_dtype(a, c, "add");

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            add_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                n);
            break;
        case DataType::Float64:
            add_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<const double*>(b.data()),
                static_cast<double*>(c.data()),
                n);
            break;
        case DataType::Int32:
            add_i32_kernel<<<grid_size, block_size>>>(
                static_cast<const int32_t*>(a.data()),
                static_cast<const int32_t*>(b.data()),
                static_cast<int32_t*>(c.data()),
                n);
            break;
        case DataType::Int64:
            add_i64_kernel<<<grid_size, block_size>>>(
                static_cast<const int64_t*>(a.data()),
                static_cast<const int64_t*>(b.data()),
                static_cast<int64_t*>(c.data()),
                n);
            break;
    }

    sync_and_check("add kernel failed");
}

GPUArray add(const GPUArray& a, const GPUArray& b) {
    validate_same_shape(a, b, "add");
    validate_same_dtype(a, b, "add");

    GPUArray c(a.shape(), a.dtype());
    add(a, b, c);
    return c;
}

// ============================================================================
// Mul kernels
// ============================================================================

__global__ void mul_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_f64_kernel(const double* a, const double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_i32_kernel(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_i64_kernel(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

void mul(const GPUArray& a, const GPUArray& b, GPUArray& c) {
    validate_same_shape(a, b, "mul");
    validate_same_dtype(a, b, "mul");
    validate_same_shape(a, c, "mul");
    validate_same_dtype(a, c, "mul");

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            mul_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                n);
            break;
        case DataType::Float64:
            mul_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<const double*>(b.data()),
                static_cast<double*>(c.data()),
                n);
            break;
        case DataType::Int32:
            mul_i32_kernel<<<grid_size, block_size>>>(
                static_cast<const int32_t*>(a.data()),
                static_cast<const int32_t*>(b.data()),
                static_cast<int32_t*>(c.data()),
                n);
            break;
        case DataType::Int64:
            mul_i64_kernel<<<grid_size, block_size>>>(
                static_cast<const int64_t*>(a.data()),
                static_cast<const int64_t*>(b.data()),
                static_cast<int64_t*>(c.data()),
                n);
            break;
    }

    sync_and_check("mul kernel failed");
}

GPUArray mul(const GPUArray& a, const GPUArray& b) {
    validate_same_shape(a, b, "mul");
    validate_same_dtype(a, b, "mul");

    GPUArray c(a.shape(), a.dtype());
    mul(a, b, c);
    return c;
}

// ============================================================================
// Matmul kernels - Tiled with Shared Memory and Double Buffering
// ============================================================================
//
// Two implementations:
// 1. L2-optimized kernel: For small matrices (<2048), uses __ldg() cache
// 2. Tiled kernel: For large matrices (>=2048), uses shared memory tiling
//
// Tiled kernel features:
// - Configurable TILE_M, TILE_N, TILE_K
// - Double-buffered prefetch to overlap compute with memory loads
// - Bank-conflict-free shared memory access
// - Coalesced global memory access
//
// ============================================================================

// Small matrix kernel block size
#define BLOCK_SIZE 16

// Tiled matmul configuration (legacy, for compatibility)
#define TILE_M 64      // Output tile height
#define TILE_N 64      // Output tile width
#define TILE_K 16      // Reduction tile depth
#define THREAD_M 4     // Elements per thread in M dimension
#define THREAD_N 4     // Elements per thread in N dimension

// RTX 3090 Ti optimized configuration
// Target: 35.6 TFLOPS (90% of 40 TFLOPS theoretical)
// 128x128 tiles with 8x8 elements per thread for high compute intensity
#define OPT_TILE_M 128     // Output tile height
#define OPT_TILE_N 128     // Output tile width
#define OPT_TILE_K 32      // Reduction tile depth (larger K = more compute per load)
#define OPT_THREAD_M 8     // Elements per thread in M dimension
#define OPT_THREAD_N 8     // Elements per thread in N dimension
// Block: (128/8, 128/8) = (16, 16) = 256 threads
// Each thread: 8x8 = 64 FMAs per K iteration
// Shared memory per buffer: (128*32 + 32*128) * 4 = 32KB (fits in 48KB default)
// Double buffer: 64KB (need to use extended shared memory)

// Threshold for switching to tiled kernel
#define TILED_MATMUL_THRESHOLD 128

// Threshold for switching to optimized kernel (larger matrices benefit more)
// DEBUG: Temporarily lowered from 2048 to 128 for testing TF32 kernel
#define OPTIMIZED_MATMUL_THRESHOLD 128

// L2-optimized matmul kernel for FP32 (Ampere+)
// Uses __ldg() for read-only cache and __restrict__ for aliasing hints
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

        // Use __ldg() for read-only loads through texture cache
        // This leverages L2 cache more efficiently on Ampere
        #pragma unroll 4
        for (size_t k = 0; k < K; ++k) {
            sum += __ldg(&A[row * K + k]) * __ldg(&B[k * N + col]);
        }

        C[row * N + col] = sum;
    }
}

// L2-optimized matmul kernel for FP64 (Ampere+)
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
// Tiled Matmul with Shared Memory and Double Buffering (FP32)
// ============================================================================
//
// Each thread block computes a TILE_M x TILE_N output tile.
// Each thread computes THREAD_M x THREAD_N elements.
// Block dimensions: (TILE_N / THREAD_N, TILE_M / THREAD_M) = (16, 16) = 256 threads
//
// Double buffering:
// - While computing with data in shared memory buffer 0
// - Prefetch next tile into shared memory buffer 1
// - Swap buffers each iteration
//
__global__ void matmul_f32_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    // Thread and block indices
    const int tx = threadIdx.x;  // 0..15 (TILE_N / THREAD_N)
    const int ty = threadIdx.y;  // 0..15 (TILE_M / THREAD_M)
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Shared memory for double buffering
    // Pad by 1 to avoid bank conflicts
    __shared__ float As[2][TILE_K][TILE_M + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];

    // Thread's output tile (THREAD_M x THREAD_N elements)
    float accum[THREAD_M][THREAD_N] = {{0.0f}};

    // Global row/col start for this thread block
    const size_t block_row_start = by * TILE_M;
    const size_t block_col_start = bx * TILE_N;

    // Linear thread ID within block
    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;  // 256

    // Number of tiles along K dimension
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Current buffer index for double buffering
    int curr_buf = 0;

    // Prefetch first tile into buffer 0
    {
        // Load A tile: TILE_M x TILE_K elements, 256 threads
        // Each thread loads multiple elements
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

        // Load B tile: TILE_K x TILE_N elements
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

    // Main loop over K tiles with double buffering
    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        int next_buf = 1 - curr_buf;

        // Prefetch next tile (if not the last tile)
        if (tile_k + 1 < num_k_tiles) {
            size_t k_offset = (tile_k + 1) * TILE_K;

            // Load A tile
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

            // Load B tile
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

        // Compute using current buffer
        // Each thread computes THREAD_M x THREAD_N elements
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            // Load A fragment for this thread
            float a_frag[THREAD_M];
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                a_frag[m] = As[curr_buf][k][ty * THREAD_M + m];
            }

            // Load B fragment and compute
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                float b_val = Bs[curr_buf][k][tx * THREAD_N + n];
                #pragma unroll
                for (int m = 0; m < THREAD_M; ++m) {
                    accum[m][n] += a_frag[m] * b_val;
                }
            }
        }

        // Sync before swapping buffers
        __syncthreads();
        curr_buf = next_buf;
    }

    // Write output
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

// Tiled Matmul for FP64
__global__ void matmul_f64_tiled_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Shared memory (smaller tiles for FP64 due to memory constraints)
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
// RTX 3090 Ti Optimized Matmul Kernel (FP32) v2
// ============================================================================
//
// High-performance SGEMM kernel optimized for Ampere:
// - 128x128 output tile per block
// - 256 threads (16x16)
// - 8x8 elements per thread
// - BK=16 for better memory bandwidth utilization
// - Vectorized memory access
//
__global__ void matmul_f32_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    // Tile configuration
    constexpr int BM = 128;  // Tile rows
    constexpr int BN = 128;  // Tile cols
    constexpr int BK = 16;   // Tile depth
    constexpr int TM = 8;    // Thread rows
    constexpr int TN = 8;    // Thread cols
    // Block: (BN/TN, BM/TM) = (16, 16) = 256 threads

    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = ty * 16 + tx;  // 0-255

    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[BK][BM + 1];  // 16 x 129 = 2064 floats
    __shared__ float Bs[BK][BN + 1];  // 16 x 129 = 2064 floats
    // Total: ~16KB (fits in 48KB)

    // Accumulators (8x8 = 64 registers per thread)
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Global base positions
    const size_t row_base = by * BM;
    const size_t col_base = bx * BN;

    // Number of K tiles
    const int num_k_tiles = (K + BK - 1) / BK;

    // Loading strategy:
    // A tile: BM x BK = 128 x 16 = 2048 elements, 256 threads = 8 each
    // B tile: BK x BN = 16 x 128 = 2048 elements, 256 threads = 8 each

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const size_t k_offset = kt * BK;

        // Load A tile with coalesced access
        // Each thread loads 8 elements in a strided pattern
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = tid + i * 256;  // 0-2047
            const int a_m = idx % BM;       // row (0-127)
            const int a_k = idx / BM;       // col (0-15)
            const size_t g_row = row_base + a_m;
            const size_t g_col = k_offset + a_k;
            float val = 0.0f;
            if (g_row < M && g_col < K) {
                val = A[g_row * K + g_col];
            }
            As[a_k][a_m] = val;
        }

        // Load B tile with coalesced access
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = tid + i * 256;  // 0-2047
            const int b_n = idx % BN;       // col (0-127) - coalesced!
            const int b_k = idx / BN;       // row (0-15)
            const size_t g_row = k_offset + b_k;
            const size_t g_col = col_base + b_n;
            float val = 0.0f;
            if (g_row < K && g_col < N) {
                val = B[g_row * N + g_col];
            }
            Bs[b_k][b_n] = val;
        }

        __syncthreads();

        // Compute 8x8 outer products
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load A fragment (8 elements)
            float a_frag[TM];
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                a_frag[m] = As[k][ty * TM + m];
            }

            // Load B fragment and compute outer product
            float b_frag[TN];
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                b_frag[n] = Bs[k][tx * TN + n];
            }

            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] = fmaf(a_frag[m], b_frag[n], acc[m][n]);
                }
            }
        }

        __syncthreads();
    }

    // Write output (8x8 per thread)
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const size_t out_row = row_base + ty * TM + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                const size_t out_col = col_base + tx * TN + n;
                if (out_col < N) {
                    C[out_row * N + out_col] = acc[m][n];
                }
            }
        }
    }
}

void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    if (c.shape()[0] != M || c.shape()[1] != N) {
        throw std::runtime_error("matmul output shape mismatch");
    }

    // Check for TF32 TensorCore mode (requires SM >= 80)
    // Note: Check on every call since env var might change
    bool tf32_enabled = false;
    int sm_version = 0;

    // Check environment variable
    const char* tf32_env = std::getenv("PYGPUKIT_ALLOW_TF32");

    // Debug output (remove in production)
    static bool debug_printed = false;
    if (!debug_printed) {
        debug_printed = true;
        printf("[PyGPUkit] PYGPUKIT_ALLOW_TF32 = %s\n", tf32_env ? tf32_env : "(null)");
        fflush(stdout);
    }

    if (tf32_env && (tf32_env[0] == '1' || tf32_env[0] == 'y' || tf32_env[0] == 'Y')) {
        // Check GPU compute capability
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        sm_version = prop.major * 10 + prop.minor;
        tf32_enabled = (sm_version >= 80);  // Ampere or newer
        if (!debug_printed) {
            fprintf(stderr, "[PyGPUkit] SM version = %d, TF32 enabled = %d\n", sm_version, tf32_enabled);
        }
    }

    // Select kernel based on matrix size and dtype
    // DEBUG: Allow small sizes for TF32 testing (M=16,N=8 or M=16,N=16)
    bool use_tf32 = tf32_enabled &&
                    (a.dtype() == DataType::Float32) &&
                    ((M >= OPTIMIZED_MATMUL_THRESHOLD &&
                      N >= OPTIMIZED_MATMUL_THRESHOLD &&
                      K >= OPTIMIZED_MATMUL_THRESHOLD) ||
                     (M == 16 && (N == 8 || N == 16)));

    bool use_optimized = !use_tf32 &&
                         (a.dtype() == DataType::Float32) &&
                         (M >= OPTIMIZED_MATMUL_THRESHOLD ||
                          N >= OPTIMIZED_MATMUL_THRESHOLD ||
                          K >= OPTIMIZED_MATMUL_THRESHOLD);

    bool use_tiled = !use_optimized && !use_tf32 &&
                     (M >= TILED_MATMUL_THRESHOLD ||
                      N >= TILED_MATMUL_THRESHOLD ||
                      K >= TILED_MATMUL_THRESHOLD);

    if (use_tf32) {
        // TF32 TensorCore kernels
        if (M == 16 && (N == 8 || N == 16)) {
            // Debug: single tile kernel for small test sizes
            tf32::launch_single_tile_verified(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        } else {
            // Full kernel for large sizes
            tf32::launch_sgemm_tf32(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        }
    } else if (use_optimized) {
        // Ampere-optimized FP32 FMA kernel with cp.async and 4-stage pipeline
        ampere::launch_sgemm_ampere(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(c.data()),
            M, N, K);
    } else if (use_tiled) {
        // Tiled kernel with shared memory and double buffering
        // Block size: (TILE_N / THREAD_N, TILE_M / THREAD_M) = (16, 16)
        dim3 block_size(TILE_N / THREAD_N, TILE_M / THREAD_M);
        dim3 grid_size(
            (N + TILE_N - 1) / TILE_N,
            (M + TILE_M - 1) / TILE_M
        );

        switch (a.dtype()) {
            case DataType::Float32:
                matmul_f32_tiled_kernel<<<grid_size, block_size>>>(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float64:
                matmul_f64_tiled_kernel<<<grid_size, block_size>>>(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float32 and float64");
        }
    } else {
        // L2-optimized kernel for small matrices (Ampere+)
        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_size(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

        switch (a.dtype()) {
            case DataType::Float32:
                matmul_f32_l2opt_kernel<<<grid_size, block_size>>>(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float64:
                matmul_f64_l2opt_kernel<<<grid_size, block_size>>>(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float32 and float64");
        }
    }

    sync_and_check("matmul kernel failed");
}

GPUArray matmul(const GPUArray& a, const GPUArray& b) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t N = b.shape()[1];

    GPUArray c({M, N}, a.dtype());
    matmul(a, b, c);
    return c;
}

// Internal helper: matmul with explicit TF32 control
static void matmul_impl(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32_explicit) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    if (c.shape()[0] != M || c.shape()[1] != N) {
        throw std::runtime_error("matmul output shape mismatch");
    }

    // Check GPU compute capability for TF32 support
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int sm_version = prop.major * 10 + prop.minor;

    // TF32 only works with float32 and SM >= 80
    bool tf32_enabled = use_tf32_explicit &&
                        (a.dtype() == DataType::Float32) &&
                        (sm_version >= 80);

    if (use_tf32_explicit && !tf32_enabled) {
        if (a.dtype() != DataType::Float32) {
            throw std::runtime_error("TF32 matmul requires float32 dtype");
        }
        if (sm_version < 80) {
            throw std::runtime_error("TF32 matmul requires SM >= 80 (Ampere or newer)");
        }
    }

    // Use TF32 kernel for explicit request and large matrices
    bool use_tf32 = tf32_enabled &&
                    ((M >= OPTIMIZED_MATMUL_THRESHOLD &&
                      N >= OPTIMIZED_MATMUL_THRESHOLD &&
                      K >= OPTIMIZED_MATMUL_THRESHOLD) ||
                     (M == 16 && (N == 8 || N == 16)));

    bool use_optimized = !use_tf32 &&
                         (a.dtype() == DataType::Float32) &&
                         (M >= OPTIMIZED_MATMUL_THRESHOLD ||
                          N >= OPTIMIZED_MATMUL_THRESHOLD ||
                          K >= OPTIMIZED_MATMUL_THRESHOLD);

    bool use_tiled = !use_optimized && !use_tf32 &&
                     (M >= TILED_MATMUL_THRESHOLD ||
                      N >= TILED_MATMUL_THRESHOLD ||
                      K >= TILED_MATMUL_THRESHOLD);

    if (use_tf32) {
        // TF32 TensorCore kernels
        if (M == 16 && (N == 8 || N == 16)) {
            tf32::launch_single_tile_verified(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        } else {
            tf32::launch_sgemm_tf32(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        }
    } else if (use_optimized) {
        ampere::launch_sgemm_ampere(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(c.data()),
            M, N, K);
    } else if (use_tiled) {
        dim3 block_size(TILE_N / THREAD_N, TILE_M / THREAD_M);
        dim3 grid_size(
            (N + TILE_N - 1) / TILE_N,
            (M + TILE_M - 1) / TILE_M
        );

        switch (a.dtype()) {
            case DataType::Float32:
                matmul_f32_tiled_kernel<<<grid_size, block_size>>>(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float64:
                matmul_f64_tiled_kernel<<<grid_size, block_size>>>(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float32 and float64");
        }
    } else {
        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_size(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

        switch (a.dtype()) {
            case DataType::Float32:
                matmul_f32_l2opt_kernel<<<grid_size, block_size>>>(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float64:
                matmul_f64_l2opt_kernel<<<grid_size, block_size>>>(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float32 and float64");
        }
    }

    sync_and_check("matmul kernel failed");
}

void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32) {
    matmul_impl(a, b, c, use_tf32);
}

GPUArray matmul(const GPUArray& a, const GPUArray& b, bool use_tf32) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t N = b.shape()[1];

    GPUArray c({M, N}, a.dtype());
    matmul_impl(a, b, c, use_tf32);
    return c;
}

} // namespace ops
} // namespace pygpukit
