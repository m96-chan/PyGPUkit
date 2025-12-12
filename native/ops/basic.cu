#include "basic.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace pygpukit {
namespace ops {

namespace {

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw CudaError(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

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

    check_cuda_error(cudaGetLastError(), "add kernel launch failed");
    check_cuda_error(cudaDeviceSynchronize(), "add kernel sync failed");
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

    check_cuda_error(cudaGetLastError(), "mul kernel launch failed");
    check_cuda_error(cudaDeviceSynchronize(), "mul kernel sync failed");
}

GPUArray mul(const GPUArray& a, const GPUArray& b) {
    validate_same_shape(a, b, "mul");
    validate_same_dtype(a, b, "mul");

    GPUArray c(a.shape(), a.dtype());
    mul(a, b, c);
    return c;
}

// ============================================================================
// Matmul kernels (Issue #26)
// ============================================================================
//
// Performance Note (RTX 3090 Ti benchmarks):
// - Naive kernel: ~2091 GFLOPS at 4096x4096, ~1410 GFLOPS at 1024x1024
// - Tiled kernel: ~1471 GFLOPS at 1024x1024 (SLOWER due to __syncthreads overhead)
//
// On modern GPUs with large L2 cache (RTX 3090 Ti has 6MB), the naive kernel
// often outperforms simple tiled implementations. The tiled kernels are kept
// for educational purposes and may benefit different GPU architectures.
//
// For production use, consider cuBLAS which achieves 20+ TFLOPS.
// ============================================================================

// Block size (also used as tile size for tiled kernel)
#define TILE_SIZE 16

// Tiled matmul kernel using shared memory for better memory bandwidth utilization
// Each thread block computes a TILE_SIZE x TILE_SIZE tile of the output matrix C
__global__ void matmul_f32_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread indices within the block
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    // Global row and column indices for this thread's output element
    size_t row = blockIdx.y * TILE_SIZE + ty;
    size_t col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for the dot product
    float sum = 0.0f;

    // Number of tiles needed to cover the K dimension
    size_t numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Iterate over tiles
    for (size_t t = 0; t < numTiles; ++t) {
        // Load tile of A into shared memory (with boundary check)
        size_t a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory (with boundary check)
        size_t b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory (with boundary check)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Tiled matmul kernel for float64
__global__ void matmul_f64_tiled_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    // Shared memory for tiles of A and B
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    // Thread indices within the block
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    // Global row and column indices for this thread's output element
    size_t row = blockIdx.y * TILE_SIZE + ty;
    size_t col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for the dot product
    double sum = 0.0;

    // Number of tiles needed to cover the K dimension
    size_t numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Iterate over tiles
    for (size_t t = 0; t < numTiles; ++t) {
        // Load tile of A into shared memory (with boundary check)
        size_t a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0;
        }

        // Load tile of B into shared memory (with boundary check)
        size_t b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory (with boundary check)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Naive matmul kernel - DEFAULT (faster on RTX 30/40 series due to L2 cache)
__global__ void matmul_f32_naive_kernel(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
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

    // Use naive kernel - faster than tiled on RTX 3090 Ti due to large L2 cache
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    switch (a.dtype()) {
        case DataType::Float32:
            matmul_f32_naive_kernel<<<grid_size, block_size>>>(
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

    check_cuda_error(cudaGetLastError(), "matmul kernel launch failed");
    check_cuda_error(cudaDeviceSynchronize(), "matmul kernel sync failed");
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

} // namespace ops
} // namespace pygpukit
