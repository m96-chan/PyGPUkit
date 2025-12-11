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
// Matmul kernels (naive implementation, can be optimized with tiling)
// ============================================================================

__global__ void matmul_f32_kernel(
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

__global__ void matmul_f64_kernel(
    const double* A, const double* B, double* C,
    size_t M, size_t N, size_t K
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
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

    dim3 block_size(16, 16);
    dim3 grid_size(
        (N + block_size.x - 1) / block_size.x,
        (M + block_size.y - 1) / block_size.y
    );

    switch (a.dtype()) {
        case DataType::Float32:
            matmul_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
            break;
        case DataType::Float64:
            matmul_f64_kernel<<<grid_size, block_size>>>(
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
