// CUDA kernels for memory operations
// PyGPUkit v0.2.4+: Single-binary distribution (driver-only mode)

#include "memory.hpp"
#include "driver_context.hpp"
#include <cuda.h>

namespace pygpukit {

namespace {

void sync_device() {
    cuCtxSynchronize();
}

} // anonymous namespace

// Kernel to fill array with ones (float32)
__global__ void fill_ones_f32_kernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1.0f;
    }
}

// Kernel to fill array with ones (float64)
__global__ void fill_ones_f64_kernel(double* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1.0;
    }
}

// Kernel to fill array with ones (int32)
__global__ void fill_ones_i32_kernel(int32_t* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1;
    }
}

// Kernel to fill array with ones (int64)
__global__ void fill_ones_i64_kernel(int64_t* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1;
    }
}

// Host function to fill with ones
void fill_ones_impl(DevicePtr ptr, size_t count, DataType dtype) {
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;

    switch (dtype) {
        case DataType::Float32:
            fill_ones_f32_kernel<<<grid_size, block_size>>>(
                static_cast<float*>(ptr), count);
            break;
        case DataType::Float64:
            fill_ones_f64_kernel<<<grid_size, block_size>>>(
                static_cast<double*>(ptr), count);
            break;
        case DataType::Int32:
            fill_ones_i32_kernel<<<grid_size, block_size>>>(
                static_cast<int32_t*>(ptr), count);
            break;
        case DataType::Int64:
            fill_ones_i64_kernel<<<grid_size, block_size>>>(
                static_cast<int64_t*>(ptr), count);
            break;
    }
    sync_device();
}

GPUArray ones(const std::vector<size_t>& shape, DataType dtype) {
    GPUArray arr(shape, dtype);
    fill_ones_impl(arr.data(), arr.size(), dtype);
    return arr;
}

} // namespace pygpukit
