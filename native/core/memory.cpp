#include "memory.hpp"
#include <cuda_runtime.h>
#include <numeric>
#include <cstring>

namespace pygpukit {

namespace {

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw CudaError(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

} // anonymous namespace

DevicePtr device_malloc(size_t size_bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size_bytes);
    check_cuda_error(err, "Failed to allocate device memory");
    return ptr;
}

void device_free(DevicePtr ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void memcpy_host_to_device(DevicePtr dst, const void* src, size_t size_bytes) {
    cudaError_t err = cudaMemcpy(dst, src, size_bytes, cudaMemcpyHostToDevice);
    check_cuda_error(err, "Failed to copy host to device");
}

void memcpy_device_to_host(void* dst, DevicePtr src, size_t size_bytes) {
    cudaError_t err = cudaMemcpy(dst, src, size_bytes, cudaMemcpyDeviceToHost);
    check_cuda_error(err, "Failed to copy device to host");
}

void memcpy_device_to_device(DevicePtr dst, DevicePtr src, size_t size_bytes) {
    cudaError_t err = cudaMemcpy(dst, src, size_bytes, cudaMemcpyDeviceToDevice);
    check_cuda_error(err, "Failed to copy device to device");
}

void device_memset(DevicePtr ptr, int value, size_t size_bytes) {
    cudaError_t err = cudaMemset(ptr, value, size_bytes);
    check_cuda_error(err, "Failed to memset device memory");
}

void get_memory_info(size_t* free_bytes, size_t* total_bytes) {
    cudaError_t err = cudaMemGetInfo(free_bytes, total_bytes);
    check_cuda_error(err, "Failed to get memory info");
}

// GPUArray implementation

GPUArray::GPUArray(const std::vector<size_t>& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), ptr_(nullptr), owns_memory_(true) {
    size_t bytes = nbytes();
    if (bytes > 0) {
        ptr_ = device_malloc(bytes);
    }
}

GPUArray::~GPUArray() {
    if (owns_memory_ && ptr_ != nullptr) {
        device_free(ptr_);
    }
}

GPUArray::GPUArray(GPUArray&& other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      ptr_(other.ptr_),
      owns_memory_(other.owns_memory_) {
    other.ptr_ = nullptr;
    other.owns_memory_ = false;
}

GPUArray& GPUArray::operator=(GPUArray&& other) noexcept {
    if (this != &other) {
        if (owns_memory_ && ptr_ != nullptr) {
            device_free(ptr_);
        }
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        ptr_ = other.ptr_;
        owns_memory_ = other.owns_memory_;
        other.ptr_ = nullptr;
        other.owns_memory_ = false;
    }
    return *this;
}

size_t GPUArray::size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(),
                          size_t(1), std::multiplies<size_t>());
}

void GPUArray::copy_from_host(const void* src) {
    memcpy_host_to_device(ptr_, src, nbytes());
}

void GPUArray::copy_to_host(void* dst) const {
    memcpy_device_to_host(dst, ptr_, nbytes());
}

void GPUArray::fill_zeros() {
    device_memset(ptr_, 0, nbytes());
}

// Factory functions

GPUArray zeros(const std::vector<size_t>& shape, DataType dtype) {
    GPUArray arr(shape, dtype);
    arr.fill_zeros();
    return arr;
}

GPUArray empty(const std::vector<size_t>& shape, DataType dtype) {
    return GPUArray(shape, dtype);
}

GPUArray from_host(const void* data, const std::vector<size_t>& shape, DataType dtype) {
    GPUArray arr(shape, dtype);
    arr.copy_from_host(data);
    return arr;
}

} // namespace pygpukit
