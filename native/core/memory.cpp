// Memory management using CUDA Driver API
// PyGPUkit v0.2.4+: Single-binary distribution (driver-only mode)

#include "memory.hpp"
#include "driver_context.hpp"
#include "driver_api.hpp"
#include <numeric>
#include <cstring>

namespace pygpukit {

namespace {

void check_driver_error(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* error_str = nullptr;
        cuGetErrorString(result, &error_str);
        throw CudaError(std::string(msg) + ": " + (error_str ? error_str : "unknown error"));
    }
}

} // anonymous namespace

DevicePtr device_malloc(size_t size_bytes) {
    // Ensure context is initialized
    driver::DriverContext::instance().set_current();

    CUdeviceptr dptr = 0;
    check_driver_error(cuMemAlloc(&dptr, size_bytes), "Failed to allocate device memory");
    return reinterpret_cast<void*>(dptr);
}

void device_free(DevicePtr ptr) {
    if (ptr != nullptr) {
        cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
    }
}

// =============================================================================
// Pinned (Page-Locked) Host Memory
// =============================================================================

void* pinned_malloc(size_t size_bytes) {
    driver::DriverContext::instance().set_current();

    void* ptr = nullptr;
    check_driver_error(
        cuMemAllocHost(&ptr, size_bytes),
        "Failed to allocate pinned host memory"
    );
    return ptr;
}

void pinned_free(void* ptr) {
    if (ptr != nullptr) {
        cuMemFreeHost(ptr);
    }
}

// =============================================================================
// Synchronous Memory Transfers
// =============================================================================

void memcpy_host_to_device(DevicePtr dst, const void* src, size_t size_bytes) {
    check_driver_error(
        cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, size_bytes),
        "Failed to copy host to device"
    );
}

void memcpy_device_to_host(void* dst, DevicePtr src, size_t size_bytes) {
    check_driver_error(
        cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), size_bytes),
        "Failed to copy device to host"
    );
}

void memcpy_device_to_device(DevicePtr dst, DevicePtr src, size_t size_bytes) {
    check_driver_error(
        cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), size_bytes),
        "Failed to copy device to device"
    );
}

// =============================================================================
// Asynchronous Memory Transfers (using CUDA Driver API)
// =============================================================================

void memcpy_host_to_device_async(DevicePtr dst, const void* src, size_t size_bytes,
                                  StreamHandle stream) {
    check_driver_error(
        cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst), src, size_bytes, stream),
        "Failed to copy host to device (async)"
    );
}

void memcpy_device_to_host_async(void* dst, DevicePtr src, size_t size_bytes,
                                  StreamHandle stream) {
    check_driver_error(
        cuMemcpyDtoHAsync(dst, reinterpret_cast<CUdeviceptr>(src), size_bytes, stream),
        "Failed to copy device to host (async)"
    );
}

void memcpy_device_to_device_async(DevicePtr dst, DevicePtr src, size_t size_bytes,
                                    StreamHandle stream) {
    check_driver_error(
        cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), size_bytes, stream),
        "Failed to copy device to device (async)"
    );
}

void device_memset(DevicePtr ptr, int value, size_t size_bytes) {
    // cuMemsetD8 sets each byte to the value
    check_driver_error(
        cuMemsetD8(reinterpret_cast<CUdeviceptr>(ptr), static_cast<unsigned char>(value), size_bytes),
        "Failed to memset device memory"
    );
}

void get_memory_info(size_t* free_bytes, size_t* total_bytes) {
    check_driver_error(cuMemGetInfo(free_bytes, total_bytes), "Failed to get memory info");
}

// GPUArray implementation

GPUArray::GPUArray(const std::vector<size_t>& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), ptr_(nullptr), owns_memory_(true) {
    size_t bytes = nbytes();
    if (bytes > 0) {
        ptr_ = device_malloc(bytes);
    }
}

// Private constructor for views (no allocation)
GPUArray::GPUArray(const std::vector<size_t>& shape, DataType dtype, DevicePtr ptr, bool owns)
    : shape_(shape), dtype_(dtype), ptr_(ptr), owns_memory_(owns) {
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

// Zero-copy view (narrow)
GPUArray GPUArray::narrow(const GPUArray& source, size_t offset_elements,
                          const std::vector<size_t>& new_shape) {
    // Calculate view size
    size_t view_size = 1;
    for (size_t dim : new_shape) {
        view_size *= dim;
    }

    // Validate bounds
    if (offset_elements + view_size > source.size()) {
        throw std::runtime_error(
            "GPUArray::narrow: view exceeds source bounds (offset=" +
            std::to_string(offset_elements) + ", view_size=" +
            std::to_string(view_size) + ", source_size=" +
            std::to_string(source.size()) + ")");
    }

    // Calculate byte offset
    size_t byte_offset = offset_elements * source.itemsize();

    // Create view with offset pointer (non-owning)
    DevicePtr view_ptr = static_cast<char*>(source.data()) + byte_offset;
    return GPUArray(new_shape, source.dtype(), view_ptr, false);
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
