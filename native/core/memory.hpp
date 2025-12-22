#pragma once

#include "types.hpp"
#include "stream.hpp"
#include <vector>

namespace pygpukit {

// =============================================================================
// Device Memory Allocation
// =============================================================================

// Allocate device memory
DevicePtr device_malloc(size_t size_bytes);

// Free device memory
void device_free(DevicePtr ptr);

// =============================================================================
// Pinned (Page-Locked) Host Memory - for faster H2D transfers
// =============================================================================

// Allocate pinned host memory
void* pinned_malloc(size_t size_bytes);

// Free pinned host memory
void pinned_free(void* ptr);

// =============================================================================
// Synchronous Memory Transfers
// =============================================================================

// Copy host to device (synchronous)
void memcpy_host_to_device(DevicePtr dst, const void* src, size_t size_bytes);

// Copy device to host (synchronous)
void memcpy_device_to_host(void* dst, DevicePtr src, size_t size_bytes);

// Copy device to device (synchronous)
void memcpy_device_to_device(DevicePtr dst, DevicePtr src, size_t size_bytes);

// =============================================================================
// Asynchronous Memory Transfers (for pipelined loading)
// =============================================================================

// Copy host to device (asynchronous on stream)
void memcpy_host_to_device_async(DevicePtr dst, const void* src, size_t size_bytes,
                                  StreamHandle stream);

// Copy device to host (asynchronous on stream)
void memcpy_device_to_host_async(void* dst, DevicePtr src, size_t size_bytes,
                                  StreamHandle stream);

// Copy device to device (asynchronous on stream)
void memcpy_device_to_device_async(DevicePtr dst, DevicePtr src, size_t size_bytes,
                                    StreamHandle stream);

// Set device memory
void device_memset(DevicePtr ptr, int value, size_t size_bytes);

// Get free and total memory
void get_memory_info(size_t* free_bytes, size_t* total_bytes);

// GPUArray class - manages a contiguous block of GPU memory
class GPUArray {
public:
    GPUArray(const std::vector<size_t>& shape, DataType dtype);
    ~GPUArray();

    // Disable copy (move only)
    GPUArray(const GPUArray&) = delete;
    GPUArray& operator=(const GPUArray&) = delete;
    GPUArray(GPUArray&& other) noexcept;
    GPUArray& operator=(GPUArray&& other) noexcept;

    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const;
    size_t nbytes() const { return size() * dtype_size(dtype_); }
    size_t itemsize() const { return dtype_size(dtype_); }
    DevicePtr data() const { return ptr_; }
    bool owns_memory() const { return owns_memory_; }

    // Data transfer
    void copy_from_host(const void* src);
    void copy_to_host(void* dst) const;

    // Fill operations
    void fill_zeros();

    // Zero-copy view (narrow) - creates a view into existing memory
    // offset_elements: offset from start in number of elements
    // new_shape: shape of the view (total elements must fit within source)
    // Returns a non-owning GPUArray pointing to source memory + offset
    static GPUArray narrow(const GPUArray& source, size_t offset_elements,
                           const std::vector<size_t>& new_shape);

private:
    // Private constructor for creating views (no allocation)
    GPUArray(const std::vector<size_t>& shape, DataType dtype, DevicePtr ptr, bool owns);

    std::vector<size_t> shape_;
    DataType dtype_;
    DevicePtr ptr_;
    bool owns_memory_;
};

// Factory functions
GPUArray zeros(const std::vector<size_t>& shape, DataType dtype);
GPUArray ones(const std::vector<size_t>& shape, DataType dtype);
GPUArray empty(const std::vector<size_t>& shape, DataType dtype);
GPUArray from_host(const void* data, const std::vector<size_t>& shape, DataType dtype);

} // namespace pygpukit
