/**
 * Error handling and validation helpers
 *
 * PyGPUkit v0.2.12+: Using CUDA Driver API only
 */
#pragma once

#include <cuda.h>
#include <stdexcept>
#include <string>
#include "../../core/memory.hpp"
#include "../../core/cuda_graph.hpp"

namespace pygpukit {
namespace ops {

// CUDA Driver API error check
inline void check_driver_error(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* error_str = nullptr;
        cuGetErrorString(result, &error_str);
        throw CudaError(std::string(msg) + ": " + (error_str ? error_str : "unknown error"));
    }
}

// Synchronize and check for errors
// Skip synchronization during CUDA Graph capture (not allowed)
inline void sync_and_check(const char* msg) {
    // Check if we're capturing - if so, skip sync (not allowed during capture)
    CUstream capture_stream = internal::get_capture_stream();
    if (capture_stream != nullptr) {
        // During capture, synchronization is not allowed.
        // Errors will be detected when graph capture ends.
        return;
    }
    check_driver_error(cuCtxSynchronize(), msg);
}

// Shape validation
inline void validate_same_shape(const GPUArray& a, const GPUArray& b, const char* op_name) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error(std::string(op_name) + " requires arrays of same shape");
    }
}

// Dtype validation
inline void validate_same_dtype(const GPUArray& a, const GPUArray& b, const char* op_name) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error(std::string(op_name) + " requires arrays of same dtype");
    }
}

// Matmul shape validation
inline void validate_matmul_shapes(const GPUArray& a, const GPUArray& b, const char* op_name) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error(std::string(op_name) + " requires 2D arrays");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error(std::string(op_name) + " dimension mismatch");
    }
}

} // namespace ops
} // namespace pygpukit
