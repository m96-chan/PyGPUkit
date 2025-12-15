/**
 * Unary operations dispatch (exp, log, relu)
 */
#include "unary_kernels.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"

namespace pygpukit {
namespace ops {

using namespace unary;

// ============================================================================
// Exp
// ============================================================================

void exp(const GPUArray& a, GPUArray& c) {
    validate_same_shape(a, c, "exp");
    validate_same_dtype(a, c, "exp");

    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float64 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("exp only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            exp_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float64:
            exp_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<double*>(c.data()), n);
            break;
        case DataType::Float16:
            exp_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            exp_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("exp kernel failed");
}

GPUArray exp(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float64 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("exp only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    exp(a, c);
    return c;
}

// ============================================================================
// Log
// ============================================================================

void log(const GPUArray& a, GPUArray& c) {
    validate_same_shape(a, c, "log");
    validate_same_dtype(a, c, "log");

    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float64 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("log only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            log_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float64:
            log_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<double*>(c.data()), n);
            break;
        case DataType::Float16:
            log_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            log_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("log kernel failed");
}

GPUArray log(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float64 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("log only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    log(a, c);
    return c;
}

// ============================================================================
// ReLU
// ============================================================================

void relu(const GPUArray& a, GPUArray& c) {
    validate_same_shape(a, c, "relu");
    validate_same_dtype(a, c, "relu");

    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float64 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("relu only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            relu_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float64:
            relu_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<double*>(c.data()), n);
            break;
        case DataType::Float16:
            relu_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            relu_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("relu kernel failed");
}

GPUArray relu(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float64 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("relu only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    relu(a, c);
    return c;
}

} // namespace ops
} // namespace pygpukit
