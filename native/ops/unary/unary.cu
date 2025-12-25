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

// ============================================================================
// Sin
// ============================================================================

void sin(const GPUArray& a, GPUArray& c) {
    validate_same_shape(a, c, "sin");
    validate_same_dtype(a, c, "sin");

    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sin only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            sin_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float16:
            sin_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            sin_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("sin kernel failed");
}

GPUArray sin(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sin only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    sin(a, c);
    return c;
}

// ============================================================================
// Cos
// ============================================================================

void cos(const GPUArray& a, GPUArray& c) {
    validate_same_shape(a, c, "cos");
    validate_same_dtype(a, c, "cos");

    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("cos only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            cos_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float16:
            cos_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            cos_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("cos kernel failed");
}

GPUArray cos(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("cos only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    cos(a, c);
    return c;
}

// ============================================================================
// Sqrt
// ============================================================================

void sqrt(const GPUArray& a, GPUArray& c) {
    validate_same_shape(a, c, "sqrt");
    validate_same_dtype(a, c, "sqrt");

    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sqrt only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            sqrt_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float16:
            sqrt_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            sqrt_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("sqrt kernel failed");
}

GPUArray sqrt(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sqrt only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    sqrt(a, c);
    return c;
}

// ============================================================================
// Rsqrt (1/sqrt(x))
// ============================================================================

void rsqrt(const GPUArray& a, GPUArray& c) {
    validate_same_shape(a, c, "rsqrt");
    validate_same_dtype(a, c, "rsqrt");

    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("rsqrt only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            rsqrt_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float16:
            rsqrt_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            rsqrt_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("rsqrt kernel failed");
}

GPUArray rsqrt(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("rsqrt only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    rsqrt(a, c);
    return c;
}

// ============================================================================
// Abs
// ============================================================================

void abs(const GPUArray& a, GPUArray& c) {
    validate_same_shape(a, c, "abs");
    validate_same_dtype(a, c, "abs");

    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("abs only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            abs_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float16:
            abs_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            abs_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("abs kernel failed");
}

GPUArray abs(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("abs only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    abs(a, c);
    return c;
}

// ============================================================================
// Neg (-x)
// ============================================================================

void neg(const GPUArray& a, GPUArray& c) {
    validate_same_shape(a, c, "neg");
    validate_same_dtype(a, c, "neg");

    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("neg only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            neg_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float16:
            neg_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            neg_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("neg kernel failed");
}

GPUArray neg(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("neg only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    neg(a, c);
    return c;
}

} // namespace ops
} // namespace pygpukit
