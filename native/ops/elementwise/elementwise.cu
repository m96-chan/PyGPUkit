/**
 * Elementwise binary operations dispatch
 */
#include "elementwise_kernels.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"

namespace pygpukit {
namespace ops {

using namespace elementwise;

// ============================================================================
// Add
// ============================================================================

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
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float64:
            add_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<const double*>(b.data()),
                static_cast<double*>(c.data()), n);
            break;
        case DataType::Int32:
            add_i32_kernel<<<grid_size, block_size>>>(
                static_cast<const int32_t*>(a.data()),
                static_cast<const int32_t*>(b.data()),
                static_cast<int32_t*>(c.data()), n);
            break;
        case DataType::Int64:
            add_i64_kernel<<<grid_size, block_size>>>(
                static_cast<const int64_t*>(a.data()),
                static_cast<const int64_t*>(b.data()),
                static_cast<int64_t*>(c.data()), n);
            break;
        case DataType::Float16:
            add_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            add_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
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
// Mul
// ============================================================================

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
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float64:
            mul_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<const double*>(b.data()),
                static_cast<double*>(c.data()), n);
            break;
        case DataType::Int32:
            mul_i32_kernel<<<grid_size, block_size>>>(
                static_cast<const int32_t*>(a.data()),
                static_cast<const int32_t*>(b.data()),
                static_cast<int32_t*>(c.data()), n);
            break;
        case DataType::Int64:
            mul_i64_kernel<<<grid_size, block_size>>>(
                static_cast<const int64_t*>(a.data()),
                static_cast<const int64_t*>(b.data()),
                static_cast<int64_t*>(c.data()), n);
            break;
        case DataType::Float16:
            mul_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            mul_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
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
// Sub
// ============================================================================

void sub(const GPUArray& a, const GPUArray& b, GPUArray& c) {
    validate_same_shape(a, b, "sub");
    validate_same_dtype(a, b, "sub");
    validate_same_shape(a, c, "sub");
    validate_same_dtype(a, c, "sub");

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            sub_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float64:
            sub_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<const double*>(b.data()),
                static_cast<double*>(c.data()), n);
            break;
        case DataType::Int32:
            sub_i32_kernel<<<grid_size, block_size>>>(
                static_cast<const int32_t*>(a.data()),
                static_cast<const int32_t*>(b.data()),
                static_cast<int32_t*>(c.data()), n);
            break;
        case DataType::Int64:
            sub_i64_kernel<<<grid_size, block_size>>>(
                static_cast<const int64_t*>(a.data()),
                static_cast<const int64_t*>(b.data()),
                static_cast<int64_t*>(c.data()), n);
            break;
        case DataType::Float16:
            sub_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            sub_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
    }
    sync_and_check("sub kernel failed");
}

GPUArray sub(const GPUArray& a, const GPUArray& b) {
    validate_same_shape(a, b, "sub");
    validate_same_dtype(a, b, "sub");
    GPUArray c(a.shape(), a.dtype());
    sub(a, b, c);
    return c;
}

// ============================================================================
// Div
// ============================================================================

void div(const GPUArray& a, const GPUArray& b, GPUArray& c) {
    validate_same_shape(a, b, "div");
    validate_same_dtype(a, b, "div");
    validate_same_shape(a, c, "div");
    validate_same_dtype(a, c, "div");

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            div_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float64:
            div_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<const double*>(b.data()),
                static_cast<double*>(c.data()), n);
            break;
        case DataType::Int32:
            div_i32_kernel<<<grid_size, block_size>>>(
                static_cast<const int32_t*>(a.data()),
                static_cast<const int32_t*>(b.data()),
                static_cast<int32_t*>(c.data()), n);
            break;
        case DataType::Int64:
            div_i64_kernel<<<grid_size, block_size>>>(
                static_cast<const int64_t*>(a.data()),
                static_cast<const int64_t*>(b.data()),
                static_cast<int64_t*>(c.data()), n);
            break;
        case DataType::Float16:
            div_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            div_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
    }
    sync_and_check("div kernel failed");
}

GPUArray div(const GPUArray& a, const GPUArray& b) {
    validate_same_shape(a, b, "div");
    validate_same_dtype(a, b, "div");
    GPUArray c(a.shape(), a.dtype());
    div(a, b, c);
    return c;
}

// ============================================================================
// Clamp
// ============================================================================

void clamp(const GPUArray& a, GPUArray& c, float min_val, float max_val) {
    validate_same_shape(a, c, "clamp");
    validate_same_dtype(a, c, "clamp");

    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("clamp only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            clamp_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(c.data()),
                min_val, max_val, n);
            break;
        case DataType::Float16:
            clamp_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()),
                min_val, max_val, n);
            break;
        case DataType::BFloat16:
            clamp_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                min_val, max_val, n);
            break;
        default:
            break;
    }
    sync_and_check("clamp kernel failed");
}

GPUArray clamp(const GPUArray& a, float min_val, float max_val) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("clamp only supports float types");
    }
    GPUArray c(a.shape(), a.dtype());
    clamp(a, c, min_val, max_val);
    return c;
}

// ============================================================================
// Where (conditional select)
// ============================================================================

void where(const GPUArray& cond, const GPUArray& a, const GPUArray& b, GPUArray& c) {
    validate_same_shape(a, b, "where");
    validate_same_shape(a, c, "where");
    validate_same_dtype(a, b, "where");
    validate_same_dtype(a, c, "where");

    if (cond.size() != a.size()) {
        throw std::runtime_error("where: condition shape must match input shape");
    }
    if (cond.dtype() != DataType::UInt8 && cond.dtype() != DataType::Int8) {
        throw std::runtime_error("where: condition must be uint8 or int8 type (boolean)");
    }

    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("where only supports float types");
    }

    size_t n = a.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            where_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const uint8_t*>(cond.data()),
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()), n);
            break;
        case DataType::Float16:
            where_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const uint8_t*>(cond.data()),
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()), n);
            break;
        case DataType::BFloat16:
            where_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const uint8_t*>(cond.data()),
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()), n);
            break;
        default:
            break;
    }
    sync_and_check("where kernel failed");
}

GPUArray where(const GPUArray& cond, const GPUArray& a, const GPUArray& b) {
    GPUArray c(a.shape(), a.dtype());
    where(cond, a, b, c);
    return c;
}

} // namespace ops
} // namespace pygpukit
