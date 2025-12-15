// Basic GPU operations using CUDA Driver API
// PyGPUkit v0.2.4+: Single-binary distribution (driver-only mode)

#include "basic.cuh"
#include "matmul_f32_ampere.cuh"
#include "matmul_f32_tf32.cuh"
#include "matmul_f32_tf32_v2.cuh"
#include "matmul_f16_bf16.cuh"
#include "matmul_f16_bf16_tc.cuh"
#include "matmul_f16_bf16_tc_generic.cuh"
#include "../core/driver_context.hpp"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <cstdlib>

namespace pygpukit {
namespace ops {

namespace {

// Helper functions for BF16 to avoid constexpr __host__ issues
// Use raw union type for conversion
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 val) {
    // BF16 is stored in upper 16 bits of FP32
    unsigned short raw;
    memcpy(&raw, &val, sizeof(raw));
    unsigned int bits = ((unsigned int)raw) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float val) {
    // BF16 truncates lower 16 bits of FP32 mantissa
    unsigned int bits;
    memcpy(&bits, &val, sizeof(bits));
    // Round to nearest even
    bits += 0x7FFF + ((bits >> 16) & 1);
    unsigned short raw = (unsigned short)(bits >> 16);
    __nv_bfloat16 result;
    memcpy(&result, &raw, sizeof(result));
    return result;
}

void check_driver_error(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* error_str = nullptr;
        cuGetErrorString(result, &error_str);
        throw CudaError(std::string(msg) + ": " + (error_str ? error_str : "unknown error"));
    }
}

void sync_and_check(const char* msg) {
    check_driver_error(cuCtxSynchronize(), msg);
}

// Get SM version using Driver API
int get_sm_version_internal() {
    auto& ctx = driver::DriverContext::instance();
    CUdevice device = ctx.get_device(ctx.current_device());
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    return major * 10 + minor;
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

__global__ void add_f16_kernel(const __half* a, const __half* b, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void add_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // BF16: convert to float, add, convert back using helper functions
        c[idx] = float_to_bf16(bf16_to_float(a[idx]) + bf16_to_float(b[idx]));
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
        case DataType::Float16:
            add_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()),
                n);
            break;
        case DataType::BFloat16:
            add_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                n);
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

__global__ void mul_f16_kernel(const __half* a, const __half* b, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void mul_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // BF16: convert to float, multiply, convert back
        c[idx] = float_to_bf16(bf16_to_float(a[idx]) * bf16_to_float(b[idx]));
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
        case DataType::Float16:
            mul_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()),
                n);
            break;
        case DataType::BFloat16:
            mul_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                n);
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
// Sub kernels
// ============================================================================

__global__ void sub_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void sub_f64_kernel(const double* a, const double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void sub_i32_kernel(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void sub_i64_kernel(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void sub_f16_kernel(const __half* a, const __half* b, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hsub(a[idx], b[idx]);
    }
}

__global__ void sub_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // BF16: convert to float, subtract, convert back
        c[idx] = float_to_bf16(bf16_to_float(a[idx]) - bf16_to_float(b[idx]));
    }
}

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
                static_cast<float*>(c.data()),
                n);
            break;
        case DataType::Float64:
            sub_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<const double*>(b.data()),
                static_cast<double*>(c.data()),
                n);
            break;
        case DataType::Int32:
            sub_i32_kernel<<<grid_size, block_size>>>(
                static_cast<const int32_t*>(a.data()),
                static_cast<const int32_t*>(b.data()),
                static_cast<int32_t*>(c.data()),
                n);
            break;
        case DataType::Int64:
            sub_i64_kernel<<<grid_size, block_size>>>(
                static_cast<const int64_t*>(a.data()),
                static_cast<const int64_t*>(b.data()),
                static_cast<int64_t*>(c.data()),
                n);
            break;
        case DataType::Float16:
            sub_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()),
                n);
            break;
        case DataType::BFloat16:
            sub_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                n);
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
// Div kernels
// ============================================================================

__global__ void div_f32_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void div_f64_kernel(const double* a, const double* b, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void div_i32_kernel(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void div_i64_kernel(const int64_t* a, const int64_t* b, int64_t* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void div_f16_kernel(const __half* a, const __half* b, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // FP16: convert to float for division, convert back
        c[idx] = __float2half(__half2float(a[idx]) / __half2float(b[idx]));
    }
}

__global__ void div_bf16_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // BF16: convert to float for division, convert back
        c[idx] = float_to_bf16(bf16_to_float(a[idx]) / bf16_to_float(b[idx]));
    }
}

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
                static_cast<float*>(c.data()),
                n);
            break;
        case DataType::Float64:
            div_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<const double*>(b.data()),
                static_cast<double*>(c.data()),
                n);
            break;
        case DataType::Int32:
            div_i32_kernel<<<grid_size, block_size>>>(
                static_cast<const int32_t*>(a.data()),
                static_cast<const int32_t*>(b.data()),
                static_cast<int32_t*>(c.data()),
                n);
            break;
        case DataType::Int64:
            div_i64_kernel<<<grid_size, block_size>>>(
                static_cast<const int64_t*>(a.data()),
                static_cast<const int64_t*>(b.data()),
                static_cast<int64_t*>(c.data()),
                n);
            break;
        case DataType::Float16:
            div_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()),
                n);
            break;
        case DataType::BFloat16:
            div_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                n);
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
// Exp kernels (float only)
// ============================================================================

__global__ void exp_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = expf(a[idx]);
    }
}

__global__ void exp_f64_kernel(const double* a, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = ::exp(a[idx]);
    }
}

__global__ void exp_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // FP16: convert to float, compute, convert back
        c[idx] = __float2half(expf(__half2float(a[idx])));
    }
}

__global__ void exp_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // BF16: convert to float, compute, convert back
        c[idx] = float_to_bf16(expf(bf16_to_float(a[idx])));
    }
}

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
                static_cast<float*>(c.data()),
                n);
            break;
        case DataType::Float64:
            exp_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<double*>(c.data()),
                n);
            break;
        case DataType::Float16:
            exp_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()),
                n);
            break;
        case DataType::BFloat16:
            exp_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                n);
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
// Log kernels
// ============================================================================

__global__ void log_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = logf(a[idx]);
    }
}

__global__ void log_f64_kernel(const double* a, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = ::log(a[idx]);
    }
}

__global__ void log_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // FP16: convert to float, compute, convert back
        c[idx] = __float2half(logf(__half2float(a[idx])));
    }
}

__global__ void log_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // BF16: convert to float, compute, convert back
        c[idx] = float_to_bf16(logf(bf16_to_float(a[idx])));
    }
}

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
                static_cast<float*>(c.data()),
                n);
            break;
        case DataType::Float64:
            log_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<double*>(c.data()),
                n);
            break;
        case DataType::Float16:
            log_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()),
                n);
            break;
        case DataType::BFloat16:
            log_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                n);
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
// ReLU kernels
// ============================================================================

__global__ void relu_f32_kernel(const float* a, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmaxf(0.0f, a[idx]);
    }
}

__global__ void relu_f64_kernel(const double* a, double* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fmax(0.0, a[idx]);
    }
}

__global__ void relu_f16_kernel(const __half* a, __half* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Convert to float for comparison, then convert result back
        float val = __half2float(a[idx]);
        c[idx] = __float2half(val > 0.0f ? val : 0.0f);
    }
}

__global__ void relu_bf16_kernel(const __nv_bfloat16* a, __nv_bfloat16* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Convert to float for comparison, then convert result back
        float val = bf16_to_float(a[idx]);
        c[idx] = float_to_bf16(val > 0.0f ? val : 0.0f);
    }
}

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
                static_cast<float*>(c.data()),
                n);
            break;
        case DataType::Float64:
            relu_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<double*>(c.data()),
                n);
            break;
        case DataType::Float16:
            relu_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(c.data()),
                n);
            break;
        case DataType::BFloat16:
            relu_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                n);
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
// Reduction Operations (sum, mean, max)
// ============================================================================

// Warp-level reduction using shuffle instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_max_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level sum reduction kernel (FP32)
__global__ void reduce_sum_f32_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    __shared__ float shared[32];  // One value per warp

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // Grid-stride loop to accumulate
    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += input[i];
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);

    // Write warp result to shared memory
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Block-level sum reduction kernel (FP64)
__global__ void reduce_sum_f64_kernel(const double* __restrict__ input, double* __restrict__ output, size_t n) {
    __shared__ double shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (size_t i = idx; i < n; i += stride) {
        sum += input[i];
    }

    sum = warp_reduce_sum_f64(sum);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0;
        sum = warp_reduce_sum_f64(sum);
        if (lane == 0) {
            // atomicAdd for double requires sm_60+
            atomicAdd(output, sum);
        }
    }
}

// Block-level max reduction kernel (FP32)
__global__ void reduce_max_f32_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float max_val = -INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        max_val = fmaxf(max_val, input[i]);
    }

    max_val = warp_reduce_max(max_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max(max_val);
        if (lane == 0) {
            // Atomic max for float - use atomicMax with int cast trick
            int* addr = (int*)output;
            int expected = *addr;
            while (max_val > __int_as_float(expected)) {
                int old = atomicCAS(addr, expected, __float_as_int(max_val));
                if (old == expected) break;
                expected = old;
            }
        }
    }
}

// Block-level max reduction kernel (FP64)
__global__ void reduce_max_f64_kernel(const double* __restrict__ input, double* __restrict__ output, size_t n) {
    __shared__ double shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    double max_val = -INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        max_val = fmax(max_val, input[i]);
    }

    max_val = warp_reduce_max_f64(max_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max_f64(max_val);
        if (lane == 0) {
            // Atomic max for double using CAS
            unsigned long long* addr = (unsigned long long*)output;
            unsigned long long expected = *addr;
            while (max_val > __longlong_as_double(expected)) {
                unsigned long long old = atomicCAS(addr, expected, __double_as_longlong(max_val));
                if (old == expected) break;
                expected = old;
            }
        }
    }
}

// FP16/BF16 reduction kernels - accumulate in FP32 for numerical stability
// The output is stored as the input dtype

__global__ void reduce_sum_f16_kernel(const __half* __restrict__ input, __half* __restrict__ output, size_t n) {
    __shared__ float shared[32];  // Accumulate in FP32

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += __half2float(input[i]);
    }

    sum = warp_reduce_sum(sum);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            // Atomic add in FP32, then convert back
            float old_val = __half2float(*output);
            *output = __float2half(old_val + sum);
        }
    }
}

__global__ void reduce_sum_bf16_kernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += bf16_to_float(input[i]);
    }

    sum = warp_reduce_sum(sum);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            float old_val = bf16_to_float(*output);
            *output = float_to_bf16(old_val + sum);
        }
    }
}

__global__ void reduce_max_f16_kernel(const __half* __restrict__ input, __half* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float max_val = -INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        max_val = fmaxf(max_val, __half2float(input[i]));
    }

    max_val = warp_reduce_max(max_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max(max_val);
        if (lane == 0) {
            float old_val = __half2float(*output);
            if (max_val > old_val) {
                *output = __float2half(max_val);
            }
        }
    }
}

__global__ void reduce_max_bf16_kernel(const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output, size_t n) {
    __shared__ float shared[32];

    const size_t tid = threadIdx.x;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    float max_val = -INFINITY;
    for (size_t i = idx; i < n; i += stride) {
        max_val = fmaxf(max_val, bf16_to_float(input[i]));
    }

    max_val = warp_reduce_max(max_val);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max(max_val);
        if (lane == 0) {
            float old_val = bf16_to_float(*output);
            if (max_val > old_val) {
                *output = float_to_bf16(max_val);
            }
        }
    }
}

// Initialize output for reduction
__global__ void init_sum_f32_kernel(float* output) { *output = 0.0f; }
__global__ void init_sum_f64_kernel(double* output) { *output = 0.0; }
__global__ void init_sum_f16_kernel(__half* output) { *output = __float2half(0.0f); }
__global__ void init_sum_bf16_kernel(__nv_bfloat16* output) { *output = float_to_bf16(0.0f); }
__global__ void init_max_f32_kernel(float* output) { *output = -INFINITY; }
__global__ void init_max_f64_kernel(double* output) { *output = -INFINITY; }
__global__ void init_max_f16_kernel(__half* output) { *output = __float2half(-INFINITY); }
__global__ void init_max_bf16_kernel(__nv_bfloat16* output) { *output = float_to_bf16(-INFINITY); }

GPUArray sum(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float64 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sum only supports float types");
    }

    GPUArray result({1}, a.dtype());
    size_t n = a.size();

    const int block_size = 256;
    const int max_blocks = 256;  // Limit blocks for efficient atomic reduction
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);

    switch (a.dtype()) {
        case DataType::Float32:
            init_sum_f32_kernel<<<1, 1>>>(static_cast<float*>(result.data()));
            reduce_sum_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float64:
            init_sum_f64_kernel<<<1, 1>>>(static_cast<double*>(result.data()));
            reduce_sum_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<double*>(result.data()),
                n);
            break;
        case DataType::Float16:
            init_sum_f16_kernel<<<1, 1>>>(static_cast<__half*>(result.data()));
            reduce_sum_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            init_sum_bf16_kernel<<<1, 1>>>(static_cast<__nv_bfloat16*>(result.data()));
            reduce_sum_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }

    sync_and_check("sum kernel failed");
    return result;
}

// Dedicated kernel for scaling a single value
__global__ void scale_f32_kernel(float* data, float scale) {
    *data *= scale;
}

__global__ void scale_f64_kernel(double* data, double scale) {
    *data *= scale;
}

__global__ void scale_f16_kernel(__half* data, float scale) {
    *data = __float2half(__half2float(*data) * scale);
}

__global__ void scale_bf16_kernel(__nv_bfloat16* data, float scale) {
    *data = float_to_bf16(bf16_to_float(*data) * scale);
}

GPUArray mean(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float64 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("mean only supports float types");
    }

    GPUArray result({1}, a.dtype());
    size_t n = a.size();

    const int block_size = 256;
    const int max_blocks = 256;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);

    switch (a.dtype()) {
        case DataType::Float32: {
            init_sum_f32_kernel<<<1, 1>>>(static_cast<float*>(result.data()));
            reduce_sum_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(result.data()),
                n);
            sync_and_check("mean sum kernel failed");
            scale_f32_kernel<<<1, 1>>>(
                static_cast<float*>(result.data()),
                1.0f / static_cast<float>(n));
            break;
        }
        case DataType::Float64: {
            init_sum_f64_kernel<<<1, 1>>>(static_cast<double*>(result.data()));
            reduce_sum_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<double*>(result.data()),
                n);
            sync_and_check("mean sum kernel failed");
            scale_f64_kernel<<<1, 1>>>(
                static_cast<double*>(result.data()),
                1.0 / static_cast<double>(n));
            break;
        }
        case DataType::Float16: {
            init_sum_f16_kernel<<<1, 1>>>(static_cast<__half*>(result.data()));
            reduce_sum_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(result.data()),
                n);
            sync_and_check("mean sum kernel failed");
            scale_f16_kernel<<<1, 1>>>(
                static_cast<__half*>(result.data()),
                1.0f / static_cast<float>(n));
            break;
        }
        case DataType::BFloat16: {
            init_sum_bf16_kernel<<<1, 1>>>(static_cast<__nv_bfloat16*>(result.data()));
            reduce_sum_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            sync_and_check("mean sum kernel failed");
            scale_bf16_kernel<<<1, 1>>>(
                static_cast<__nv_bfloat16*>(result.data()),
                1.0f / static_cast<float>(n));
            break;
        }
        default:
            break;
    }

    sync_and_check("mean kernel failed");
    return result;
}

GPUArray max(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float64 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("max only supports float types");
    }

    GPUArray result({1}, a.dtype());
    size_t n = a.size();

    const int block_size = 256;
    const int max_blocks = 256;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);

    switch (a.dtype()) {
        case DataType::Float32:
            init_max_f32_kernel<<<1, 1>>>(static_cast<float*>(result.data()));
            reduce_max_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float64:
            init_max_f64_kernel<<<1, 1>>>(static_cast<double*>(result.data()));
            reduce_max_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(a.data()),
                static_cast<double*>(result.data()),
                n);
            break;
        case DataType::Float16:
            init_max_f16_kernel<<<1, 1>>>(static_cast<__half*>(result.data()));
            reduce_max_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            init_max_bf16_kernel<<<1, 1>>>(static_cast<__nv_bfloat16*>(result.data()));
            reduce_max_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }

    sync_and_check("max kernel failed");
    return result;
}

// ============================================================================
// Matmul kernels - Tiled with Shared Memory and Double Buffering
// ============================================================================
//
// Two implementations:
// 1. L2-optimized kernel: For small matrices (<2048), uses __ldg() cache
// 2. Tiled kernel: For large matrices (>=2048), uses shared memory tiling
//
// Tiled kernel features:
// - Configurable TILE_M, TILE_N, TILE_K
// - Double-buffered prefetch to overlap compute with memory loads
// - Bank-conflict-free shared memory access
// - Coalesced global memory access
//
// ============================================================================

// Small matrix kernel block size
#define BLOCK_SIZE 16

// Tiled matmul configuration (legacy, for compatibility)
#define TILE_M 64      // Output tile height
#define TILE_N 64      // Output tile width
#define TILE_K 16      // Reduction tile depth
#define THREAD_M 4     // Elements per thread in M dimension
#define THREAD_N 4     // Elements per thread in N dimension

// RTX 3090 Ti optimized configuration
// Target: 35.6 TFLOPS (90% of 40 TFLOPS theoretical)
// 128x128 tiles with 8x8 elements per thread for high compute intensity
#define OPT_TILE_M 128     // Output tile height
#define OPT_TILE_N 128     // Output tile width
#define OPT_TILE_K 32      // Reduction tile depth (larger K = more compute per load)
#define OPT_THREAD_M 8     // Elements per thread in M dimension
#define OPT_THREAD_N 8     // Elements per thread in N dimension
// Block: (128/8, 128/8) = (16, 16) = 256 threads
// Each thread: 8x8 = 64 FMAs per K iteration
// Shared memory per buffer: (128*32 + 32*128) * 4 = 32KB (fits in 48KB default)
// Double buffer: 64KB (need to use extended shared memory)

// Threshold for switching to tiled kernel
#define TILED_MATMUL_THRESHOLD 128

// Threshold for switching to optimized kernel (larger matrices benefit more)
// DEBUG: Temporarily lowered from 2048 to 128 for testing TF32 kernel
#define OPTIMIZED_MATMUL_THRESHOLD 128

// L2-optimized matmul kernel for FP32 (Ampere+)
// Uses __ldg() for read-only cache and __restrict__ for aliasing hints
__global__ void matmul_f32_l2opt_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Use __ldg() for read-only loads through texture cache
        // This leverages L2 cache more efficiently on Ampere
        #pragma unroll 4
        for (size_t k = 0; k < K; ++k) {
            sum += __ldg(&A[row * K + k]) * __ldg(&B[k * N + col]);
        }

        C[row * N + col] = sum;
    }
}

// L2-optimized matmul kernel for FP64 (Ampere+)
__global__ void matmul_f64_l2opt_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;

        #pragma unroll 4
        for (size_t k = 0; k < K; ++k) {
            sum += __ldg(&A[row * K + k]) * __ldg(&B[k * N + col]);
        }

        C[row * N + col] = sum;
    }
}

// ============================================================================
// Tiled Matmul with Shared Memory and Double Buffering (FP32)
// ============================================================================
//
// Each thread block computes a TILE_M x TILE_N output tile.
// Each thread computes THREAD_M x THREAD_N elements.
// Block dimensions: (TILE_N / THREAD_N, TILE_M / THREAD_M) = (16, 16) = 256 threads
//
// Double buffering:
// - While computing with data in shared memory buffer 0
// - Prefetch next tile into shared memory buffer 1
// - Swap buffers each iteration
//
__global__ void matmul_f32_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    // Thread and block indices
    const int tx = threadIdx.x;  // 0..15 (TILE_N / THREAD_N)
    const int ty = threadIdx.y;  // 0..15 (TILE_M / THREAD_M)
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Shared memory for double buffering
    // Pad by 1 to avoid bank conflicts
    __shared__ float As[2][TILE_K][TILE_M + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];

    // Thread's output tile (THREAD_M x THREAD_N elements)
    float accum[THREAD_M][THREAD_N] = {{0.0f}};

    // Global row/col start for this thread block
    const size_t block_row_start = by * TILE_M;
    const size_t block_col_start = bx * TILE_N;

    // Linear thread ID within block
    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;  // 256

    // Number of tiles along K dimension
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Current buffer index for double buffering
    int curr_buf = 0;

    // Prefetch first tile into buffer 0
    {
        // Load A tile: TILE_M x TILE_K elements, 256 threads
        // Each thread loads multiple elements
        const int a_loads_per_thread = (TILE_M * TILE_K + num_threads - 1) / num_threads;
        for (int i = 0; i < a_loads_per_thread; ++i) {
            int load_idx = tid + i * num_threads;
            if (load_idx < TILE_M * TILE_K) {
                int a_row = load_idx / TILE_K;
                int a_col = load_idx % TILE_K;
                size_t global_row = block_row_start + a_row;
                size_t global_col = a_col;
                if (global_row < M && global_col < K) {
                    As[0][a_col][a_row] = A[global_row * K + global_col];
                } else {
                    As[0][a_col][a_row] = 0.0f;
                }
            }
        }

        // Load B tile: TILE_K x TILE_N elements
        const int b_loads_per_thread = (TILE_K * TILE_N + num_threads - 1) / num_threads;
        for (int i = 0; i < b_loads_per_thread; ++i) {
            int load_idx = tid + i * num_threads;
            if (load_idx < TILE_K * TILE_N) {
                int b_row = load_idx / TILE_N;
                int b_col = load_idx % TILE_N;
                size_t global_row = b_row;
                size_t global_col = block_col_start + b_col;
                if (global_row < K && global_col < N) {
                    Bs[0][b_row][b_col] = B[global_row * N + global_col];
                } else {
                    Bs[0][b_row][b_col] = 0.0f;
                }
            }
        }
    }
    __syncthreads();

    // Main loop over K tiles with double buffering
    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        int next_buf = 1 - curr_buf;

        // Prefetch next tile (if not the last tile)
        if (tile_k + 1 < num_k_tiles) {
            size_t k_offset = (tile_k + 1) * TILE_K;

            // Load A tile
            const int a_loads_per_thread = (TILE_M * TILE_K + num_threads - 1) / num_threads;
            for (int i = 0; i < a_loads_per_thread; ++i) {
                int load_idx = tid + i * num_threads;
                if (load_idx < TILE_M * TILE_K) {
                    int a_row = load_idx / TILE_K;
                    int a_col = load_idx % TILE_K;
                    size_t global_row = block_row_start + a_row;
                    size_t global_col = k_offset + a_col;
                    if (global_row < M && global_col < K) {
                        As[next_buf][a_col][a_row] = A[global_row * K + global_col];
                    } else {
                        As[next_buf][a_col][a_row] = 0.0f;
                    }
                }
            }

            // Load B tile
            const int b_loads_per_thread = (TILE_K * TILE_N + num_threads - 1) / num_threads;
            for (int i = 0; i < b_loads_per_thread; ++i) {
                int load_idx = tid + i * num_threads;
                if (load_idx < TILE_K * TILE_N) {
                    int b_row = load_idx / TILE_N;
                    int b_col = load_idx % TILE_N;
                    size_t global_row = k_offset + b_row;
                    size_t global_col = block_col_start + b_col;
                    if (global_row < K && global_col < N) {
                        Bs[next_buf][b_row][b_col] = B[global_row * N + global_col];
                    } else {
                        Bs[next_buf][b_row][b_col] = 0.0f;
                    }
                }
            }
        }

        // Compute using current buffer
        // Each thread computes THREAD_M x THREAD_N elements
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            // Load A fragment for this thread
            float a_frag[THREAD_M];
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                a_frag[m] = As[curr_buf][k][ty * THREAD_M + m];
            }

            // Load B fragment and compute
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                float b_val = Bs[curr_buf][k][tx * THREAD_N + n];
                #pragma unroll
                for (int m = 0; m < THREAD_M; ++m) {
                    accum[m][n] += a_frag[m] * b_val;
                }
            }
        }

        // Sync before swapping buffers
        __syncthreads();
        curr_buf = next_buf;
    }

    // Write output
    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        size_t out_row = block_row_start + ty * THREAD_M + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                size_t out_col = block_col_start + tx * THREAD_N + n;
                if (out_col < N) {
                    C[out_row * N + out_col] = accum[m][n];
                }
            }
        }
    }
}

// Tiled Matmul for FP64
__global__ void matmul_f64_tiled_kernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Shared memory (smaller tiles for FP64 due to memory constraints)
    constexpr int TILE_K_F64 = 8;
    __shared__ double As[2][TILE_K_F64][TILE_M + 1];
    __shared__ double Bs[2][TILE_K_F64][TILE_N + 1];

    double accum[THREAD_M][THREAD_N] = {{0.0}};

    const size_t block_row_start = by * TILE_M;
    const size_t block_col_start = bx * TILE_N;

    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;

    const int num_k_tiles = (K + TILE_K_F64 - 1) / TILE_K_F64;

    int curr_buf = 0;

    // Prefetch first tile
    {
        const int a_loads_per_thread = (TILE_M * TILE_K_F64 + num_threads - 1) / num_threads;
        for (int i = 0; i < a_loads_per_thread; ++i) {
            int load_idx = tid + i * num_threads;
            if (load_idx < TILE_M * TILE_K_F64) {
                int a_row = load_idx / TILE_K_F64;
                int a_col = load_idx % TILE_K_F64;
                size_t global_row = block_row_start + a_row;
                size_t global_col = a_col;
                if (global_row < M && global_col < K) {
                    As[0][a_col][a_row] = A[global_row * K + global_col];
                } else {
                    As[0][a_col][a_row] = 0.0;
                }
            }
        }

        const int b_loads_per_thread = (TILE_K_F64 * TILE_N + num_threads - 1) / num_threads;
        for (int i = 0; i < b_loads_per_thread; ++i) {
            int load_idx = tid + i * num_threads;
            if (load_idx < TILE_K_F64 * TILE_N) {
                int b_row = load_idx / TILE_N;
                int b_col = load_idx % TILE_N;
                size_t global_row = b_row;
                size_t global_col = block_col_start + b_col;
                if (global_row < K && global_col < N) {
                    Bs[0][b_row][b_col] = B[global_row * N + global_col];
                } else {
                    Bs[0][b_row][b_col] = 0.0;
                }
            }
        }
    }
    __syncthreads();

    for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
        int next_buf = 1 - curr_buf;

        if (tile_k + 1 < num_k_tiles) {
            size_t k_offset = (tile_k + 1) * TILE_K_F64;

            const int a_loads_per_thread = (TILE_M * TILE_K_F64 + num_threads - 1) / num_threads;
            for (int i = 0; i < a_loads_per_thread; ++i) {
                int load_idx = tid + i * num_threads;
                if (load_idx < TILE_M * TILE_K_F64) {
                    int a_row = load_idx / TILE_K_F64;
                    int a_col = load_idx % TILE_K_F64;
                    size_t global_row = block_row_start + a_row;
                    size_t global_col = k_offset + a_col;
                    if (global_row < M && global_col < K) {
                        As[next_buf][a_col][a_row] = A[global_row * K + global_col];
                    } else {
                        As[next_buf][a_col][a_row] = 0.0;
                    }
                }
            }

            const int b_loads_per_thread = (TILE_K_F64 * TILE_N + num_threads - 1) / num_threads;
            for (int i = 0; i < b_loads_per_thread; ++i) {
                int load_idx = tid + i * num_threads;
                if (load_idx < TILE_K_F64 * TILE_N) {
                    int b_row = load_idx / TILE_N;
                    int b_col = load_idx % TILE_N;
                    size_t global_row = k_offset + b_row;
                    size_t global_col = block_col_start + b_col;
                    if (global_row < K && global_col < N) {
                        Bs[next_buf][b_row][b_col] = B[global_row * N + global_col];
                    } else {
                        Bs[next_buf][b_row][b_col] = 0.0;
                    }
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < TILE_K_F64; ++k) {
            double a_frag[THREAD_M];
            #pragma unroll
            for (int m = 0; m < THREAD_M; ++m) {
                a_frag[m] = As[curr_buf][k][ty * THREAD_M + m];
            }

            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                double b_val = Bs[curr_buf][k][tx * THREAD_N + n];
                #pragma unroll
                for (int m = 0; m < THREAD_M; ++m) {
                    accum[m][n] += a_frag[m] * b_val;
                }
            }
        }

        __syncthreads();
        curr_buf = next_buf;
    }

    #pragma unroll
    for (int m = 0; m < THREAD_M; ++m) {
        size_t out_row = block_row_start + ty * THREAD_M + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_N; ++n) {
                size_t out_col = block_col_start + tx * THREAD_N + n;
                if (out_col < N) {
                    C[out_row * N + out_col] = accum[m][n];
                }
            }
        }
    }
}

// ============================================================================
// RTX 3090 Ti Optimized Matmul Kernel (FP32) v2
// ============================================================================
//
// High-performance SGEMM kernel optimized for Ampere:
// - 128x128 output tile per block
// - 256 threads (16x16)
// - 8x8 elements per thread
// - BK=16 for better memory bandwidth utilization
// - Vectorized memory access
//
__global__ void matmul_f32_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t M, size_t N, size_t K
) {
    // Tile configuration
    constexpr int BM = 128;  // Tile rows
    constexpr int BN = 128;  // Tile cols
    constexpr int BK = 16;   // Tile depth
    constexpr int TM = 8;    // Thread rows
    constexpr int TN = 8;    // Thread cols
    // Block: (BN/TN, BM/TM) = (16, 16) = 256 threads

    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = ty * 16 + tx;  // 0-255

    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[BK][BM + 1];  // 16 x 129 = 2064 floats
    __shared__ float Bs[BK][BN + 1];  // 16 x 129 = 2064 floats
    // Total: ~16KB (fits in 48KB)

    // Accumulators (8x8 = 64 registers per thread)
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Global base positions
    const size_t row_base = by * BM;
    const size_t col_base = bx * BN;

    // Number of K tiles
    const int num_k_tiles = (K + BK - 1) / BK;

    // Loading strategy:
    // A tile: BM x BK = 128 x 16 = 2048 elements, 256 threads = 8 each
    // B tile: BK x BN = 16 x 128 = 2048 elements, 256 threads = 8 each

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const size_t k_offset = kt * BK;

        // Load A tile with coalesced access
        // Each thread loads 8 elements in a strided pattern
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = tid + i * 256;  // 0-2047
            const int a_m = idx % BM;       // row (0-127)
            const int a_k = idx / BM;       // col (0-15)
            const size_t g_row = row_base + a_m;
            const size_t g_col = k_offset + a_k;
            float val = 0.0f;
            if (g_row < M && g_col < K) {
                val = A[g_row * K + g_col];
            }
            As[a_k][a_m] = val;
        }

        // Load B tile with coalesced access
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = tid + i * 256;  // 0-2047
            const int b_n = idx % BN;       // col (0-127) - coalesced!
            const int b_k = idx / BN;       // row (0-15)
            const size_t g_row = k_offset + b_k;
            const size_t g_col = col_base + b_n;
            float val = 0.0f;
            if (g_row < K && g_col < N) {
                val = B[g_row * N + g_col];
            }
            Bs[b_k][b_n] = val;
        }

        __syncthreads();

        // Compute 8x8 outer products
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load A fragment (8 elements)
            float a_frag[TM];
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                a_frag[m] = As[k][ty * TM + m];
            }

            // Load B fragment and compute outer product
            float b_frag[TN];
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                b_frag[n] = Bs[k][tx * TN + n];
            }

            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] = fmaf(a_frag[m], b_frag[n], acc[m][n]);
                }
            }
        }

        __syncthreads();
    }

    // Write output (8x8 per thread)
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const size_t out_row = row_base + ty * TM + m;
        if (out_row < M) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                const size_t out_col = col_base + tx * TN + n;
                if (out_col < N) {
                    C[out_row * N + out_col] = acc[m][n];
                }
            }
        }
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

    // Check for TensorCore modes (requires SM >= 80)
    // Note: Check on every call since env var might change
    bool tf32_enabled = false;
    bool fp16_tc_enabled = false;
    int sm_version = 0;

    // Check environment variables
    const char* tf32_env = std::getenv("PYGPUKIT_ALLOW_TF32");
    const char* fp16_tc_env = std::getenv("PYGPUKIT_ALLOW_FP16_TC");

    // Debug output (remove in production)
    static bool debug_printed = false;
    if (!debug_printed) {
        debug_printed = true;
        printf("[PyGPUkit] PYGPUKIT_ALLOW_TF32 = %s\n", tf32_env ? tf32_env : "(null)");
        printf("[PyGPUkit] PYGPUKIT_ALLOW_FP16_TC = %s\n", fp16_tc_env ? fp16_tc_env : "(null)");
        fflush(stdout);
    }

    // Check SM version once if any TensorCore mode is requested
    if ((tf32_env && (tf32_env[0] == '1' || tf32_env[0] == 'y' || tf32_env[0] == 'Y')) ||
        (fp16_tc_env && (fp16_tc_env[0] == '1' || fp16_tc_env[0] == 'y' || fp16_tc_env[0] == 'Y'))) {
        sm_version = get_sm_version_internal();
    }

    if (tf32_env && (tf32_env[0] == '1' || tf32_env[0] == 'y' || tf32_env[0] == 'Y')) {
        tf32_enabled = (sm_version >= 80);  // Ampere or newer
    }

    if (fp16_tc_env && (fp16_tc_env[0] == '1' || fp16_tc_env[0] == 'y' || fp16_tc_env[0] == 'Y')) {
        fp16_tc_enabled = (sm_version >= 80);  // Ampere or newer
    }

    // Select kernel based on matrix size and dtype
    // DEBUG: Allow small sizes for TF32 testing (M=16,N=8 or M=16,N=16)
    bool use_tf32 = tf32_enabled &&
                    (a.dtype() == DataType::Float32) &&
                    ((M >= OPTIMIZED_MATMUL_THRESHOLD &&
                      N >= OPTIMIZED_MATMUL_THRESHOLD &&
                      K >= OPTIMIZED_MATMUL_THRESHOLD) ||
                     (M == 16 && (N == 8 || N == 16)));

    // FP16/BF16 TensorCore FAST: requires sizes to be exact multiples of tile size
    // BM=128, BN=128, BK=32 in fp16_bf16_tc namespace
    bool use_fp16_tc_fast = fp16_tc_enabled &&
                            (a.dtype() == DataType::Float16 || a.dtype() == DataType::BFloat16) &&
                            (M >= 128 && N >= 128 && K >= 32) &&
                            (M % 128 == 0 && N % 128 == 0 && K % 32 == 0);

    // FP16/BF16 TensorCore GENERIC: supports M,N >= 16, K % 8 == 0
    // Slower than FAST but more flexible
    bool use_fp16_tc_generic = !use_fp16_tc_fast && fp16_tc_enabled &&
                               (a.dtype() == DataType::Float16 || a.dtype() == DataType::BFloat16) &&
                               (M >= 16 && N >= 16 && K >= 8) &&
                               (K % 8 == 0);

    bool use_optimized = !use_tf32 && !use_fp16_tc_fast && !use_fp16_tc_generic &&
                         (a.dtype() == DataType::Float32) &&
                         (M >= OPTIMIZED_MATMUL_THRESHOLD ||
                          N >= OPTIMIZED_MATMUL_THRESHOLD ||
                          K >= OPTIMIZED_MATMUL_THRESHOLD);

    bool use_tiled = !use_optimized && !use_tf32 && !use_fp16_tc_fast && !use_fp16_tc_generic &&
                     (M >= TILED_MATMUL_THRESHOLD ||
                      N >= TILED_MATMUL_THRESHOLD ||
                      K >= TILED_MATMUL_THRESHOLD);

    if (use_tf32) {
        // TF32 TensorCore kernels
        if (M == 16 && (N == 8 || N == 16)) {
            // Debug: single tile kernel for small test sizes
            tf32::launch_single_tile_verified(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        } else {
            // Full kernel for large sizes
            tf32::launch_sgemm_tf32(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        }
    } else if (use_fp16_tc_fast) {
        // FP16/BF16 TensorCore FAST kernels with mma.sync.m16n8k16
        if (a.dtype() == DataType::Float16) {
            fp16_bf16_tc::launch_sgemm_f16_tc(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()),
                M, N, K);
        } else {
            fp16_bf16_tc::launch_sgemm_bf16_tc(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                M, N, K);
        }
    } else if (use_fp16_tc_generic) {
        // FP16/BF16 TensorCore GENERIC kernels with mma.sync.m16n8k8 (boundary handling)
        if (a.dtype() == DataType::Float16) {
            fp16_bf16_tc_generic::launch_sgemm_f16_tc_generic(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()),
                M, N, K);
        } else {
            fp16_bf16_tc_generic::launch_sgemm_bf16_tc_generic(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                M, N, K);
        }
    } else if (use_optimized) {
        // Ampere-optimized FP32 FMA kernel with cp.async and 4-stage pipeline
        ampere::launch_sgemm_ampere(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(c.data()),
            M, N, K);
    } else if (use_tiled) {
        // Tiled kernel with shared memory and double buffering
        // Block size: (TILE_N / THREAD_N, TILE_M / THREAD_M) = (16, 16)
        dim3 block_size(TILE_N / THREAD_N, TILE_M / THREAD_M);
        dim3 grid_size(
            (N + TILE_N - 1) / TILE_N,
            (M + TILE_M - 1) / TILE_M
        );

        switch (a.dtype()) {
            case DataType::Float32:
                matmul_f32_tiled_kernel<<<grid_size, block_size>>>(
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
            case DataType::Float16:
                fp16_bf16_matmul::launch_sgemm_f16(
                    static_cast<const __half*>(a.data()),
                    static_cast<const __half*>(b.data()),
                    static_cast<__half*>(c.data()),
                    M, N, K);
                break;
            case DataType::BFloat16:
                fp16_bf16_matmul::launch_sgemm_bf16(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<const __nv_bfloat16*>(b.data()),
                    static_cast<__nv_bfloat16*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float types");
        }
    } else {
        // L2-optimized kernel for small matrices (Ampere+)
        // or FP16/BF16 kernels
        switch (a.dtype()) {
            case DataType::Float32: {
                dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
                dim3 grid_size(
                    (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (M + BLOCK_SIZE - 1) / BLOCK_SIZE
                );
                matmul_f32_l2opt_kernel<<<grid_size, block_size>>>(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            }
            case DataType::Float64: {
                dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
                dim3 grid_size(
                    (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (M + BLOCK_SIZE - 1) / BLOCK_SIZE
                );
                matmul_f64_l2opt_kernel<<<grid_size, block_size>>>(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            }
            case DataType::Float16:
                fp16_bf16_matmul::launch_sgemm_f16(
                    static_cast<const __half*>(a.data()),
                    static_cast<const __half*>(b.data()),
                    static_cast<__half*>(c.data()),
                    M, N, K);
                break;
            case DataType::BFloat16:
                fp16_bf16_matmul::launch_sgemm_bf16(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<const __nv_bfloat16*>(b.data()),
                    static_cast<__nv_bfloat16*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float types");
        }
    }

    sync_and_check("matmul kernel failed");
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

// Internal helper: matmul with explicit TF32 control
static void matmul_impl(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32_explicit) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    if (c.shape()[0] != M || c.shape()[1] != N) {
        throw std::runtime_error("matmul output shape mismatch");
    }

    // Check GPU compute capability for TF32 support
    // (using internal helper for driver-only compatibility)
    int sm_version = get_sm_version_internal();

    // TF32 only works with float32 and SM >= 80
    bool tf32_enabled = use_tf32_explicit &&
                        (a.dtype() == DataType::Float32) &&
                        (sm_version >= 80);

    if (use_tf32_explicit && !tf32_enabled) {
        if (a.dtype() != DataType::Float32) {
            throw std::runtime_error("TF32 matmul requires float32 dtype");
        }
        if (sm_version < 80) {
            throw std::runtime_error("TF32 matmul requires SM >= 80 (Ampere or newer)");
        }
    }

    // Use TF32 kernel for explicit request and large matrices
    bool use_tf32 = tf32_enabled &&
                    ((M >= OPTIMIZED_MATMUL_THRESHOLD &&
                      N >= OPTIMIZED_MATMUL_THRESHOLD &&
                      K >= OPTIMIZED_MATMUL_THRESHOLD) ||
                     (M == 16 && (N == 8 || N == 16)));

    bool use_optimized = !use_tf32 &&
                         (a.dtype() == DataType::Float32) &&
                         (M >= OPTIMIZED_MATMUL_THRESHOLD ||
                          N >= OPTIMIZED_MATMUL_THRESHOLD ||
                          K >= OPTIMIZED_MATMUL_THRESHOLD);

    bool use_tiled = !use_optimized && !use_tf32 &&
                     (M >= TILED_MATMUL_THRESHOLD ||
                      N >= TILED_MATMUL_THRESHOLD ||
                      K >= TILED_MATMUL_THRESHOLD);

    if (use_tf32) {
        // TF32 TensorCore kernels
        if (M == 16 && (N == 8 || N == 16)) {
            tf32::launch_single_tile_verified(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        } else {
            // Check for v2 kernel (optimized) via environment variable
            const char* use_v2 = std::getenv("PYGPUKIT_TF32_V2");
            if (use_v2 && std::string(use_v2) == "1") {
                tf32_v2::launch_sgemm_tf32_v2(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
            } else {
                tf32::launch_sgemm_tf32(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
            }
        }
    } else if (use_optimized) {
        ampere::launch_sgemm_ampere(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(c.data()),
            M, N, K);
    } else if (use_tiled) {
        dim3 block_size(TILE_N / THREAD_N, TILE_M / THREAD_M);
        dim3 grid_size(
            (N + TILE_N - 1) / TILE_N,
            (M + TILE_M - 1) / TILE_M
        );

        switch (a.dtype()) {
            case DataType::Float32:
                matmul_f32_tiled_kernel<<<grid_size, block_size>>>(
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
            case DataType::Float16:
                fp16_bf16_matmul::launch_sgemm_f16(
                    static_cast<const __half*>(a.data()),
                    static_cast<const __half*>(b.data()),
                    static_cast<__half*>(c.data()),
                    M, N, K);
                break;
            case DataType::BFloat16:
                fp16_bf16_matmul::launch_sgemm_bf16(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<const __nv_bfloat16*>(b.data()),
                    static_cast<__nv_bfloat16*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float32, float64, float16, and bfloat16");
        }
    } else {
        dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_size(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

        switch (a.dtype()) {
            case DataType::Float32:
                matmul_f32_l2opt_kernel<<<grid_size, block_size>>>(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float64:
                matmul_f64_l2opt_kernel<<<grid_size, block_size>>>(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float16:
                fp16_bf16_matmul::launch_sgemm_f16(
                    static_cast<const __half*>(a.data()),
                    static_cast<const __half*>(b.data()),
                    static_cast<__half*>(c.data()),
                    M, N, K);
                break;
            case DataType::BFloat16:
                fp16_bf16_matmul::launch_sgemm_bf16(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<const __nv_bfloat16*>(b.data()),
                    static_cast<__nv_bfloat16*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float32, float64, float16, and bfloat16");
        }
    }

    sync_and_check("matmul kernel failed");
}

void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32) {
    matmul_impl(a, b, c, use_tf32);
}

GPUArray matmul(const GPUArray& a, const GPUArray& b, bool use_tf32) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t N = b.shape()[1];

    GPUArray c({M, N}, a.dtype());
    matmul_impl(a, b, c, use_tf32);
    return c;
}

} // namespace ops
} // namespace pygpukit
