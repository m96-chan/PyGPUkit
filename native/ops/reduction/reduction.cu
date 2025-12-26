/**
 * Reduction operations dispatch (sum, mean, max)
 */
#include "reduction_kernels.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"
#include <algorithm>

namespace pygpukit {
namespace ops {

using namespace reduction;

// ============================================================================
// Sum
// ============================================================================

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

// ============================================================================
// Mean
// ============================================================================

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

// ============================================================================
// Max
// ============================================================================

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
// Min
// ============================================================================

GPUArray min(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("min only supports float types (f32, f16, bf16)");
    }

    GPUArray result({1}, a.dtype());
    size_t n = a.size();

    const int block_size = 256;
    const int max_blocks = 256;
    const int grid_size = std::min((int)((n + block_size - 1) / block_size), max_blocks);

    switch (a.dtype()) {
        case DataType::Float32:
            init_min_f32_kernel<<<1, 1>>>(static_cast<float*>(result.data()));
            reduce_min_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float16:
            init_min_f16_kernel<<<1, 1>>>(static_cast<__half*>(result.data()));
            reduce_min_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            init_min_bf16_kernel<<<1, 1>>>(static_cast<__nv_bfloat16*>(result.data()));
            reduce_min_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }

    sync_and_check("min kernel failed");
    return result;
}

// ============================================================================
// Argmax
// ============================================================================

GPUArray argmax(const GPUArray& a) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("argmax only supports float types (f32, f16, bf16)");
    }

    GPUArray result({1}, DataType::Int64);
    size_t n = a.size();

    // Single block reduction for simplicity - argmax needs coordination
    const int block_size = 256;
    const int grid_size = 1;  // Single block for global argmax

    switch (a.dtype()) {
        case DataType::Float32:
            argmax_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<int64_t*>(result.data()),
                n);
            break;
        case DataType::Float16:
            argmax_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<int64_t*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            argmax_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<int64_t*>(result.data()),
                n);
            break;
        default:
            break;
    }

    sync_and_check("argmax kernel failed");
    return result;
}

// ============================================================================
// Sum with axis
// ============================================================================

GPUArray sum_axis(const GPUArray& a, int axis) {
    if (a.dtype() != DataType::Float32 &&
        a.dtype() != DataType::Float16 && a.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sum_axis only supports float types (f32, f16, bf16)");
    }
    if (a.ndim() != 2) {
        throw std::runtime_error("sum_axis only supports 2D tensors");
    }
    if (axis != 0 && axis != 1) {
        throw std::runtime_error("sum_axis: axis must be 0 or 1");
    }

    int M = a.shape()[0];
    int N = a.shape()[1];

    std::vector<size_t> out_shape;
    if (axis == 0) {
        out_shape = {static_cast<size_t>(N)};
    } else {
        out_shape = {static_cast<size_t>(M)};
    }

    GPUArray result(out_shape, a.dtype());

    const int block_size = 256;

    if (axis == 0) {
        // Sum along rows -> output [N]
        const int grid_size = (N + block_size - 1) / block_size;
        switch (a.dtype()) {
            case DataType::Float32:
                sum_axis0_f32_kernel<<<grid_size, block_size>>>(
                    static_cast<const float*>(a.data()),
                    static_cast<float*>(result.data()),
                    M, N);
                break;
            case DataType::Float16:
                sum_axis0_f16_kernel<<<grid_size, block_size>>>(
                    static_cast<const __half*>(a.data()),
                    static_cast<__half*>(result.data()),
                    M, N);
                break;
            case DataType::BFloat16:
                sum_axis0_bf16_kernel<<<grid_size, block_size>>>(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<__nv_bfloat16*>(result.data()),
                    M, N);
                break;
            default:
                break;
        }
    } else {
        // Sum along columns -> output [M]
        const int grid_size = (M + block_size - 1) / block_size;
        switch (a.dtype()) {
            case DataType::Float32:
                sum_axis1_f32_kernel<<<grid_size, block_size>>>(
                    static_cast<const float*>(a.data()),
                    static_cast<float*>(result.data()),
                    M, N);
                break;
            case DataType::Float16:
                sum_axis1_f16_kernel<<<grid_size, block_size>>>(
                    static_cast<const __half*>(a.data()),
                    static_cast<__half*>(result.data()),
                    M, N);
                break;
            case DataType::BFloat16:
                sum_axis1_bf16_kernel<<<grid_size, block_size>>>(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<__nv_bfloat16*>(result.data()),
                    M, N);
                break;
            default:
                break;
        }
    }

    sync_and_check("sum_axis kernel failed");
    return result;
}

} // namespace ops
} // namespace pygpukit
