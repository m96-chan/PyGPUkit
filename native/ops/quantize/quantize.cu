/**
 * Quantization operations dispatch
 */
#include "quantize_kernels.cuh"
#include "../common/error.cuh"
#include "../../core/memory.hpp"

namespace pygpukit {
namespace ops {

using namespace quantize;

// ============================================================================
// Dequantization
// ============================================================================

GPUArray dequantize_int8(const GPUArray& input, const GPUArray& scale, DataType output_dtype) {
    if (input.dtype() != DataType::Int8) {
        throw std::runtime_error("dequantize_int8: input must be Int8");
    }
    if (input.ndim() != 2) {
        throw std::runtime_error("dequantize_int8: input must be 2D [rows, cols]");
    }
    if (scale.ndim() != 1) {
        throw std::runtime_error("dequantize_int8: scale must be 1D [rows]");
    }

    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];

    // Per-row scale
    if (scale.shape()[0] != rows) {
        throw std::runtime_error("dequantize_int8: scale size must match rows");
    }

    GPUArray result({rows, cols}, output_dtype);

    size_t total = rows * cols;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    if (output_dtype == DataType::Float16) {
        if (scale.dtype() != DataType::Float16) {
            throw std::runtime_error("dequantize_int8: scale dtype must match output dtype");
        }
        dequantize_int8_to_f16_kernel<<<grid_size, block_size>>>(
            static_cast<const int8_t*>(input.data()),
            static_cast<const __half*>(scale.data()),
            static_cast<__half*>(result.data()),
            rows, cols);
    } else if (output_dtype == DataType::Float32) {
        if (scale.dtype() != DataType::Float32) {
            throw std::runtime_error("dequantize_int8: scale dtype must match output dtype");
        }
        dequantize_int8_to_f32_kernel<<<grid_size, block_size>>>(
            static_cast<const int8_t*>(input.data()),
            static_cast<const float*>(scale.data()),
            static_cast<float*>(result.data()),
            rows, cols);
    } else {
        throw std::runtime_error("dequantize_int8: output dtype must be Float16 or Float32");
    }

    sync_and_check("dequantize_int8 kernel failed");
    return result;
}

// ============================================================================
// Quantized Linear Layer
// ============================================================================

GPUArray linear_int8(
    const GPUArray& activation,   // [M, K] FP16
    const GPUArray& weight_int8,  // [N, K] INT8
    const GPUArray& scale,        // [N] FP16
    const GPUArray* bias          // [N] FP16 (optional)
) {
    if (activation.dtype() != DataType::Float16) {
        throw std::runtime_error("linear_int8: activation must be Float16");
    }
    if (weight_int8.dtype() != DataType::Int8) {
        throw std::runtime_error("linear_int8: weight must be Int8");
    }
    if (scale.dtype() != DataType::Float16) {
        throw std::runtime_error("linear_int8: scale must be Float16");
    }
    if (activation.ndim() != 2 || weight_int8.ndim() != 2) {
        throw std::runtime_error("linear_int8: activation and weight must be 2D");
    }

    int M = activation.shape()[0];
    int K = activation.shape()[1];
    int N = weight_int8.shape()[0];

    if (weight_int8.shape()[1] != K) {
        throw std::runtime_error("linear_int8: weight K dimension mismatch");
    }
    if (scale.shape()[0] != N) {
        throw std::runtime_error("linear_int8: scale size must match N");
    }

    GPUArray result({(size_t)M, (size_t)N}, DataType::Float16);

    // Use tiled kernel for better performance
    dim3 block(Q_TILE_N, Q_TILE_M);
    dim3 grid((N + Q_TILE_N - 1) / Q_TILE_N, (M + Q_TILE_M - 1) / Q_TILE_M);

    linear_int8_f16_tiled_kernel<<<grid, block>>>(
        static_cast<const __half*>(activation.data()),
        static_cast<const int8_t*>(weight_int8.data()),
        static_cast<const __half*>(scale.data()),
        static_cast<__half*>(result.data()),
        M, N, K);

    sync_and_check("linear_int8 kernel failed");

    // Add bias if provided
    if (bias != nullptr) {
        if (bias->dtype() != DataType::Float16) {
            throw std::runtime_error("linear_int8: bias must be Float16");
        }
        if (bias->shape()[0] != N) {
            throw std::runtime_error("linear_int8: bias size must match N");
        }
        // TODO: fuse bias add into kernel
        // For now, use separate bias_add
        // bias_add_inplace(result, *bias);
    }

    return result;
}

// ============================================================================
// Quantization
// ============================================================================

std::pair<GPUArray, GPUArray> quantize_to_int8(const GPUArray& input) {
    if (input.ndim() != 2) {
        throw std::runtime_error("quantize_to_int8: input must be 2D [rows, cols]");
    }

    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];

    GPUArray output({rows, cols}, DataType::Int8);
    // Per-row scale: one scale per row (output channel)
    GPUArray scale({rows}, input.dtype());

    const int block_size = 256;
    size_t smem_size = block_size * sizeof(float);

    if (input.dtype() == DataType::Float16) {
        // Launch one block per row
        quantize_f16_to_int8_kernel<<<rows, block_size, smem_size>>>(
            static_cast<const __half*>(input.data()),
            static_cast<int8_t*>(output.data()),
            static_cast<__half*>(scale.data()),
            rows, cols);
    } else if (input.dtype() == DataType::Float32) {
        quantize_f32_to_int8_kernel<<<rows, block_size, smem_size>>>(
            static_cast<const float*>(input.data()),
            static_cast<int8_t*>(output.data()),
            static_cast<float*>(scale.data()),
            rows, cols);
    } else {
        throw std::runtime_error("quantize_to_int8: input must be Float16 or Float32");
    }

    sync_and_check("quantize_to_int8 kernel failed");
    return {std::move(output), std::move(scale)};
}

} // namespace ops
} // namespace pygpukit
