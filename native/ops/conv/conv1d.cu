// Conv1d CUDA Dispatcher
// native/ops/conv/conv1d.cu

#include "conv1d_kernels.cuh"
#include "../common/error.cuh"

#include <stdexcept>
#include <string>

namespace pygpukit {
namespace ops {

// Helper to compute output length
inline int compute_conv1d_output_length(int length, int kernel_size, int stride, int padding) {
    return (length + 2 * padding - kernel_size) / stride + 1;
}

// Conv1d dispatcher
void conv1d(
    const GPUArray& input,
    const GPUArray& weight,
    const GPUArray* bias,
    GPUArray& output,
    int stride,
    int padding
) {
    // Validate input dimensions: [batch, in_channels, length]
    if (input.ndim() != 3) {
        throw std::invalid_argument("conv1d: input must be 3D [batch, in_channels, length]");
    }

    // Validate weight dimensions: [out_channels, in_channels, kernel_size]
    if (weight.ndim() != 3) {
        throw std::invalid_argument("conv1d: weight must be 3D [out_channels, in_channels, kernel_size]");
    }

    // Extract dimensions
    int batch = input.shape()[0];
    int in_channels = input.shape()[1];
    int length = input.shape()[2];
    int out_channels = weight.shape()[0];
    int weight_in_channels = weight.shape()[1];
    int kernel_size = weight.shape()[2];

    // Validate channel match
    if (in_channels != weight_in_channels) {
        throw std::invalid_argument(
            "conv1d: input channels (" + std::to_string(in_channels) +
            ") != weight in_channels (" + std::to_string(weight_in_channels) + ")"
        );
    }

    // Validate bias if present
    if (bias != nullptr) {
        if (bias->ndim() != 1 || bias->shape()[0] != out_channels) {
            throw std::invalid_argument(
                "conv1d: bias must be 1D with size " + std::to_string(out_channels)
            );
        }
    }

    // Validate dtypes match
    if (input.dtype() != weight.dtype()) {
        throw std::invalid_argument("conv1d: input and weight must have same dtype");
    }
    if (bias != nullptr && bias->dtype() != input.dtype()) {
        throw std::invalid_argument("conv1d: bias must have same dtype as input");
    }

    // Compute output length
    int out_length = compute_conv1d_output_length(length, kernel_size, stride, padding);
    if (out_length <= 0) {
        throw std::invalid_argument(
            "conv1d: invalid parameters result in non-positive output length"
        );
    }

    // Validate output shape
    if (output.ndim() != 3 ||
        output.shape()[0] != batch ||
        output.shape()[1] != out_channels ||
        output.shape()[2] != out_length) {
        throw std::invalid_argument(
            "conv1d: output shape mismatch, expected [" +
            std::to_string(batch) + ", " +
            std::to_string(out_channels) + ", " +
            std::to_string(out_length) + "]"
        );
    }

    // Kernel configuration
    int total_elements = batch * out_channels * out_length;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    // Type-based dispatch
    switch (input.dtype()) {
        case DataType::Float32: {
            const float* bias_ptr = (bias != nullptr)
                ? static_cast<const float*>(bias->data()) : nullptr;

            conv1d_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(weight.data()),
                bias_ptr,
                static_cast<float*>(output.data()),
                batch, in_channels, length,
                out_channels, kernel_size,
                stride, padding, out_length
            );
            break;
        }

        case DataType::BFloat16: {
            const __nv_bfloat16* bias_ptr = (bias != nullptr)
                ? static_cast<const __nv_bfloat16*>(bias->data()) : nullptr;

            conv1d_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<const __nv_bfloat16*>(weight.data()),
                bias_ptr,
                static_cast<__nv_bfloat16*>(output.data()),
                batch, in_channels, length,
                out_channels, kernel_size,
                stride, padding, out_length
            );
            break;
        }

        case DataType::Float16: {
            const __half* bias_ptr = (bias != nullptr)
                ? static_cast<const __half*>(bias->data()) : nullptr;

            conv1d_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<const __half*>(weight.data()),
                bias_ptr,
                static_cast<__half*>(output.data()),
                batch, in_channels, length,
                out_channels, kernel_size,
                stride, padding, out_length
            );
            break;
        }

        default:
            throw std::invalid_argument(
                "conv1d: unsupported dtype (only float32, float16, bfloat16 supported)"
            );
    }

    sync_and_check("conv1d kernel failed");
}

// Convenience overload: allocates output
GPUArray conv1d(
    const GPUArray& input,
    const GPUArray& weight,
    const GPUArray* bias,
    int stride,
    int padding
) {
    // Compute output shape
    int batch = input.shape()[0];
    int out_channels = weight.shape()[0];
    int kernel_size = weight.shape()[2];
    int length = input.shape()[2];
    int out_length = compute_conv1d_output_length(length, kernel_size, stride, padding);

    // Allocate output
    GPUArray output({static_cast<size_t>(batch), static_cast<size_t>(out_channels), static_cast<size_t>(out_length)}, input.dtype());

    // Call in-place version
    conv1d(input, weight, bias, output, stride, padding);

    return output;
}

// Overload without bias pointer (for pybind11)
GPUArray conv1d_no_bias(
    const GPUArray& input,
    const GPUArray& weight,
    int stride,
    int padding
) {
    return conv1d(input, weight, nullptr, stride, padding);
}

GPUArray conv1d_with_bias(
    const GPUArray& input,
    const GPUArray& weight,
    const GPUArray& bias,
    int stride,
    int padding
) {
    return conv1d(input, weight, &bias, stride, padding);
}

}  // namespace ops
}  // namespace pygpukit
