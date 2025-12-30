/**
 * Linear layer and bias operations
 * - bias_add_inplace: output += bias
 * - linear: y = xW^T + b
 * - softmax: softmax normalization
 */

namespace pygpukit {
namespace ops {

using namespace nn;

// ============================================================================
// Bias Add
// ============================================================================

// In-place bias add: output[batch, features] += bias[features]
void bias_add_inplace(GPUArray& output, const GPUArray& bias) {
    if (output.ndim() != 2) {
        throw std::runtime_error("bias_add expects 2D output tensor [batch, features]");
    }
    if (bias.ndim() != 1) {
        throw std::runtime_error("bias_add expects 1D bias tensor [features]");
    }
    if (output.dtype() != bias.dtype()) {
        throw std::runtime_error("bias_add: dtype mismatch");
    }

    size_t batch_size = output.shape()[0];
    size_t features = output.shape()[1];

    if (bias.shape()[0] != features) {
        throw std::runtime_error("bias_add: bias size must match output features");
    }

    size_t n = batch_size * features;
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    // Use capture stream for CUDA Graph compatibility
    cudaStream_t stream = internal::get_capture_stream();

    switch (output.dtype()) {
        case DataType::Float32:
            bias_add_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(output.data()),
                static_cast<const float*>(bias.data()),
                batch_size, features);
            break;
        case DataType::Float64:
            bias_add_f64_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<double*>(output.data()),
                static_cast<const double*>(bias.data()),
                batch_size, features);
            break;
        case DataType::Float16:
            bias_add_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__half*>(output.data()),
                static_cast<const __half*>(bias.data()),
                batch_size, features);
            break;
        case DataType::BFloat16:
            bias_add_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(output.data()),
                static_cast<const __nv_bfloat16*>(bias.data()),
                batch_size, features);
            break;
        default:
            throw std::runtime_error("bias_add only supports float types");
    }

    sync_and_check("bias_add kernel failed");
}

// ============================================================================
// Linear Layer: y = xW^T + b
// ============================================================================

GPUArray linear(const GPUArray& input, const GPUArray& weight, const GPUArray* bias) {
    // input: [batch, in_features]
    // weight: [out_features, in_features]
    // output: [batch, out_features]

    if (input.ndim() != 2) {
        throw std::runtime_error("linear expects 2D input [batch, in_features]");
    }
    if (weight.ndim() != 2) {
        throw std::runtime_error("linear expects 2D weight [out_features, in_features]");
    }
    if (input.dtype() != weight.dtype()) {
        throw std::runtime_error("linear: input and weight dtype mismatch");
    }

    size_t batch = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::runtime_error("linear: weight in_features must match input");
    }

    // Skip bias for now in basic implementation
    (void)bias;

    throw std::runtime_error("linear: not yet implemented - use matmul + bias_add separately for MVP");
}

// ============================================================================
// Softmax
// ============================================================================

GPUArray softmax(const GPUArray& input) {
    if (input.ndim() != 2) {
        throw std::runtime_error("softmax expects 2D input [batch, features]");
    }
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float64 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("softmax only supports float types");
    }

    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];

    GPUArray result(input.shape(), input.dtype());

    // One block per row
    int block_size = std::min(256, (int)((features + 31) / 32 * 32));
    block_size = std::max(32, block_size);

    switch (input.dtype()) {
        case DataType::Float32:
            nn::softmax_f32_kernel<<<batch_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                batch_size, features);
            break;
        case DataType::Float64:
            nn::softmax_f64_kernel<<<batch_size, block_size>>>(
                static_cast<const double*>(input.data()),
                static_cast<double*>(result.data()),
                batch_size, features);
            break;
        case DataType::Float16:
            nn::softmax_f16_kernel<<<batch_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                batch_size, features);
            break;
        case DataType::BFloat16:
            nn::softmax_bf16_kernel<<<batch_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                batch_size, features);
            break;
        default:
            break;
    }

    sync_and_check("softmax kernel failed");
    return result;
}

}  // namespace ops
}  // namespace pygpukit
