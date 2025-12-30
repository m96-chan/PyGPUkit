/**
 * RMSNorm (Root Mean Square Normalization)
 */

namespace pygpukit {
namespace ops {

// Internal helper for rmsnorm kernel dispatch
static void rmsnorm_dispatch(
    const GPUArray& input,
    const GPUArray& gamma,
    GPUArray& result,
    float eps
) {
    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];

    // One block per row, use enough threads to cover features
    int block_size = std::min(256, (int)((features + 31) / 32 * 32));
    block_size = std::max(32, block_size);

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::rmsnorm_f32_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(gamma.data()),
                static_cast<float*>(result.data()),
                batch_size, features, eps);
            break;
        case DataType::Float64:
            nn::rmsnorm_f64_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const double*>(input.data()),
                static_cast<const double*>(gamma.data()),
                static_cast<double*>(result.data()),
                batch_size, features, (double)eps);
            break;
        case DataType::Float16:
            nn::rmsnorm_f16_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<const __half*>(gamma.data()),
                static_cast<__half*>(result.data()),
                batch_size, features, eps);
            break;
        case DataType::BFloat16:
            nn::rmsnorm_bf16_kernel<<<batch_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<const __nv_bfloat16*>(gamma.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                batch_size, features, eps);
            break;
        default:
            throw std::runtime_error("rmsnorm only supports float types");
    }
}

GPUArray rmsnorm(const GPUArray& input, const GPUArray& gamma, float eps) {
    // input: [batch, features]
    // gamma: [features]

    if (input.ndim() != 2) {
        throw std::runtime_error("rmsnorm expects 2D input [batch, features]");
    }
    if (gamma.ndim() != 1) {
        throw std::runtime_error("rmsnorm expects 1D gamma");
    }
    if (input.dtype() != gamma.dtype()) {
        throw std::runtime_error("rmsnorm: dtype mismatch");
    }

    size_t features = input.shape()[1];

    if (gamma.shape()[0] != features) {
        throw std::runtime_error("rmsnorm: gamma size must match features");
    }

    GPUArray result(input.shape(), input.dtype());
    rmsnorm_dispatch(input, gamma, result, eps);
    sync_and_check("rmsnorm kernel failed");
    return result;
}

// In-place variant for CUDA Graph capture
void rmsnorm(const GPUArray& input, const GPUArray& gamma, GPUArray& out, float eps) {
    // input: [batch, features]
    // gamma: [features]
    // out: [batch, features]

    if (input.ndim() != 2) {
        throw std::runtime_error("rmsnorm expects 2D input [batch, features]");
    }
    if (gamma.ndim() != 1) {
        throw std::runtime_error("rmsnorm expects 1D gamma");
    }
    if (out.ndim() != 2) {
        throw std::runtime_error("rmsnorm expects 2D output");
    }
    if (input.dtype() != gamma.dtype() || input.dtype() != out.dtype()) {
        throw std::runtime_error("rmsnorm: dtype mismatch");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("rmsnorm: input and output shape mismatch");
    }

    size_t features = input.shape()[1];

    if (gamma.shape()[0] != features) {
        throw std::runtime_error("rmsnorm: gamma size must match features");
    }

    rmsnorm_dispatch(input, gamma, out, eps);
    sync_and_check("rmsnorm kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
