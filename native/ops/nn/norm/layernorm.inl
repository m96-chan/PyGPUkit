/**
 * LayerNorm (Layer Normalization)
 */

namespace pygpukit {
namespace ops {

using namespace nn;

GPUArray layernorm(const GPUArray& input, const GPUArray& gamma, const GPUArray& beta, float eps) {
    // input: [batch, features]
    // gamma: [features]
    // beta: [features]

    if (input.ndim() != 2) {
        throw std::runtime_error("layernorm expects 2D input [batch, features]");
    }
    if (gamma.ndim() != 1 || beta.ndim() != 1) {
        throw std::runtime_error("layernorm expects 1D gamma and beta");
    }
    if (input.dtype() != gamma.dtype() || input.dtype() != beta.dtype()) {
        throw std::runtime_error("layernorm: dtype mismatch");
    }

    size_t batch_size = input.shape()[0];
    size_t features = input.shape()[1];

    if (gamma.shape()[0] != features || beta.shape()[0] != features) {
        throw std::runtime_error("layernorm: gamma/beta size must match features");
    }

    GPUArray result(input.shape(), input.dtype());

    // One block per row, use enough threads to cover features
    int block_size = std::min(256, (int)((features + 31) / 32 * 32));
    block_size = std::max(32, block_size);

    switch (input.dtype()) {
        case DataType::Float32:
            layernorm_f32_kernel<<<batch_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<const float*>(gamma.data()),
                static_cast<const float*>(beta.data()),
                static_cast<float*>(result.data()),
                batch_size, features, eps);
            break;
        case DataType::Float64:
            layernorm_f64_kernel<<<batch_size, block_size>>>(
                static_cast<const double*>(input.data()),
                static_cast<const double*>(gamma.data()),
                static_cast<const double*>(beta.data()),
                static_cast<double*>(result.data()),
                batch_size, features, (double)eps);
            break;
        case DataType::Float16:
            layernorm_f16_kernel<<<batch_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<const __half*>(gamma.data()),
                static_cast<const __half*>(beta.data()),
                static_cast<__half*>(result.data()),
                batch_size, features, eps);
            break;
        case DataType::BFloat16:
            layernorm_bf16_kernel<<<batch_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<const __nv_bfloat16*>(gamma.data()),
                static_cast<const __nv_bfloat16*>(beta.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                batch_size, features, eps);
            break;
        default:
            throw std::runtime_error("layernorm only supports float types");
    }

    sync_and_check("layernorm kernel failed");
    return result;
}

}  // namespace ops
}  // namespace pygpukit
