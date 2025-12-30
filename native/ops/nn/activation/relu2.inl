/**
 * ReLU squared (ReLU^2) activation: (max(0, x))^2
 *
 * Introduced in the Primer paper (Google, 2021).
 * Benefits: stronger sparsity, continuous first derivative.
 */

namespace pygpukit {
namespace ops {

// Internal dispatch helper with capture stream support
static void relu2_dispatch(const GPUArray& input, GPUArray& result) {
    size_t n = input.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::relu2_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float16:
            nn::relu2_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            nn::relu2_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }
}

GPUArray relu2(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("relu2 only supports float32, float16, bfloat16");
    }

    GPUArray result(input.shape(), input.dtype());
    relu2_dispatch(input, result);
    sync_and_check("relu2 kernel failed");
    return result;
}

// ReLU squared with output buffer (for CUDA Graph capture)
void relu2(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("relu2 only supports float32, float16, bfloat16");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("relu2: dtype mismatch between input and output");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("relu2: shape mismatch between input and output");
    }

    relu2_dispatch(input, out);
    sync_and_check("relu2 kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
