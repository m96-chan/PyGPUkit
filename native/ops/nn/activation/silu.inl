/**
 * SiLU (Swish) activation: x * sigmoid(x)
 */

namespace pygpukit {
namespace ops {

// Internal dispatch helper with capture stream support
static void silu_dispatch(const GPUArray& input, GPUArray& result) {
    size_t n = input.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::silu_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float64:
            nn::silu_f64_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const double*>(input.data()),
                static_cast<double*>(result.data()),
                n);
            break;
        case DataType::Float16:
            nn::silu_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            nn::silu_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }
}

GPUArray silu(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float64 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("silu only supports float types");
    }

    GPUArray result(input.shape(), input.dtype());
    silu_dispatch(input, result);
    sync_and_check("silu kernel failed");
    return result;
}

// SiLU with output buffer (for CUDA Graph capture)
void silu(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float64 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("silu only supports float types");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("silu: dtype mismatch between input and output");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("silu: shape mismatch between input and output");
    }

    silu_dispatch(input, out);
    sync_and_check("silu kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
