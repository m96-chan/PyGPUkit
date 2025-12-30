/**
 * Tanh activation
 */

namespace pygpukit {
namespace ops {

static void tanh_dispatch(const GPUArray& input, GPUArray& result) {
    size_t n = input.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::tanh_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float16:
            nn::tanh_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            nn::tanh_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }
}

GPUArray tanh(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("tanh only supports float types (f32, f16, bf16)");
    }

    GPUArray result(input.shape(), input.dtype());
    tanh_dispatch(input, result);
    sync_and_check("tanh kernel failed");
    return result;
}

void tanh(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("tanh only supports float types (f32, f16, bf16)");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("tanh: dtype mismatch between input and output");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("tanh: shape mismatch between input and output");
    }

    tanh_dispatch(input, out);
    sync_and_check("tanh kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
