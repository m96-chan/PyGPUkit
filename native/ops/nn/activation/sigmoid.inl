/**
 * Sigmoid activation: 1 / (1 + exp(-x))
 */

namespace pygpukit {
namespace ops {

static void sigmoid_dispatch(const GPUArray& input, GPUArray& result) {
    size_t n = input.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::sigmoid_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float16:
            nn::sigmoid_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            nn::sigmoid_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }
}

GPUArray sigmoid(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sigmoid only supports float types (f32, f16, bf16)");
    }

    GPUArray result(input.shape(), input.dtype());
    sigmoid_dispatch(input, result);
    sync_and_check("sigmoid kernel failed");
    return result;
}

void sigmoid(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sigmoid only supports float types (f32, f16, bf16)");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("sigmoid: dtype mismatch between input and output");
    }
    if (input.shape() != out.shape()) {
        throw std::runtime_error("sigmoid: shape mismatch between input and output");
    }

    sigmoid_dispatch(input, out);
    sync_and_check("sigmoid kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
