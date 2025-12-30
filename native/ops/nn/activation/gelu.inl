/**
 * GELU (Gaussian Error Linear Unit) activation
 */

namespace pygpukit {
namespace ops {

using namespace nn;

GPUArray gelu(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float64 &&
        input.dtype() != DataType::Float16 && input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("gelu only supports float types");
    }

    GPUArray result(input.shape(), input.dtype());
    size_t n = input.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    switch (input.dtype()) {
        case DataType::Float32:
            gelu_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                n);
            break;
        case DataType::Float64:
            gelu_f64_kernel<<<grid_size, block_size>>>(
                static_cast<const double*>(input.data()),
                static_cast<double*>(result.data()),
                n);
            break;
        case DataType::Float16:
            gelu_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                n);
            break;
        case DataType::BFloat16:
            gelu_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                n);
            break;
        default:
            break;
    }

    sync_and_check("gelu kernel failed");
    return result;
}

}  // namespace ops
}  // namespace pygpukit
