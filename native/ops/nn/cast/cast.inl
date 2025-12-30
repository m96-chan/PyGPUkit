/**
 * Dtype cast operations
 * - cast_f32_to_bf16
 * - cast_f32_to_f16
 * - cast_bf16_to_f32
 * - cast_f16_to_f32
 */

namespace pygpukit {
namespace ops {

GPUArray cast_f32_to_bf16(const GPUArray& src) {
    if (src.dtype() != DataType::Float32) {
        throw std::runtime_error("cast_f32_to_bf16: input must be float32");
    }

    GPUArray dst(src.shape(), DataType::BFloat16);
    size_t n = src.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_f32_to_bf16_kernel<<<grid_size, block_size>>>(
        static_cast<const float*>(src.data()),
        static_cast<__nv_bfloat16*>(dst.data()), n);

    sync_and_check("cast_f32_to_bf16 kernel failed");
    return dst;
}

void cast_f32_to_bf16(const GPUArray& src, GPUArray& dst) {
    if (src.dtype() != DataType::Float32) {
        throw std::runtime_error("cast_f32_to_bf16: input must be float32");
    }
    if (dst.dtype() != DataType::BFloat16) {
        throw std::runtime_error("cast_f32_to_bf16: output must be bfloat16");
    }
    if (src.size() != dst.size()) {
        throw std::runtime_error("cast_f32_to_bf16: size mismatch");
    }

    size_t n = src.size();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_f32_to_bf16_kernel<<<grid_size, block_size>>>(
        static_cast<const float*>(src.data()),
        static_cast<__nv_bfloat16*>(dst.data()), n);

    sync_and_check("cast_f32_to_bf16 kernel failed");
}

GPUArray cast_f32_to_f16(const GPUArray& src) {
    if (src.dtype() != DataType::Float32) {
        throw std::runtime_error("cast_f32_to_f16: input must be float32");
    }

    GPUArray dst(src.shape(), DataType::Float16);
    size_t n = src.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_f32_to_f16_kernel<<<grid_size, block_size>>>(
        static_cast<const float*>(src.data()),
        static_cast<__half*>(dst.data()), n);

    sync_and_check("cast_f32_to_f16 kernel failed");
    return dst;
}

GPUArray cast_bf16_to_f32(const GPUArray& src) {
    if (src.dtype() != DataType::BFloat16) {
        throw std::runtime_error("cast_bf16_to_f32: input must be bfloat16");
    }

    GPUArray dst(src.shape(), DataType::Float32);
    size_t n = src.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_bf16_to_f32_kernel<<<grid_size, block_size>>>(
        static_cast<const __nv_bfloat16*>(src.data()),
        static_cast<float*>(dst.data()), n);

    sync_and_check("cast_bf16_to_f32 kernel failed");
    return dst;
}

GPUArray cast_f16_to_f32(const GPUArray& src) {
    if (src.dtype() != DataType::Float16) {
        throw std::runtime_error("cast_f16_to_f32: input must be float16");
    }

    GPUArray dst(src.shape(), DataType::Float32);
    size_t n = src.size();

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    nn::cast_f16_to_f32_kernel<<<grid_size, block_size>>>(
        static_cast<const __half*>(src.data()),
        static_cast<float*>(dst.data()), n);

    sync_and_check("cast_f16_to_f32 kernel failed");
    return dst;
}

}  // namespace ops
}  // namespace pygpukit
