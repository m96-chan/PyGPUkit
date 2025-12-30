/**
 * In-place elementwise operations
 * - add_inplace: a += b
 * - mul_inplace: a *= b
 * - copy_to: GPU-to-GPU copy
 */

namespace pygpukit {
namespace ops {

void add_inplace(GPUArray& a, const GPUArray& b) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("add_inplace: dtype mismatch");
    }
    size_t n = a.size();
    if (n != b.size()) {
        throw std::runtime_error("add_inplace: size mismatch");
    }

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (a.dtype()) {
        case DataType::Float16:
            nn::add_inplace_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__half*>(a.data()),
                static_cast<const __half*>(b.data()), n);
            break;
        case DataType::BFloat16:
            nn::add_inplace_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()), n);
            break;
        case DataType::Float32:
            nn::add_inplace_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(a.data()),
                static_cast<const float*>(b.data()), n);
            break;
        case DataType::Float64:
            nn::add_inplace_f64_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<double*>(a.data()),
                static_cast<const double*>(b.data()), n);
            break;
        default:
            throw std::runtime_error("add_inplace: unsupported dtype");
    }

    sync_and_check("add_inplace kernel failed");
}

void mul_inplace(GPUArray& a, const GPUArray& b) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("mul_inplace: dtype mismatch");
    }
    size_t n = a.size();
    if (n != b.size()) {
        throw std::runtime_error("mul_inplace: size mismatch");
    }

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (a.dtype()) {
        case DataType::Float16:
            nn::mul_inplace_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__half*>(a.data()),
                static_cast<const __half*>(b.data()), n);
            break;
        case DataType::BFloat16:
            nn::mul_inplace_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<__nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()), n);
            break;
        case DataType::Float32:
            nn::mul_inplace_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(a.data()),
                static_cast<const float*>(b.data()), n);
            break;
        case DataType::Float64:
            nn::mul_inplace_f64_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<double*>(a.data()),
                static_cast<const double*>(b.data()), n);
            break;
        default:
            throw std::runtime_error("mul_inplace: unsupported dtype");
    }

    sync_and_check("mul_inplace kernel failed");
}

void copy_to(const GPUArray& src, GPUArray& dst) {
    if (src.dtype() != dst.dtype()) {
        throw std::runtime_error("copy_to: dtype mismatch");
    }
    size_t n = src.size();
    if (n != dst.size()) {
        throw std::runtime_error("copy_to: size mismatch");
    }

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (src.dtype()) {
        case DataType::Float16:
            nn::copy_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(src.data()),
                static_cast<__half*>(dst.data()), n);
            break;
        case DataType::BFloat16:
            nn::copy_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(src.data()),
                static_cast<__nv_bfloat16*>(dst.data()), n);
            break;
        case DataType::Float32:
            nn::copy_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(src.data()),
                static_cast<float*>(dst.data()), n);
            break;
        case DataType::Int32:
            nn::copy_i32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const int*>(src.data()),
                static_cast<int*>(dst.data()), n);
            break;
        default:
            throw std::runtime_error("copy_to: unsupported dtype");
    }

    sync_and_check("copy_to kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
