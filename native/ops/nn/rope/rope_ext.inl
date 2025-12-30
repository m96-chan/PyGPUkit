/**
 * Extended RoPE dispatch functions for context length extension
 *
 * Provides:
 * - rope_init_ntk_aware: NTK-aware frequency scaling
 * - rope_init_yarn: YaRN dimension-wise interpolation
 * - rope_init_linear: Simple linear position interpolation
 */

#include "rope_ext_kernels.cuh"

namespace pygpukit {
namespace ops {

std::pair<GPUArray, GPUArray> rope_init_ntk_aware(
    int max_seq_len,
    int head_dim,
    float base,
    float scale
) {
    // NTK-aware interpolation: scales base frequency instead of positions
    // base' = base * scale^(dim / (dim - 2))

    GPUArray cos_table({(size_t)max_seq_len, (size_t)head_dim}, DataType::Float32);
    GPUArray sin_table({(size_t)max_seq_len, (size_t)head_dim}, DataType::Float32);

    int total = max_seq_len * head_dim;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    nn::rope_init_ntk_aware_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<float*>(cos_table.data()),
        static_cast<float*>(sin_table.data()),
        max_seq_len,
        head_dim,
        base,
        scale);

    sync_and_check("rope_init_ntk_aware kernel failed");
    return {std::move(cos_table), std::move(sin_table)};
}

std::pair<GPUArray, GPUArray> rope_init_yarn(
    int max_seq_len,
    int head_dim,
    float base,
    float scale,
    int original_max_len,
    float beta_fast,
    float beta_slow,
    float mscale
) {
    // YaRN: dimension-wise interpolation with attention scaling
    // Different scaling for different frequency bands

    GPUArray cos_table({(size_t)max_seq_len, (size_t)head_dim}, DataType::Float32);
    GPUArray sin_table({(size_t)max_seq_len, (size_t)head_dim}, DataType::Float32);

    int total = max_seq_len * head_dim;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    nn::rope_init_yarn_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<float*>(cos_table.data()),
        static_cast<float*>(sin_table.data()),
        max_seq_len,
        head_dim,
        base,
        scale,
        original_max_len,
        beta_fast,
        beta_slow,
        mscale);

    sync_and_check("rope_init_yarn kernel failed");
    return {std::move(cos_table), std::move(sin_table)};
}

std::pair<GPUArray, GPUArray> rope_init_linear(
    int max_seq_len,
    int head_dim,
    float base,
    float scale
) {
    // Linear position interpolation (PI): pos' = pos / scale
    // Simple baseline, degrades quality at high scales

    GPUArray cos_table({(size_t)max_seq_len, (size_t)head_dim}, DataType::Float32);
    GPUArray sin_table({(size_t)max_seq_len, (size_t)head_dim}, DataType::Float32);

    int total = max_seq_len * head_dim;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    cudaStream_t stream = internal::get_capture_stream();

    nn::rope_init_linear_interpolation_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<float*>(cos_table.data()),
        static_cast<float*>(sin_table.data()),
        max_seq_len,
        head_dim,
        base,
        scale);

    sync_and_check("rope_init_linear kernel failed");
    return {std::move(cos_table), std::move(sin_table)};
}

}  // namespace ops
}  // namespace pygpukit
