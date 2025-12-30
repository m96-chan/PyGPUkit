/**
 * Tensor manipulation operations
 * - transpose (2D, 3D, 4D variants)
 * - reshape_copy
 * - concat_axis0
 * - split_qkv_batch
 * - repeat_interleave_axis1
 */

namespace pygpukit {
namespace ops {

using namespace nn;

// ============================================================================
// 2D Transpose
// ============================================================================

GPUArray transpose(const GPUArray& input) {
    if (input.ndim() != 2) {
        throw std::runtime_error("transpose expects 2D input [rows, cols]");
    }

    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];

    // Output shape is [cols, rows]
    GPUArray result({cols, rows}, input.dtype());

    // Use 32x32 tiles with 32x8 threads
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

    switch (input.dtype()) {
        case DataType::Float32:
            transpose_f32_kernel<<<grid, block>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()),
                rows, cols);
            break;
        case DataType::Float64:
            transpose_f64_kernel<<<grid, block>>>(
                static_cast<const double*>(input.data()),
                static_cast<double*>(result.data()),
                rows, cols);
            break;
        case DataType::Float16:
            transpose_f16_kernel<<<grid, block>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()),
                rows, cols);
            break;
        case DataType::BFloat16:
            transpose_bf16_kernel<<<grid, block>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                rows, cols);
            break;
        default:
            throw std::runtime_error("transpose only supports float types");
    }

    sync_and_check("transpose kernel failed");
    return result;
}

// ============================================================================
// 3D Transpose: [d0, d1, d2] -> [d1, d0, d2] (swaps first two axes)
// ============================================================================

static void transpose_3d_021_dispatch(
    const GPUArray& input, GPUArray& result,
    size_t dim0, size_t dim1, size_t dim2
) {
    size_t total = input.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::transpose_021_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()), dim0, dim1, dim2);
            break;
        case DataType::Float16:
            nn::transpose_021_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()), dim0, dim1, dim2);
            break;
        case DataType::BFloat16:
            nn::transpose_021_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()), dim0, dim1, dim2);
            break;
        default:
            throw std::runtime_error("transpose_3d_021: unsupported dtype");
    }
}

GPUArray transpose_3d_021(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_3d_021: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3) {
        throw std::runtime_error("transpose_3d_021: expects 3D tensor");
    }

    size_t dim0 = input.shape()[0], dim1 = input.shape()[1], dim2 = input.shape()[2];
    std::vector<size_t> out_shape = {dim1, dim0, dim2};
    GPUArray result(out_shape, input.dtype());

    transpose_3d_021_dispatch(input, result, dim0, dim1, dim2);
    sync_and_check("transpose_3d_021 kernel failed");
    return result;
}

void transpose_3d_021(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_3d_021: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("transpose_3d_021: expects 3D tensors");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("transpose_3d_021: dtype mismatch");
    }

    size_t dim0 = input.shape()[0], dim1 = input.shape()[1], dim2 = input.shape()[2];
    if (out.shape()[0] != dim1 || out.shape()[1] != dim0 || out.shape()[2] != dim2) {
        throw std::runtime_error("transpose_3d_021: output shape mismatch");
    }

    transpose_3d_021_dispatch(input, out, dim0, dim1, dim2);
    sync_and_check("transpose_3d_021 kernel failed");
}

// ============================================================================
// 3D Transpose: [d0, d1, d2] -> [d0, d2, d1] (swaps last two axes)
// ============================================================================

static void transpose_3d_012_dispatch(
    const GPUArray& input, GPUArray& result,
    size_t dim0, size_t dim1, size_t dim2
) {
    size_t total = input.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::transpose_012_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()), dim0, dim1, dim2);
            break;
        case DataType::Float16:
            nn::transpose_012_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()), dim0, dim1, dim2);
            break;
        case DataType::BFloat16:
            nn::transpose_012_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()), dim0, dim1, dim2);
            break;
        default:
            throw std::runtime_error("transpose_3d_012: unsupported dtype");
    }
}

GPUArray transpose_3d_012(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_3d_012: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3) {
        throw std::runtime_error("transpose_3d_012: expects 3D tensor");
    }

    size_t dim0 = input.shape()[0], dim1 = input.shape()[1], dim2 = input.shape()[2];
    std::vector<size_t> out_shape = {dim0, dim2, dim1};
    GPUArray result(out_shape, input.dtype());

    transpose_3d_012_dispatch(input, result, dim0, dim1, dim2);
    sync_and_check("transpose_3d_012 kernel failed");
    return result;
}

void transpose_3d_012(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_3d_012: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("transpose_3d_012: expects 3D tensors");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("transpose_3d_012: dtype mismatch");
    }

    size_t dim0 = input.shape()[0], dim1 = input.shape()[1], dim2 = input.shape()[2];
    if (out.shape()[0] != dim0 || out.shape()[1] != dim2 || out.shape()[2] != dim1) {
        throw std::runtime_error("transpose_3d_012: output shape mismatch");
    }

    transpose_3d_012_dispatch(input, out, dim0, dim1, dim2);
    sync_and_check("transpose_3d_012 kernel failed");
}

// ============================================================================
// 4D Transpose: [d0, d1, d2, d3] -> [d0, d2, d1, d3]
// ============================================================================

static void transpose_4d_0213_dispatch(
    const GPUArray& input, GPUArray& result,
    size_t dim0, size_t dim1, size_t dim2, size_t dim3
) {
    size_t total = input.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::transpose_0213_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()), dim0, dim1, dim2, dim3);
            break;
        case DataType::Float16:
            nn::transpose_0213_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()), dim0, dim1, dim2, dim3);
            break;
        case DataType::BFloat16:
            nn::transpose_0213_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()), dim0, dim1, dim2, dim3);
            break;
        default:
            throw std::runtime_error("transpose_4d_0213: unsupported dtype");
    }
}

GPUArray transpose_4d_0213(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_4d_0213: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0213: expects 4D tensor");
    }

    size_t dim0 = input.shape()[0], dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2], dim3 = input.shape()[3];
    std::vector<size_t> out_shape = {dim0, dim2, dim1, dim3};
    GPUArray result(out_shape, input.dtype());

    transpose_4d_0213_dispatch(input, result, dim0, dim1, dim2, dim3);
    sync_and_check("transpose_4d_0213 kernel failed");
    return result;
}

void transpose_4d_0213(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_4d_0213: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 4 || out.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0213: expects 4D tensors");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("transpose_4d_0213: dtype mismatch");
    }

    size_t dim0 = input.shape()[0], dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2], dim3 = input.shape()[3];
    if (out.shape()[0] != dim0 || out.shape()[1] != dim2 ||
        out.shape()[2] != dim1 || out.shape()[3] != dim3) {
        throw std::runtime_error("transpose_4d_0213: output shape mismatch");
    }

    transpose_4d_0213_dispatch(input, out, dim0, dim1, dim2, dim3);
    sync_and_check("transpose_4d_0213 kernel failed");
}

// ============================================================================
// 4D Transpose: [d0, d1, d2, d3] -> [d0, d1, d3, d2] (swaps last two axes)
// ============================================================================

static void transpose_4d_0132_dispatch(
    const GPUArray& input, GPUArray& result,
    size_t dim0, size_t dim1, size_t dim2, size_t dim3
) {
    size_t total = input.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::transpose_0132_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()), dim0, dim1, dim2, dim3);
            break;
        case DataType::Float16:
            nn::transpose_0132_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()), dim0, dim1, dim2, dim3);
            break;
        case DataType::BFloat16:
            nn::transpose_0132_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()), dim0, dim1, dim2, dim3);
            break;
        default:
            throw std::runtime_error("transpose_4d_0132: unsupported dtype");
    }
}

GPUArray transpose_4d_0132(const GPUArray& input) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_4d_0132: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0132: expects 4D tensor");
    }

    size_t dim0 = input.shape()[0], dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2], dim3 = input.shape()[3];
    std::vector<size_t> out_shape = {dim0, dim1, dim3, dim2};
    GPUArray result(out_shape, input.dtype());

    transpose_4d_0132_dispatch(input, result, dim0, dim1, dim2, dim3);
    sync_and_check("transpose_4d_0132 kernel failed");
    return result;
}

void transpose_4d_0132(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("transpose_4d_0132: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 4 || out.ndim() != 4) {
        throw std::runtime_error("transpose_4d_0132: expects 4D tensors");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("transpose_4d_0132: dtype mismatch");
    }

    size_t dim0 = input.shape()[0], dim1 = input.shape()[1];
    size_t dim2 = input.shape()[2], dim3 = input.shape()[3];
    if (out.shape()[0] != dim0 || out.shape()[1] != dim1 ||
        out.shape()[2] != dim3 || out.shape()[3] != dim2) {
        throw std::runtime_error("transpose_4d_0132: output shape mismatch");
    }

    transpose_4d_0132_dispatch(input, out, dim0, dim1, dim2, dim3);
    sync_and_check("transpose_4d_0132 kernel failed");
}

// ============================================================================
// Reshape with Copy
// ============================================================================

static void reshape_copy_dispatch(const GPUArray& input, GPUArray& result, size_t total_size) {
    const int block_size = 256;
    const int grid_size = (total_size + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (input.dtype()) {
        case DataType::Float32:
            nn::copy_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()), total_size);
            break;
        case DataType::Float16:
            nn::copy_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()), total_size);
            break;
        case DataType::BFloat16:
            nn::copy_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()), total_size);
            break;
        default:
            throw std::runtime_error("reshape_copy: unsupported dtype");
    }
}

GPUArray reshape_copy(const GPUArray& input, const std::vector<size_t>& new_shape) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("reshape_copy: only float32/float16/bfloat16 supported");
    }

    size_t input_size = input.size();
    size_t output_size = 1;
    for (size_t dim : new_shape) output_size *= dim;

    if (input_size != output_size) {
        throw std::runtime_error("reshape_copy: total size mismatch");
    }

    GPUArray result(new_shape, input.dtype());
    reshape_copy_dispatch(input, result, input_size);
    sync_and_check("reshape_copy kernel failed");
    return result;
}

void reshape_copy(const GPUArray& input, GPUArray& out) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("reshape_copy: only float32/float16/bfloat16 supported");
    }
    if (input.dtype() != out.dtype()) {
        throw std::runtime_error("reshape_copy: dtype mismatch");
    }
    if (input.size() != out.size()) {
        throw std::runtime_error("reshape_copy: total size mismatch");
    }

    reshape_copy_dispatch(input, out, input.size());
    sync_and_check("reshape_copy kernel failed");
}

// ============================================================================
// Concat Axis 0
// ============================================================================

GPUArray concat_axis0(const GPUArray& a, const GPUArray& b) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("concat: dtype mismatch");
    }
    if (a.dtype() != DataType::Float32 && a.dtype() != DataType::Float16 &&
        a.dtype() != DataType::BFloat16 && a.dtype() != DataType::UInt8) {
        throw std::runtime_error("concat: only float32/float16/bfloat16/uint8 supported");
    }
    if (a.ndim() < 1 || b.ndim() < 1 || a.ndim() != b.ndim()) {
        throw std::runtime_error("concat: dimension mismatch");
    }

    for (size_t i = 1; i < a.ndim(); i++) {
        if (a.shape()[i] != b.shape()[i]) {
            throw std::runtime_error("concat: shape mismatch on non-concat axis");
        }
    }

    std::vector<size_t> out_shape = a.shape();
    out_shape[0] = a.shape()[0] + b.shape()[0];
    GPUArray result(out_shape, a.dtype());

    size_t stride = 1;
    for (size_t i = 1; i < a.ndim(); i++) stride *= a.shape()[i];

    size_t total = result.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    switch (a.dtype()) {
        case DataType::Float32:
            nn::concat_axis0_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(result.data()),
                a.shape()[0], b.shape()[0], stride);
            break;
        case DataType::Float16:
            nn::concat_axis0_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(result.data()),
                a.shape()[0], b.shape()[0], stride);
            break;
        case DataType::BFloat16:
            nn::concat_axis0_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(result.data()),
                a.shape()[0], b.shape()[0], stride);
            break;
        case DataType::UInt8:
            nn::concat_axis0_u8_kernel<<<grid_size, block_size>>>(
                static_cast<const uint8_t*>(a.data()),
                static_cast<const uint8_t*>(b.data()),
                static_cast<uint8_t*>(result.data()),
                a.shape()[0], b.shape()[0], stride);
            break;
        default:
            break;
    }

    sync_and_check("concat_axis0 kernel failed");
    return result;
}

// ============================================================================
// Repeat Interleave Axis 1
// ============================================================================

GPUArray repeat_interleave_axis1(const GPUArray& input, size_t repeats) {
    if (input.dtype() != DataType::Float32 && input.dtype() != DataType::Float16 &&
        input.dtype() != DataType::BFloat16) {
        throw std::runtime_error("repeat_interleave: only float32/float16/bfloat16 supported");
    }
    if (input.ndim() != 3) {
        throw std::runtime_error("repeat_interleave: expects 3D tensor [dim0, dim1, dim2]");
    }

    size_t dim0 = input.shape()[0], dim1 = input.shape()[1], dim2 = input.shape()[2];
    std::vector<size_t> out_shape = {dim0, dim1 * repeats, dim2};
    GPUArray result(out_shape, input.dtype());

    size_t total = result.size();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    switch (input.dtype()) {
        case DataType::Float32:
            nn::repeat_interleave_axis1_f32_kernel<<<grid_size, block_size>>>(
                static_cast<const float*>(input.data()),
                static_cast<float*>(result.data()), dim0, dim1, dim2, repeats);
            break;
        case DataType::Float16:
            nn::repeat_interleave_axis1_f16_kernel<<<grid_size, block_size>>>(
                static_cast<const __half*>(input.data()),
                static_cast<__half*>(result.data()), dim0, dim1, dim2, repeats);
            break;
        case DataType::BFloat16:
            nn::repeat_interleave_axis1_bf16_kernel<<<grid_size, block_size>>>(
                static_cast<const __nv_bfloat16*>(input.data()),
                static_cast<__nv_bfloat16*>(result.data()), dim0, dim1, dim2, repeats);
            break;
        default:
            break;
    }

    sync_and_check("repeat_interleave_axis1 kernel failed");
    return result;
}

// ============================================================================
// Split QKV Batch
// ============================================================================

void split_qkv_batch(
    const GPUArray& qkv, GPUArray& q_out, GPUArray& k_out, GPUArray& v_out,
    int q_dim, int k_dim, int v_dim
) {
    if (qkv.ndim() != 2) {
        throw std::runtime_error("split_qkv_batch: qkv must be 2D [seq_len, total_dim]");
    }

    int seq_len = static_cast<int>(qkv.shape()[0]);
    int total_dim = q_dim + k_dim + v_dim;

    if (static_cast<int>(qkv.shape()[1]) != total_dim) {
        throw std::runtime_error("split_qkv_batch: qkv dim mismatch");
    }

    int total_elements = seq_len * total_dim;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (qkv.dtype()) {
        case DataType::Float16:
            nn::split_qkv_batch_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(qkv.data()),
                static_cast<__half*>(q_out.data()),
                static_cast<__half*>(k_out.data()),
                static_cast<__half*>(v_out.data()),
                seq_len, q_dim, k_dim, v_dim);
            break;
        case DataType::Float32:
            nn::split_qkv_batch_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(qkv.data()),
                static_cast<float*>(q_out.data()),
                static_cast<float*>(k_out.data()),
                static_cast<float*>(v_out.data()),
                seq_len, q_dim, k_dim, v_dim);
            break;
        case DataType::BFloat16:
            nn::split_qkv_batch_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(qkv.data()),
                static_cast<__nv_bfloat16*>(q_out.data()),
                static_cast<__nv_bfloat16*>(k_out.data()),
                static_cast<__nv_bfloat16*>(v_out.data()),
                seq_len, q_dim, k_dim, v_dim);
            break;
        default:
            throw std::runtime_error("split_qkv_batch: unsupported dtype");
    }

    sync_and_check("split_qkv_batch kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
