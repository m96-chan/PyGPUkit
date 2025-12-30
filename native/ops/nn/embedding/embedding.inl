/**
 * Embedding and KV Cache operations
 * - embedding_lookup (single, ptr, batch)
 * - slice_rows_range_ptr
 * - kv_cache_update/prefill (standard and GQA variants)
 */

namespace pygpukit {
namespace ops {

// ============================================================================
// Embedding Lookup
// ============================================================================

void embedding_lookup(const GPUArray& embed_matrix, GPUArray& out, int token_id) {
    if (embed_matrix.ndim() != 2) {
        throw std::runtime_error("embedding_lookup: embed_matrix must be 2D");
    }
    if (embed_matrix.dtype() != out.dtype()) {
        throw std::runtime_error("embedding_lookup: dtype mismatch");
    }

    int hidden_size = static_cast<int>(embed_matrix.shape()[1]);
    const int block_size = 256;
    const int grid_size = (hidden_size + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (embed_matrix.dtype()) {
        case DataType::Float16:
            nn::embedding_lookup_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(embed_matrix.data()),
                static_cast<__half*>(out.data()), hidden_size, token_id);
            break;
        case DataType::BFloat16:
            nn::embedding_lookup_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(embed_matrix.data()),
                static_cast<__nv_bfloat16*>(out.data()), hidden_size, token_id);
            break;
        case DataType::Float32:
            nn::embedding_lookup_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(embed_matrix.data()),
                static_cast<float*>(out.data()), hidden_size, token_id);
            break;
        default:
            throw std::runtime_error("embedding_lookup: unsupported dtype");
    }

    sync_and_check("embedding_lookup kernel failed");
}

void embedding_lookup_ptr(
    const GPUArray& embed_matrix, GPUArray& out, const GPUArray& token_id_buf
) {
    if (embed_matrix.ndim() != 2) {
        throw std::runtime_error("embedding_lookup_ptr: embed_matrix must be 2D");
    }
    if (embed_matrix.dtype() != out.dtype()) {
        throw std::runtime_error("embedding_lookup_ptr: dtype mismatch");
    }
    if (token_id_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("embedding_lookup_ptr: token_id_buf must be int32");
    }

    int hidden_size = static_cast<int>(embed_matrix.shape()[1]);
    const int block_size = 256;
    const int grid_size = (hidden_size + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (embed_matrix.dtype()) {
        case DataType::Float16:
            nn::embedding_lookup_f16_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(embed_matrix.data()),
                static_cast<__half*>(out.data()), hidden_size,
                static_cast<const int*>(token_id_buf.data()));
            break;
        case DataType::BFloat16:
            nn::embedding_lookup_bf16_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(embed_matrix.data()),
                static_cast<__nv_bfloat16*>(out.data()), hidden_size,
                static_cast<const int*>(token_id_buf.data()));
            break;
        case DataType::Float32:
            nn::embedding_lookup_f32_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(embed_matrix.data()),
                static_cast<float*>(out.data()), hidden_size,
                static_cast<const int*>(token_id_buf.data()));
            break;
        default:
            throw std::runtime_error("embedding_lookup_ptr: unsupported dtype");
    }

    sync_and_check("embedding_lookup_ptr kernel failed");
}

void embedding_lookup_batch(
    const GPUArray& embed_matrix, GPUArray& out,
    const GPUArray& token_ids_buf, int batch_size
) {
    if (embed_matrix.ndim() != 2) {
        throw std::runtime_error("embedding_lookup_batch: embed_matrix must be 2D");
    }
    if (embed_matrix.dtype() != out.dtype()) {
        throw std::runtime_error("embedding_lookup_batch: dtype mismatch");
    }
    if (token_ids_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("embedding_lookup_batch: token_ids_buf must be int32");
    }

    int hidden_size = static_cast<int>(embed_matrix.shape()[1]);
    int total_elements = batch_size * hidden_size;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (embed_matrix.dtype()) {
        case DataType::Float16:
            nn::embedding_lookup_batch_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(embed_matrix.data()),
                static_cast<__half*>(out.data()),
                static_cast<const int*>(token_ids_buf.data()),
                batch_size, hidden_size);
            break;
        case DataType::BFloat16:
            nn::embedding_lookup_batch_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(embed_matrix.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                static_cast<const int*>(token_ids_buf.data()),
                batch_size, hidden_size);
            break;
        case DataType::Float32:
            nn::embedding_lookup_batch_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(embed_matrix.data()),
                static_cast<float*>(out.data()),
                static_cast<const int*>(token_ids_buf.data()),
                batch_size, hidden_size);
            break;
        default:
            throw std::runtime_error("embedding_lookup_batch: unsupported dtype");
    }

    sync_and_check("embedding_lookup_batch kernel failed");
}

void slice_rows_range_ptr(
    const GPUArray& table, GPUArray& out,
    const GPUArray& start_pos_buf, int count
) {
    if (table.ndim() != 2) {
        throw std::runtime_error("slice_rows_range_ptr: table must be 2D");
    }
    if (table.dtype() != out.dtype()) {
        throw std::runtime_error("slice_rows_range_ptr: dtype mismatch");
    }
    if (start_pos_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("slice_rows_range_ptr: start_pos_buf must be int32");
    }

    int row_dim = static_cast<int>(table.shape()[1]);
    int total_elements = count * row_dim;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (table.dtype()) {
        case DataType::Float16:
            nn::slice_rows_range_ptr_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(table.data()),
                static_cast<__half*>(out.data()),
                static_cast<const int*>(start_pos_buf.data()), count, row_dim);
            break;
        case DataType::BFloat16:
            nn::slice_rows_range_ptr_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(table.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                static_cast<const int*>(start_pos_buf.data()), count, row_dim);
            break;
        case DataType::Float32:
            nn::slice_rows_range_ptr_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(table.data()),
                static_cast<float*>(out.data()),
                static_cast<const int*>(start_pos_buf.data()), count, row_dim);
            break;
        default:
            throw std::runtime_error("slice_rows_range_ptr: unsupported dtype");
    }

    sync_and_check("slice_rows_range_ptr kernel failed");
}

// ============================================================================
// KV Cache Operations
// ============================================================================

void kv_cache_update(const GPUArray& new_kv, GPUArray& cache, int position) {
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_update: expected 3D tensors");
    }
    if (new_kv.shape()[0] != 1) {
        throw std::runtime_error("kv_cache_update: new_kv should have seq_len=1");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_update: dtype mismatch");
    }
    if (new_kv.shape()[1] != cache.shape()[1] || new_kv.shape()[2] != cache.shape()[2]) {
        throw std::runtime_error("kv_cache_update: shape mismatch (num_kv_heads, head_dim)");
    }

    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int total_elements = num_kv_heads * head_dim;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_update_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()), num_kv_heads, head_dim, position);
            break;
        case DataType::BFloat16:
            nn::kv_cache_update_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()), num_kv_heads, head_dim, position);
            break;
        case DataType::Float32:
            nn::kv_cache_update_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()), num_kv_heads, head_dim, position);
            break;
        default:
            throw std::runtime_error("kv_cache_update: unsupported dtype");
    }

    sync_and_check("kv_cache_update kernel failed");
}

void kv_cache_prefill(const GPUArray& new_kv, GPUArray& cache, int start_pos) {
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_prefill: expected 3D tensors");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_prefill: dtype mismatch");
    }
    if (new_kv.shape()[1] != cache.shape()[1] || new_kv.shape()[2] != cache.shape()[2]) {
        throw std::runtime_error("kv_cache_prefill: shape mismatch (num_kv_heads, head_dim)");
    }

    int seq_len = static_cast<int>(new_kv.shape()[0]);
    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int total_elements = seq_len * num_kv_heads * head_dim;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_prefill_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()), num_kv_heads, head_dim, start_pos, seq_len);
            break;
        case DataType::BFloat16:
            nn::kv_cache_prefill_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()), num_kv_heads, head_dim, start_pos, seq_len);
            break;
        case DataType::Float32:
            nn::kv_cache_prefill_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()), num_kv_heads, head_dim, start_pos, seq_len);
            break;
        default:
            throw std::runtime_error("kv_cache_prefill: unsupported dtype");
    }

    sync_and_check("kv_cache_prefill kernel failed");
}

// ============================================================================
// GQA KV Cache Operations
// ============================================================================

void kv_cache_update_gqa(
    const GPUArray& new_kv, GPUArray& cache, int num_heads, int position
) {
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_update_gqa: expected 3D tensors");
    }
    if (new_kv.shape()[0] != 1) {
        throw std::runtime_error("kv_cache_update_gqa: new_kv should have seq_len=1");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_update_gqa: dtype mismatch");
    }
    if (static_cast<int>(cache.shape()[0]) != num_heads) {
        throw std::runtime_error("kv_cache_update_gqa: cache shape[0] should equal num_heads");
    }

    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int max_seq_len = static_cast<int>(cache.shape()[1]);
    int total_elements = num_heads * head_dim;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_update_gqa_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, position);
            break;
        case DataType::BFloat16:
            nn::kv_cache_update_gqa_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, position);
            break;
        case DataType::Float32:
            nn::kv_cache_update_gqa_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, position);
            break;
        default:
            throw std::runtime_error("kv_cache_update_gqa: unsupported dtype");
    }

    sync_and_check("kv_cache_update_gqa kernel failed");
}

void kv_cache_update_gqa_ptr(
    const GPUArray& new_kv, GPUArray& cache, int num_heads, const GPUArray& position_buf
) {
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: expected 3D tensors");
    }
    if (new_kv.shape()[0] != 1) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: new_kv should have seq_len=1");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: dtype mismatch");
    }
    if (static_cast<int>(cache.shape()[0]) != num_heads) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: cache shape[0] should equal num_heads");
    }
    if (position_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("kv_cache_update_gqa_ptr: position_buf must be int32");
    }

    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int max_seq_len = static_cast<int>(cache.shape()[1]);
    int total_elements = num_heads * head_dim;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_update_gqa_f16_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len,
                static_cast<const int*>(position_buf.data()));
            break;
        case DataType::BFloat16:
            nn::kv_cache_update_gqa_bf16_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len,
                static_cast<const int*>(position_buf.data()));
            break;
        case DataType::Float32:
            nn::kv_cache_update_gqa_f32_kernel_ptr<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len,
                static_cast<const int*>(position_buf.data()));
            break;
        default:
            throw std::runtime_error("kv_cache_update_gqa_ptr: unsupported dtype");
    }

    sync_and_check("kv_cache_update_gqa_ptr kernel failed");
}

void kv_cache_prefill_gqa(
    const GPUArray& new_kv, GPUArray& cache, int num_heads, int start_pos
) {
    if (new_kv.ndim() != 3 || cache.ndim() != 3) {
        throw std::runtime_error("kv_cache_prefill_gqa: expected 3D tensors");
    }
    if (new_kv.dtype() != cache.dtype()) {
        throw std::runtime_error("kv_cache_prefill_gqa: dtype mismatch");
    }
    if (static_cast<int>(cache.shape()[0]) != num_heads) {
        throw std::runtime_error("kv_cache_prefill_gqa: cache shape[0] should equal num_heads");
    }

    int seq_len = static_cast<int>(new_kv.shape()[0]);
    int num_kv_heads = static_cast<int>(new_kv.shape()[1]);
    int head_dim = static_cast<int>(new_kv.shape()[2]);
    int max_seq_len = static_cast<int>(cache.shape()[1]);
    int total_elements = seq_len * num_heads * head_dim;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    cudaStream_t stream = internal::get_capture_stream();

    switch (new_kv.dtype()) {
        case DataType::Float16:
            nn::kv_cache_prefill_gqa_f16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __half*>(new_kv.data()),
                static_cast<__half*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, start_pos, seq_len);
            break;
        case DataType::BFloat16:
            nn::kv_cache_prefill_gqa_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(new_kv.data()),
                static_cast<__nv_bfloat16*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, start_pos, seq_len);
            break;
        case DataType::Float32:
            nn::kv_cache_prefill_gqa_f32_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(new_kv.data()),
                static_cast<float*>(cache.data()),
                num_heads, num_kv_heads, head_dim, max_seq_len, start_pos, seq_len);
            break;
        default:
            throw std::runtime_error("kv_cache_prefill_gqa: unsupported dtype");
    }

    sync_and_check("kv_cache_prefill_gqa kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
