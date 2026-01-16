/**
 * Scaled Dot-Product Attention (SDPA) with Causal Mask
 *
 * Supports:
 * - Standard SDPA (O(n^2) memory)
 * - Flash Attention 2 (O(n) memory, tiled computation)
 * - Flash Attention 3 (SM120+, MMA-based, warp specialization)
 * - Flash-Decoding (optimized for decode phase with q_len=1)
 */

#include "flash_attention_3.cuh"
#include "../../common/device.cuh"

namespace pygpukit {
namespace ops {

// =============================================================================
// Flash Attention 3 Environment Control
// =============================================================================

// PYGPUKIT_FA3: 0=off, 1=on (auto on SM120+), -1=auto (default)
static int get_fa3_mode() {
    static int cached = -999;
    if (cached == -999) {
        const char* env = std::getenv("PYGPUKIT_FA3");
        if (env) {
            cached = std::atoi(env);
        } else {
            cached = -1;  // Auto mode by default
        }
    }
    return cached;
}

// Check if FA3 should be used
static bool should_use_fa3(int head_dim, int seq_len) {
    int fa3_mode = get_fa3_mode();

    // Force off
    if (fa3_mode == 0) return false;

    // Check SM version (FA3 requires SM120+)
    static int sm_version = -1;
    if (sm_version == -1) {
        sm_version = ops::get_sm_version();
    }

    if (sm_version < 120) return false;

    // Currently only support head_dim=128
    if (head_dim != 128) return false;

    // Force on
    if (fa3_mode == 1) return true;

    // Auto mode: use FA3 for sequences > 256 on SM120+
    return seq_len > 256;
}

// Flash Attention mode:
// - "0" or "false": Always use standard SDPA
// - "1" or "true": Always use Flash Attention
// - "auto" or unset: Auto-select based on sequence length (>2048 uses Flash)
static int get_flash_attention_mode() {
    static int cached = -2;  // -2 = not checked, -1 = auto, 0 = off, 1 = on
    if (cached == -2) {
        const char* env = std::getenv("PYGPUKIT_FLASH_ATTENTION");
        if (env == nullptr || std::string(env) == "auto") {
            cached = -1;  // auto mode
        } else if (std::string(env) == "1" || std::string(env) == "true") {
            cached = 1;   // force on
        } else {
            cached = 0;   // force off
        }
    }
    return cached;
}

// Threshold for auto-selecting Flash Attention (sequence length)
constexpr int FLASH_ATTENTION_SEQ_THRESHOLD = 2048;

// Flash-Decoding workspace manager (lazy allocation, auto-expanding)
class FlashDecodingWorkspace {
public:
    static float* get(int n_heads, int head_dim, int kv_len) {
        static FlashDecodingWorkspace instance;
        size_t required = flash_decoding::flash_decoding_workspace_size(n_heads, head_dim, kv_len);
        if (required > instance.size_) {
            instance.resize(required);
        }
        return instance.buffer_;
    }

private:
    FlashDecodingWorkspace() : buffer_(nullptr), size_(0) {}

    ~FlashDecodingWorkspace() {
        if (buffer_) {
            device_free(buffer_);
        }
    }

    void resize(size_t new_size) {
        if (buffer_) {
            device_free(buffer_);
        }
        buffer_ = static_cast<float*>(device_malloc(new_size));
        size_ = new_size;
    }

    float* buffer_;
    size_t size_;
};

// Environment variable control for Flash-Decoding
// PYGPUKIT_FLASH_DECODING: 0=off, 1=on, -1=auto (default)
static int get_flash_decoding_mode() {
    static int cached = -999;
    if (cached == -999) {
        const char* env = std::getenv("PYGPUKIT_FLASH_DECODING");
        if (env) {
            cached = std::atoi(env);
        } else {
            cached = -1;  // Auto mode by default
        }
    }
    return cached;
}

// Internal helper for SDPA kernel dispatch
// context_len: if > 0, use this as kv_len (for fixed-length cache)
//              if <= 0, use K.shape()[1] as kv_len
static void sdpa_causal_dispatch(
    const GPUArray& Q, const GPUArray& K, const GPUArray& V,
    GPUArray& result, float scale, int context_len = 0
) {
    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];
    // kv_stride: actual K/V tensor size (for pointer calculations)
    int kv_stride = static_cast<int>(K.shape()[1]);
    // kv_len: number of KV positions to attend to (for masking)
    int kv_len = (context_len > 0) ? context_len : kv_stride;

    // Compute scale if not provided
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }

    // Causal offset for proper masking
    int causal_offset = kv_len - q_len;

    // Grid: one block per (head, query_position) pair
    dim3 grid(n_heads, q_len);
    int block_size = 128;  // Enough threads for reduction

    // Use capture stream if available
    cudaStream_t stream = internal::get_capture_stream();

    // Flash-Decoding: Optimized for decode phase (q_len=1)
    // Parallelizes over KV sequence length for better GPU utilization
    int flash_decoding_mode = get_flash_decoding_mode();
    bool use_flash_decoding = false;
    if (q_len == 1 && head_dim <= 128) {
        if (flash_decoding_mode == 1) {
            // Force on
            use_flash_decoding = true;
        } else if (flash_decoding_mode == -1) {
            // Auto: use Flash-Decoding when it provides benefit
            // Crossover point is around kv_len=1024 (4 chunks with chunk_size=256)
            // Only enable for long contexts where parallelism benefit > kernel launch overhead
            use_flash_decoding = (kv_len >= 1024);
        }
    }

    if (use_flash_decoding) {
        // Flash-Decoding: chunk-parallel attention for decode phase
        float* workspace = FlashDecodingWorkspace::get(n_heads, head_dim, kv_len);

        switch (Q.dtype()) {
            case DataType::Float16:
                flash_decoding::flash_decoding_f16(
                    static_cast<const __half*>(Q.data()),
                    static_cast<const __half*>(K.data()),
                    static_cast<const __half*>(V.data()),
                    static_cast<__half*>(result.data()),
                    workspace,
                    n_heads, head_dim, kv_len, kv_stride, stream
                );
                return;
            default:
                // Fall through to standard SDPA for unsupported dtypes
                break;
        }
    }

    // =========================================================================
    // Flash Attention 3 (SM120+, MMA-based with warp specialization)
    // =========================================================================
    // FA3 uses 4D layout [batch, num_heads, seq, head_dim]
    // Current SDPA uses 3D layout [n_heads, seq, head_dim]
    // Treat as batch_size=1 for compatibility
    if (should_use_fa3(head_dim, kv_len)) {
        cudaError_t err = cudaSuccess;

        switch (Q.dtype()) {
            case DataType::BFloat16:
                err = nn::fa3::launch_flash_attention_3<__nv_bfloat16>(
                    static_cast<const __nv_bfloat16*>(Q.data()),
                    static_cast<const __nv_bfloat16*>(K.data()),
                    static_cast<const __nv_bfloat16*>(V.data()),
                    static_cast<__nv_bfloat16*>(result.data()),
                    1,              // batch_size = 1
                    n_heads,
                    q_len,
                    kv_len,
                    head_dim,
                    scale,
                    true,           // causal = true
                    stream
                );
                if (err == cudaSuccess) return;
                // Fall through if FA3 launch failed
                break;

            case DataType::Float16:
                // TODO: Add FP16 support when implemented
                break;

            default:
                // FA3 only supports BF16/FP16, fall through to FA2/SDPA
                break;
        }
    }

    // Determine whether to use Flash Attention
    // - Auto mode: use Flash for long sequences (>2048) where memory savings matter
    // - Force mode: respect user preference
    int flash_mode = get_flash_attention_mode();
    bool use_flash = false;
    if (flash_mode == 1) {
        // Force on
        use_flash = (head_dim <= 128);
    } else if (flash_mode == -1) {
        // Auto: use Flash for long sequences
        use_flash = (head_dim <= 128) && (kv_len > FLASH_ATTENTION_SEQ_THRESHOLD);
    }
    // flash_mode == 0: force off, use_flash stays false

    if (use_flash) {
        // Flash Attention 2: O(n) memory, tiled computation
        size_t shared_mem_size = nn::flash_attention_smem_size(head_dim);

        switch (Q.dtype()) {
            case DataType::Float32:
                nn::flash_attention_f32_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const float*>(Q.data()),
                    static_cast<const float*>(K.data()),
                    static_cast<const float*>(V.data()),
                    static_cast<float*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            case DataType::Float16:
                nn::flash_attention_f16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __half*>(Q.data()),
                    static_cast<const __half*>(K.data()),
                    static_cast<const __half*>(V.data()),
                    static_cast<__half*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            case DataType::BFloat16:
                nn::flash_attention_bf16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __nv_bfloat16*>(Q.data()),
                    static_cast<const __nv_bfloat16*>(K.data()),
                    static_cast<const __nv_bfloat16*>(V.data()),
                    static_cast<__nv_bfloat16*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            default:
                throw std::runtime_error("sdpa only supports Float32, Float16, BFloat16");
        }
    } else {
        // Standard SDPA: O(n^2) memory for attention scores
        size_t shared_mem_size = kv_len * sizeof(float);

        switch (Q.dtype()) {
            case DataType::Float32:
                nn::sdpa_causal_f32_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const float*>(Q.data()),
                    static_cast<const float*>(K.data()),
                    static_cast<const float*>(V.data()),
                    static_cast<float*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            case DataType::Float16:
                nn::sdpa_causal_f16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __half*>(Q.data()),
                    static_cast<const __half*>(K.data()),
                    static_cast<const __half*>(V.data()),
                    static_cast<__half*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            case DataType::BFloat16:
                nn::sdpa_causal_bf16_kernel<<<grid, block_size, shared_mem_size, stream>>>(
                    static_cast<const __nv_bfloat16*>(Q.data()),
                    static_cast<const __nv_bfloat16*>(K.data()),
                    static_cast<const __nv_bfloat16*>(V.data()),
                    static_cast<__nv_bfloat16*>(result.data()),
                    n_heads, q_len, kv_len, kv_stride, head_dim, scale, causal_offset);
                break;
            default:
                throw std::runtime_error("sdpa only supports Float32, Float16, BFloat16");
        }
    }
}

GPUArray sdpa_causal(const GPUArray& Q, const GPUArray& K, const GPUArray& V, float scale) {
    // Q: [n_heads, q_len, head_dim]
    // K: [n_heads, kv_len, head_dim]
    // V: [n_heads, kv_len, head_dim]
    // Output: [n_heads, q_len, head_dim]

    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3) {
        throw std::runtime_error("sdpa expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype()) {
        throw std::runtime_error("sdpa: dtype mismatch");
    }

    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];

    if (K.shape()[0] != n_heads || V.shape()[0] != n_heads) {
        throw std::runtime_error("sdpa: n_heads mismatch");
    }
    if (K.shape()[2] != head_dim || V.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: head_dim mismatch");
    }
    if (K.shape()[1] != V.shape()[1]) {
        throw std::runtime_error("sdpa: K and V seq_len mismatch");
    }

    GPUArray result({(size_t)n_heads, (size_t)q_len, (size_t)head_dim}, Q.dtype());
    sdpa_causal_dispatch(Q, K, V, result, scale);
    sync_and_check("sdpa kernel failed");
    return result;
}

// SDPA with output buffer (for CUDA Graph capture)
void sdpa_causal(const GPUArray& Q, const GPUArray& K, const GPUArray& V, GPUArray& out, float scale) {
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("sdpa expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype() || Q.dtype() != out.dtype()) {
        throw std::runtime_error("sdpa: dtype mismatch");
    }

    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];

    if (K.shape()[0] != n_heads || V.shape()[0] != n_heads) {
        throw std::runtime_error("sdpa: n_heads mismatch");
    }
    if (K.shape()[2] != head_dim || V.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: head_dim mismatch");
    }
    if (K.shape()[1] != V.shape()[1]) {
        throw std::runtime_error("sdpa: K and V seq_len mismatch");
    }
    if (out.shape()[0] != n_heads || out.shape()[1] != q_len || out.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: output shape mismatch");
    }

    sdpa_causal_dispatch(Q, K, V, out, scale);
    sync_and_check("sdpa kernel failed");
}

// SDPA with fixed-length KV cache support
// context_len: actual number of valid tokens in KV cache (K/V may have max_seq_len)
void sdpa_causal_fixed_cache(
    const GPUArray& Q, const GPUArray& K, const GPUArray& V,
    GPUArray& out, int context_len, float scale
) {
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("sdpa expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype() || Q.dtype() != out.dtype()) {
        throw std::runtime_error("sdpa: dtype mismatch");
    }

    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];

    if (K.shape()[0] != n_heads || V.shape()[0] != n_heads) {
        throw std::runtime_error("sdpa: n_heads mismatch");
    }
    if (K.shape()[2] != head_dim || V.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: head_dim mismatch");
    }
    if (K.shape()[1] != V.shape()[1]) {
        throw std::runtime_error("sdpa: K and V seq_len mismatch");
    }
    if (out.shape()[0] != n_heads || out.shape()[1] != q_len || out.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: output shape mismatch");
    }
    if (context_len <= 0 || context_len > static_cast<int>(K.shape()[1])) {
        throw std::runtime_error("sdpa: invalid context_len");
    }

    sdpa_causal_dispatch(Q, K, V, out, scale, context_len);
    sync_and_check("sdpa kernel failed");
}

// SDPA with fixed-length KV cache using pointer-based context_len (for CUDA Graph)
// context_len_buf: GPU buffer containing actual context_len (read at runtime)
// max_kv_len: Maximum KV length (for shared memory allocation during graph capture)
void sdpa_causal_fixed_cache_ptr(
    const GPUArray& Q, const GPUArray& K, const GPUArray& V,
    GPUArray& out, const GPUArray& context_len_buf, int max_kv_len, float scale
) {
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("sdpa expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype() || Q.dtype() != out.dtype()) {
        throw std::runtime_error("sdpa: dtype mismatch");
    }
    if (context_len_buf.dtype() != DataType::Int32) {
        throw std::runtime_error("sdpa: context_len_buf must be int32");
    }

    int n_heads = Q.shape()[0];
    int q_len = Q.shape()[1];
    int head_dim = Q.shape()[2];
    int kv_stride = static_cast<int>(K.shape()[1]);

    if (K.shape()[0] != n_heads || V.shape()[0] != n_heads) {
        throw std::runtime_error("sdpa: n_heads mismatch");
    }
    if (K.shape()[2] != head_dim || V.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: head_dim mismatch");
    }
    if (K.shape()[1] != V.shape()[1]) {
        throw std::runtime_error("sdpa: K and V seq_len mismatch");
    }
    if (out.shape()[0] != n_heads || out.shape()[1] != q_len || out.shape()[2] != head_dim) {
        throw std::runtime_error("sdpa: output shape mismatch");
    }
    if (max_kv_len <= 0 || max_kv_len > kv_stride) {
        throw std::runtime_error("sdpa: invalid max_kv_len");
    }

    // Compute scale if not provided
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }

    // Grid: one block per (head, query_position) pair
    dim3 grid(n_heads, q_len);
    int block_size = 128;

    // Allocate shared memory for max_kv_len (allows dynamic context_len at runtime)
    size_t shared_mem_size = max_kv_len * sizeof(float);

    cudaStream_t stream = internal::get_capture_stream();

    switch (Q.dtype()) {
        case DataType::Float32:
            nn::sdpa_causal_f32_kernel_ptr<<<grid, block_size, shared_mem_size, stream>>>(
                static_cast<const float*>(Q.data()),
                static_cast<const float*>(K.data()),
                static_cast<const float*>(V.data()),
                static_cast<float*>(out.data()),
                static_cast<const int*>(context_len_buf.data()),
                n_heads, q_len, kv_stride, head_dim, scale);
            break;
        case DataType::Float16:
            nn::sdpa_causal_f16_kernel_ptr<<<grid, block_size, shared_mem_size, stream>>>(
                static_cast<const __half*>(Q.data()),
                static_cast<const __half*>(K.data()),
                static_cast<const __half*>(V.data()),
                static_cast<__half*>(out.data()),
                static_cast<const int*>(context_len_buf.data()),
                n_heads, q_len, kv_stride, head_dim, scale);
            break;
        case DataType::BFloat16:
            nn::sdpa_causal_bf16_kernel_ptr<<<grid, block_size, shared_mem_size, stream>>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                static_cast<const int*>(context_len_buf.data()),
                n_heads, q_len, kv_stride, head_dim, scale);
            break;
        default:
            throw std::runtime_error("sdpa: unsupported dtype");
    }

    sync_and_check("sdpa_causal_fixed_cache_ptr kernel failed");
}

}  // namespace ops
}  // namespace pygpukit
