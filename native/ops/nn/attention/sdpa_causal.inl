/**
 * Scaled Dot-Product Attention (SDPA) with Causal Mask
 *
 * Supports:
 * - Standard SDPA (O(n^2) memory)
 * - Flash Attention 2 (O(n) memory, tiled computation)
 * - Flash Attention 3 (SM120+, MMA-based, warp specialization)
 * - Flash Attention 3 TMA (SM90+, async TMA loading)
 * - Flash-Decoding (optimized for decode phase with q_len=1)
 */

#include "flash_attention_3.cuh"
#include "flash_attention_3_tma.cuh"
#include "flash_attention_3_sm120.cuh"
#include "flash_attention_3_fp8_sm120.cuh"
#include "../../matmul/gemm/fp8_block_scale/test_mma_direct.cuh"
#include "../../common/device.cuh"
#include "../../common/tma_utils.cuh"
#include "../../common/tma_descriptor_cache.cuh"
#include <type_traits>

namespace pygpukit {
namespace ops {

// =============================================================================
// Flash Attention 3 Environment Control
// =============================================================================

// PYGPUKIT_FA3_SM120_VERSION: 0-6 to select SM120 config version
// 0 = baseline (TILE_Q=32, TILE_KV=64, 2-stage, 4+8 warps)
// 1 = TILE_Q=64
// 2 = 3-stage pipeline
// 3 = 2+10 warps
// 4 = 4+12 warps (best for stability)
// 5 = 2+14 warps (minimal producers)
// 6 = 4+16 warps (max consumers)
static int get_fa3_sm120_version() {
    static int cached = -1;
    if (cached == -1) {
        const char* env = std::getenv("PYGPUKIT_FA3_SM120_VERSION");
        if (env) {
            cached = std::atoi(env);
            if (cached < 0 || cached > 6) {
                fprintf(stderr, "[FA3 SM120] Invalid version %d, using 0\n", cached);
                cached = 0;
            }
        } else {
            cached = 0;  // Default: baseline
        }
    }
    return cached;
}

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

// PYGPUKIT_FA3_TMA: 0=off, 1=on (use TMA variant), -1=auto (default: on for SM90+)
static int get_fa3_tma_mode() {
    static int cached = -999;
    if (cached == -999) {
        const char* env = std::getenv("PYGPUKIT_FA3_TMA");
        if (env) {
            cached = std::atoi(env);
        } else {
            cached = -1;  // Auto mode by default
        }
    }
    return cached;
}

// Check if FA3 TMA should be used
static bool should_use_fa3_tma(int head_dim, int seq_len) {
    int tma_mode = get_fa3_tma_mode();

    // Force off
    if (tma_mode == 0) return false;

    // Check SM version (TMA requires SM90+)
    static int sm_version = -1;
    if (sm_version == -1) {
        sm_version = ops::get_sm_version();
    }

    if (sm_version < 90) return false;

    // Currently only support head_dim=128
    if (head_dim != 128) return false;

    // Force on
    if (tma_mode == 1) return true;

    // Auto mode: use TMA for sequences > 512 on SM90+
    return seq_len > 512;
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

// =============================================================================
// FA3 TMA Launcher (SM120 Version Dispatch)
// =============================================================================

/**
 * Inner launcher template - uses specific SM120 config version.
 */
template<typename Element, typename Config>
static cudaError_t try_launch_fa3_tma_impl(
    const Element* Q,
    const Element* K,
    const Element* V,
    Element* output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream
) {
    // Only support BF16 for now
    if constexpr (!std::is_same_v<Element, __nv_bfloat16>) {
        return cudaErrorNotSupported;
    }

    auto& cache = tma::TmaDescriptorCache::instance();

    CUtensorMap* d_q_desc = nullptr;
    CUtensorMap* d_k_desc = nullptr;
    CUtensorMap* d_v_desc = nullptr;

    // Q: [num_heads, seq_q, head_dim]
    d_q_desc = cache.get_or_create_3d_bf16(
        const_cast<Element*>(Q),
        head_dim,
        seq_q,
        num_heads,
        head_dim,
        seq_q * head_dim,
        Config::HEAD_DIM,
        Config::TILE_Q,
        tma::SwizzleMode::None,
        stream
    );
    if (!d_q_desc) return cudaErrorUnknown;

    // K: [num_heads, seq_kv, head_dim]
    d_k_desc = cache.get_or_create_3d_bf16(
        const_cast<Element*>(K),
        head_dim,
        seq_kv,
        num_heads,
        head_dim,
        seq_kv * head_dim,
        Config::HEAD_DIM,
        Config::TILE_KV,
        tma::SwizzleMode::None,
        stream
    );
    if (!d_k_desc) return cudaErrorUnknown;

    // V: [num_heads, seq_kv, head_dim]
    d_v_desc = cache.get_or_create_3d_bf16(
        const_cast<Element*>(V),
        head_dim,
        seq_kv,
        num_heads,
        head_dim,
        seq_kv * head_dim,
        Config::HEAD_DIM,
        Config::TILE_KV,
        tma::SwizzleMode::None,
        stream
    );
    if (!d_v_desc) return cudaErrorUnknown;

    // Use SM120 namespace for versioned launch
    return nn::fa3_sm120::launch_flash_attention_3_tma_cached<Config>(
        d_q_desc, d_k_desc, d_v_desc, output,
        batch_size, num_heads, seq_q, seq_kv,
        scale, causal, stream
    );
}

/**
 * Try to launch FA3 with TMA (cached descriptors).
 * Uses TMA descriptor cache to avoid per-call overhead.
 * Dispatches to SM120 config version based on environment variable.
 *
 * Returns cudaSuccess if TMA launch succeeded, error code otherwise.
 */
template<typename Element>
static cudaError_t try_launch_fa3_tma(
    const Element* Q,
    const Element* K,
    const Element* V,
    Element* output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream
) {
    using namespace nn::fa3_sm120;

    // Dispatch based on SM120 version
    int version = get_fa3_sm120_version();

    switch (version) {
        case 1:
            return try_launch_fa3_tma_impl<Element, SM120Config<1>>(
                Q, K, V, output, batch_size, num_heads, seq_q, seq_kv,
                head_dim, scale, causal, stream);
        case 2:
            return try_launch_fa3_tma_impl<Element, SM120Config<2>>(
                Q, K, V, output, batch_size, num_heads, seq_q, seq_kv,
                head_dim, scale, causal, stream);
        case 3:
            return try_launch_fa3_tma_impl<Element, SM120Config<3>>(
                Q, K, V, output, batch_size, num_heads, seq_q, seq_kv,
                head_dim, scale, causal, stream);
        case 4:
            return try_launch_fa3_tma_impl<Element, SM120Config<4>>(
                Q, K, V, output, batch_size, num_heads, seq_q, seq_kv,
                head_dim, scale, causal, stream);
        case 5:
            return try_launch_fa3_tma_impl<Element, SM120Config<5>>(
                Q, K, V, output, batch_size, num_heads, seq_q, seq_kv,
                head_dim, scale, causal, stream);
        case 6:
            return try_launch_fa3_tma_impl<Element, SM120Config<6>>(
                Q, K, V, output, batch_size, num_heads, seq_q, seq_kv,
                head_dim, scale, causal, stream);
        default:  // version 0 or unknown
            return try_launch_fa3_tma_impl<Element, SM120Config<0>>(
                Q, K, V, output, batch_size, num_heads, seq_q, seq_kv,
                head_dim, scale, causal, stream);
    }
}

// Legacy compatibility - keep the old code path for non-SM120 dispatch
template<typename Element>
static cudaError_t try_launch_fa3_tma_legacy(
    const Element* Q,
    const Element* K,
    const Element* V,
    Element* output,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_kv,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream
) {
    using namespace nn::fa3::tma_kernel;
    using Config = TmaFA3Config<120>;

    // Only support BF16 for now
    if constexpr (!std::is_same_v<Element, __nv_bfloat16>) {
        return cudaErrorNotSupported;
    }

    // Get cached TMA descriptors (device pointers)
    // Cache key: (base_ptr, dimensions, strides, tile sizes, swizzle)
    // On cache miss: creates host descriptor, allocates device memory, copies once
    // On cache hit: returns existing device pointer immediately

    auto& cache = tma::TmaDescriptorCache::instance();

    CUtensorMap* d_q_desc = nullptr;
    CUtensorMap* d_k_desc = nullptr;
    CUtensorMap* d_v_desc = nullptr;

    // Q: [num_heads, seq_q, head_dim]
    d_q_desc = cache.get_or_create_3d_bf16(
        const_cast<Element*>(Q),
        head_dim,                 // dim0: head_dim
        seq_q,                    // dim1: seq_q
        num_heads,                // dim2: num_heads
        head_dim,                 // stride1: elements between seq positions
        seq_q * head_dim,         // stride2: elements between heads
        Config::HEAD_DIM,         // tile0: full head_dim
        Config::TILE_Q,           // tile1: Q tile size
        tma::SwizzleMode::None,
        stream
    );
    if (!d_q_desc) {
        return cudaErrorUnknown;
    }

    // K: [num_heads, seq_kv, head_dim]
    d_k_desc = cache.get_or_create_3d_bf16(
        const_cast<Element*>(K),
        head_dim,
        seq_kv,
        num_heads,
        head_dim,
        seq_kv * head_dim,
        Config::HEAD_DIM,
        Config::TILE_KV,
        tma::SwizzleMode::None,
        stream
    );
    if (!d_k_desc) {
        return cudaErrorUnknown;
    }

    // V: [num_heads, seq_kv, head_dim]
    d_v_desc = cache.get_or_create_3d_bf16(
        const_cast<Element*>(V),
        head_dim,
        seq_kv,
        num_heads,
        head_dim,
        seq_kv * head_dim,
        Config::HEAD_DIM,
        Config::TILE_KV,
        tma::SwizzleMode::None,
        stream
    );
    if (!d_v_desc) {
        return cudaErrorUnknown;
    }

    // Launch TMA kernel with cached device descriptors
    return launch_flash_attention_3_tma_cached<Config>(
        d_q_desc,
        d_k_desc,
        d_v_desc,
        output,
        batch_size,
        num_heads,
        seq_q,
        seq_kv,
        scale,
        causal,
        stream
    );
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
    // Flash Attention 3 TMA (SM90+, async TMA loading)
    // =========================================================================
    // Try TMA variant first if enabled and supported
    if (should_use_fa3_tma(head_dim, kv_len)) {
        cudaError_t err = cudaSuccess;

        switch (Q.dtype()) {
            case DataType::BFloat16:
                err = try_launch_fa3_tma<__nv_bfloat16>(
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
                if (err == cudaSuccess) {
                    return;
                }
                // Fall through if TMA launch failed
                break;

            default:
                // TMA only supports BF16 for now
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

// =============================================================================
// Timed SDPA (for benchmarking kernel-only time)
// =============================================================================

/**
 * Inner timed launcher template - uses specific SM120 config version.
 */
template<typename Config>
static cudaError_t try_launch_fa3_tma_timed_impl(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* output,
    int n_heads,
    int seq_q,
    int seq_kv,
    int head_dim,
    float scale,
    float* kernel_time_us
) {
    auto& cache = tma::TmaDescriptorCache::instance();

    CUtensorMap* d_q_desc = cache.get_or_create_3d_bf16(
        const_cast<__nv_bfloat16*>(Q),
        head_dim, seq_q, n_heads,
        head_dim, seq_q * head_dim,
        Config::HEAD_DIM, Config::TILE_Q,
        tma::SwizzleMode::None, nullptr
    );
    CUtensorMap* d_k_desc = cache.get_or_create_3d_bf16(
        const_cast<__nv_bfloat16*>(K),
        head_dim, seq_kv, n_heads,
        head_dim, seq_kv * head_dim,
        Config::HEAD_DIM, Config::TILE_KV,
        tma::SwizzleMode::None, nullptr
    );
    CUtensorMap* d_v_desc = cache.get_or_create_3d_bf16(
        const_cast<__nv_bfloat16*>(V),
        head_dim, seq_kv, n_heads,
        head_dim, seq_kv * head_dim,
        Config::HEAD_DIM, Config::TILE_KV,
        tma::SwizzleMode::None, nullptr
    );

    if (!d_q_desc || !d_k_desc || !d_v_desc) {
        return cudaErrorUnknown;
    }

    return nn::fa3_sm120::launch_flash_attention_3_tma_timed<Config>(
        d_q_desc, d_k_desc, d_v_desc, output,
        1,  // batch_size
        n_heads, seq_q, seq_kv,
        scale, true,  // causal
        nullptr,  // default stream
        kernel_time_us
    );
}

void sdpa_causal_timed(
    const GPUArray& Q, const GPUArray& K, const GPUArray& V,
    GPUArray& out, float scale, float* kernel_time_us
) {
    using namespace nn::fa3_sm120;

    // Validate inputs
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("sdpa_causal_timed expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sdpa_causal_timed only supports BFloat16 (for FA3 TMA)");
    }

    int n_heads = Q.shape()[0];
    int seq_q = Q.shape()[1];
    int seq_kv = K.shape()[1];
    int head_dim = Q.shape()[2];

    // Check SM version
    int sm = ops::get_sm_version();
    if (sm < 90) {
        throw std::runtime_error("sdpa_causal_timed requires SM90+ (TMA support)");
    }

    // Compute scale if not provided
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }

    // Dispatch based on SM120 version
    int version = get_fa3_sm120_version();
    cudaError_t err = cudaSuccess;

    switch (version) {
        case 1:
            err = try_launch_fa3_tma_timed_impl<SM120Config<1>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                n_heads, seq_q, seq_kv, head_dim, scale, kernel_time_us);
            break;
        case 2:
            err = try_launch_fa3_tma_timed_impl<SM120Config<2>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                n_heads, seq_q, seq_kv, head_dim, scale, kernel_time_us);
            break;
        case 3:
            err = try_launch_fa3_tma_timed_impl<SM120Config<3>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                n_heads, seq_q, seq_kv, head_dim, scale, kernel_time_us);
            break;
        case 4:
            err = try_launch_fa3_tma_timed_impl<SM120Config<4>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                n_heads, seq_q, seq_kv, head_dim, scale, kernel_time_us);
            break;
        case 5:
            err = try_launch_fa3_tma_timed_impl<SM120Config<5>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                n_heads, seq_q, seq_kv, head_dim, scale, kernel_time_us);
            break;
        case 6:
            err = try_launch_fa3_tma_timed_impl<SM120Config<6>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                n_heads, seq_q, seq_kv, head_dim, scale, kernel_time_us);
            break;
        default:  // version 0 or unknown
            err = try_launch_fa3_tma_timed_impl<SM120Config<0>>(
                static_cast<const __nv_bfloat16*>(Q.data()),
                static_cast<const __nv_bfloat16*>(K.data()),
                static_cast<const __nv_bfloat16*>(V.data()),
                static_cast<__nv_bfloat16*>(out.data()),
                n_heads, seq_q, seq_kv, head_dim, scale, kernel_time_us);
            break;
    }

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("sdpa_causal_timed failed: ") + cudaGetErrorString(err));
    }
}

// =============================================================================
// TMA Cache Utilities
// =============================================================================

void print_tma_cache_stats() {
    tma::TmaDescriptorCache::instance().print_stats();
}

void clear_tma_cache() {
    tma::TmaDescriptorCache::instance().clear();
}

// =============================================================================
// FA3 FP8 (SM120+, FP8 Q@K^T with block-scale MMA)
// =============================================================================

/**
 * Check if FA3 FP8 is available on current device.
 */
bool fa3_fp8_available() {
    int sm = ops::get_sm_version();
    return sm >= 120;
}

/**
 * FA3 FP8: FP8 Q@K^T with block-scale MMA, BF16 P@V
 *
 * Input Q, K, V are BF16 (auto-quantized to FP8 internally).
 * ~50% memory bandwidth reduction for Q, K.
 * ~0.25% expected error vs BF16 FA3.
 */
void sdpa_causal_fp8(
    const GPUArray& Q, const GPUArray& K, const GPUArray& V,
    GPUArray& out, float scale
) {
    // Validate inputs
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3 || out.ndim() != 3) {
        throw std::runtime_error("sdpa_causal_fp8 expects 3D inputs [n_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != DataType::BFloat16) {
        throw std::runtime_error("sdpa_causal_fp8 only supports BFloat16 input");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype() || Q.dtype() != out.dtype()) {
        throw std::runtime_error("sdpa_causal_fp8: dtype mismatch");
    }

    int n_heads = Q.shape()[0];
    int seq_q = Q.shape()[1];
    int seq_kv = K.shape()[1];
    int head_dim = Q.shape()[2];

    if (head_dim != 128) {
        throw std::runtime_error("sdpa_causal_fp8 only supports head_dim=128");
    }

    // Check SM version
    int sm = ops::get_sm_version();
    if (sm < 120) {
        throw std::runtime_error("sdpa_causal_fp8 requires SM120+");
    }

    // Compute scale if not provided
    if (scale <= 0.0f) {
        scale = 1.0f / sqrtf((float)head_dim);
    }

    cudaStream_t stream = internal::get_capture_stream();

    cudaError_t err = nn::fa3_fp8_sm120::flash_attention_3_fp8_sm120<>(
        static_cast<const __nv_bfloat16*>(Q.data()),
        static_cast<const __nv_bfloat16*>(K.data()),
        static_cast<const __nv_bfloat16*>(V.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        1,  // batch_size
        n_heads,
        seq_q,
        seq_kv,
        scale,
        true,  // causal
        stream
    );

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("sdpa_causal_fp8 failed: ") + cudaGetErrorString(err));
    }

    sync_and_check("sdpa_causal_fp8 kernel failed");
}

/**
 * Test FP8 MMA directly to debug C fragment layout.
 */
void test_fp8_mma_direct() {
#if __CUDACC_VER_MAJOR__ >= 13
    int sm = ops::get_sm_version();
    if (sm < 120) {
        fprintf(stderr, "test_fp8_mma_direct requires SM120+\n");
        return;
    }
    matmul::fp8_mma_test::test_mma_direct();
#else
    fprintf(stderr, "test_fp8_mma_direct requires CUDA 13.x+\n");
#endif
}

}  // namespace ops
}  // namespace pygpukit
