/**
 * Matrix multiplication dispatch
 */
#include "gemm/f32/f32/generic/f32_naive.cuh"
#include "../common/error.cuh"
#include "../common/device.cuh"
#include "../../core/memory.hpp"
#include "../../core/cuda_graph.hpp"
#include "../ops.cuh"  // For transpose()

// Include existing optimized kernels
#include "gemm/f32/f32/generic/f32_ampere.cuh"
#include "gemm/f32/f32/generic/tf32_wmma.cuh"
#include "gemm/f32/f32/generic/tf32_mma.cuh"
#include "gemm/bf16/bf16/generic/bf16_naive.cuh"
#include "gemm/bf16/bf16/generic/bf16_wmma.cuh"
#include "gemm/bf16/bf16/generic/bf16_wmma_generic.cuh"
#include "cublaslt.cuh"
#include "gemm/bf16/bf16/sm80/bf16_cutlass.cuh"

#include <cstdlib>
#include <algorithm>

// CUTLASS GEMM (extern declarations from matmul_cutlass.cu)
extern "C" {
    cudaError_t cutlass_gemm_tf32(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);
    cudaError_t cutlass_gemm_fp16(const __half* A, const __half* B, __half* C, int M, int N, int K, cudaStream_t stream);
    cudaError_t cutlass_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K, cudaStream_t stream);
    bool cutlass_is_compatible(int M, int N, int K);
    bool cutlass_is_sm_supported();

    // BiasGELU fused operations
    cudaError_t cutlass_gemm_tf32_bias_gelu(const float* A, const float* B, const float* bias, float* D, int M, int N, int K, cudaStream_t stream);
    cudaError_t cutlass_gemm_fp16_bias_gelu(const __half* A, const __half* B, const __half* bias, __half* D, int M, int N, int K, cudaStream_t stream);
    cudaError_t cutlass_gemm_bf16_bias_gelu(const __nv_bfloat16* A, const __nv_bfloat16* B, const __nv_bfloat16* bias, __nv_bfloat16* D, int M, int N, int K, cudaStream_t stream);
}

namespace pygpukit {
namespace ops {

// Thresholds for kernel selection
constexpr int TILED_MATMUL_THRESHOLD = 128;
constexpr int OPTIMIZED_MATMUL_THRESHOLD = 128;

void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    if (c.shape()[0] != M || c.shape()[1] != N) {
        throw std::runtime_error("matmul output shape mismatch");
    }

    // v0.2.6: CUTLASS is the default backend
    // Environment variables:
    //   PYGPUKIT_NO_CUTLASS=1  - Disable CUTLASS entirely, use native kernels
    //   PYGPUKIT_NO_TF32=1     - Disable TF32 for FP32 inputs (use native FP32 kernel)
    const char* no_cutlass_env = std::getenv("PYGPUKIT_NO_CUTLASS");
    const char* no_tf32_env = std::getenv("PYGPUKIT_NO_TF32");

    bool cutlass_disabled = no_cutlass_env &&
        (no_cutlass_env[0] == '1' || no_cutlass_env[0] == 'y' || no_cutlass_env[0] == 'Y');
    bool tf32_disabled = no_tf32_env &&
        (no_tf32_env[0] == '1' || no_tf32_env[0] == 'y' || no_tf32_env[0] == 'Y');

    // CUTLASS enabled by default if dimensions are compatible AND SM >= 86
    // v0.2.7+ requires SM >= 86 (RTX 30xx and newer)
    // For FP32: skip CUTLASS TF32 if NO_TF32 is set (will use native FP32 kernel)
    bool cutlass_enabled = !cutlass_disabled && cutlass_is_compatible(M, N, K) && cutlass_is_sm_supported();
    bool cutlass_tf32_enabled = cutlass_enabled && !tf32_disabled;

    // Fallback to native TensorCore kernels
    bool tf32_enabled = false;
    bool fp16_tc_enabled = false;
    int sm_version = 0;

    // Only check native TensorCore settings if CUTLASS is disabled
    if (!cutlass_enabled) {
        sm_version = get_sm_version();
        const char* tf32_env = std::getenv("PYGPUKIT_ALLOW_TF32");
        const char* fp16_tc_env = std::getenv("PYGPUKIT_ALLOW_FP16_TC");

        // On SM 120+ where CUTLASS doesn't work, automatically enable TF32 TensorCore
        // This provides good performance fallback for Blackwell GeForce (RTX 5090)
        bool auto_tf32 = (sm_version >= 120);

        if (auto_tf32 || (tf32_env && (tf32_env[0] == '1' || tf32_env[0] == 'y' || tf32_env[0] == 'Y'))) {
            tf32_enabled = (sm_version >= MIN_SM_VERSION);
        }

        if ((fp16_tc_env && (fp16_tc_env[0] == '1' || fp16_tc_env[0] == 'y' || fp16_tc_env[0] == 'Y'))) {
            fp16_tc_enabled = (sm_version >= MIN_SM_VERSION);
        }
    }

    // Kernel selection
    bool use_tf32 = tf32_enabled &&
                    (a.dtype() == DataType::Float32) &&
                    ((M >= OPTIMIZED_MATMUL_THRESHOLD &&
                      N >= OPTIMIZED_MATMUL_THRESHOLD &&
                      K >= OPTIMIZED_MATMUL_THRESHOLD) ||
                     (M == 16 && (N == 8 || N == 16)));

    bool use_fp16_tc_fast = fp16_tc_enabled &&
                            (a.dtype() == DataType::Float16 || a.dtype() == DataType::BFloat16) &&
                            (M >= 128 && N >= 128 && K >= 32) &&
                            (M % 128 == 0 && N % 128 == 0 && K % 32 == 0);

    bool use_fp16_tc_generic = !use_fp16_tc_fast && fp16_tc_enabled &&
                               (a.dtype() == DataType::Float16 || a.dtype() == DataType::BFloat16) &&
                               (M >= 16 && N >= 16 && K >= 8) &&
                               (K % 8 == 0);

    bool use_optimized = !use_tf32 && !use_fp16_tc_fast && !use_fp16_tc_generic &&
                         (a.dtype() == DataType::Float32) &&
                         (M >= OPTIMIZED_MATMUL_THRESHOLD ||
                          N >= OPTIMIZED_MATMUL_THRESHOLD ||
                          K >= OPTIMIZED_MATMUL_THRESHOLD);

    bool use_tiled = !use_optimized && !use_tf32 && !use_fp16_tc_fast && !use_fp16_tc_generic &&
                     (M >= TILED_MATMUL_THRESHOLD ||
                      N >= TILED_MATMUL_THRESHOLD ||
                      K >= TILED_MATMUL_THRESHOLD);

    // cuBLASLt for small M (batch size) where CUTLASS is not compatible
    // Cache environment variable and availability check for performance
    static bool cublaslt_checked = false;
    static bool cublaslt_available = false;
    if (!cublaslt_checked) {
        const char* no_cublaslt_env = std::getenv("PYGPUKIT_NO_CUBLASLT");
        bool cublaslt_disabled = no_cublaslt_env &&
            (no_cublaslt_env[0] == '1' || no_cublaslt_env[0] == 'y' || no_cublaslt_env[0] == 'Y');
        cublaslt_available = !cublaslt_disabled && cublaslt_gemm::is_available();
        cublaslt_checked = true;
    }

    // Get current stream (capture stream if in CUDA Graph mode, otherwise nullptr for default)
    cudaStream_t stream = internal::get_capture_stream();

    // Use cuBLASLt for small M (< 16) or when CUTLASS is not compatible
    bool use_cublaslt = cublaslt_available &&
                        (M < 16 || !cutlass_is_compatible(M, N, K));

    // cuBLASLt dispatch (for small batch sizes and CUTLASS-incompatible dimensions)
    // Note: cuBLASLt may fail on some CUDA versions, fall back to native kernels in that case
    if (use_cublaslt) {
        cudaError_t err = cudaSuccess;

        switch (a.dtype()) {
            case DataType::Float32:
                err = cublaslt_gemm::gemm_fp32(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K, stream);
                break;
            case DataType::Float16:
                err = cublaslt_gemm::gemm_fp16(
                    static_cast<const __half*>(a.data()),
                    static_cast<const __half*>(b.data()),
                    static_cast<__half*>(c.data()),
                    M, N, K, stream);
                break;
            case DataType::BFloat16:
                err = cublaslt_gemm::gemm_bf16(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<const __nv_bfloat16*>(b.data()),
                    static_cast<__nv_bfloat16*>(c.data()),
                    M, N, K, stream);
                break;
            default:
                break;  // Fall through to native kernels
        }

        if (err == cudaSuccess) {
            sync_and_check("cuBLASLt matmul kernel failed");
            return;
        }
        // cuBLASLt failed - fall through to native kernels
    }

    // CUTLASS dispatch (highest priority when enabled)
    // FP32 uses TF32 TensorCore (can be disabled with PYGPUKIT_NO_TF32)
    // FP16/BF16 always use CUTLASS when available
    if (cutlass_enabled || cutlass_tf32_enabled) {
        cudaError_t err = cudaSuccess;
        bool used_cutlass = false;

        // Get current stream (capture stream if available, otherwise default)
        cudaStream_t stream = internal::get_capture_stream();

        switch (a.dtype()) {
            case DataType::Float32:
                if (cutlass_tf32_enabled) {
                    err = cutlass_gemm_tf32(
                        static_cast<const float*>(a.data()),
                        static_cast<const float*>(b.data()),
                        static_cast<float*>(c.data()),
                        M, N, K, stream);
                    used_cutlass = true;
                }
                break;
            case DataType::Float16:
                if (cutlass_enabled) {
                    err = cutlass_gemm_fp16(
                        static_cast<const __half*>(a.data()),
                        static_cast<const __half*>(b.data()),
                        static_cast<__half*>(c.data()),
                        M, N, K, stream);
                    used_cutlass = true;
                }
                break;
            case DataType::BFloat16:
                if (cutlass_enabled) {
                    err = cutlass_gemm_bf16(
                        static_cast<const __nv_bfloat16*>(a.data()),
                        static_cast<const __nv_bfloat16*>(b.data()),
                        static_cast<__nv_bfloat16*>(c.data()),
                        M, N, K, stream);
                    used_cutlass = true;
                }
                break;
            default:
                break;
        }

        if (used_cutlass) {
            if (err != cudaSuccess) {
                throw std::runtime_error("CUTLASS GEMM failed");
            }
            sync_and_check("CUTLASS matmul kernel failed");
            return;
        }
    }

    if (use_tf32) {
        if (M == 16 && (N == 8 || N == 16)) {
            tf32::launch_single_tile_verified(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        } else {
            tf32::launch_sgemm_tf32(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        }
    } else if (use_fp16_tc_fast) {
        if (a.dtype() == DataType::Float16) {
            fp16_bf16_tc::launch_sgemm_f16_tc(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()),
                M, N, K);
        } else {
            fp16_bf16_tc::launch_sgemm_bf16_tc(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                M, N, K);
        }
    } else if (use_fp16_tc_generic) {
        if (a.dtype() == DataType::Float16) {
            fp16_bf16_tc_generic::launch_sgemm_f16_tc_generic(
                static_cast<const __half*>(a.data()),
                static_cast<const __half*>(b.data()),
                static_cast<__half*>(c.data()),
                M, N, K);
        } else {
            fp16_bf16_tc_generic::launch_sgemm_bf16_tc_generic(
                static_cast<const __nv_bfloat16*>(a.data()),
                static_cast<const __nv_bfloat16*>(b.data()),
                static_cast<__nv_bfloat16*>(c.data()),
                M, N, K);
        }
    } else if (use_optimized) {
        ampere::launch_sgemm_ampere(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(c.data()),
            M, N, K);
    } else if (use_tiled) {
        switch (a.dtype()) {
            case DataType::Float32:
                matmul_fp32::launch_tiled_f32(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float64:
                matmul_fp32::launch_tiled_f64(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float16:
                fp16_bf16_matmul::launch_sgemm_f16(
                    static_cast<const __half*>(a.data()),
                    static_cast<const __half*>(b.data()),
                    static_cast<__half*>(c.data()),
                    M, N, K);
                break;
            case DataType::BFloat16:
                fp16_bf16_matmul::launch_sgemm_bf16(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<const __nv_bfloat16*>(b.data()),
                    static_cast<__nv_bfloat16*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float types");
        }
    } else {
        switch (a.dtype()) {
            case DataType::Float32:
                matmul_fp32::launch_l2opt_f32(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float64:
                matmul_fp32::launch_l2opt_f64(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float16:
                fp16_bf16_matmul::launch_sgemm_f16(
                    static_cast<const __half*>(a.data()),
                    static_cast<const __half*>(b.data()),
                    static_cast<__half*>(c.data()),
                    M, N, K);
                break;
            case DataType::BFloat16:
                fp16_bf16_matmul::launch_sgemm_bf16(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<const __nv_bfloat16*>(b.data()),
                    static_cast<__nv_bfloat16*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float types");
        }
    }

    sync_and_check("matmul kernel failed");
}

GPUArray matmul(const GPUArray& a, const GPUArray& b) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t N = b.shape()[1];

    GPUArray c({M, N}, a.dtype());
    matmul(a, b, c);
    return c;
}

// Internal helper: matmul with explicit TF32 control
static void matmul_impl(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32_explicit) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    if (c.shape()[0] != M || c.shape()[1] != N) {
        throw std::runtime_error("matmul output shape mismatch");
    }

    int sm_version = get_sm_version();

    bool tf32_enabled = use_tf32_explicit &&
                        (a.dtype() == DataType::Float32) &&
                        (sm_version >= MIN_SM_VERSION);

    if (use_tf32_explicit && !tf32_enabled) {
        if (a.dtype() != DataType::Float32) {
            throw std::runtime_error("TF32 matmul requires float32 dtype");
        }
        if (sm_version < MIN_SM_VERSION) {
            throw std::runtime_error("TF32 matmul requires SM >= 80 (Ampere or newer)");
        }
    }

    bool use_tf32 = tf32_enabled &&
                    ((M >= OPTIMIZED_MATMUL_THRESHOLD &&
                      N >= OPTIMIZED_MATMUL_THRESHOLD &&
                      K >= OPTIMIZED_MATMUL_THRESHOLD) ||
                     (M == 16 && (N == 8 || N == 16)));

    bool use_optimized = !use_tf32 &&
                         (a.dtype() == DataType::Float32) &&
                         (M >= OPTIMIZED_MATMUL_THRESHOLD ||
                          N >= OPTIMIZED_MATMUL_THRESHOLD ||
                          K >= OPTIMIZED_MATMUL_THRESHOLD);

    bool use_tiled = !use_optimized && !use_tf32 &&
                     (M >= TILED_MATMUL_THRESHOLD ||
                      N >= TILED_MATMUL_THRESHOLD ||
                      K >= TILED_MATMUL_THRESHOLD);

    if (use_tf32) {
        if (M == 16 && (N == 8 || N == 16)) {
            tf32::launch_single_tile_verified(
                static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(c.data()),
                M, N, K);
        } else {
            const char* use_v2 = std::getenv("PYGPUKIT_TF32_V2");
            if (use_v2 && std::string(use_v2) == "1") {
                tf32_v2::launch_sgemm_tf32_v2(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
            } else {
                tf32::launch_sgemm_tf32(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
            }
        }
    } else if (use_optimized) {
        ampere::launch_sgemm_ampere(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(c.data()),
            M, N, K);
    } else if (use_tiled) {
        switch (a.dtype()) {
            case DataType::Float32:
                matmul_fp32::launch_tiled_f32(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float64:
                matmul_fp32::launch_tiled_f64(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float16:
                fp16_bf16_matmul::launch_sgemm_f16(
                    static_cast<const __half*>(a.data()),
                    static_cast<const __half*>(b.data()),
                    static_cast<__half*>(c.data()),
                    M, N, K);
                break;
            case DataType::BFloat16:
                fp16_bf16_matmul::launch_sgemm_bf16(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<const __nv_bfloat16*>(b.data()),
                    static_cast<__nv_bfloat16*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float32, float64, float16, and bfloat16");
        }
    } else {
        switch (a.dtype()) {
            case DataType::Float32:
                matmul_fp32::launch_l2opt_f32(
                    static_cast<const float*>(a.data()),
                    static_cast<const float*>(b.data()),
                    static_cast<float*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float64:
                matmul_fp32::launch_l2opt_f64(
                    static_cast<const double*>(a.data()),
                    static_cast<const double*>(b.data()),
                    static_cast<double*>(c.data()),
                    M, N, K);
                break;
            case DataType::Float16:
                fp16_bf16_matmul::launch_sgemm_f16(
                    static_cast<const __half*>(a.data()),
                    static_cast<const __half*>(b.data()),
                    static_cast<__half*>(c.data()),
                    M, N, K);
                break;
            case DataType::BFloat16:
                fp16_bf16_matmul::launch_sgemm_bf16(
                    static_cast<const __nv_bfloat16*>(a.data()),
                    static_cast<const __nv_bfloat16*>(b.data()),
                    static_cast<__nv_bfloat16*>(c.data()),
                    M, N, K);
                break;
            default:
                throw std::runtime_error("matmul only supports float32, float64, float16, and bfloat16");
        }
    }

    sync_and_check("matmul kernel failed");
}

void matmul(const GPUArray& a, const GPUArray& b, GPUArray& c, bool use_tf32) {
    matmul_impl(a, b, c, use_tf32);
}

GPUArray matmul(const GPUArray& a, const GPUArray& b, bool use_tf32) {
    validate_matmul_shapes(a, b, "matmul");
    validate_same_dtype(a, b, "matmul");

    size_t M = a.shape()[0];
    size_t N = b.shape()[1];

    GPUArray c({M, N}, a.dtype());
    matmul_impl(a, b, c, use_tf32);
    return c;
}

// ============================================================================
// Fused Operations (CUTLASS Epilogue Fusion)
// ============================================================================

GPUArray linear_bias_gelu(const GPUArray& input, const GPUArray& weight, const GPUArray& bias) {
    // Validate shapes: input [batch, in_features], weight [out_features, in_features], bias [out_features]
    if (input.ndim() != 2) {
        throw std::runtime_error("linear_bias_gelu: input must be 2D [batch, in_features]");
    }
    if (weight.ndim() != 2) {
        throw std::runtime_error("linear_bias_gelu: weight must be 2D [out_features, in_features]");
    }
    if (bias.ndim() != 1) {
        throw std::runtime_error("linear_bias_gelu: bias must be 1D [out_features]");
    }

    size_t batch = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::runtime_error("linear_bias_gelu: weight.shape[1] must match input.shape[1]");
    }
    if (bias.shape()[0] != out_features) {
        throw std::runtime_error("linear_bias_gelu: bias.shape[0] must match weight.shape[0]");
    }

    // Validate dtypes
    if (input.dtype() != weight.dtype() || input.dtype() != bias.dtype()) {
        throw std::runtime_error("linear_bias_gelu: all inputs must have the same dtype");
    }

    // Check if CUTLASS fused kernel can be used
    // Requirements: dimensions must be multiples of 16 AND SM >= 86
    bool use_cutlass = cutlass_is_compatible(batch, out_features, in_features) && cutlass_is_sm_supported();

    // Also check if CUTLASS is disabled via environment variable
    const char* no_cutlass_env = std::getenv("PYGPUKIT_NO_CUTLASS");
    if (no_cutlass_env && (no_cutlass_env[0] == '1' || no_cutlass_env[0] == 'y' || no_cutlass_env[0] == 'Y')) {
        use_cutlass = false;
    }

    // Transpose weight for both paths (needed for input @ weight^T)
    GPUArray weight_T = transpose(weight);  // [in_features, out_features]

    // Allocate output
    GPUArray output({batch, out_features}, input.dtype());

    if (use_cutlass) {
        // CUTLASS fused BiasGELU kernel path
        cudaError_t err = cudaSuccess;
        cudaStream_t stream = internal::get_capture_stream();

        switch (input.dtype()) {
            case DataType::Float32:
                err = cutlass_gemm_tf32_bias_gelu(
                    static_cast<const float*>(input.data()),
                    static_cast<const float*>(weight_T.data()),
                    static_cast<const float*>(bias.data()),
                    static_cast<float*>(output.data()),
                    batch, out_features, in_features, stream);
                break;
            case DataType::Float16:
                err = cutlass_gemm_fp16_bias_gelu(
                    static_cast<const __half*>(input.data()),
                    static_cast<const __half*>(weight_T.data()),
                    static_cast<const __half*>(bias.data()),
                    static_cast<__half*>(output.data()),
                    batch, out_features, in_features, stream);
                break;
            case DataType::BFloat16:
                err = cutlass_gemm_bf16_bias_gelu(
                    static_cast<const __nv_bfloat16*>(input.data()),
                    static_cast<const __nv_bfloat16*>(weight_T.data()),
                    static_cast<const __nv_bfloat16*>(bias.data()),
                    static_cast<__nv_bfloat16*>(output.data()),
                    batch, out_features, in_features, stream);
                break;
            default:
                throw std::runtime_error("linear_bias_gelu only supports float32, float16, and bfloat16");
        }

        // If CUTLASS fails (e.g., not compiled in), fall back to native path
        if (err == cudaSuccess) {
            sync_and_check("linear_bias_gelu CUTLASS kernel failed");
            return output;
        }
        // Fall through to native path if CUTLASS returns error
    }

    // Native fallback path: matmul + bias_add_inplace + gelu
    // This works for any dimensions and when CUTLASS is not available
    matmul(input, weight_T, output);
    bias_add_inplace(output, bias);
    output = gelu(output);

    return output;
}

// ============================================================================
// Batched GEMM Implementation
// ============================================================================

void batched_matmul_fp32(const GPUArray& A, const GPUArray& B, GPUArray& C,
                         int M, int N, int K, int batch_count,
                         int64_t strideA, int64_t strideB, int64_t strideC) {
    // Validate inputs
    if (A.dtype() != DataType::Float32 || B.dtype() != DataType::Float32 || C.dtype() != DataType::Float32) {
        throw std::runtime_error("batched_matmul_fp32: all inputs must be float32");
    }

#if PYGPUKIT_HAS_CUTLASS
    // Use CUTLASS batched GEMM
    cudaError_t err = cutlass_gemm::gemm_batched_fp32(
        static_cast<const float*>(A.data()),
        static_cast<const float*>(B.data()),
        static_cast<float*>(C.data()),
        M, N, K,
        batch_count,
        strideA, strideB, strideC,
        1.0f, 0.0f,  // alpha, beta
        internal::get_capture_stream()
    );
    if (err != cudaSuccess) {
        throw std::runtime_error("batched_matmul_fp32: CUTLASS kernel failed");
    }
    sync_and_check("batched_matmul_fp32 CUTLASS kernel failed");
#else
    throw std::runtime_error("batched_matmul_fp32: CUTLASS not available");
#endif
}

} // namespace ops
} // namespace pygpukit
