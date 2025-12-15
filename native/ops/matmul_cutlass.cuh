/**
 * CUTLASS-based GEMM kernels for PyGPUkit
 *
 * Provides high-performance matrix multiplication using NVIDIA CUTLASS library.
 * Multi-SM support with runtime dispatch for optimal performance across GPU architectures.
 *
 * Supported architectures:
 * - SM 80 (A100): 4-stage pipeline (data center baseline)
 * - SM 86 (RTX 30xx): 5-stage pipeline, 100KB shared memory
 * - SM 89 (RTX 40xx): Ada Lovelace, uses SM86 configuration
 * - SM 90 (H100): Hopper (CUTLASS 3.x WGMMA is future work)
 * - SM 100-120: Future architectures (forward compatible)
 *
 * NOT supported:
 * - SM < 80 (Turing and older)
 *
 * Supported dtypes:
 * - FP32 (with TF32 TensorCore acceleration)
 * - FP16 (native TensorCore)
 * - BF16 (native TensorCore)
 *
 * Epilogue variants:
 * - Default: D = alpha * A @ B + beta * C
 * - Bias: D = A @ B + bias (with broadcast)
 * - BiasGELU: D = gelu(A @ B + bias)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/util/device_memory.h"

namespace pygpukit {
namespace ops {
namespace cutlass_gemm {

// ============================================================================
// SM Version Detection
// ============================================================================

// Cached SM version (initialized on first use)
inline int get_cached_sm_version() {
    static int sm_version = -1;
    if (sm_version < 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);
        sm_version = props.major * 10 + props.minor;
    }
    return sm_version;
}

// Minimum supported SM version
constexpr int MIN_SM_VERSION = 80;

// Check if SM version is supported
inline bool is_sm_supported() {
    return get_cached_sm_version() >= MIN_SM_VERSION;
}

// Check if SM >= 86 for optimized 5-stage pipeline
inline bool use_5stage_pipeline() {
    return get_cached_sm_version() >= 86;
}

// ============================================================================
// TF32 GEMM (FP32 input/output, TF32 TensorCore)
// ============================================================================

// TF32 GEMM: FP32 in -> TF32 TensorCore -> FP32 out
// For row-major inputs, use all-ColumnMajor with transpose trick:
//   C (MxN row) = A (MxK row) @ B (KxN row)
//   becomes: C^T (NxM col) = B^T (NxK col) @ A^T (KxM col)
// where row-major X = col-major X^T in memory

// SM80 (A100): 4-stage pipeline, optimized for data center
using TF32Gemm_Sm80 = cutlass::gemm::device::Gemm<
    float,                                      // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    float,                                      // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    float,                                      // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere)
    cutlass::gemm::GemmShape<128, 128, 16>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 16>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 8>,         // InstructionShape (mma.sync)
    cutlass::epilogue::thread::LinearCombination<
        float, 128 / cutlass::sizeof_bits<float>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                           // Stages (4-stage for SM80)
>;

// SM86+ (RTX 30xx/40xx/H100+): 5-stage pipeline, 100KB+ shared memory
using TF32Gemm_Sm86 = cutlass::gemm::device::Gemm<
    float,                                      // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    float,                                      // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    float,                                      // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere TensorCore compatible)
    cutlass::gemm::GemmShape<128, 128, 16>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 16>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 8>,         // InstructionShape (mma.sync)
    cutlass::epilogue::thread::LinearCombination<
        float, 128 / cutlass::sizeof_bits<float>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5                                           // Stages (5-stage for SM86+)
>;

// Default alias (SM80 for backward compatibility)
using TF32Gemm = TF32Gemm_Sm80;

// ============================================================================
// FP16 GEMM (FP16 input/output, FP16 TensorCore)
// ============================================================================

// SM80 (A100): FP16 GEMM with 4-stage pipeline
using FP16Gemm_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                            // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::half_t,                            // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::half_t,                            // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape (mma.sync.m16n8k16)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                           // Stages (4-stage for SM80)
>;

// SM86+ (RTX 30xx/40xx/H100+): FP16 GEMM with 5-stage pipeline
using FP16Gemm_Sm86 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                            // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::half_t,                            // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::half_t,                            // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere TensorCore compatible)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape (mma.sync.m16n8k16)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5                                           // Stages (5-stage for SM86+)
>;

// Default alias (SM80 for backward compatibility)
using FP16Gemm = FP16Gemm_Sm80;

// ============================================================================
// BF16 GEMM (BF16 input/output, BF16 TensorCore)
// ============================================================================

// SM80 (A100): BF16 GEMM with 4-stage pipeline
using BF16Gemm_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,                        // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::bfloat16_t,                        // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::bfloat16_t,                        // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                           // Stages (4-stage for SM80)
>;

// SM86+ (RTX 30xx/40xx/H100+): BF16 GEMM with 5-stage pipeline
using BF16Gemm_Sm86 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,                        // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::bfloat16_t,                        // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::bfloat16_t,                        // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere TensorCore compatible)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5                                           // Stages (5-stage for SM86+)
>;

// Default alias (SM80 for backward compatibility)
using BF16Gemm = BF16Gemm_Sm80;

// ============================================================================
// BiasGELU GEMM Types (Epilogue Fusion: D = gelu(A @ B + bias))
// ============================================================================

// TF32 BiasGELU - SM80 (4-stage)
using TF32GemmBiasGELU_Sm80 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

// TF32 BiasGELU - SM86+ (5-stage)
using TF32GemmBiasGELU_Sm86 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5
>;

using TF32GemmBiasGELU = TF32GemmBiasGELU_Sm80;

// FP16 BiasGELU - SM80 (4-stage)
using FP16GemmBiasGELU_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

// FP16 BiasGELU - SM86+ (5-stage)
using FP16GemmBiasGELU_Sm86 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5
>;

using FP16GemmBiasGELU = FP16GemmBiasGELU_Sm80;

// BF16 BiasGELU - SM80 (4-stage)
using BF16GemmBiasGELU_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

// BF16 BiasGELU - SM86+ (5-stage)
using BF16GemmBiasGELU_Sm86 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5
>;

using BF16GemmBiasGELU = BF16GemmBiasGELU_Sm80;

// ============================================================================
// Wrapper functions
// ============================================================================

/**
 * TF32 GEMM: C = alpha * A @ B + beta * C
 *
 * @param A Input matrix A (M x K), row-major, FP32
 * @param B Input matrix B (K x N), row-major, FP32
 * @param C Output matrix C (M x N), row-major, FP32
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A @ B
 * @param beta Scalar multiplier for C (set to 0 for C = A @ B)
 * @param stream CUDA stream
 * @return cudaError_t
 *
 * Layout trick for row-major inputs with RowMajorxColumnMajor kernel:
 * - CUTLASS kernel: D (MxN row) = A (MxK row) @ B (KxN col)
 * - Our inputs: C (MxN row) = A (MxK row) @ B (KxN row)
 *
 * Key insight: row-major B (KxN) = column-major B^T (NxK) in memory
 *
 * We compute: C^T (NxM row) = B^T (NxK row) @ A^T (KxM col)
 * Which is equivalent to: C (MxN row) = A (MxK row) @ B (KxN row)
 *
 * For the kernel:
 * - M' = N, N' = M, K' = K
 * - A' = B^T (NxK row-major), pointer = B, ld = N (stride between rows)
 * - B' = A^T (KxM col-major) = A (MxK row-major) in memory, pointer = A, ld = K
 * - C' = C^T (NxM row-major), pointer = C, ld = M (stride between rows)
 */
// Template helper for GEMM dispatch
template<typename GemmOp>
inline cudaError_t run_gemm(
    cutlass::gemm::GemmCoord problem_size,
    const void* A, int ldA,
    const void* B, int ldB,
    void* C, int ldC,
    void* D, int ldD,
    float alpha, float beta,
    cudaStream_t stream
) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementC = typename GemmOp::ElementC;

    typename GemmOp::Arguments arguments{
        problem_size,
        {static_cast<const ElementA*>(A), ldA},
        {static_cast<const ElementB*>(B), ldB},
        {static_cast<ElementC*>(C), ldC},
        {static_cast<ElementC*>(D), ldD},
        {alpha, beta}
    };

    GemmOp gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = GemmOp::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

inline cudaError_t gemm_tf32(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    // Transpose trick: C^T (NxM col) = B^T (NxK col) @ A^T (KxM col)
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Runtime SM dispatch: SM86+ uses 5-stage pipeline
    if (use_5stage_pipeline()) {
        return run_gemm<TF32Gemm_Sm86>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    } else {
        return run_gemm<TF32Gemm_Sm80>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    }
}

/**
 * FP16 GEMM: C = alpha * A @ B + beta * C (row-major inputs)
 * Uses same transpose trick as TF32
 */
inline cudaError_t gemm_fp16(
    const __half* A,
    const __half* B,
    __half* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Runtime SM dispatch: SM86+ uses 5-stage pipeline
    if (use_5stage_pipeline()) {
        return run_gemm<FP16Gemm_Sm86>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    } else {
        return run_gemm<FP16Gemm_Sm80>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    }
}

/**
 * BF16 GEMM: C = alpha * A @ B + beta * C (row-major inputs)
 * Uses same transpose trick as TF32
 */
inline cudaError_t gemm_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Runtime SM dispatch: SM86+ uses 5-stage pipeline
    if (use_5stage_pipeline()) {
        return run_gemm<BF16Gemm_Sm86>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    } else {
        return run_gemm<BF16Gemm_Sm80>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    }
}

// ============================================================================
// BiasGELU Wrapper functions
// ============================================================================

// Template helper for BiasGELU GEMM dispatch (with bias broadcast stride=0)
template<typename GemmOp>
inline cudaError_t run_gemm_bias_gelu(
    cutlass::gemm::GemmCoord problem_size,
    const void* A, int ldA,
    const void* B, int ldB,
    const void* bias,  // stride=0 for broadcast
    void* D, int ldD,
    cudaStream_t stream
) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementC = typename GemmOp::ElementC;

    typename GemmOp::Arguments arguments{
        problem_size,
        {static_cast<const ElementA*>(A), ldA},
        {static_cast<const ElementB*>(B), ldB},
        {static_cast<const ElementC*>(bias), 0},  // stride=0 for broadcast
        {static_cast<ElementC*>(D), ldD},
        {1.0f, 1.0f}  // alpha=1, beta=1
    };

    GemmOp gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = GemmOp::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

/**
 * TF32 GEMM with fused BiasGELU: D = gelu(A @ B + bias)
 * Uses transpose trick, bias broadcast with stride=0
 */
inline cudaError_t gemm_tf32_bias_gelu(
    const float* A,
    const float* B,
    const float* bias,
    float* D,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Runtime SM dispatch: SM86+ uses 5-stage pipeline
    if (use_5stage_pipeline()) {
        return run_gemm_bias_gelu<TF32GemmBiasGELU_Sm86>(
            problem_size, B, N, A, K, bias, D, N, stream);
    } else {
        return run_gemm_bias_gelu<TF32GemmBiasGELU_Sm80>(
            problem_size, B, N, A, K, bias, D, N, stream);
    }
}

/**
 * FP16 GEMM with fused BiasGELU: D = gelu(A @ B + bias)
 */
inline cudaError_t gemm_fp16_bias_gelu(
    const __half* A,
    const __half* B,
    const __half* bias,
    __half* D,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Runtime SM dispatch: SM86+ uses 5-stage pipeline
    if (use_5stage_pipeline()) {
        return run_gemm_bias_gelu<FP16GemmBiasGELU_Sm86>(
            problem_size, B, N, A, K, bias, D, N, stream);
    } else {
        return run_gemm_bias_gelu<FP16GemmBiasGELU_Sm80>(
            problem_size, B, N, A, K, bias, D, N, stream);
    }
}

/**
 * BF16 GEMM with fused BiasGELU: D = gelu(A @ B + bias)
 */
inline cudaError_t gemm_bf16_bias_gelu(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    const __nv_bfloat16* bias,
    __nv_bfloat16* D,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Runtime SM dispatch: SM86+ uses 5-stage pipeline
    if (use_5stage_pipeline()) {
        return run_gemm_bias_gelu<BF16GemmBiasGELU_Sm86>(
            problem_size, B, N, A, K, bias, D, N, stream);
    } else {
        return run_gemm_bias_gelu<BF16GemmBiasGELU_Sm80>(
            problem_size, B, N, A, K, bias, D, N, stream);
    }
}

// ============================================================================
// Dispatch function for runtime dtype selection
// ============================================================================

enum class GemmDtype {
    FP32_TF32,  // FP32 input, TF32 TensorCore
    FP16,       // FP16 TensorCore
    BF16        // BF16 TensorCore
};

/**
 * Check if matrix dimensions are compatible with CUTLASS TensorCore kernels
 * TensorCore requires alignment to tile sizes
 */
inline bool is_cutlass_compatible(int M, int N, int K) {
    // Minimum alignment for TensorCore (based on ThreadBlockShape)
    // TF32: 128x128x16, FP16/BF16: 128x128x32
    // For simplicity, require 16-alignment on all dimensions
    return (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0);
}

}  // namespace cutlass_gemm
}  // namespace ops
}  // namespace pygpukit
