/**
 * CUTLASS-based GEMM kernels for PyGPUkit
 *
 * Provides high-performance matrix multiplication using NVIDIA CUTLASS library.
 * Targets SM 86 (RTX 30 series) with TensorCore support.
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
// TF32 GEMM (FP32 input/output, TF32 TensorCore)
// ============================================================================

// TF32 GEMM: FP32 in -> TF32 TensorCore -> FP32 out
// For row-major inputs, use all-ColumnMajor with transpose trick:
//   C (MxN row) = A (MxK row) @ B (KxN row)
//   becomes: C^T (NxM col) = B^T (NxK col) @ A^T (KxM col)
// where row-major X = col-major X^T in memory
using TF32Gemm = cutlass::gemm::device::Gemm<
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
    4                                           // Stages (pipeline depth)
>;

// ============================================================================
// FP16 GEMM (FP16 input/output, FP16 TensorCore)
// ============================================================================

// FP16 GEMM with same transpose trick as TF32 (all ColumnMajor)
using FP16Gemm = cutlass::gemm::device::Gemm<
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
    4                                           // Stages
>;

// ============================================================================
// BF16 GEMM (BF16 input/output, BF16 TensorCore)
// ============================================================================

// BF16 GEMM with same transpose trick as TF32 (all ColumnMajor)
using BF16Gemm = cutlass::gemm::device::Gemm<
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
    4                                           // Stages
>;

// ============================================================================
// BiasGELU GEMM Types (Epilogue Fusion: D = gelu(A @ B + bias))
// ============================================================================

// TF32 GEMM with BiasGELU epilogue
// D = gelu(alpha * A @ B + beta * bias)
// For bias broadcast: set bias stride = 0
using TF32GemmBiasGELU = cutlass::gemm::device::Gemm<
    float,                                      // ElementA
    cutlass::layout::ColumnMajor,               // LayoutA
    float,                                      // ElementB
    cutlass::layout::ColumnMajor,               // LayoutB
    float,                                      // ElementC (bias)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere)
    cutlass::gemm::GemmShape<128, 128, 16>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 16>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 8>,         // InstructionShape
    cutlass::epilogue::thread::LinearCombinationGELU<
        float, 128 / cutlass::sizeof_bits<float>::value,
        float, float>,                          // BiasGELU epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                           // Stages
>;

// FP16 GEMM with BiasGELU epilogue
using FP16GemmBiasGELU = cutlass::gemm::device::Gemm<
    cutlass::half_t,                            // ElementA
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::half_t,                            // ElementB
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::half_t,                            // ElementC (bias)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float, float>,                          // BiasGELU epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                           // Stages
>;

// BF16 GEMM with BiasGELU epilogue
using BF16GemmBiasGELU = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,                        // ElementA
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::bfloat16_t,                        // ElementB
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::bfloat16_t,                        // ElementC (bias)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
        float, float>,                          // BiasGELU epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                           // Stages
>;

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
inline cudaError_t gemm_tf32(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    // Transpose trick for row-major inputs with all-ColumnMajor kernel:
    //   C (MxN row) = A (MxK row) @ B (KxN row)
    //   becomes: C^T (NxM col) = B^T (NxK col) @ A^T (KxM col)
    //
    // Memory equivalence: row-major X (RxC) = col-major X^T (CxR)
    // So we reinterpret pointers without copying:
    //   - B (KxN row) in memory = B^T (NxK col), which is our "A" operand
    //   - A (MxK row) in memory = A^T (KxM col), which is our "B" operand
    //   - C (MxN row) in memory = C^T (NxM col), which is our output
    //
    // problem_size(M', N', K') for output M'xN' = (N, M, K)
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // For column-major matrices, leading dimension = number of rows
    // - B^T is NxK col-major, ld = N (num rows)
    // - A^T is KxM col-major, ld = K (num rows)
    // - C^T is NxM col-major, ld = N (num rows)
    typename TF32Gemm::Arguments arguments{
        problem_size,
        {B, N},         // "A" operand: B^T (NxK col-major), ld = N
        {A, K},         // "B" operand: A^T (KxM col-major), ld = K
        {C, N},         // "C" operand: C^T (NxM col-major), ld = N
        {C, N},         // D = C
        {alpha, beta}   // Epilogue params
    };

    TF32Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = TF32Gemm::get_workspace_size(arguments);

    if (workspace_size == 0) {
        status = gemm_op.initialize(arguments, nullptr, stream);
    } else {
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        status = gemm_op.initialize(arguments, workspace.get(), stream);
    }

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
    // Same transpose trick as TF32: compute C^T = B^T @ A^T
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Cast to CUTLASS types
    const cutlass::half_t* A_cutlass = reinterpret_cast<const cutlass::half_t*>(A);
    const cutlass::half_t* B_cutlass = reinterpret_cast<const cutlass::half_t*>(B);
    cutlass::half_t* C_cutlass = reinterpret_cast<cutlass::half_t*>(C);

    // Leading dimensions for col-major transpose trick (ld = num rows)
    typename FP16Gemm::Arguments arguments{
        problem_size,
        {B_cutlass, N},  // "A" = B^T (NxK col-major), ld = N
        {A_cutlass, K},  // "B" = A^T (KxM col-major), ld = K
        {C_cutlass, N},  // "C" = C^T (NxM col-major), ld = N
        {C_cutlass, N},  // D = C
        {alpha, beta}
    };

    FP16Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = FP16Gemm::get_workspace_size(arguments);
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
    // Same transpose trick as TF32: compute C^T = B^T @ A^T
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Cast to CUTLASS types
    const cutlass::bfloat16_t* A_cutlass = reinterpret_cast<const cutlass::bfloat16_t*>(A);
    const cutlass::bfloat16_t* B_cutlass = reinterpret_cast<const cutlass::bfloat16_t*>(B);
    cutlass::bfloat16_t* C_cutlass = reinterpret_cast<cutlass::bfloat16_t*>(C);

    // Leading dimensions for col-major transpose trick (ld = num rows)
    typename BF16Gemm::Arguments arguments{
        problem_size,
        {B_cutlass, N},  // "A" = B^T (NxK col-major), ld = N
        {A_cutlass, K},  // "B" = A^T (KxM col-major), ld = K
        {C_cutlass, N},  // "C" = C^T (NxM col-major), ld = N
        {C_cutlass, N},  // D = C
        {alpha, beta}
    };

    BF16Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = BF16Gemm::get_workspace_size(arguments);
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

// ============================================================================
// BiasGELU Wrapper functions
// ============================================================================

/**
 * TF32 GEMM with fused BiasGELU: D = gelu(A @ B + bias)
 *
 * @param A Input matrix A (M x K), row-major, FP32
 * @param B Input matrix B (K x N), row-major, FP32
 * @param bias Bias vector (N), FP32 - broadcast across M dimension
 * @param D Output matrix D (M x N), row-major, FP32
 * @param M Number of rows in A and D
 * @param N Number of columns in B and D (and length of bias)
 * @param K Number of columns in A and rows in B
 * @param stream CUDA stream
 * @return cudaError_t
 *
 * Uses same transpose trick as base GEMM.
 * Bias is broadcast using stride=0 in CUTLASS.
 */
inline cudaError_t gemm_tf32_bias_gelu(
    const float* A,
    const float* B,
    const float* bias,
    float* D,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    // Transpose trick: D^T (NxM col) = B^T (NxK col) @ A^T (KxM col) + bias
    // Bias is broadcast across all M columns (stride=0)
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // alpha=1, beta=1 for D = gelu(AB + bias)
    typename TF32GemmBiasGELU::Arguments arguments{
        problem_size,
        {B, N},           // "A" operand: B^T (NxK col-major), ld = N
        {A, K},           // "B" operand: A^T (KxM col-major), ld = K
        {bias, 0},        // "C" operand: bias (N), stride=0 for broadcast
        {D, N},           // D = output D^T (NxM col-major), ld = N
        {1.0f, 1.0f}      // alpha=1, beta=1
    };

    TF32GemmBiasGELU gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = TF32GemmBiasGELU::get_workspace_size(arguments);
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
 * FP16 GEMM with fused BiasGELU: D = gelu(A @ B + bias)
 * Uses same transpose trick as TF32
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

    const cutlass::half_t* A_cutlass = reinterpret_cast<const cutlass::half_t*>(A);
    const cutlass::half_t* B_cutlass = reinterpret_cast<const cutlass::half_t*>(B);
    const cutlass::half_t* bias_cutlass = reinterpret_cast<const cutlass::half_t*>(bias);
    cutlass::half_t* D_cutlass = reinterpret_cast<cutlass::half_t*>(D);

    typename FP16GemmBiasGELU::Arguments arguments{
        problem_size,
        {B_cutlass, N},      // "A" = B^T (NxK col-major), ld = N
        {A_cutlass, K},      // "B" = A^T (KxM col-major), ld = K
        {bias_cutlass, 0},   // "C" = bias, stride=0 for broadcast
        {D_cutlass, N},      // D = output
        {1.0f, 1.0f}
    };

    FP16GemmBiasGELU gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = FP16GemmBiasGELU::get_workspace_size(arguments);
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
 * BF16 GEMM with fused BiasGELU: D = gelu(A @ B + bias)
 * Uses same transpose trick as TF32
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

    const cutlass::bfloat16_t* A_cutlass = reinterpret_cast<const cutlass::bfloat16_t*>(A);
    const cutlass::bfloat16_t* B_cutlass = reinterpret_cast<const cutlass::bfloat16_t*>(B);
    const cutlass::bfloat16_t* bias_cutlass = reinterpret_cast<const cutlass::bfloat16_t*>(bias);
    cutlass::bfloat16_t* D_cutlass = reinterpret_cast<cutlass::bfloat16_t*>(D);

    typename BF16GemmBiasGELU::Arguments arguments{
        problem_size,
        {B_cutlass, N},      // "A" = B^T (NxK col-major), ld = N
        {A_cutlass, K},      // "B" = A^T (KxM col-major), ld = K
        {bias_cutlass, 0},   // "C" = bias, stride=0 for broadcast
        {D_cutlass, N},      // D = output
        {1.0f, 1.0f}
    };

    BF16GemmBiasGELU gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = BF16GemmBiasGELU::get_workspace_size(arguments);
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
