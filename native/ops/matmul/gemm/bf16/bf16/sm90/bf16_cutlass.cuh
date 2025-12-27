/**
 * CUTLASS 3.x GEMM kernels for SM90+ architectures
 *
 * Uses CUTLASS 3.x CollectiveBuilder API for optimal performance on:
 * - SM 90 (H100): Hopper with WGMMA/TMA
 * - SM 100 (B100/B200): Blackwell
 * - SM 103: Blackwell variant
 * - SM 110/120/121: Future architectures
 *
 * Features:
 * - TMA (Tensor Memory Accelerator) for efficient data loading
 * - WGMMA (Warp Group Matrix Multiply-Accumulate) instructions
 * - Warp-specialized kernel scheduling
 * - Automatic pipeline stage selection
 *
 * This file requires CUDA 12.0+ and SM90+ GPU.
 */
#pragma once

// Only compile for SM90+ with CUTLASS 3.x support
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) || defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace pygpukit {
namespace ops {
namespace cutlass_gemm_sm90 {

using namespace cute;

// ============================================================================
// Common Type Definitions
// ============================================================================

// Kernel scheduling modes
using KernelScheduleAuto = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueScheduleAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
using StageCountAuto = cutlass::gemm::collective::StageCountAuto;

// ============================================================================
// TF32 GEMM for SM90+ (Hopper/Blackwell)
// ============================================================================

// TF32: FP32 input with TensorCore acceleration
// Uses TMA for data loading and WGMMA for compute
struct TF32GemmSm90 {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementD = float;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 4;  // 16B / 4B = 4 elements
    static constexpr int AlignmentB = 4;
    static constexpr int AlignmentC = 4;
    static constexpr int AlignmentD = 4;

    // Tile shape: optimized for H100
    using TileShape = Shape<_128, _128, _32>;  // M, N, K
    using ClusterShape = Shape<_1, _1, _1>;

    // Epilogue operation: D = alpha * A @ B + beta * C
    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;

    // Build collective epilogue
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueScheduleAuto,
        EpilogueOp
    >::CollectiveOp;

    // Build collective mainloop with auto stage carveout
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelScheduleAuto
    >::CollectiveOp;

    // Kernel type
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,  // ProblemShape: M, N, K, L (batch)
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    // Device adapter
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

// ============================================================================
// FP16 GEMM for SM90+ (Hopper/Blackwell)
// ============================================================================

struct FP16GemmSm90 {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 8;  // 16B / 2B = 8 elements
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    static constexpr int AlignmentD = 8;

    // Tile shape: optimized for FP16 on H100
    using TileShape = Shape<_128, _256, _64>;  // Larger N for FP16
    using ClusterShape = Shape<_1, _1, _1>;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueScheduleAuto,
        EpilogueOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

// ============================================================================
// BF16 GEMM for SM90+ (Hopper/Blackwell)
// ============================================================================

struct BF16GemmSm90 {
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    static constexpr int AlignmentD = 8;

    using TileShape = Shape<_128, _256, _64>;
    using ClusterShape = Shape<_1, _1, _1>;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueScheduleAuto,
        EpilogueOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

// ============================================================================
// Wrapper Functions for SM90+ GEMM
// ============================================================================

template<typename GemmType>
inline cudaError_t run_gemm_sm90(
    const void* A, const void* B,
    void* C, void* D,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    using Gemm = typename GemmType::Gemm;
    using ProblemShape = typename Gemm::GemmKernel::ProblemShape;
    using StrideA = typename GemmType::StrideA;
    using StrideB = typename GemmType::StrideB;
    using StrideC = typename GemmType::StrideC;
    using StrideD = typename GemmType::StrideD;

    // Problem shape: M, N, K, batch=1
    ProblemShape problem_size{M, N, K, 1};

    // Compute strides for row-major layouts
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Hardware info
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = 0;  // Auto-detect
    cudaDeviceGetAttribute(&hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);

    // Arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {
            static_cast<const typename GemmType::ElementA*>(A), stride_a,
            static_cast<const typename GemmType::ElementB*>(B), stride_b
        },
        {
            {alpha, beta},
            static_cast<typename GemmType::ElementC*>(C), stride_c,
            static_cast<typename GemmType::ElementD*>(D), stride_d
        },
        hw_info
    };

    // Instantiate GEMM
    Gemm gemm_op;

    // Check if arguments are valid
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    // Get workspace size
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Initialize
    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    // Run
    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorLaunchFailure;
    }

    return cudaSuccess;
}

// ============================================================================
// Public API Functions
// ============================================================================

/**
 * TF32 GEMM for SM90+: C = alpha * A @ B + beta * C
 */
inline cudaError_t gemm_tf32_sm90(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    return run_gemm_sm90<TF32GemmSm90>(A, B, C, C, M, N, K, alpha, beta, stream);
}

/**
 * FP16 GEMM for SM90+: C = alpha * A @ B + beta * C
 */
inline cudaError_t gemm_fp16_sm90(
    const __half* A,
    const __half* B,
    __half* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    return run_gemm_sm90<FP16GemmSm90>(A, B, C, C, M, N, K, alpha, beta, stream);
}

/**
 * BF16 GEMM for SM90+: C = alpha * A @ B + beta * C
 */
inline cudaError_t gemm_bf16_sm90(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    return run_gemm_sm90<BF16GemmSm90>(A, B, C, C, M, N, K, alpha, beta, stream);
}

// ============================================================================
// SM90+ Check
// ============================================================================

inline bool is_sm90_supported() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    return (props.major * 10 + props.minor) >= 90;
}

}  // namespace cutlass_gemm_sm90
}  // namespace ops
}  // namespace pygpukit

#endif  // CUTLASS_ARCH_MMA_SM90_SUPPORTED
