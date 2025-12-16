/**
 * CUTLASS 4.x GEMM kernels for SM100 (Blackwell datacenter) architecture
 *
 * Uses CUTLASS 4.x CollectiveBuilder API for optimal performance on:
 * - SM 100 (B100/B200): Blackwell datacenter GPUs
 * - SM 101, SM 103: Blackwell variants
 *
 * Features specific to SM100:
 * - 232KB shared memory per SM (vs 100KB on SM120)
 * - Multi-SM cluster support (2x2x1 clusters)
 * - TMA multicast for inter-SM data sharing
 * - 2SM MMA with 256x128x64 tile sizes
 *
 * This file requires CUDA 12.8+ and SM100 GPU (B200).
 */
#pragma once

// Only compile for SM100+ with CUTLASS 4.x support
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

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
namespace cutlass_gemm_sm100 {

using namespace cute;

// ============================================================================
// Common Type Definitions for SM100
// ============================================================================

using KernelScheduleAuto = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueScheduleAuto = cutlass::epilogue::collective::EpilogueScheduleAuto;
using StageCountAuto = cutlass::gemm::collective::StageCountAuto;

// ============================================================================
// TF32 GEMM for SM100 (Blackwell datacenter)
// Optimized for B200's 232KB shared memory and 2SM MMA
// ============================================================================

struct TF32GemmSm100 {
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

    // Tile shape: optimized for B200 with 232KB shared memory
    // SM100 supports 2SM MMA for larger tiles
    using TileShape = Shape<_256, _128, _64>;  // 2SM MMA tile
    using ClusterShape = Shape<_2, _2, _1>;    // Multi-SM cluster

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
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
        cutlass::arch::Sm100,
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
// FP16 GEMM for SM100 (Blackwell datacenter)
// ============================================================================

struct FP16GemmSm100 {
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

    // Larger tiles for FP16 on SM100
    using TileShape = Shape<_256, _256, _64>;
    using ClusterShape = Shape<_2, _2, _1>;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
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
        cutlass::arch::Sm100,
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
// BF16 GEMM for SM100 (Blackwell datacenter)
// ============================================================================

struct BF16GemmSm100 {
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

    using TileShape = Shape<_256, _256, _64>;
    using ClusterShape = Shape<_2, _2, _1>;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
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
        cutlass::arch::Sm100,
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
// Wrapper Functions for SM100 GEMM
// ============================================================================

template<typename GemmType>
inline cudaError_t run_gemm_sm100(
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

    ProblemShape problem_size{M, N, K, 1};

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = 0;
    cudaDeviceGetAttribute(&hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);

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

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorLaunchFailure;
    }

    return cudaSuccess;
}

// ============================================================================
// Public API Functions
// ============================================================================

inline cudaError_t gemm_tf32_sm100(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    return run_gemm_sm100<TF32GemmSm100>(A, B, C, C, M, N, K, alpha, beta, stream);
}

inline cudaError_t gemm_fp16_sm100(
    const __half* A,
    const __half* B,
    __half* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    return run_gemm_sm100<FP16GemmSm100>(A, B, C, C, M, N, K, alpha, beta, stream);
}

inline cudaError_t gemm_bf16_sm100(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    return run_gemm_sm100<BF16GemmSm100>(A, B, C, C, M, N, K, alpha, beta, stream);
}

// ============================================================================
// SM100 Check
// ============================================================================

inline bool is_sm100_supported() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    int sm = props.major * 10 + props.minor;
    return sm >= 100 && sm < 120;  // SM100, SM101, SM103
}

}  // namespace cutlass_gemm_sm100
}  // namespace ops
}  // namespace pygpukit

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
