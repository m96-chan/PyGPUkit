/**
 * FP8 GEMM v2 for SM120 - Tile size tuning
 * Multiple tile configurations for benchmarking
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

#define PYGPUKIT_ENABLE_FP8_SM120

#if (defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)) && defined(PYGPUKIT_ENABLE_FP8_SM120)

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#define PYGPUKIT_PATCH_CUTLASS_LDSM_POST 1
#include "../../../../common/aligned_copy_sm120.cuh"

using namespace cute;

namespace pygpukit {
namespace ops {
namespace fp8_fp8_gemm_sm120_v2 {

using ElementA = cutlass::float_e4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = cutlass::float_e4m3_t;
using ElementD = cutlass::float_e4m3_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ClusterShape_MNK = Shape<_1, _1, _1>;

// Base kernel with auto schedule (default: cooperative)
template<typename MmaTileShape>
struct FP8GemmKernel {
    using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape, ClusterShape_MNK,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, cute::tuple<LayoutATag, LayoutSFA>, AlignmentA,
        ElementB, cute::tuple<LayoutBTag, LayoutSFB>, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
};

// Note: Ping-pong schedule is NOT supported for FP8 blockwise scaling on SM120
// KernelTmaWarpSpecializedPingpong fails with "Could not build a collective"
// Only cooperative schedule works with FP8 blockwise scaling

__global__ void fill_unity_kernel(float* scales, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) scales[idx] = 1.0f;
}

template<typename MmaTileShape>
cudaError_t run_gemm(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    using Kernel = FP8GemmKernel<MmaTileShape>;
    using Gemm = typename Kernel::Gemm;
    using ScaleConfig = typename Kernel::ScaleConfig;
    using LayoutSFA = typename Kernel::LayoutSFA;
    using LayoutSFB = typename Kernel::LayoutSFB;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideC = typename Kernel::StrideC;
    using StrideD = typename Kernel::StrideD;

    int64_t size_D = static_cast<int64_t>(M) * N;
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_C(size_D);
    auto* d_C = buf_C.get();

    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    size_t sfa_size = size(filter_zeros(layout_SFA));
    size_t sfb_size = size(filter_zeros(layout_SFB));
    size_t sfa_padded = std::max(sfa_size, size_t(32));
    size_t sfb_padded = std::max(sfb_size, size_t(32));

    cutlass::device_memory::allocation<float> buf_SFA(sfa_padded);
    cutlass::device_memory::allocation<float> buf_SFB(sfb_padded);

    int threads = 256;
    fill_unity_kernel<<<(sfa_padded + threads - 1) / threads, threads, 0, stream>>>(buf_SFA.get(), sfa_padded);
    fill_unity_kernel<<<(sfb_padded + threads - 1) / threads, threads, 0, stream>>>(buf_SFB.get(), sfb_padded);

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<const cutlass::float_e4m3_t*>(A), stride_a,
            reinterpret_cast<const cutlass::float_e4m3_t*>(B), stride_b,
            buf_SFA.get(), layout_SFA,
            buf_SFB.get(), layout_SFB
        },
        {
            {},
            d_C, stride_c,
            reinterpret_cast<cutlass::float_e4m3_t*>(D), stride_d
        }
    };
    arguments.epilogue.thread.alpha = alpha;
    arguments.epilogue.thread.beta = beta;

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorLaunchFailure;
    }

    return cudaSuccess;
}

}  // namespace fp8_fp8_gemm_sm120_v2
}  // namespace ops
}  // namespace pygpukit

extern "C" {

// V2: 128x128x128 - same tile as v1, template version for comparison
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v2(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream
) {
    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    return pygpukit::ops::fp8_fp8_gemm_sm120_v2::run_gemm<TileShape>(A, B, D, M, N, K, alpha, beta, stream);
}

// V3: Same as V2 (ping-pong not supported for FP8 blockwise)
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v3(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream
) {
    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    return pygpukit::ops::fp8_fp8_gemm_sm120_v2::run_gemm<TileShape>(A, B, D, M, N, K, alpha, beta, stream);
}

// V4: Stub (tile exploration TBD)
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v4(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream
) {
    // Same as v2 for now
    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    return pygpukit::ops::fp8_fp8_gemm_sm120_v2::run_gemm<TileShape>(A, B, D, M, N, K, alpha, beta, stream);
}

}  // extern "C"

#else  // !SM120

extern "C" {
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v2(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v3(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v4(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t) { return cudaErrorNotSupported; }
}

#endif
