/**
 * FP8 GEMM v3 for SM120 - Using BlockScaledTensorOp like NVF4
 *
 * Key insight: NVF4 achieves 446 TFLOPS using OpClassBlockScaledTensorOp
 * which supports pingpong schedule. Try same approach for FP8.
 *
 * Note: If BlockScaledTensorOp doesn't work with FP8, fall back to
 * tile size tuning with the regular approach.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <mutex>

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
namespace fp8_fp8_gemm_sm120_v3 {

// ============================================================================
// Scale Factor Cache (avoid per-call allocation)
// ============================================================================
namespace {
    constexpr size_t MAX_SCALE_SIZE = 1024 * 1024;  // 1M floats = 4MB
    float* g_scale_buffer_a = nullptr;
    float* g_scale_buffer_b = nullptr;
    size_t g_scale_capacity = 0;
    std::mutex g_scale_mutex;

    __global__ void fill_unity_kernel(float* scales, size_t n) {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < n) scales[idx] = 1.0f;
    }

    cudaError_t ensure_scale_buffers(size_t required_size, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(g_scale_mutex);
        if (g_scale_capacity >= required_size) return cudaSuccess;

        size_t new_capacity = std::max(required_size, size_t(32768));  // At least 32K
        new_capacity = std::min(new_capacity, MAX_SCALE_SIZE);

        if (g_scale_buffer_a) cudaFree(g_scale_buffer_a);
        if (g_scale_buffer_b) cudaFree(g_scale_buffer_b);

        cudaError_t err = cudaMalloc(&g_scale_buffer_a, new_capacity * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&g_scale_buffer_b, new_capacity * sizeof(float));
        if (err != cudaSuccess) { cudaFree(g_scale_buffer_a); return err; }

        // Initialize to 1.0
        int threads = 256;
        int blocks = (new_capacity + threads - 1) / threads;
        fill_unity_kernel<<<blocks, threads, 0, stream>>>(g_scale_buffer_a, new_capacity);
        fill_unity_kernel<<<blocks, threads, 0, stream>>>(g_scale_buffer_b, new_capacity);
        cudaStreamSynchronize(stream);

        g_scale_capacity = new_capacity;
        return cudaSuccess;
    }
}

// ============================================================================
// GEMM Configuration - Same as v2 but with cached scale buffers
// ============================================================================

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

// ============================================================================
// Kernel Template with cached scale buffers
// FP8 blockscaled GEMM only supports cooperative schedule (not pingpong)
// ============================================================================

template<typename MmaTileShape>
struct FP8GemmKernelCached {
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

template<typename MmaTileShape>
cudaError_t run_gemm_cached(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    using Kernel = FP8GemmKernelCached<MmaTileShape>;
    using Gemm = typename Kernel::Gemm;
    using ScaleConfig = typename Kernel::ScaleConfig;
    using LayoutSFA = typename Kernel::LayoutSFA;
    using LayoutSFB = typename Kernel::LayoutSFB;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideC = typename Kernel::StrideC;
    using StrideD = typename Kernel::StrideD;

    // Allocate temporary C buffer
    int64_t size_D = static_cast<int64_t>(M) * N;
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_C(size_D);
    auto* d_C = buf_C.get();

    // Compute scale factor layouts
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    size_t sfa_size = size(filter_zeros(layout_SFA));
    size_t sfb_size = size(filter_zeros(layout_SFB));
    size_t max_scale_size = std::max(sfa_size, sfb_size);

    // Use cached scale buffers
    cudaError_t err = ensure_scale_buffers(max_scale_size, stream);
    if (err != cudaSuccess) return err;

    // Build strides
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
            g_scale_buffer_a, layout_SFA,
            g_scale_buffer_b, layout_SFB
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

}  // namespace fp8_fp8_gemm_sm120_v3
}  // namespace ops
}  // namespace pygpukit

extern "C" {

// V5: 128x128x128 with cached scale buffers
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v5(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream
) {
    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    return pygpukit::ops::fp8_fp8_gemm_sm120_v3::run_gemm_cached<TileShape>(A, B, D, M, N, K, alpha, beta, stream);
}

// V6: 128x256x64 (matches v2's best tile)
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v6(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream
) {
    using TileShape = cute::Shape<cute::_128, cute::_256, cute::_64>;
    return pygpukit::ops::fp8_fp8_gemm_sm120_v3::run_gemm_cached<TileShape>(A, B, D, M, N, K, alpha, beta, stream);
}

// V7: 256x128x64
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v7(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream
) {
    using TileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;
    return pygpukit::ops::fp8_fp8_gemm_sm120_v3::run_gemm_cached<TileShape>(A, B, D, M, N, K, alpha, beta, stream);
}

// V8: 128x128x64 with cached buffers
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v8(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream
) {
    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
    return pygpukit::ops::fp8_fp8_gemm_sm120_v3::run_gemm_cached<TileShape>(A, B, D, M, N, K, alpha, beta, stream);
}

// Cleanup function
void pygpukit_gemm_fp8_fp8_sm120_cleanup() {
    std::lock_guard<std::mutex> lock(pygpukit::ops::fp8_fp8_gemm_sm120_v3::g_scale_mutex);
    if (pygpukit::ops::fp8_fp8_gemm_sm120_v3::g_scale_buffer_a) {
        cudaFree(pygpukit::ops::fp8_fp8_gemm_sm120_v3::g_scale_buffer_a);
        pygpukit::ops::fp8_fp8_gemm_sm120_v3::g_scale_buffer_a = nullptr;
    }
    if (pygpukit::ops::fp8_fp8_gemm_sm120_v3::g_scale_buffer_b) {
        cudaFree(pygpukit::ops::fp8_fp8_gemm_sm120_v3::g_scale_buffer_b);
        pygpukit::ops::fp8_fp8_gemm_sm120_v3::g_scale_buffer_b = nullptr;
    }
    pygpukit::ops::fp8_fp8_gemm_sm120_v3::g_scale_capacity = 0;
}

}  // extern "C"

#else  // !SM120

extern "C" {
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v5(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v6(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v7(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t) { return cudaErrorNotSupported; }
cudaError_t pygpukit_gemm_fp8_fp8_sm120_v8(const uint8_t*, const uint8_t*, uint8_t*, int, int, int, float, float, cudaStream_t) { return cudaErrorNotSupported; }
void pygpukit_gemm_fp8_fp8_sm120_cleanup() {}
}

#endif
