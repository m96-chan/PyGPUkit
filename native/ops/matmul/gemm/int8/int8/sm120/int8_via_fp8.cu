/**
 * Int8 GEMM for SM120 (Blackwell GeForce) via FP8 TensorCore
 *
 * SM120 does NOT have native Int8 TensorCore support (only SM100/SM101/SM110 do).
 * This implementation uses FP8 TensorCore as an approximation:
 *   1. Convert Int8 inputs to FP8 (with scaling)
 *   2. Run fast FP8xFP8 GEMM
 *   3. Convert output back to Int8/Int32
 *
 * Performance: ~200+ TFLOPS (matches FP8 ceiling)
 * Precision: Approximate (FP8 E4M3 has non-uniform precision)
 *
 * For true Int8 GEMM, use SM100/SM101/SM110 or SIMT fallback.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

// Enable FP8 SM120
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
namespace int8_gemm_sm120 {

// ============================================================================
// FP8 GEMM Configuration (reuse from fp8_cutlass.cu)
// ============================================================================

using ElementA = cutlass::float_e4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// Use BF16 output to avoid FP8 saturation - allows full accumulator range
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
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
    MmaTileShape_MNK, ClusterShape_MNK,
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

// ============================================================================
// Conversion Kernels
// ============================================================================

// Int8 to FP8 with scaling
// FP8 E4M3 range: [-448, 448]
// Int8 range: [-128, 127]
// Scale factor: 1.0 works for typical quantized data
__global__ void convert_int8_to_fp8_kernel(
    const int8_t* __restrict__ input,
    cutlass::float_e4m3_t* __restrict__ output,
    size_t num_elements,
    float scale
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = static_cast<float>(input[idx]) * scale;
    output[idx] = cutlass::float_e4m3_t(val);
}

// BF16 to Int32 with descaling
__global__ void convert_bf16_to_int32_kernel(
    const cutlass::bfloat16_t* __restrict__ input,
    int32_t* __restrict__ output,
    size_t num_elements,
    float descale
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = static_cast<float>(input[idx]) * descale;
    // Clamp to Int32 range
    val = fminf(fmaxf(val, -2147483648.0f), 2147483647.0f);
    output[idx] = static_cast<int32_t>(roundf(val));
}

// BF16 to Int8 with descaling (for output quantization)
__global__ void convert_bf16_to_int8_kernel(
    const cutlass::bfloat16_t* __restrict__ input,
    int8_t* __restrict__ output,
    size_t num_elements,
    float descale
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = static_cast<float>(input[idx]) * descale;
    // Clamp to Int8 range
    val = fminf(fmaxf(val, -128.0f), 127.0f);
    output[idx] = static_cast<int8_t>(roundf(val));
}

// Unity scale factor kernel (reuse)
__global__ void fill_unity_kernel(float* scales, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) scales[idx] = 1.0f;
}

// Thread-local cached scale buffers
static thread_local cutlass::device_memory::allocation<float> s_cached_SFA;
static thread_local cutlass::device_memory::allocation<float> s_cached_SFB;
static thread_local size_t s_cached_sfa_size = 0;
static thread_local size_t s_cached_sfb_size = 0;

// ============================================================================
// Int8 GEMM via FP8 TensorCore
// ============================================================================

cudaError_t gemm_int8_via_fp8(
    const int8_t* A,        // [M, K] Int8 input (RowMajor)
    const int8_t* B,        // [N, K] Int8 input (ColumnMajor, stored as transposed)
    int32_t* D,             // [M, N] Int32 output
    int M, int N, int K,
    float scale_A,          // Scale for A (typically 1.0 for normalized data)
    float scale_B,          // Scale for B
    float descale_D,        // Descale for D output
    cudaStream_t stream
) {
    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(N) * K;
    int64_t size_D = static_cast<int64_t>(M) * N;

    // Allocate FP8 buffers for A and B, BF16 for D (to avoid saturation)
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_A_fp8(size_A);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_B_fp8(size_B);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_D_bf16(size_D);

    int threads = 256;

    // 1. Convert Int8 inputs to FP8
    int blocks_A = (size_A + threads - 1) / threads;
    int blocks_B = (size_B + threads - 1) / threads;
    convert_int8_to_fp8_kernel<<<blocks_A, threads, 0, stream>>>(
        A, buf_A_fp8.get(), size_A, scale_A
    );
    convert_int8_to_fp8_kernel<<<blocks_B, threads, 0, stream>>>(
        B, buf_B_fp8.get(), size_B, scale_B
    );

    // Calculate scale layouts
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    size_t sfa_size = size(filter_zeros(layout_SFA));
    size_t sfb_size = size(filter_zeros(layout_SFB));
    size_t sfa_padded = std::max(sfa_size, size_t(32));
    size_t sfb_padded = std::max(sfb_size, size_t(32));

    // Use cached scale buffers
    if (s_cached_sfa_size < sfa_padded) {
        s_cached_SFA.reset(sfa_padded);
        s_cached_sfa_size = sfa_padded;
        int blocks_sfa = (sfa_padded + threads - 1) / threads;
        fill_unity_kernel<<<blocks_sfa, threads, 0, stream>>>(s_cached_SFA.get(), sfa_padded);
    }
    if (s_cached_sfb_size < sfb_padded) {
        s_cached_SFB.reset(sfb_padded);
        s_cached_sfb_size = sfb_padded;
        int blocks_sfb = (sfb_padded + threads - 1) / threads;
        fill_unity_kernel<<<blocks_sfb, threads, 0, stream>>>(s_cached_SFB.get(), sfb_padded);
    }

    // 2. Run FP8 GEMM
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            buf_A_fp8.get(), stride_a,
            buf_B_fp8.get(), stride_b,
            s_cached_SFA.get(), layout_SFA,
            s_cached_SFB.get(), layout_SFB
        },
        {
            {},
            buf_D_bf16.get(), stride_c,
            buf_D_bf16.get(), stride_d
        }
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

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

    // 3. Convert BF16 output to Int32
    int blocks_D = (size_D + threads - 1) / threads;
    convert_bf16_to_int32_kernel<<<blocks_D, threads, 0, stream>>>(
        buf_D_bf16.get(), D, size_D, descale_D
    );

    return cudaSuccess;
}

// Int8xInt8->Int8 version (for quantized inference)
cudaError_t gemm_int8_via_fp8_int8_out(
    const int8_t* A,        // [M, K] Int8 input
    const int8_t* B,        // [N, K] Int8 input (transposed)
    int8_t* D,              // [M, N] Int8 output
    int M, int N, int K,
    float scale_A,
    float scale_B,
    float descale_D,
    cudaStream_t stream
) {
    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(N) * K;
    int64_t size_D = static_cast<int64_t>(M) * N;

    // Allocate FP8 buffers for A and B, BF16 for D (to avoid saturation)
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_A_fp8(size_A);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_B_fp8(size_B);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_D_bf16(size_D);

    int threads = 256;

    // Convert inputs
    int blocks_A = (size_A + threads - 1) / threads;
    int blocks_B = (size_B + threads - 1) / threads;
    convert_int8_to_fp8_kernel<<<blocks_A, threads, 0, stream>>>(
        A, buf_A_fp8.get(), size_A, scale_A
    );
    convert_int8_to_fp8_kernel<<<blocks_B, threads, 0, stream>>>(
        B, buf_B_fp8.get(), size_B, scale_B
    );

    // Scale layouts
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    size_t sfa_size = size(filter_zeros(layout_SFA));
    size_t sfb_size = size(filter_zeros(layout_SFB));
    size_t sfa_padded = std::max(sfa_size, size_t(32));
    size_t sfb_padded = std::max(sfb_size, size_t(32));

    if (s_cached_sfa_size < sfa_padded) {
        s_cached_SFA.reset(sfa_padded);
        s_cached_sfa_size = sfa_padded;
        fill_unity_kernel<<<(sfa_padded + threads - 1) / threads, threads, 0, stream>>>(
            s_cached_SFA.get(), sfa_padded);
    }
    if (s_cached_sfb_size < sfb_padded) {
        s_cached_SFB.reset(sfb_padded);
        s_cached_sfb_size = sfb_padded;
        fill_unity_kernel<<<(sfb_padded + threads - 1) / threads, threads, 0, stream>>>(
            s_cached_SFB.get(), sfb_padded);
    }

    // GEMM
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            buf_A_fp8.get(), stride_a,
            buf_B_fp8.get(), stride_b,
            s_cached_SFA.get(), layout_SFA,
            s_cached_SFB.get(), layout_SFB
        },
        {
            {},
            buf_D_bf16.get(), stride_c,
            buf_D_bf16.get(), stride_d
        }
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return cudaErrorInvalidValue;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) return cudaErrorInvalidValue;

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;

    // Convert BF16 to Int8
    int blocks_D = (size_D + threads - 1) / threads;
    convert_bf16_to_int8_kernel<<<blocks_D, threads, 0, stream>>>(
        buf_D_bf16.get(), D, size_D, descale_D
    );

    return cudaSuccess;
}

bool is_available() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    return (props.major * 10 + props.minor) >= 120;
}

}  // namespace int8_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

extern "C" {

cudaError_t pygpukit_gemm_int8_int8_int32_sm120(
    const int8_t* A, const int8_t* B, int32_t* D,
    int M, int N, int K,
    float scale_A, float scale_B, float descale_D,
    cudaStream_t stream
) {
    return pygpukit::ops::int8_gemm_sm120::gemm_int8_via_fp8(
        A, B, D, M, N, K, scale_A, scale_B, descale_D, stream
    );
}

cudaError_t pygpukit_gemm_int8_int8_int8_sm120(
    const int8_t* A, const int8_t* B, int8_t* D,
    int M, int N, int K,
    float scale_A, float scale_B, float descale_D,
    cudaStream_t stream
) {
    return pygpukit::ops::int8_gemm_sm120::gemm_int8_via_fp8_int8_out(
        A, B, D, M, N, K, scale_A, scale_B, descale_D, stream
    );
}

bool pygpukit_int8_gemm_sm120_available() {
    return pygpukit::ops::int8_gemm_sm120::is_available();
}

}  // extern "C"

#else  // !SM120

extern "C" {

cudaError_t pygpukit_gemm_int8_int8_int32_sm120(
    const int8_t*, const int8_t*, int32_t*,
    int, int, int,
    float, float, float,
    cudaStream_t
) {
    return cudaErrorNotSupported;
}

cudaError_t pygpukit_gemm_int8_int8_int8_sm120(
    const int8_t*, const int8_t*, int8_t*,
    int, int, int,
    float, float, float,
    cudaStream_t
) {
    return cudaErrorNotSupported;
}

bool pygpukit_int8_gemm_sm120_available() {
    return false;
}

}  // extern "C"

#endif
