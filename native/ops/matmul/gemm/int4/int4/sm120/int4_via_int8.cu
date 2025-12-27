/**
 * Int4 GEMM for SM120 (Blackwell GeForce) via Int8/FP8 TensorCore
 *
 * SM120 does NOT have native Int4 TensorCore support for signed integers.
 * This implementation uses a two-stage approach:
 *   1. Unpack Int4 (2 values per byte) to Int8
 *   2. Run Int8 GEMM via FP8 TensorCore (using our existing implementation)
 *   3. Convert output to Int8/Int32
 *
 * Performance: Slightly lower than Int8 due to unpacking overhead
 * Precision: Approximate (FP8 E4M3 has non-uniform precision)
 *
 * Int4 storage format: Two signed 4-bit values packed per byte
 *   byte = (high_nibble << 4) | (low_nibble & 0xF)
 *   low_nibble: bits 0-3 (first value)
 *   high_nibble: bits 4-7 (second value)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

// Enable Int4 SM120
#define PYGPUKIT_ENABLE_INT4_SM120

#if (defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)) && defined(PYGPUKIT_ENABLE_INT4_SM120)

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
namespace int4_gemm_sm120 {

// ============================================================================
// FP8 GEMM Configuration (reuse from int8_via_fp8.cu)
// ============================================================================

using ElementA = cutlass::float_e4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// Use BF16 output to avoid FP8 saturation
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
// Int4 Unpacking and Conversion Kernels
// ============================================================================

// Unpack Int4 (packed 2 per byte) to Int8
// Input: packed_bytes[n/2] where each byte contains 2 Int4 values
// Output: unpacked_int8[n]
__global__ void unpack_int4_to_int8_kernel(
    const uint8_t* __restrict__ packed,
    int8_t* __restrict__ unpacked,
    size_t num_elements  // Number of Int4 values (must be even)
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t byte_idx = idx;  // One thread per packed byte

    if (byte_idx >= num_elements / 2) return;

    uint8_t packed_byte = packed[byte_idx];

    // Low nibble (bits 0-3) - sign extend from 4-bit to 8-bit
    int8_t low = static_cast<int8_t>(packed_byte << 4) >> 4;  // Sign extend

    // High nibble (bits 4-7) - sign extend from 4-bit to 8-bit
    int8_t high = static_cast<int8_t>(packed_byte) >> 4;  // Sign extend

    // Write two Int8 values
    unpacked[byte_idx * 2] = low;
    unpacked[byte_idx * 2 + 1] = high;
}

// Int8 to FP8 conversion (reuse from int8_via_fp8.cu)
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
    val = fminf(fmaxf(val, -2147483648.0f), 2147483647.0f);
    output[idx] = static_cast<int32_t>(roundf(val));
}

// BF16 to Int8 with descaling
__global__ void convert_bf16_to_int8_kernel(
    const cutlass::bfloat16_t* __restrict__ input,
    int8_t* __restrict__ output,
    size_t num_elements,
    float descale
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = static_cast<float>(input[idx]) * descale;
    val = fminf(fmaxf(val, -128.0f), 127.0f);
    output[idx] = static_cast<int8_t>(roundf(val));
}

// Unity scale factor kernel
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
// Int4 GEMM via Int8/FP8 TensorCore
// ============================================================================

cudaError_t gemm_int4_via_int8(
    const uint8_t* A_packed,    // [M, K/2] packed Int4 (RowMajor, 2 values per byte)
    const uint8_t* B_packed,    // [N, K/2] packed Int4 (ColumnMajor transposed, 2 values per byte)
    int32_t* D,                 // [M, N] Int32 output
    int M, int N, int K,        // K must be even
    float scale_A,              // Scale for A (typically 1.0)
    float scale_B,              // Scale for B
    float descale_D,            // Descale for D output
    cudaStream_t stream
) {
    if (K % 2 != 0) {
        return cudaErrorInvalidValue;  // K must be even for Int4 packing
    }

    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(N) * K;
    int64_t size_D = static_cast<int64_t>(M) * N;

    // Allocate buffers: Int8 unpacked + FP8 converted + BF16 output
    cutlass::device_memory::allocation<int8_t> buf_A_int8(size_A);
    cutlass::device_memory::allocation<int8_t> buf_B_int8(size_B);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_A_fp8(size_A);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_B_fp8(size_B);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_D_bf16(size_D);

    int threads = 256;

    // 1. Unpack Int4 to Int8
    int blocks_A_unpack = (size_A / 2 + threads - 1) / threads;
    int blocks_B_unpack = (size_B / 2 + threads - 1) / threads;
    unpack_int4_to_int8_kernel<<<blocks_A_unpack, threads, 0, stream>>>(
        A_packed, buf_A_int8.get(), size_A
    );
    unpack_int4_to_int8_kernel<<<blocks_B_unpack, threads, 0, stream>>>(
        B_packed, buf_B_int8.get(), size_B
    );

    // 2. Convert Int8 to FP8
    int blocks_A = (size_A + threads - 1) / threads;
    int blocks_B = (size_B + threads - 1) / threads;
    convert_int8_to_fp8_kernel<<<blocks_A, threads, 0, stream>>>(
        buf_A_int8.get(), buf_A_fp8.get(), size_A, scale_A
    );
    convert_int8_to_fp8_kernel<<<blocks_B, threads, 0, stream>>>(
        buf_B_int8.get(), buf_B_fp8.get(), size_B, scale_B
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

    // 3. Run FP8 GEMM
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

    // 4. Convert BF16 output to Int32
    int blocks_D = (size_D + threads - 1) / threads;
    convert_bf16_to_int32_kernel<<<blocks_D, threads, 0, stream>>>(
        buf_D_bf16.get(), D, size_D, descale_D
    );

    return cudaSuccess;
}

// Int4xInt4->Int8 version
cudaError_t gemm_int4_via_int8_int8_out(
    const uint8_t* A_packed,
    const uint8_t* B_packed,
    int8_t* D,
    int M, int N, int K,
    float scale_A,
    float scale_B,
    float descale_D,
    cudaStream_t stream
) {
    if (K % 2 != 0) {
        return cudaErrorInvalidValue;
    }

    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(N) * K;
    int64_t size_D = static_cast<int64_t>(M) * N;

    cutlass::device_memory::allocation<int8_t> buf_A_int8(size_A);
    cutlass::device_memory::allocation<int8_t> buf_B_int8(size_B);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_A_fp8(size_A);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_B_fp8(size_B);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_D_bf16(size_D);

    int threads = 256;

    // Unpack
    int blocks_A_unpack = (size_A / 2 + threads - 1) / threads;
    int blocks_B_unpack = (size_B / 2 + threads - 1) / threads;
    unpack_int4_to_int8_kernel<<<blocks_A_unpack, threads, 0, stream>>>(
        A_packed, buf_A_int8.get(), size_A
    );
    unpack_int4_to_int8_kernel<<<blocks_B_unpack, threads, 0, stream>>>(
        B_packed, buf_B_int8.get(), size_B
    );

    // Convert to FP8
    int blocks_A = (size_A + threads - 1) / threads;
    int blocks_B = (size_B + threads - 1) / threads;
    convert_int8_to_fp8_kernel<<<blocks_A, threads, 0, stream>>>(
        buf_A_int8.get(), buf_A_fp8.get(), size_A, scale_A
    );
    convert_int8_to_fp8_kernel<<<blocks_B, threads, 0, stream>>>(
        buf_B_int8.get(), buf_B_fp8.get(), size_B, scale_B
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

    // Convert to Int8
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

}  // namespace int4_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

extern "C" {

cudaError_t pygpukit_gemm_int4_int4_int32_sm120(
    const uint8_t* A_packed, const uint8_t* B_packed, int32_t* D,
    int M, int N, int K,
    float scale_A, float scale_B, float descale_D,
    cudaStream_t stream
) {
    return pygpukit::ops::int4_gemm_sm120::gemm_int4_via_int8(
        A_packed, B_packed, D, M, N, K, scale_A, scale_B, descale_D, stream
    );
}

cudaError_t pygpukit_gemm_int4_int4_int8_sm120(
    const uint8_t* A_packed, const uint8_t* B_packed, int8_t* D,
    int M, int N, int K,
    float scale_A, float scale_B, float descale_D,
    cudaStream_t stream
) {
    return pygpukit::ops::int4_gemm_sm120::gemm_int4_via_int8_int8_out(
        A_packed, B_packed, D, M, N, K, scale_A, scale_B, descale_D, stream
    );
}

bool pygpukit_int4_gemm_sm120_available() {
    return pygpukit::ops::int4_gemm_sm120::is_available();
}

}  // extern "C"

#else  // !SM120

extern "C" {

cudaError_t pygpukit_gemm_int4_int4_int32_sm120(
    const uint8_t*, const uint8_t*, int32_t*,
    int, int, int,
    float, float, float,
    cudaStream_t
) {
    return cudaErrorNotSupported;
}

cudaError_t pygpukit_gemm_int4_int4_int8_sm120(
    const uint8_t*, const uint8_t*, int8_t*,
    int, int, int,
    float, float, float,
    cudaStream_t
) {
    return cudaErrorNotSupported;
}

bool pygpukit_int4_gemm_sm120_available() {
    return false;
}

}  // extern "C"

#endif
