/**
 * FP8 GEMM implementation for SM100 (Blackwell datacenter)
 *
 * Path:
 * 1. FP32 input
 * 2. FP8 quantization with blockwise scaling
 * 3. FP8 CUTLASS GEMM (SM100 tcgen05)
 * 4. FP32 output
 *
 * Based on CUTLASS example 81: blackwell_gemm_blockwise
 *
 * This serves as potential fallback for SM120 (Blackwell GeForce).
 * SM100 and SM120 are both Blackwell architecture - the kernel might work.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

// Only compile for SM100+
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

namespace pygpukit {
namespace ops {
namespace fp8_gemm_sm100 {

// ============================================================================
// GEMM Configuration: FP8 E4M3 x FP8 E4M3 -> FP32 with blockwise scaling
// Based on CUTLASS example 81
// ============================================================================

// A matrix: FP8 E4M3, RowMajor
using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

// B matrix: FP8 E4M3, ColumnMajor
using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

// Output: FP32 (we use bfloat16 internally then convert)
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = AlignmentC;

// Accumulator type
using ElementAccumulator = float;
using ElementCompute = float;

// SM100 Blackwell architecture
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// Tile and cluster shapes - using smaller tiles for better compatibility
using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

// Scale config for blockwise scaling
using ScaleConfig = decltype(cutlass::detail::sm100_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

// Epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Mainloop with blockwise scaling
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    cutlass::gemm::KernelScheduleSm100Blockwise
>::CollectiveOp;

// GEMM Kernel
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
// FP32 -> FP8 Quantization
// ============================================================================

constexpr float FP8_E4M3_MAX = 448.0f;

__device__ __forceinline__
uint8_t float_to_fp8_e4m3_scaled(float val, float inv_scale) {
    val = val * inv_scale;
    val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);

    if (fabsf(val) < 1e-7f) return 0;

    uint32_t bits = __float_as_uint(val);
    uint8_t sign = (bits >> 24) & 0x80;
    int exp = ((bits >> 23) & 0xFF) - 127 + 7;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp <= 0) return sign;
    if (exp >= 15) return sign | 0x7E;

    return sign | (static_cast<uint8_t>(exp) << 3) | static_cast<uint8_t>(mant >> 20);
}

__global__ void quantize_fp32_to_fp8_kernel(
    const float* __restrict__ input,
    cutlass::float_e4m3_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    uint8_t fp8 = float_to_fp8_e4m3_scaled(input[idx], 1.0f);
    output[idx] = cutlass::float_e4m3_t::bitcast(fp8);
}

__global__ void transpose_quantize_fp32_to_fp8_kernel(
    const float* __restrict__ input,
    cutlass::float_e4m3_t* __restrict__ output,
    int K, int N
) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= K || n >= N) return;

    float val = input[k * N + n];
    uint8_t fp8 = float_to_fp8_e4m3_scaled(val, 1.0f);
    output[k + n * K] = cutlass::float_e4m3_t::bitcast(fp8);
}

__global__ void fill_scale_factors_unity_kernel(
    float* __restrict__ scales,
    size_t num_scales
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_scales) return;
    scales[idx] = 1.0f;
}

__global__ void bf16_to_fp32_kernel(
    const cutlass::bfloat16_t* __restrict__ input,
    float* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    output[idx] = static_cast<float>(input[idx]);
}

// ============================================================================
// FP8 GEMM Entry Point
// ============================================================================

cudaError_t gemm_fp8(
    const float* A,
    const float* B,
    float* D,
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // Sizes
    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(K) * N;
    int64_t size_D = static_cast<int64_t>(M) * N;

    // Allocate FP8 buffers
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_A_fp8(size_A);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_B_fp8(size_B);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_C_bf16(size_D);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_D_bf16(size_D);

    auto* d_A_fp8 = buf_A_fp8.get();
    auto* d_B_fp8 = buf_B_fp8.get();
    auto* d_C_bf16 = buf_C_bf16.get();
    auto* d_D_bf16 = buf_D_bf16.get();

    // Scale factor sizes
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    size_t sfa_size = size(filter_zeros(layout_SFA));
    size_t sfb_size = size(filter_zeros(layout_SFB));

    cutlass::device_memory::allocation<float> buf_SFA(sfa_size);
    cutlass::device_memory::allocation<float> buf_SFB(sfb_size);

    auto* d_SFA = buf_SFA.get();
    auto* d_SFB = buf_SFB.get();

    // Quantize
    int threads = 256;
    int blocks_A = (size_A + threads - 1) / threads;

    quantize_fp32_to_fp8_kernel<<<blocks_A, threads, 0, stream>>>(A, d_A_fp8, size_A);

    dim3 block_B(16, 16);
    dim3 grid_B((N + 15) / 16, (K + 15) / 16);
    transpose_quantize_fp32_to_fp8_kernel<<<grid_B, block_B, 0, stream>>>(B, d_B_fp8, K, N);

    // Fill scale factors
    int blocks_SFA = (sfa_size + threads - 1) / threads;
    int blocks_SFB = (sfb_size + threads - 1) / threads;
    fill_scale_factors_unity_kernel<<<blocks_SFA, threads, 0, stream>>>(d_SFA, sfa_size);
    fill_scale_factors_unity_kernel<<<blocks_SFB, threads, 0, stream>>>(d_SFB, sfb_size);

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    // Build strides
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Build arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A_fp8, stride_a, d_B_fp8, stride_b, d_SFA, layout_SFA, d_SFB, layout_SFB},
        {{alpha, beta}, d_C_bf16, stride_c, d_D_bf16, stride_d}
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 GEMM SM100] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 GEMM SM100] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 GEMM SM100] run failed: %d\n", static_cast<int>(status));
        return cudaErrorLaunchFailure;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[FP8 GEMM SM100] sync failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Convert BF16 to FP32
    int blocks_D = (size_D + threads - 1) / threads;
    bf16_to_fp32_kernel<<<blocks_D, threads, 0, stream>>>(d_D_bf16, D, size_D);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

bool is_available() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    // SM100+ (Blackwell datacenter and consumer)
    return (props.major * 10 + props.minor) >= 100;
}

}  // namespace fp8_gemm_sm100
}  // namespace ops
}  // namespace pygpukit

extern "C" {
    cudaError_t pygpukit_gemm_fp8_sm100(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return pygpukit::ops::fp8_gemm_sm100::gemm_fp8(A, B, D, M, N, K, alpha, beta, stream);
    }

    bool pygpukit_fp8_sm100_available() {
        return pygpukit::ops::fp8_gemm_sm100::is_available();
    }
}

#else  // !SM100

namespace pygpukit {
namespace ops {
namespace fp8_gemm_sm100 {

cudaError_t gemm_fp8(
    const float* A, const float* B, float* D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

bool is_available() {
    return false;
}

}  // namespace fp8_gemm_sm100
}  // namespace ops
}  // namespace pygpukit

extern "C" {
    cudaError_t pygpukit_gemm_fp8_sm100(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return cudaErrorNotSupported;
    }

    bool pygpukit_fp8_sm100_available() {
        return false;
    }
}

#endif
