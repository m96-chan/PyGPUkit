/**
 * W8A16 GEMM for SM120 (Blackwell GeForce) using CUTLASS
 *
 * Strategy: Quantize BF16 activations to FP8 on-the-fly, then use FP8xFP8 TensorCore
 * This is faster than dequantizing FP8 weights to BF16 because:
 * 1. FP8 TensorCore is highly efficient on Blackwell
 * 2. BF16->FP8 quantization is cheap (truncation)
 * 3. No need to store dequantized weights in shared memory
 *
 * Data Flow:
 *   A: [M, K] BF16 activation -> quantize to FP8 ->
 *   B: [N, K] FP8 weight (transposed storage for ColumnMajor) ->
 *   FP8 x FP8 CUTLASS GEMM with blockwise scaling ->
 *   D: [M, N] BF16 output (FP8 accumulator converted to BF16)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>

#define PYGPUKIT_ENABLE_W8A16_SM120

#if (defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)) && defined(PYGPUKIT_ENABLE_W8A16_SM120)

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
namespace w8a16_cutlass_sm120 {

// ============================================================================
// GEMM Configuration: FP8 x FP8 -> BF16 with blockwise scaling
// Exactly matching fp8_blockwise.cu configuration
// ============================================================================

// A matrix: FP8 E4M3 (quantized from BF16 activation), RowMajor
using ElementA = cutlass::float_e4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

// B matrix: FP8 E4M3 (weight), ColumnMajor [K, N] (stored as [N, K])
using ElementB = cutlass::float_e4m3_t;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// Output: BF16
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

// Scale configuration
using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

// Epilogue - outputs BF16
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Mainloop with scale factor layouts
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
// BF16 to FP8 quantization kernel
// ============================================================================

constexpr float FP8_E4M3_MAX = 448.0f;

__device__ __forceinline__
uint8_t bf16_to_fp8_e4m3(float val) {
    val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    if (fabsf(val) < 1e-7f) return 0;

    uint32_t bits = __float_as_uint(val);
    uint8_t sign = (bits >> 24) & 0x80;
    int exp = ((bits >> 23) & 0xFF) - 127 + 7;  // FP8 E4M3 bias = 7
    uint32_t mant = bits & 0x7FFFFF;

    if (exp <= 0) return sign;
    if (exp >= 15) return sign | 0x7E;  // Max FP8 E4M3

    return sign | (static_cast<uint8_t>(exp) << 3) | static_cast<uint8_t>(mant >> 20);
}

__global__ void quantize_bf16_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ input,
    cutlass::float_e4m3_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Use same conversion as fp8_blockwise.cu (FP32 -> FP8)
    float val = __bfloat162float(input[idx]);
    uint8_t fp8 = bf16_to_fp8_e4m3(val);
    output[idx] = cutlass::float_e4m3_t::bitcast(fp8);
}

// Alternative: use same pattern as fp8_blockwise.cu
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

__global__ void quantize_bf16_to_fp8_v2_kernel(
    const __nv_bfloat16* __restrict__ input,
    cutlass::float_e4m3_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float val = __bfloat162float(input[idx]);
    uint8_t fp8 = float_to_fp8_e4m3_scaled(val, 1.0f);
    output[idx] = cutlass::float_e4m3_t::bitcast(fp8);
}

__global__ void fill_scale_factors_unity_kernel(
    float* __restrict__ scales,
    size_t num_scales
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_scales) return;
    scales[idx] = 1.0f;
}

// ============================================================================
// W8A16 GEMM Entry Point
// ============================================================================

cudaError_t gemm_w8a16(
    const cutlass::bfloat16_t* A_bf16,  // [M, K] BF16 activation
    const cutlass::float_e4m3_t* B,      // [N, K] FP8 weight (transposed for ColumnMajor)
    cutlass::bfloat16_t* D,              // [M, N] BF16 output
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    fprintf(stderr, "[W8A16 CUTLASS SM120] Starting M=%d, N=%d, K=%d\n", M, N, K);

    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(N) * K;
    int64_t size_D = static_cast<int64_t>(M) * N;

    // Allocate all internal buffers (guaranteed 128-byte alignment)
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_A_fp8(size_A);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_B_fp8(size_B);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_C(size_D);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_D(size_D);

    auto* d_A_fp8 = buf_A_fp8.get();
    auto* d_B_fp8 = buf_B_fp8.get();
    auto* d_C = buf_C.get();
    auto* d_D = buf_D.get();

    fprintf(stderr, "[W8A16 CUTLASS SM120] Alignment check:\n");
    fprintf(stderr, "  A_fp8 mod 128 = %llu\n", (unsigned long long)((uintptr_t)d_A_fp8 % 128));
    fprintf(stderr, "  B_fp8 mod 128 = %llu\n", (unsigned long long)((uintptr_t)d_B_fp8 % 128));
    fprintf(stderr, "  D_bf16 mod 128 = %llu\n", (unsigned long long)((uintptr_t)d_D % 128));

    // Quantize BF16 activations to FP8
    int threads = 256;
    int blocks = (size_A + threads - 1) / threads;
    quantize_bf16_to_fp8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A_bf16),
        d_A_fp8,
        size_A
    );

    // Copy B to aligned buffer
    cudaMemcpyAsync(d_B_fp8, B, size_B * sizeof(cutlass::float_e4m3_t),
                    cudaMemcpyDeviceToDevice, stream);

    // Calculate scale factor sizes
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    size_t sfa_size = size(filter_zeros(layout_SFA));
    size_t sfb_size = size(filter_zeros(layout_SFB));
    size_t sfa_padded = std::max(sfa_size, size_t(32));
    size_t sfb_padded = std::max(sfb_size, size_t(32));

    fprintf(stderr, "[W8A16 CUTLASS SM120] Scale sizes: SFA=%zu, SFB=%zu\n", sfa_size, sfb_size);

    cutlass::device_memory::allocation<float> buf_SFA(sfa_padded);
    cutlass::device_memory::allocation<float> buf_SFB(sfb_padded);

    // Fill scale factors with 1.0
    fill_scale_factors_unity_kernel<<<(sfa_padded + threads - 1) / threads, threads, 0, stream>>>(
        buf_SFA.get(), sfa_padded);
    fill_scale_factors_unity_kernel<<<(sfb_padded + threads - 1) / threads, threads, 0, stream>>>(
        buf_SFB.get(), sfb_padded);

    // Sync before CUTLASS GEMM
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[W8A16 CUTLASS SM120] Prep sync failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    fprintf(stderr, "[W8A16 CUTLASS SM120] Prep OK\n");

    // Build strides (matching fp8_blockwise.cu exactly)
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    fprintf(stderr, "[W8A16 CUTLASS SM120] Strides:\n");
    fprintf(stderr, "  stride_a: (%lld, %lld, %lld)\n",
            (long long)cute::get<0>(stride_a), (long long)cute::get<1>(stride_a), (long long)cute::get<2>(stride_a));
    fprintf(stderr, "  stride_b: (%lld, %lld, %lld)\n",
            (long long)cute::get<0>(stride_b), (long long)cute::get<1>(stride_b), (long long)cute::get<2>(stride_b));

    // Build CUTLASS arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            d_A_fp8, stride_a,
            d_B_fp8, stride_b,
            buf_SFA.get(), layout_SFA,
            buf_SFB.get(), layout_SFB
        },
        {
            {},
            d_C, stride_c,
            d_D, stride_d
        }
    };

    arguments.epilogue.thread.alpha = alpha;
    arguments.epilogue.thread.beta = beta;

    fprintf(stderr, "[W8A16 CUTLASS SM120] Arguments built\n");

    // Run GEMM
    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[W8A16 CUTLASS SM120] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }
    fprintf(stderr, "[W8A16 CUTLASS SM120] can_implement OK\n");

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    fprintf(stderr, "[W8A16 CUTLASS SM120] Workspace: %zu bytes\n", workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[W8A16 CUTLASS SM120] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }
    fprintf(stderr, "[W8A16 CUTLASS SM120] initialize OK\n");

    // Run without stream argument (matching fp8_blockwise.cu)
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[W8A16 CUTLASS SM120] run failed: %d\n", static_cast<int>(status));
        return cudaErrorLaunchFailure;
    }

    // Sync and check
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[W8A16 CUTLASS SM120] GEMM sync failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[W8A16 CUTLASS SM120] GEMM error: %s\n", cudaGetErrorString(err));
        return err;
    }
    fprintf(stderr, "[W8A16 CUTLASS SM120] GEMM OK\n");

    // Copy output to user buffer
    cudaMemcpy(D, d_D, size_D * sizeof(cutlass::bfloat16_t), cudaMemcpyDeviceToDevice);

    fprintf(stderr, "[W8A16 CUTLASS SM120] Complete\n");
    return cudaSuccess;
}

}  // namespace w8a16_cutlass_sm120
}  // namespace ops
}  // namespace pygpukit

// ============================================================================
// C API
// ============================================================================

extern "C" cudaError_t pygpukit_w8a16_cutlass_sm120(
    const void* A,      // [M, K] BF16 activation
    const void* B,      // [N, K] FP8 weight (transposed for ColumnMajor)
    void* D,            // [M, N] BF16 output
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    return pygpukit::ops::w8a16_cutlass_sm120::gemm_w8a16(
        reinterpret_cast<const cutlass::bfloat16_t*>(A),
        reinterpret_cast<const cutlass::float_e4m3_t*>(B),
        reinterpret_cast<cutlass::bfloat16_t*>(D),
        M, N, K, alpha, beta, stream
    );
}

#else  // !SM120

extern "C" cudaError_t pygpukit_w8a16_cutlass_sm120(
    const void* A, const void* B, void* D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

#endif
