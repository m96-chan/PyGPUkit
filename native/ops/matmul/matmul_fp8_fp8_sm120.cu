/**
 * Pure FP8 GEMM implementation for SM120 (Blackwell GeForce)
 *
 * Path:
 * 1. FP8 E4M3 input (A, B already quantized)
 * 2. FP8 CUTLASS GEMM with blockwise scaling
 * 3. FP8 E4M3 output (direct, no conversion)
 *
 * This is the "true" FP8 GEMM for FP8 models (Llama 3.1 FP8, etc.)
 * where weights and activations are already in FP8 format.
 *
 * Implementation based on CUTLASS example 87a:
 * "87a_blackwell_geforce_fp8_bf16_gemm_blockwise"
 * Modified for FP8 output instead of BF16.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

// Enable FP8 SM120
#define PYGPUKIT_ENABLE_FP8_SM120

// Only compile for SM120+ AND when explicitly enabled
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

// Alignment patch for Issue #2902 workaround
#define PYGPUKIT_PATCH_CUTLASS_LDSM_POST 1
#include "aligned_copy_sm120.cuh"

using namespace cute;

namespace pygpukit {
namespace ops {
namespace fp8_fp8_gemm_sm120 {

// ============================================================================
// GEMM Configuration: FP8 E4M3 x FP8 E4M3 -> FP8 E4M3 with blockwise scaling
// ============================================================================

// A matrix: FP8 E4M3, RowMajor
using ElementA = cutlass::float_e4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

// B matrix: FP8 E4M3, ColumnMajor
using ElementB = cutlass::float_e4m3_t;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// Output: FP8 E4M3 (Pure FP8 output!)
using ElementC = cutlass::float_e4m3_t;
using ElementD = cutlass::float_e4m3_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = AlignmentC;

// Accumulator type (still float for precision)
using ElementAccumulator = float;
using ElementCompute = float;

// SM120 GeForce architecture with TensorOp
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// MMA and Cluster Tile Shapes
using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ClusterShape_MNK = Shape<_1, _1, _1>;  // GeForce: no cluster support

// Scale configuration (trivial blockwise scaling from example 87a)
using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

// Epilogue - outputs FP8
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
    void  // Default CLC scheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Stride and Layout types
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// ============================================================================
// Scale factor initialization (unity for now, can be extended for per-tensor/block)
// ============================================================================

__global__ void fill_scale_factors_unity_kernel(
    float* __restrict__ scales,
    size_t num_scales
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_scales) return;
    scales[idx] = 1.0f;
}

// ============================================================================
// FP8 -> FP8 GEMM Entry Point
// ============================================================================

cudaError_t gemm_fp8_fp8(
    const cutlass::float_e4m3_t* A,  // [M, K] FP8 input (RowMajor)
    const cutlass::float_e4m3_t* B,  // [K, N] FP8 input (ColumnMajor, pre-transposed)
    cutlass::float_e4m3_t* D,        // [M, N] FP8 output
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // Sizes
    int64_t size_D = static_cast<int64_t>(M) * N;

    // Allocate C buffer for epilogue (even with beta=0, CUTLASS needs valid pointer)
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_C(size_D);
    auto* d_C = buf_C.get();

    // Calculate scale factor sizes using ScaleConfig
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    size_t sfa_size = size(filter_zeros(layout_SFA));
    size_t sfb_size = size(filter_zeros(layout_SFB));

    // Pad to 32 floats (128 bytes) for TMA alignment
    size_t sfa_padded = std::max(sfa_size, size_t(32));
    size_t sfb_padded = std::max(sfb_size, size_t(32));

    cutlass::device_memory::allocation<float> buf_SFA(sfa_padded);
    cutlass::device_memory::allocation<float> buf_SFB(sfb_padded);

    auto* d_SFA = buf_SFA.get();
    auto* d_SFB = buf_SFB.get();

    // Fill scale factors with 1.0
    int threads = 256;
    int blocks_SFA_fill = (sfa_padded + threads - 1) / threads;
    int blocks_SFB_fill = (sfb_padded + threads - 1) / threads;
    fill_scale_factors_unity_kernel<<<blocks_SFA_fill, threads, 0, stream>>>(d_SFA, sfa_padded);
    fill_scale_factors_unity_kernel<<<blocks_SFB_fill, threads, 0, stream>>>(d_SFB, sfb_padded);

    // Build strides
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Build CUTLASS arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {  // Mainloop arguments
            A, stride_a,
            B, stride_b,
            d_SFA, layout_SFA,
            d_SFB, layout_SFB
        },
        {  // Epilogue arguments
            {},  // epilogue.thread
            d_C, stride_c,
            D, stride_d
        }
    };

    // Set alpha/beta
    arguments.epilogue.thread.alpha = alpha;
    arguments.epilogue.thread.beta = beta;

    // Instantiate and run GEMM
    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8_FP8 GEMM SM120] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8_FP8 GEMM SM120] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8_FP8 GEMM SM120] run failed: %d\n", static_cast<int>(status));
        return cudaErrorLaunchFailure;
    }

    return cudaSuccess;
}

// Wrapper for raw uint8_t pointers (for Python binding convenience)
cudaError_t gemm_fp8_fp8_raw(
    const uint8_t* A,  // [M, K] FP8 as raw bytes
    const uint8_t* B,  // [K, N] FP8 as raw bytes (ColumnMajor)
    uint8_t* D,        // [M, N] FP8 as raw bytes
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    return gemm_fp8_fp8(
        reinterpret_cast<const cutlass::float_e4m3_t*>(A),
        reinterpret_cast<const cutlass::float_e4m3_t*>(B),
        reinterpret_cast<cutlass::float_e4m3_t*>(D),
        M, N, K, alpha, beta, stream
    );
}

// ============================================================================
// Get scale factor sizes for a given problem size
// ============================================================================

void get_scale_sizes(int M, int N, int K, size_t* sfa_size, size_t* sfb_size) {
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    *sfa_size = size(filter_zeros(layout_SFA));
    *sfb_size = size(filter_zeros(layout_SFB));
}

// ============================================================================
// FP8 -> FP8 GEMM with Blockwise Scaling
// ============================================================================

cudaError_t gemm_fp8_fp8_blockwise(
    const cutlass::float_e4m3_t* A,  // [M, K] FP8 input (RowMajor)
    const cutlass::float_e4m3_t* B,  // [K, N] FP8 input (ColumnMajor, pre-transposed)
    cutlass::float_e4m3_t* D,        // [M, N] FP8 output
    const float* scale_A,            // Scale factors for A
    const float* scale_B,            // Scale factors for B
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // Sizes
    int64_t size_D = static_cast<int64_t>(M) * N;

    // Allocate C buffer for epilogue
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_C(size_D);
    auto* d_C = buf_C.get();

    // Calculate scale factor layouts
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    // Build strides
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Build CUTLASS arguments with user-provided scale factors
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {  // Mainloop arguments
            A, stride_a,
            B, stride_b,
            scale_A, layout_SFA,
            scale_B, layout_SFB
        },
        {  // Epilogue arguments
            {},  // epilogue.thread
            d_C, stride_c,
            D, stride_d
        }
    };

    // Set alpha/beta
    arguments.epilogue.thread.alpha = alpha;
    arguments.epilogue.thread.beta = beta;

    // Instantiate and run GEMM
    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8_FP8 Blockwise GEMM SM120] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8_FP8 Blockwise GEMM SM120] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8_FP8 Blockwise GEMM SM120] run failed: %d\n", static_cast<int>(status));
        return cudaErrorLaunchFailure;
    }

    return cudaSuccess;
}

// Wrapper for raw uint8_t pointers
cudaError_t gemm_fp8_fp8_blockwise_raw(
    const uint8_t* A,
    const uint8_t* B,
    uint8_t* D,
    const float* scale_A,
    const float* scale_B,
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    return gemm_fp8_fp8_blockwise(
        reinterpret_cast<const cutlass::float_e4m3_t*>(A),
        reinterpret_cast<const cutlass::float_e4m3_t*>(B),
        reinterpret_cast<cutlass::float_e4m3_t*>(D),
        scale_A, scale_B,
        M, N, K, alpha, beta, stream
    );
}

bool is_available() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    return (props.major * 10 + props.minor) >= 120;
}

}  // namespace fp8_fp8_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

// Extern C for linking
extern "C" {
    cudaError_t pygpukit_gemm_fp8_fp8_sm120(
        const uint8_t* A, const uint8_t* B, uint8_t* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return pygpukit::ops::fp8_fp8_gemm_sm120::gemm_fp8_fp8_raw(
            A, B, D, M, N, K, alpha, beta, stream
        );
    }

    bool pygpukit_fp8_fp8_sm120_available() {
        return pygpukit::ops::fp8_fp8_gemm_sm120::is_available();
    }

    // Blockwise scaled version
    cudaError_t pygpukit_gemm_fp8_fp8_blockwise_sm120(
        const uint8_t* A, const uint8_t* B, uint8_t* D,
        const float* scale_A, const float* scale_B,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return pygpukit::ops::fp8_fp8_gemm_sm120::gemm_fp8_fp8_blockwise_raw(
            A, B, D, scale_A, scale_B, M, N, K, alpha, beta, stream
        );
    }

    // Get scale factor sizes for a given problem
    void pygpukit_fp8_fp8_get_scale_sizes(
        int M, int N, int K,
        size_t* sfa_size, size_t* sfb_size
    ) {
        pygpukit::ops::fp8_fp8_gemm_sm120::get_scale_sizes(M, N, K, sfa_size, sfb_size);
    }
}

#else  // !SM120

namespace pygpukit {
namespace ops {
namespace fp8_fp8_gemm_sm120 {

cudaError_t gemm_fp8_fp8_raw(
    const uint8_t* A, const uint8_t* B, uint8_t* D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

bool is_available() {
    return false;
}

}  // namespace fp8_fp8_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

extern "C" {
    cudaError_t pygpukit_gemm_fp8_fp8_sm120(
        const uint8_t* A, const uint8_t* B, uint8_t* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return cudaErrorNotSupported;
    }

    bool pygpukit_fp8_fp8_sm120_available() {
        return false;
    }

    cudaError_t pygpukit_gemm_fp8_fp8_blockwise_sm120(
        const uint8_t* A, const uint8_t* B, uint8_t* D,
        const float* scale_A, const float* scale_B,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return cudaErrorNotSupported;
    }

    void pygpukit_fp8_fp8_get_scale_sizes(
        int M, int N, int K,
        size_t* sfa_size, size_t* sfb_size
    ) {
        *sfa_size = 0;
        *sfb_size = 0;
    }
}

#endif
