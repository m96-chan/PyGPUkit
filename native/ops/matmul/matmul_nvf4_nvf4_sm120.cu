/**
 * NVF4 GEMM implementation for SM120 (Blackwell GeForce) - Pure NVF4 I/O
 *
 * Based on CUTLASS example 79a: blackwell_geforce_nvfp4_bf16_gemm
 *
 * This version takes pre-quantized NVF4 inputs directly to measure
 * pure GEMM kernel performance without quantization overhead.
 *
 * Data Flow:
 *   NVF4 input (packed) + Scale Factors -> CUTLASS GEMM -> BF16 output
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

// Enable NVF4 SM120
#define PYGPUKIT_ENABLE_NVF4_SM120

// Only compile for SM120+
#if (defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)) && defined(PYGPUKIT_ENABLE_NVF4_SM120)

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

namespace pygpukit {
namespace ops {
namespace nvf4_nvf4_gemm_sm120 {

// ============================================================================
// GEMM Configuration (from example 79a)
// ============================================================================

// A matrix configuration
using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVF4 wrapper type
using LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;  // Memory access granularity

// B matrix configuration
using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVF4 wrapper type
using LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

// C/D matrix configuration (BF16 output)
using ElementC    = cutlass::bfloat16_t;
using ElementD    = cutlass::bfloat16_t;
using LayoutCTag  = cutlass::layout::RowMajor;
using LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

// Kernel config
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Tile shapes - 128x128x128 (baseline, optimal for SM120)
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;  // GeForce: no cluster support

// Epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Mainloop - Pingpong schedule with 3-stage pipeline (optimal for SM120)
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCount<3>,  // 3 stages optimal (2=base, 4=too much smem)
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

// GEMM Kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Types for data layout
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Data types for raw storage
using DataTypeA = typename ElementA::DataType;           // float_e2m1_t
using ScaleFactorType = typename ElementA::ScaleFactorType;  // float_ue4m3_t

// ============================================================================
// NVF4 GEMM Entry Point (Pre-quantized NVF4 I/O)
// ============================================================================

cudaError_t gemm_nvf4_nvf4(
    const uint8_t* A_packed,     // [M, K] NVF4 packed (M*K/2 bytes), RowMajor
    const uint8_t* B_packed,     // [N, K] NVF4 packed (N*K/2 bytes), ColMajor
    const uint8_t* SFA,          // Scale factors for A
    const uint8_t* SFB,          // Scale factors for B
    nv_bfloat16* D,              // [M, N] BF16 output (device)
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // For SFA and SFB tensors layouts
    using Sm1xxBlkScaledConfigLocal = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    // Build strides and layouts
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = Sm1xxBlkScaledConfigLocal::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfigLocal::tile_atom_to_shape_SFB(problem_shape);

    // Compute sizes
    int64_t size_C = static_cast<int64_t>(M) * N;
    int64_t size_D = size_C;

    // Allocate output buffers
    cutlass::device_memory::allocation<ElementC> dev_C(size_C);
    cutlass::device_memory::allocation<ElementD> dev_D_out(size_D);

    cudaError_t err;

    // Initialize C to zero
    err = cudaMemsetAsync(dev_C.get(), 0, size_C * sizeof(ElementC), stream);
    if (err != cudaSuccess) return err;

    // Build GEMM arguments using pre-quantized device memory
    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
            reinterpret_cast<const DataTypeA*>(A_packed), stride_A,
            reinterpret_cast<const DataTypeA*>(B_packed), stride_B,
            reinterpret_cast<const ScaleFactorType*>(SFA), layout_SFA,
            reinterpret_cast<const ScaleFactorType*>(SFB), layout_SFB
        },
        { // Epilogue arguments
            {alpha, beta},
            dev_C.get(), stride_C,
            dev_D_out.get(), stride_D
        }
    };

    // Run GEMM
    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[NVF4 GEMM] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[NVF4 GEMM] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[NVF4 GEMM] run failed: %d\n", static_cast<int>(status));
        return cudaErrorLaunchFailure;
    }

    // Copy result from CUTLASS output buffer to user-provided D buffer (D2D only!)
    err = cudaMemcpyAsync(D, dev_D_out.get(),
                          size_D * sizeof(nv_bfloat16),
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        return err;
    }

    return cudaSuccess;
}

// ============================================================================
// Benchmark helper: prepare pre-quantized data and run GEMM
// ============================================================================

// Initialize scale factors to 1.0 (UE4M3 encoding: 0x38)
__global__ void init_scale_factors_kernel(uint8_t* sf, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    sf[idx] = 0x38;  // float_ue4m3_t(1.0f) = 0x38
}

// Initialize NVF4 data to 1.0 (E2M1 encoding: 0x22 = two 1.0 values packed)
__global__ void init_nvf4_ones_kernel(uint8_t* data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    // E2M1 1.0 = 0x2, packed: low nibble = 0x2, high nibble = 0x2 -> 0x22
    data[idx] = 0x22;
}

// Benchmark entry point: allocates, initializes, and runs GEMM (all inline)
cudaError_t benchmark_gemm_nvf4(
    nv_bfloat16* D,              // [M, N] BF16 output (device, pre-allocated)
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    using Sm1xxBlkScaledConfigLocal = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    // Build strides and layouts
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = Sm1xxBlkScaledConfigLocal::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfigLocal::tile_atom_to_shape_SFB(problem_shape);

    // Compute sizes
    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(K) * N;
    int64_t size_C = static_cast<int64_t>(M) * N;
    int64_t size_D = size_C;

    size_t sfa_size = cute::size(cute::filter_zeros(layout_SFA));
    size_t sfb_size = cute::size(cute::filter_zeros(layout_SFB));

    // WORKAROUND: Blackwell driver TMA bug requires >= 128KB allocations
    constexpr size_t MIN_ALLOC_128KB = 128 * 1024;
    size_t min_sf_elements = MIN_ALLOC_128KB / sizeof(ScaleFactorType);

    size_t sfa_padded = std::max(sfa_size, min_sf_elements);
    size_t sfb_padded = std::max(sfb_size, min_sf_elements);

    // NVF4 packed sizes (with 128KB minimum)
    size_t size_A_packed = (size_A + 1) / 2;
    size_t size_B_packed = (size_B + 1) / 2;
    size_t size_A_padded = std::max(size_A_packed, MIN_ALLOC_128KB);
    size_t size_B_padded = std::max(size_B_packed, MIN_ALLOC_128KB);

    // Allocate device memory (no need to allocate D - use user buffer directly)
    cutlass::device_memory::allocation<uint8_t> dev_A(size_A_padded);
    cutlass::device_memory::allocation<uint8_t> dev_B(size_B_padded);
    cutlass::device_memory::allocation<uint8_t> dev_SFA(sfa_padded);
    cutlass::device_memory::allocation<uint8_t> dev_SFB(sfb_padded);
    cutlass::device_memory::allocation<ElementC> dev_C(size_C);

    cudaError_t err;

    // Initialize C to zero
    err = cudaMemsetAsync(dev_C.get(), 0, size_C * sizeof(ElementC), stream);
    if (err != cudaSuccess) return err;

    constexpr int BLOCK_SIZE = 256;

    // Initialize A and B to 1.0
    {
        int grid_a = (size_A_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int grid_b = (size_B_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_nvf4_ones_kernel<<<grid_a, BLOCK_SIZE, 0, stream>>>(dev_A.get(), size_A_padded);
        init_nvf4_ones_kernel<<<grid_b, BLOCK_SIZE, 0, stream>>>(dev_B.get(), size_B_padded);
    }

    // Initialize scale factors to 1.0
    {
        int grid_sfa = (sfa_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int grid_sfb = (sfb_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_scale_factors_kernel<<<grid_sfa, BLOCK_SIZE, 0, stream>>>(dev_SFA.get(), sfa_padded);
        init_scale_factors_kernel<<<grid_sfb, BLOCK_SIZE, 0, stream>>>(dev_SFB.get(), sfb_padded);
    }

    // Sync before GEMM
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    // Build GEMM arguments - use D directly (no intermediate buffer)
    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
            reinterpret_cast<DataTypeA*>(dev_A.get()), stride_A,
            reinterpret_cast<DataTypeA*>(dev_B.get()), stride_B,
            reinterpret_cast<ScaleFactorType*>(dev_SFA.get()), layout_SFA,
            reinterpret_cast<ScaleFactorType*>(dev_SFB.get()), layout_SFB
        },
        { // Epilogue arguments - write directly to user buffer
            {alpha, beta},
            dev_C.get(), stride_C,
            reinterpret_cast<ElementD*>(D), stride_D
        }
    };

    // Run GEMM
    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[NVF4 Bench] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[NVF4 Bench] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[NVF4 Bench] run failed: %d\n", static_cast<int>(status));
        return cudaErrorLaunchFailure;
    }

    // No D2D copy needed - CUTLASS writes directly to user buffer D
    return cudaSuccess;
}

bool is_available() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    return (props.major == 12 && (props.minor == 0 || props.minor == 1));
}

}  // namespace nvf4_nvf4_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

// Extern C for linking
extern "C" {
    cudaError_t pygpukit_gemm_nvf4_nvf4_sm120(
        const uint8_t* A_packed, const uint8_t* B_packed,
        const uint8_t* SFA, const uint8_t* SFB,
        nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return pygpukit::ops::nvf4_nvf4_gemm_sm120::gemm_nvf4_nvf4(
            A_packed, B_packed, SFA, SFB, D, M, N, K, alpha, beta, stream
        );
    }

    cudaError_t pygpukit_benchmark_gemm_nvf4_sm120(
        nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return pygpukit::ops::nvf4_nvf4_gemm_sm120::benchmark_gemm_nvf4(
            D, M, N, K, alpha, beta, stream
        );
    }

    bool pygpukit_nvf4_nvf4_sm120_available() {
        return pygpukit::ops::nvf4_nvf4_gemm_sm120::is_available();
    }
}

#else  // !SM120

namespace pygpukit {
namespace ops {
namespace nvf4_nvf4_gemm_sm120 {

cudaError_t gemm_nvf4_nvf4(
    const uint8_t* A_packed, const uint8_t* B_packed,
    const uint8_t* SFA, const uint8_t* SFB,
    nv_bfloat16* D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t benchmark_gemm_nvf4(
    nv_bfloat16* D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

bool is_available() {
    return false;
}

}  // namespace nvf4_nvf4_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

extern "C" {
    cudaError_t pygpukit_gemm_nvf4_nvf4_sm120(
        const uint8_t* A_packed, const uint8_t* B_packed,
        const uint8_t* SFA, const uint8_t* SFB,
        nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return cudaErrorNotSupported;
    }

    cudaError_t pygpukit_benchmark_gemm_nvf4_sm120(
        nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return cudaErrorNotSupported;
    }

    bool pygpukit_nvf4_nvf4_sm120_available() {
        return false;
    }
}

#endif
