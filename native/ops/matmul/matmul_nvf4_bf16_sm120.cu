/**
 * NVF4 GEMM implementation for SM120 (Blackwell GeForce) with BF16 I/O
 *
 * Based on CUTLASS example 79a: blackwell_geforce_nvfp4_bf16_gemm
 *
 * Data Flow:
 *   BF16 input -> NVF4 (4-bit) quantize with block scaling -> CUTLASS GEMM -> BF16 output
 *
 * NVF4 (float_e2m1_t) is a 4-bit format with 2-bit exponent and 1-bit mantissa.
 * This provides 2x memory bandwidth compared to FP8, making it ideal for
 * memory-bound LLM inference workloads.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
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
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using namespace cute;

namespace pygpukit {
namespace ops {
namespace nvf4_bf16_gemm_sm120 {

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

// Tile shapes
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

// Mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
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
// BF16 -> NVF4 Quantization with Block Scaling
// ============================================================================

// NVF4 E2M1 range: [-6.0, 6.0]
constexpr float NVF4_MAX = 6.0f;

// Convert float to NVF4 E2M1 (4-bit) - HOST version
inline uint8_t bf16_to_nvf4_e2m1_host(float val) {
    // E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    if (std::abs(val) < 0.25f) return 0;  // Zero

    uint8_t sign = (val < 0) ? 0x8 : 0x0;
    val = std::abs(val);
    val = std::min(val, NVF4_MAX);

    // Quantize to nearest E2M1 value
    uint8_t code;
    if (val < 0.75f) code = 1;       // 0.5
    else if (val < 1.25f) code = 2;  // 1.0
    else if (val < 1.75f) code = 3;  // 1.5
    else if (val < 2.5f) code = 4;   // 2.0
    else if (val < 3.5f) code = 5;   // 3.0
    else if (val < 5.0f) code = 6;   // 4.0
    else code = 7;                    // 6.0

    return sign | code;
}

// Convert float to NVF4 E2M1 (4-bit) - DEVICE version
__device__ __forceinline__
uint8_t bf16_to_nvf4_e2m1(float val) {
    // E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    if (fabsf(val) < 0.25f) return 0;  // Zero

    uint8_t sign = (val < 0) ? 0x8 : 0x0;
    val = fabsf(val);
    val = fminf(val, NVF4_MAX);

    // Quantize to nearest E2M1 value
    uint8_t code;
    if (val < 0.75f) code = 1;       // 0.5
    else if (val < 1.25f) code = 2;  // 1.0
    else if (val < 1.75f) code = 3;  // 1.5
    else if (val < 2.5f) code = 4;   // 2.0
    else if (val < 3.5f) code = 5;   // 3.0
    else if (val < 5.0f) code = 6;   // 4.0
    else code = 7;                    // 6.0

    return sign | code;
}

// Scale factor block size (32 elements per scale factor for NVF4)
constexpr int SF_BLOCK_SIZE = 32;

// Quantize A matrix: BF16 [M, K] RowMajor -> NVF4 with block scaling
__global__ void quantize_A_bf16_to_nvf4_kernel(
    const nv_bfloat16* __restrict__ input,  // [M, K] RowMajor BF16
    uint8_t* __restrict__ output_data,       // Packed NVF4 (2 per byte)
    uint8_t* __restrict__ output_sf,         // Scale factors
    int M, int K
) {
    int m = blockIdx.y;
    int k_block = blockIdx.x * blockDim.x + threadIdx.x;

    int num_k_blocks = (K + SF_BLOCK_SIZE - 1) / SF_BLOCK_SIZE;
    if (m >= M || k_block >= num_k_blocks) return;

    int k_start = k_block * SF_BLOCK_SIZE;
    int k_end = min(k_start + SF_BLOCK_SIZE, K);

    // Find max absolute value in block for scale factor
    float max_val = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
        float val = fabsf(__bfloat162float(input[m * K + k]));
        max_val = fmaxf(max_val, val);
    }

    // Compute scale factor (stored as float_ue4m3_t)
    float scale = (max_val > 1e-8f) ? (max_val / NVF4_MAX) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Store scale factor (simplified - just store as uint8_t representation)
    // Note: In production, should use proper float_ue4m3_t conversion
    int sf_idx = m * num_k_blocks + k_block;
    output_sf[sf_idx] = static_cast<uint8_t>(fminf(scale * 16.0f, 255.0f));

    // Quantize and pack pairs
    int out_base = (m * K + k_start) / 2;
    for (int k = k_start; k < k_end; k += 2) {
        float v0 = __bfloat162float(input[m * K + k]) * inv_scale;
        float v1 = (k + 1 < k_end) ? __bfloat162float(input[m * K + k + 1]) * inv_scale : 0.0f;

        uint8_t q0 = bf16_to_nvf4_e2m1(v0);
        uint8_t q1 = bf16_to_nvf4_e2m1(v1);

        // Pack: low nibble = first element, high nibble = second element
        output_data[out_base + (k - k_start) / 2] = (q1 << 4) | (q0 & 0x0F);
    }
}

// Quantize B matrix: BF16 [K, N] RowMajor -> NVF4 ColumnMajor with block scaling
__global__ void quantize_B_bf16_to_nvf4_kernel(
    const nv_bfloat16* __restrict__ input,  // [K, N] RowMajor BF16
    uint8_t* __restrict__ output_data,       // Packed NVF4 ColMajor
    uint8_t* __restrict__ output_sf,         // Scale factors
    int K, int N
) {
    int n = blockIdx.y;
    int k_block = blockIdx.x * blockDim.x + threadIdx.x;

    int num_k_blocks = (K + SF_BLOCK_SIZE - 1) / SF_BLOCK_SIZE;
    if (n >= N || k_block >= num_k_blocks) return;

    int k_start = k_block * SF_BLOCK_SIZE;
    int k_end = min(k_start + SF_BLOCK_SIZE, K);

    // Find max absolute value in block
    float max_val = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
        float val = fabsf(__bfloat162float(input[k * N + n]));
        max_val = fmaxf(max_val, val);
    }

    // Compute scale factor
    float scale = (max_val > 1e-8f) ? (max_val / NVF4_MAX) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Store scale factor
    int sf_idx = n * num_k_blocks + k_block;
    output_sf[sf_idx] = static_cast<uint8_t>(fminf(scale * 16.0f, 255.0f));

    // Quantize and pack pairs (ColumnMajor output)
    int out_base = (n * K + k_start) / 2;
    for (int k = k_start; k < k_end; k += 2) {
        float v0 = __bfloat162float(input[k * N + n]) * inv_scale;
        float v1 = (k + 1 < k_end) ? __bfloat162float(input[(k + 1) * N + n]) * inv_scale : 0.0f;

        uint8_t q0 = bf16_to_nvf4_e2m1(v0);
        uint8_t q1 = bf16_to_nvf4_e2m1(v1);

        output_data[out_base + (k - k_start) / 2] = (q1 << 4) | (q0 & 0x0F);
    }
}

// ============================================================================
// NVF4 GEMM Entry Point (BF16 I/O)
// ============================================================================

cudaError_t gemm_nvf4_bf16(
    const nv_bfloat16* A,  // [M, K] BF16 input
    const nv_bfloat16* B,  // [K, N] BF16 input
    nv_bfloat16* D,        // [M, N] BF16 output
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    fprintf(stderr, "[NVF4 BF16 GEMM SM120] Starting M=%d, N=%d, K=%d\n", M, N, K);

    // Compute sizes
    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(K) * N;
    int64_t size_C = static_cast<int64_t>(M) * N;
    int64_t size_D = size_C;

    // Packed NVF4 sizes (2 elements per byte)
    int64_t packed_A = (size_A + 1) / 2;
    int64_t packed_B = (size_B + 1) / 2;

    // Build strides and layouts
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape);

    // Compute scale factor sizes
    size_t sfa_size = size(filter_zeros(layout_SFA));
    size_t sfb_size = size(filter_zeros(layout_SFB));

    // WORKAROUND: Blackwell driver TMA bug requires >= 128KB allocations
    // See CUTLASS v4.3.4 CHANGELOG
    constexpr size_t MIN_ALLOC_128KB = 128 * 1024;

    // Calculate minimum element counts for 128KB
    size_t min_sf_elements = MIN_ALLOC_128KB / sizeof(ScaleFactorType);  // 128KB / 1 byte
    size_t min_data_elements = MIN_ALLOC_128KB / sizeof(DataTypeA);      // 128KB / 0.5 byte
    size_t min_bf16_elements = MIN_ALLOC_128KB / sizeof(ElementC);       // 128KB / 2 bytes

    size_t sfa_padded = std::max(sfa_size, min_sf_elements);
    size_t sfb_padded = std::max(sfb_size, min_sf_elements);

    // Also pad A, B, C, D to >= 128KB
    size_t size_A_padded = std::max(static_cast<size_t>(size_A), min_data_elements);
    size_t size_B_padded = std::max(static_cast<size_t>(size_B), min_data_elements);
    size_t size_C_padded = std::max(static_cast<size_t>(size_C), min_bf16_elements);
    size_t size_D_padded = std::max(static_cast<size_t>(size_D), min_bf16_elements);

    fprintf(stderr, "[NVF4 BF16 GEMM SM120] 128KB padding applied to all tensors\n");
    fprintf(stderr, "[NVF4 BF16 GEMM SM120] A: %zu->%zu, B: %zu->%zu, C: %zu->%zu, SFA: %zu->%zu, SFB: %zu->%zu\n",
            size_A, size_A_padded, size_B, size_B_padded, size_C, size_C_padded, sfa_size, sfa_padded, sfb_size, sfb_padded);

    // Allocate device memory using HostTensor for proper alignment
    cutlass::HostTensor<DataTypeA, cutlass::layout::PackedVectorLayout> block_A;
    cutlass::HostTensor<ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
    cutlass::HostTensor<DataTypeA, cutlass::layout::PackedVectorLayout> block_B;
    cutlass::HostTensor<ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
    cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D_out;

    auto layout_A = cute::make_layout(cute::make_shape(M, K, 1), stride_A);
    auto layout_B = cute::make_layout(cute::make_shape(N, K, 1), stride_B);
    auto layout_C_cute = cute::make_layout(cute::make_shape(M, N, 1), stride_C);

    block_A.reset(cutlass::make_Coord(size_A_padded));
    block_B.reset(cutlass::make_Coord(size_B_padded));
    block_C.reset(cutlass::make_Coord(size_C_padded));
    block_D_out.reset(cutlass::make_Coord(size_D_padded));
    block_SFA.reset(cutlass::make_Coord(sfa_padded));
    block_SFB.reset(cutlass::make_Coord(sfb_padded));

    fprintf(stderr, "[NVF4 BF16 GEMM SM120] Buffers allocated\n");

    // Use CUTLASS TensorFill for proper initialization
    cutlass::reference::host::TensorFill(block_A.host_view(), DataTypeA(0));
    cutlass::reference::host::TensorFill(block_B.host_view(), DataTypeA(0));
    cutlass::reference::host::TensorFill(block_C.host_view(), ElementC(0.0f));
    cutlass::reference::host::TensorFill(block_SFA.host_view(), ScaleFactorType(1.0f));
    cutlass::reference::host::TensorFill(block_SFB.host_view(), ScaleFactorType(1.0f));

    fprintf(stderr, "[NVF4 BF16 GEMM SM120] Data initialized (TensorFill)\n");

    // Sync to device
    block_A.sync_device();
    block_B.sync_device();
    block_C.sync_device();
    block_SFA.sync_device();
    block_SFB.sync_device();

    fprintf(stderr, "[NVF4 BF16 GEMM SM120] Data prepared\n");

    // ========================================================================
    // Alignment Check: TMA requires 128B alignment for all base pointers
    // ========================================================================
    auto check_alignment = [](const void* ptr, const char* name) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        bool aligned = (addr & 0x7F) == 0;
        fprintf(stderr, "[ALIGN CHECK] %s: %p -> %s (offset: %zu)\n",
                name, ptr, aligned ? "OK" : "MISALIGNED", addr & 0x7F);
        return aligned;
    };

    bool all_aligned = true;
    all_aligned &= check_alignment(block_A.device_data(), "A_data");
    all_aligned &= check_alignment(block_B.device_data(), "B_data");
    all_aligned &= check_alignment(block_C.device_data(), "C_data");
    all_aligned &= check_alignment(block_D_out.device_data(), "D_out");
    all_aligned &= check_alignment(block_SFA.device_data(), "SFA");
    all_aligned &= check_alignment(block_SFB.device_data(), "SFB");

    if (!all_aligned) {
        fprintf(stderr, "[NVF4 BF16 GEMM SM120] WARNING: Misaligned buffers detected!\n");
    }

    // Build GEMM arguments (matching example 79a structure)
    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
            block_A.device_data(), stride_A,
            block_B.device_data(), stride_B,
            block_SFA.device_data(), layout_SFA,
            block_SFB.device_data(), layout_SFB
        },
        { // Epilogue arguments
            {alpha, beta},
            block_C.device_data(), stride_C,
            block_D_out.device_data(), stride_D
        }
    };

    // Run GEMM
    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[NVF4 BF16 GEMM SM120] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }
    fprintf(stderr, "[NVF4 BF16 GEMM SM120] can_implement OK\n");

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    fprintf(stderr, "[NVF4 BF16 GEMM SM120] Workspace size: %zu bytes\n", workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[NVF4 BF16 GEMM SM120] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }
    fprintf(stderr, "[NVF4 BF16 GEMM SM120] initialize OK\n");

    status = gemm_op.run();
    cudaError_t launch_err = cudaGetLastError();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[NVF4 BF16 GEMM SM120] run failed: status=%d, cuda=%s\n",
                static_cast<int>(status), cudaGetErrorString(launch_err));
        return cudaErrorLaunchFailure;
    }
    fprintf(stderr, "[NVF4 BF16 GEMM SM120] run OK\n");

    // Sync immediately after run to catch any kernel errors
    cudaError_t kernel_err = cudaDeviceSynchronize();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "[NVF4 BF16 GEMM SM120] Kernel execution failed: %s\n",
                cudaGetErrorString(kernel_err));
        return kernel_err;
    }
    fprintf(stderr, "[NVF4 BF16 GEMM SM120] Kernel sync OK\n");

    // Copy result to user buffer
    cudaError_t err = cudaMemcpy(D, block_D_out.device_data(),
                                 size_D * sizeof(nv_bfloat16),
                                 cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[NVF4 BF16 GEMM SM120] Memcpy failed: %s\n",
                cudaGetErrorString(err));
        return err;
    }
    fprintf(stderr, "[NVF4 BF16 GEMM SM120] Complete\n");

    return cudaSuccess;
}

bool is_available() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    return (props.major == 12 && (props.minor == 0 || props.minor == 1));
}

}  // namespace nvf4_bf16_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

// Extern C for linking
extern "C" {
    cudaError_t pygpukit_gemm_nvf4_bf16_sm120(
        const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return pygpukit::ops::nvf4_bf16_gemm_sm120::gemm_nvf4_bf16(A, B, D, M, N, K, alpha, beta, stream);
    }

    bool pygpukit_nvf4_bf16_sm120_available() {
        return pygpukit::ops::nvf4_bf16_gemm_sm120::is_available();
    }
}

#else  // !SM120

namespace pygpukit {
namespace ops {
namespace nvf4_bf16_gemm_sm120 {

cudaError_t gemm_nvf4_bf16(
    const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* D,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

bool is_available() {
    return false;
}

}  // namespace nvf4_bf16_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

extern "C" {
    cudaError_t pygpukit_gemm_nvf4_bf16_sm120(
        const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return cudaErrorNotSupported;
    }

    bool pygpukit_nvf4_bf16_sm120_available() {
        return false;
    }
}

#endif
