/**
 * FP8 GEMM implementation for SM120 (Blackwell GeForce) with BF16 I/O
 *
 * Data Flow:
 *   BF16 input -> FP8 E4M3 quantize -> CUTLASS GEMM -> BF16 output
 *
 * This kernel takes BF16 inputs and produces BF16 output, using FP8
 * for the internal matrix multiplication for higher throughput.
 *
 * Based on matmul_fp8_sm120.cu (FP32 version)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

// Enable FP8 SM120 with alignment patch
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
namespace fp8_bf16_gemm_sm120 {

// ============================================================================
// GEMM Configuration: FP8 E4M3 x FP8 E4M3 -> BF16 with blockwise scaling
// ============================================================================

// A matrix: FP8 E4M3, RowMajor
using ElementA = cutlass::float_e4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

// B matrix: FP8 E4M3, ColumnMajor
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

// Accumulator type
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

// Epilogue
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
// BF16 -> FP8 E4M3 Quantization
// ============================================================================

constexpr float FP8_E4M3_MAX = 448.0f;

__device__ __forceinline__
uint8_t bf16_to_fp8_e4m3_scaled(nv_bfloat16 val_bf16, float inv_scale) {
    // Convert BF16 to FP32
    float val = __bfloat162float(val_bf16);

    // Apply inverse scale
    val = val * inv_scale;

    // Clamp to FP8 E4M3 range
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

// BF16 -> FP8 conversion kernel (unity scale)
__global__ void quantize_bf16_to_fp8_kernel(
    const nv_bfloat16* __restrict__ input,
    cutlass::float_e4m3_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    uint8_t fp8 = bf16_to_fp8_e4m3_scaled(input[idx], 1.0f);
    output[idx] = cutlass::float_e4m3_t::bitcast(fp8);
}

// Transpose and quantize B from RowMajor [K,N] to ColumnMajor [K,N]
__global__ void transpose_quantize_bf16_to_fp8_kernel(
    const nv_bfloat16* __restrict__ input,  // [K, N] RowMajor
    cutlass::float_e4m3_t* __restrict__ output,  // [K, N] ColumnMajor
    int K, int N
) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= K || n >= N) return;

    // Read from RowMajor: B[k,n] = input[k * N + n]
    nv_bfloat16 val = input[k * N + n];

    // Write to ColumnMajor: B[k,n] = output[k + n * K]
    uint8_t fp8 = bf16_to_fp8_e4m3_scaled(val, 1.0f);
    output[k + n * K] = cutlass::float_e4m3_t::bitcast(fp8);
}

// Fill scale factors with unity (1.0f)
__global__ void fill_scale_factors_unity_kernel(
    float* __restrict__ scales,
    size_t num_scales
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_scales) return;
    scales[idx] = 1.0f;
}

// ============================================================================
// FP8 GEMM Entry Point (BF16 I/O)
// ============================================================================

cudaError_t gemm_fp8_bf16(
    const nv_bfloat16* A,  // [M, K] BF16 input
    const nv_bfloat16* B,  // [K, N] BF16 input (will be transposed internally)
    nv_bfloat16* D,        // [M, N] BF16 output
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    fprintf(stderr, "[FP8 BF16 GEMM SM120] Starting M=%d, N=%d, K=%d\n", M, N, K);
    fprintf(stderr, "[FP8 BF16 GEMM SM120] Input pointers: A=%p, B=%p, D=%p\n", (void*)A, (void*)B, (void*)D);

    // Sizes
    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(K) * N;
    int64_t size_D = static_cast<int64_t>(M) * N;

    // Allocate FP8 data buffers
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_A_fp8(size_A);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_B_fp8(size_B);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_C_bf16(size_D);  // For epilogue C input

    auto* d_A_fp8 = buf_A_fp8.get();
    auto* d_B_fp8 = buf_B_fp8.get();
    auto* d_C_bf16 = buf_C_bf16.get();

    // Calculate scale factor sizes using ScaleConfig
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    fprintf(stderr, "[FP8 BF16 GEMM SM120] Scale layouts computed\n");

    size_t sfa_size = static_cast<size_t>(size(filter_zeros(layout_SFA)));
    size_t sfb_size = static_cast<size_t>(size(filter_zeros(layout_SFB)));

    // Pad to at least 32 floats (128 bytes) for TMA alignment
    size_t sfa_padded = (sfa_size > 32) ? sfa_size : 32;
    size_t sfb_padded = (sfb_size > 32) ? sfb_size : 32;

    cutlass::device_memory::allocation<float> buf_SFA(sfa_padded);
    cutlass::device_memory::allocation<float> buf_SFB(sfb_padded);

    auto* d_SFA = buf_SFA.get();
    auto* d_SFB = buf_SFB.get();

    fprintf(stderr, "[FP8 BF16 GEMM SM120] Buffers allocated\n");

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
    all_aligned &= check_alignment(d_A_fp8, "A_fp8");
    all_aligned &= check_alignment(d_B_fp8, "B_fp8");
    all_aligned &= check_alignment(d_C_bf16, "C_bf16");
    all_aligned &= check_alignment(d_SFA, "SFA");
    all_aligned &= check_alignment(d_SFB, "SFB");

    if (!all_aligned) {
        fprintf(stderr, "[FP8 BF16 GEMM SM120] WARNING: Misaligned buffers detected!\n");
    }

    // Quantize A and B
    int threads = 256;
    int blocks_A_data = (size_A + threads - 1) / threads;

    // Convert A: BF16 -> FP8 (keep RowMajor)
    quantize_bf16_to_fp8_kernel<<<blocks_A_data, threads, 0, stream>>>(
        A, d_A_fp8, size_A
    );

    // Convert B: BF16 RowMajor -> FP8 ColumnMajor
    dim3 block_B(16, 16);
    dim3 grid_B((N + 15) / 16, (K + 15) / 16);
    transpose_quantize_bf16_to_fp8_kernel<<<grid_B, block_B, 0, stream>>>(
        B, d_B_fp8, K, N
    );

    // Fill scale factors with 1.0
    int blocks_SFA_fill = (sfa_padded + threads - 1) / threads;
    int blocks_SFB_fill = (sfb_padded + threads - 1) / threads;
    fill_scale_factors_unity_kernel<<<blocks_SFA_fill, threads, 0, stream>>>(d_SFA, sfa_padded);
    fill_scale_factors_unity_kernel<<<blocks_SFB_fill, threads, 0, stream>>>(d_SFB, sfb_padded);

    // Sync and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[FP8 BF16 GEMM SM120] Quantization failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    fprintf(stderr, "[FP8 BF16 GEMM SM120] Quantization OK\n");

    // Build strides
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Allocate internal output buffer (aligned)
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_D_bf16(size_D);
    auto* d_D_internal = buf_D_bf16.get();

    fprintf(stderr, "[FP8 BF16 GEMM SM120] Output buffer: internal=%p, user=%p\n", (void*)d_D_internal, (void*)D);
    check_alignment(d_D_internal, "D_internal");
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {  // Mainloop arguments
            d_A_fp8, stride_a,
            d_B_fp8, stride_b,
            d_SFA, layout_SFA,
            d_SFB, layout_SFB
        },
        {  // Epilogue arguments
            {},  // epilogue.thread (will be filled below)
            d_C_bf16, stride_c,  // C pointer (valid even with beta=0)
            d_D_internal, stride_d   // D pointer (internal buffer)
        }
    };

    // Set alpha/beta
    arguments.epilogue.thread.alpha = alpha;
    arguments.epilogue.thread.beta = beta;

    // Instantiate and run GEMM
    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 BF16 GEMM SM120] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }
    fprintf(stderr, "[FP8 BF16 GEMM SM120] can_implement OK\n");

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    fprintf(stderr, "[FP8 BF16 GEMM SM120] Workspace size: %zu bytes\n", workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 BF16 GEMM SM120] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }
    fprintf(stderr, "[FP8 BF16 GEMM SM120] initialize OK\n");

    status = gemm_op.run();
    cudaError_t launch_err = cudaGetLastError();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 BF16 GEMM SM120] run failed: status=%d, cuda=%s\n",
                static_cast<int>(status), cudaGetErrorString(launch_err));
        return cudaErrorLaunchFailure;
    }
    fprintf(stderr, "[FP8 BF16 GEMM SM120] run OK\n");

    // Sync before returning
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[FP8 BF16 GEMM SM120] sync failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    fprintf(stderr, "[FP8 BF16 GEMM SM120] Complete\n");

    return cudaSuccess;
}

bool is_available() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    return (props.major * 10 + props.minor) >= 120;
}

}  // namespace fp8_bf16_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

// Extern C for linking
extern "C" {
    cudaError_t pygpukit_gemm_fp8_bf16_sm120(
        const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return pygpukit::ops::fp8_bf16_gemm_sm120::gemm_fp8_bf16(A, B, D, M, N, K, alpha, beta, stream);
    }

    bool pygpukit_fp8_bf16_sm120_available() {
        return pygpukit::ops::fp8_bf16_gemm_sm120::is_available();
    }
}

#else  // !SM120

namespace pygpukit {
namespace ops {
namespace fp8_bf16_gemm_sm120 {

cudaError_t gemm_fp8_bf16(
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

}  // namespace fp8_bf16_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

extern "C" {
    cudaError_t pygpukit_gemm_fp8_bf16_sm120(
        const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return cudaErrorNotSupported;
    }

    bool pygpukit_fp8_bf16_sm120_available() {
        return false;
    }
}

#endif
