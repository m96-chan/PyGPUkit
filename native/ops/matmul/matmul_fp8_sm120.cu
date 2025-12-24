/**
 * FP8 GEMM implementation for SM120 (Blackwell GeForce)
 *
 * Path:
 * 1. FP32 input
 * 2. FP8 quantization (A scale, B scale separate)
 * 3. FP8 CUTLASS GEMM
 * 4. BF16 accumulate
 * 5. FP32 output (if needed)
 *
 * Implementation based on CUTLASS example 87a:
 * "87a_blackwell_geforce_fp8_bf16_gemm_blockwise"
 *
 * IMPORTANT: This is the ONLY backend for SM120. No cuBLAS fallback.
 *
 * WORKAROUND for CUTLASS bug #2902:
 * - partition_S() drops alignment from 1024 to 8 bytes
 * - SM75_U32x4_LDSM_N requires 16-byte alignment
 * - We patch the LDSM copy operations to handle misalignment
 * - Tracking issue: https://github.com/NVIDIA/cutlass/issues/2902
 * - Local issue: https://github.com/m96-chan/PyGPUkit/issues/107
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

// ============================================================================
// ALIGNMENT PATCH: Include AFTER CUTLASS headers
// Provides alignment-safe LDSM operations for Issue #2902 workaround
// ============================================================================
#define PYGPUKIT_PATCH_CUTLASS_LDSM_POST 1
#include "aligned_copy_sm120.cuh"

using namespace cute;

namespace pygpukit {
namespace ops {
namespace fp8_gemm_sm120 {

// ============================================================================
// GEMM Configuration: FP8 E4M3 x FP8 E4M3 -> BF16 with blockwise scaling
// Based on CUTLASS example 87a_blackwell_geforce_fp8_bf16_gemm_blockwise
// Using OpClassTensorOp for SM120 GeForce (NOT OpClassBlockScaledTensorOp)
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
// FP32 -> FP8 E4M3 Quantization with blockwise scaling
// ============================================================================

constexpr float FP8_E4M3_MAX = 448.0f;

__device__ __forceinline__
uint8_t float_to_fp8_e4m3_scaled(float val, float inv_scale) {
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

// Simple FP32 -> FP8 conversion kernel (unity scale for testing)
__global__ void quantize_fp32_to_fp8_kernel(
    const float* __restrict__ input,
    cutlass::float_e4m3_t* __restrict__ output,
    int64_t num_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Simple quantization with unity scale (inv_scale = 1.0)
    uint8_t fp8 = float_to_fp8_e4m3_scaled(input[idx], 1.0f);
    output[idx] = cutlass::float_e4m3_t::bitcast(fp8);
}

// Transpose and quantize B from RowMajor [K,N] to ColumnMajor [K,N]
// Input:  B_row[k,n] = B[k * N + n]  (RowMajor)
// Output: B_col[k,n] = B[k + n * K]  (ColumnMajor)
__global__ void transpose_quantize_fp32_to_fp8_kernel(
    const float* __restrict__ input,  // [K, N] RowMajor
    cutlass::float_e4m3_t* __restrict__ output,  // [K, N] ColumnMajor
    int K, int N
) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= K || n >= N) return;

    // Read from RowMajor: B[k,n] = input[k * N + n]
    float val = input[k * N + n];

    // Write to ColumnMajor: B[k,n] = output[k + n * K]
    uint8_t fp8 = float_to_fp8_e4m3_scaled(val, 1.0f);
    output[k + n * K] = cutlass::float_e4m3_t::bitcast(fp8);
}

// Fill scale factors with unity (1.0f)
// Example 87a uses float scale factors, not E8M0
__global__ void fill_scale_factors_unity_kernel(
    float* __restrict__ scales,
    size_t num_scales
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_scales) return;

    scales[idx] = 1.0f;
}

// ============================================================================
// BF16 -> FP32 Conversion
// ============================================================================

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
    const float* A,      // [M, K] FP32 input
    const float* B,      // [K, N] FP32 input (will be transposed internally)
    float* D,            // [M, N] FP32 output
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    fprintf(stderr, "[FP8 GEMM SM120] BUILD_VER=2024-12-24-A\n");
    fprintf(stderr, "[FP8 GEMM SM120] Starting M=%d, N=%d, K=%d\n", M, N, K);

    // Check input/output alignment
    fprintf(stderr, "[FP8 GEMM SM120] Alignment check:\n");
    fprintf(stderr, "  A ptr alignment mod 128 = %llu\n", (unsigned long long)((uintptr_t)A % 128));
    fprintf(stderr, "  B ptr alignment mod 128 = %llu\n", (unsigned long long)((uintptr_t)B % 128));
    fprintf(stderr, "  D ptr alignment mod 128 = %llu\n", (unsigned long long)((uintptr_t)D % 128));

    // Sizes
    int64_t size_A = static_cast<int64_t>(M) * K;
    int64_t size_B = static_cast<int64_t>(K) * N;
    int64_t size_D = static_cast<int64_t>(M) * N;

    // Allocate FP8 data buffers
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_A_fp8(size_A);
    cutlass::device_memory::allocation<cutlass::float_e4m3_t> buf_B_fp8(size_B);
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_C_bf16(size_D);  // For epilogue C input
    cutlass::device_memory::allocation<cutlass::bfloat16_t> buf_D_bf16(size_D);

    auto* d_A_fp8 = buf_A_fp8.get();
    auto* d_B_fp8 = buf_B_fp8.get();
    auto* d_C_bf16 = buf_C_bf16.get();
    auto* d_D_bf16 = buf_D_bf16.get();

    fprintf(stderr, "[FP8 GEMM SM120] FP8 buffers allocated: A=%p, B=%p, D_bf16=%p\n",
            (void*)d_A_fp8, (void*)d_B_fp8, (void*)d_D_bf16);
    fprintf(stderr, "[FP8 GEMM SM120] Internal alignment check:\n");
    fprintf(stderr, "  A_fp8 mod 128 = %llu\n", (unsigned long long)((uintptr_t)d_A_fp8 % 128));
    fprintf(stderr, "  B_fp8 mod 128 = %llu\n", (unsigned long long)((uintptr_t)d_B_fp8 % 128));
    fprintf(stderr, "  D_bf16 mod 128 = %llu\n", (unsigned long long)((uintptr_t)d_D_bf16 % 128));

    // Calculate scale factor sizes using ScaleConfig (from example 87a)
    auto problem_shape = cute::make_shape(M, N, K, 1);
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    size_t sfa_size = size(filter_zeros(layout_SFA));
    size_t sfb_size = size(filter_zeros(layout_SFB));

    fprintf(stderr, "[FP8 GEMM SM120] Scale factor sizes: SFA=%zu, SFB=%zu\n", sfa_size, sfb_size);
    fprintf(stderr, "[FP8 GEMM SM120] Scale factor layouts:\n");
    cute::print("  layout_SFA: "); cute::print(layout_SFA); cute::print("\n");
    cute::print("  layout_SFB: "); cute::print(layout_SFB); cute::print("\n");

    // Allocate scale factor buffers (float, not E8M0)
    // TMA requires 128-byte alignment for each scale factor access
    // Pad to at least 32 floats (128 bytes) to ensure TMA alignment
    size_t sfa_padded = std::max(sfa_size, size_t(32));
    size_t sfb_padded = std::max(sfb_size, size_t(32));
    fprintf(stderr, "[FP8 GEMM SM120] Scale factor padded sizes: SFA=%zu->%zu, SFB=%zu->%zu\n",
            sfa_size, sfa_padded, sfb_size, sfb_padded);

    cutlass::device_memory::allocation<float> buf_SFA(sfa_padded);
    cutlass::device_memory::allocation<float> buf_SFB(sfb_padded);

    auto* d_SFA = buf_SFA.get();
    auto* d_SFB = buf_SFB.get();

    fprintf(stderr, "[FP8 GEMM SM120] Scale factor alignment:\n");
    fprintf(stderr, "  SFA mod 128 = %llu\n", (unsigned long long)((uintptr_t)d_SFA % 128));
    fprintf(stderr, "  SFB mod 128 = %llu\n", (unsigned long long)((uintptr_t)d_SFB % 128));

    // Quantize A and B
    int threads = 256;
    int blocks_A_data = (size_A + threads - 1) / threads;

    // Convert A: FP32 -> FP8 (keep RowMajor)
    quantize_fp32_to_fp8_kernel<<<blocks_A_data, threads, 0, stream>>>(
        A, d_A_fp8, size_A
    );

    // Convert B: FP32 RowMajor -> FP8 ColumnMajor (transpose during quantization)
    // B input is [K, N] RowMajor, output needs to be [K, N] ColumnMajor
    dim3 block_B(16, 16);
    dim3 grid_B((N + 15) / 16, (K + 15) / 16);
    transpose_quantize_fp32_to_fp8_kernel<<<grid_B, block_B, 0, stream>>>(
        B, d_B_fp8, K, N
    );
    fprintf(stderr, "[FP8 GEMM SM120] B transposed from RowMajor to ColumnMajor\n");

    // Fill scale factors with 1.0 (fill entire padded buffer)
    int blocks_SFA_fill = (sfa_padded + threads - 1) / threads;
    int blocks_SFB_fill = (sfb_padded + threads - 1) / threads;
    fill_scale_factors_unity_kernel<<<blocks_SFA_fill, threads, 0, stream>>>(d_SFA, sfa_padded);
    fill_scale_factors_unity_kernel<<<blocks_SFB_fill, threads, 0, stream>>>(d_SFB, sfb_padded);

    // Sync and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[FP8 GEMM SM120] Quantization sync failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    fprintf(stderr, "[FP8 GEMM SM120] Quantization OK\n");

    // Build strides (from example 87a)
    // For CUTLASS 3.x with cute layouts:
    // - StrideA for RowMajor A[M,K]: packed stride from shape (M, K, L)
    // - StrideB for ColumnMajor B[K,N]: packed stride from shape (N, K, L)
    // Note: The shape passed to make_cute_packed_stride is the logical GEMM shape,
    // not the memory layout shape. CUTLASS handles the layout internally.
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Debug: Print stride values
    fprintf(stderr, "[FP8 GEMM SM120] Stride debug:\n");
    fprintf(stderr, "  stride_a: (%lld, %lld, %lld)\n",
            (long long)cute::get<0>(stride_a), (long long)cute::get<1>(stride_a), (long long)cute::get<2>(stride_a));
    fprintf(stderr, "  stride_b: (%lld, %lld, %lld)\n",
            (long long)cute::get<0>(stride_b), (long long)cute::get<1>(stride_b), (long long)cute::get<2>(stride_b));
    fprintf(stderr, "  stride_c: (%lld, %lld, %lld)\n",
            (long long)cute::get<0>(stride_c), (long long)cute::get<1>(stride_c), (long long)cute::get<2>(stride_c));
    fprintf(stderr, "  stride_d: (%lld, %lld, %lld)\n",
            (long long)cute::get<0>(stride_d), (long long)cute::get<1>(stride_d), (long long)cute::get<2>(stride_d));

    // Build CUTLASS arguments (following example 87a structure)
    // Note: Even with beta=0, we must pass a valid C pointer (CUTLASS may dereference it)
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
            d_D_bf16, stride_d   // D pointer
        }
    };

    // Set alpha/beta
    arguments.epilogue.thread.alpha = alpha;
    arguments.epilogue.thread.beta = beta;

    fprintf(stderr, "[FP8 GEMM SM120] Arguments built, alpha=%f, beta=%f\n", alpha, beta);

    // Instantiate and run GEMM
    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 GEMM SM120] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }
    fprintf(stderr, "[FP8 GEMM SM120] can_implement OK\n");

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    fprintf(stderr, "[FP8 GEMM SM120] Workspace size: %zu bytes\n", workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 GEMM SM120] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }
    fprintf(stderr, "[FP8 GEMM SM120] initialize OK\n");

    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 GEMM SM120] run failed: %d\n", static_cast<int>(status));
        return cudaErrorLaunchFailure;
    }

    // Sync and check for kernel errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[FP8 GEMM SM120] GEMM sync failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[FP8 GEMM SM120] GEMM kernel error: %s\n", cudaGetErrorString(err));
        return err;
    }
    fprintf(stderr, "[FP8 GEMM SM120] GEMM completed OK\n");

    // Convert BF16 output to FP32
    int blocks_D = (size_D + threads - 1) / threads;
    bf16_to_fp32_kernel<<<blocks_D, threads, 0, stream>>>(d_D_bf16, D, size_D);

    // Sync before RAII cleanup
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[FP8 GEMM SM120] BF16->FP32 sync failed: %s\n", cudaGetErrorString(err));
        return err;
    }
    fprintf(stderr, "[FP8 GEMM SM120] Complete\n");

    return cudaSuccess;
}

bool is_available() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    return (props.major * 10 + props.minor) >= 120;
}

}  // namespace fp8_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

// Extern C for linking
extern "C" {
    cudaError_t pygpukit_gemm_fp8_sm120(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return pygpukit::ops::fp8_gemm_sm120::gemm_fp8(A, B, D, M, N, K, alpha, beta, stream);
    }

    bool pygpukit_fp8_sm120_available() {
        return pygpukit::ops::fp8_gemm_sm120::is_available();
    }
}

#else  // !SM120

namespace pygpukit {
namespace ops {
namespace fp8_gemm_sm120 {

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

}  // namespace fp8_gemm_sm120
}  // namespace ops
}  // namespace pygpukit

extern "C" {
    cudaError_t pygpukit_gemm_fp8_sm120(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return cudaErrorNotSupported;
    }

    bool pygpukit_fp8_sm120_available() {
        return false;
    }
}

#endif
