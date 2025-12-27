/**
 * FP8 GEMM implementation for SM90 (Hopper)
 *
 * Path:
 * 1. FP32 input
 * 2. FP8 quantization with per-tensor scaling
 * 3. FP8 CUTLASS GEMM (Hopper TMA + WGMMA)
 * 4. FP32 output
 *
 * Based on CUTLASS example 54: hopper_fp8_warp_specialized_gemm
 *
 * This serves as fallback for SM120 (Blackwell GeForce) until CUTLASS
 * fixes the blockwise scaling alignment bug (#2902).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

// Only compile for SM90+
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

using namespace cute;

namespace pygpukit {
namespace ops {
namespace fp8_gemm_sm90 {

// ============================================================================
// GEMM Configuration: FP8 E4M3 x FP8 E4M3 -> FP32 with per-tensor scaling
// Based on CUTLASS example 54
// ============================================================================

// A matrix: FP8 E4M3, RowMajor
using ElementA = cutlass::float_e4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

// B matrix: FP8 E4M3, ColumnMajor
using ElementB = cutlass::float_e4m3_t;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

// Output: FP32 (we'll convert internally)
using ElementC = float;
using ElementD = float;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 4
constexpr int AlignmentD = AlignmentC;

// Accumulator type
using ElementAccumulator = float;
using ElementCompute = float;

// SM90 Hopper architecture
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

// Tile and cluster shapes for Hopper
using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;  // Simple 1x1x1 cluster for compatibility

// Kernel schedule
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

// Epilogue (simple linear combination: D = alpha * A @ B + beta * C)
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

// Mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
>::CollectiveOp;

// GEMM Kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// ============================================================================
// FP32 -> FP8 Quantization with per-tensor scaling
// ============================================================================

constexpr float FP8_E4M3_MAX = 448.0f;

// Find max absolute value in tensor (for computing scale)
__global__ void find_absmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ absmax,
    int64_t num_elements
) {
    __shared__ float shared_max[256];

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    float local_max = 0.0f;

    // Grid-stride loop
    for (int64_t i = idx; i < num_elements; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<int*>(absmax),
                  __float_as_int(shared_max[0]));
    }
}

// Quantize FP32 to FP8 with scale
__device__ __forceinline__
uint8_t float_to_fp8_e4m3_scaled(float val, float inv_scale) {
    val = val * inv_scale;
    val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);

    if (fabsf(val) < 1e-7f) return 0;

    uint32_t bits = __float_as_uint(val);
    uint8_t sign = (bits >> 24) & 0x80;
    int exp = ((bits >> 23) & 0xFF) - 127 + 7;  // FP8 E4M3 bias = 7
    uint32_t mant = bits & 0x7FFFFF;

    if (exp <= 0) return sign;
    if (exp >= 15) return sign | 0x7E;

    return sign | (static_cast<uint8_t>(exp) << 3) | static_cast<uint8_t>(mant >> 20);
}

__global__ void quantize_fp32_to_fp8_scaled_kernel(
    const float* __restrict__ input,
    cutlass::float_e4m3_t* __restrict__ output,
    float inv_scale,
    int64_t num_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    uint8_t fp8 = float_to_fp8_e4m3_scaled(input[idx], inv_scale);
    output[idx] = cutlass::float_e4m3_t::bitcast(fp8);
}

// Transpose and quantize B from RowMajor [K,N] to ColumnMajor [K,N]
__global__ void transpose_quantize_fp32_to_fp8_kernel(
    const float* __restrict__ input,  // [K, N] RowMajor
    cutlass::float_e4m3_t* __restrict__ output,  // [K, N] ColumnMajor
    float inv_scale,
    int K, int N
) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= K || n >= N) return;

    float val = input[k * N + n];
    uint8_t fp8 = float_to_fp8_e4m3_scaled(val, inv_scale);
    output[k + n * K] = cutlass::float_e4m3_t::bitcast(fp8);
}

// ============================================================================
// FP8 GEMM Entry Point
// ============================================================================

cudaError_t gemm_fp8(
    const float* A,      // [M, K] FP32 input
    const float* B,      // [K, N] FP32 input
    float* D,            // [M, N] FP32 output
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
    cutlass::device_memory::allocation<float> buf_C(size_D);  // For beta * C

    auto* d_A_fp8 = buf_A_fp8.get();
    auto* d_B_fp8 = buf_B_fp8.get();
    auto* d_C = buf_C.get();

    // Compute scale factors (find absmax for each tensor)
    cutlass::device_memory::allocation<float> buf_absmax_A(1);
    cutlass::device_memory::allocation<float> buf_absmax_B(1);

    cudaMemsetAsync(buf_absmax_A.get(), 0, sizeof(float), stream);
    cudaMemsetAsync(buf_absmax_B.get(), 0, sizeof(float), stream);

    int threads = 256;
    int blocks_A = std::min(1024, static_cast<int>((size_A + threads - 1) / threads));
    int blocks_B = std::min(1024, static_cast<int>((size_B + threads - 1) / threads));

    find_absmax_kernel<<<blocks_A, threads, 0, stream>>>(A, buf_absmax_A.get(), size_A);
    find_absmax_kernel<<<blocks_B, threads, 0, stream>>>(B, buf_absmax_B.get(), size_B);

    // Copy absmax to host to compute scales
    float absmax_A = 0.0f, absmax_B = 0.0f;
    cudaMemcpyAsync(&absmax_A, buf_absmax_A.get(), sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&absmax_B, buf_absmax_B.get(), sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Compute scales: scale = absmax / FP8_MAX, inv_scale = FP8_MAX / absmax
    float scale_A = (absmax_A > 0.0f) ? (absmax_A / FP8_E4M3_MAX) : 1.0f;
    float scale_B = (absmax_B > 0.0f) ? (absmax_B / FP8_E4M3_MAX) : 1.0f;
    float inv_scale_A = (absmax_A > 0.0f) ? (FP8_E4M3_MAX / absmax_A) : 1.0f;
    float inv_scale_B = (absmax_B > 0.0f) ? (FP8_E4M3_MAX / absmax_B) : 1.0f;

    // Quantize A (keep RowMajor)
    int blocks_A_q = (size_A + threads - 1) / threads;
    quantize_fp32_to_fp8_scaled_kernel<<<blocks_A_q, threads, 0, stream>>>(
        A, d_A_fp8, inv_scale_A, size_A
    );

    // Quantize and transpose B (RowMajor -> ColumnMajor)
    dim3 block_B(16, 16);
    dim3 grid_B((N + 15) / 16, (K + 15) / 16);
    transpose_quantize_fp32_to_fp8_kernel<<<grid_B, block_B, 0, stream>>>(
        B, d_B_fp8, inv_scale_B, K, N
    );

    // Initialize C buffer (for beta=0, we can skip)
    if (beta != 0.0f) {
        cudaMemsetAsync(d_C, 0, size_D * sizeof(float), stream);
    }

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    // Build strides
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Adjusted alpha to account for FP8 scaling
    // Result = scale_A * scale_B * (A_fp8 @ B_fp8)
    // So we multiply alpha by scale_A * scale_B
    float adjusted_alpha = alpha * scale_A * scale_B;

    // Build CUTLASS arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A_fp8, stride_a, d_B_fp8, stride_b},
        {{adjusted_alpha, beta}, d_C, stride_c, D, stride_d}
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 GEMM SM90] can_implement failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 GEMM SM90] initialize failed: %d\n", static_cast<int>(status));
        return cudaErrorInvalidValue;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8 GEMM SM90] run failed: %d\n", static_cast<int>(status));
        return cudaErrorLaunchFailure;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[FP8 GEMM SM90] sync failed: %s\n", cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}

bool is_available() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    // SM90 only (Hopper) - TMA-based kernels may not work on Blackwell (SM100/SM120)
    // Blackwell has different TMA behavior that causes CUTLASS initialization failures
    int sm = props.major * 10 + props.minor;
    return (sm >= 90 && sm < 100);
}

}  // namespace fp8_gemm_sm90
}  // namespace ops
}  // namespace pygpukit

// Extern C for linking
extern "C" {
    cudaError_t pygpukit_gemm_fp8_sm90(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return pygpukit::ops::fp8_gemm_sm90::gemm_fp8(A, B, D, M, N, K, alpha, beta, stream);
    }

    bool pygpukit_fp8_sm90_available() {
        return pygpukit::ops::fp8_gemm_sm90::is_available();
    }
}

#else  // !SM90

namespace pygpukit {
namespace ops {
namespace fp8_gemm_sm90 {

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

}  // namespace fp8_gemm_sm90
}  // namespace ops
}  // namespace pygpukit

extern "C" {
    cudaError_t pygpukit_gemm_fp8_sm90(
        const float* A, const float* B, float* D,
        int M, int N, int K,
        float alpha, float beta,
        cudaStream_t stream
    ) {
        return cudaErrorNotSupported;
    }

    bool pygpukit_fp8_sm90_available() {
        return false;
    }
}

#endif
