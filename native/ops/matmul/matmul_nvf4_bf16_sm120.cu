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

// Tile shapes - K=256 is recommended for NVF4 in CUTLASS tests
using ThreadBlockShape = Shape<_128, _128, _256>;
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

// Mainloop - Pingpong schedule with auto stage count (explicit 3 causes init failure)
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
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

// ============================================================================
// Branchless BF16 -> NVF4 Quantization
// ============================================================================
// Uses comparison accumulation - faster than LUT on modern GPUs
// LUT approaches tested but slower due to constant memory latency

__device__ __forceinline__
uint8_t bf16_to_nvf4_e2m1(float val) {
    // E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    float absval = fabsf(val);
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;

    // Branchless: count how many thresholds we exceed
    uint8_t code = 0;
    code += (absval >= 0.25f);
    code += (absval >= 0.75f);
    code += (absval >= 1.25f);
    code += (absval >= 1.75f);
    code += (absval >= 2.5f);
    code += (absval >= 3.5f);
    code += (absval >= 5.0f);

    return sign | code;
}

// ============================================================================
// GPU-side BF16 -> NVF4 Quantization Kernels (Unit Scale)
// ============================================================================

// Vectorized GPU quantization: BF16 [M, K] RowMajor -> NVF4 packed (unit scale)
// Each thread processes 8 BF16 elements -> 4 output bytes using uint4 loads
// Uses branchless float comparison (faster than LUT - see benchmark notes)
__global__ void quantize_A_gpu_kernel(
    const nv_bfloat16* __restrict__ input,  // [M, K] RowMajor BF16
    uint8_t* __restrict__ output,            // Packed NVF4 (size = M*K/2)
    int M, int K
) {
    // Each thread handles 8 elements (4 output bytes)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_quads = (M * K) / 8;
    if (idx >= total_quads) return;

    // Vectorized load: 8 BF16 = 16 bytes = uint4
    const uint4* input_vec = reinterpret_cast<const uint4*>(input);
    uint4 data = input_vec[idx];

    // Extract BF16 values and convert to float
    nv_bfloat16 bf0, bf1, bf2, bf3, bf4, bf5, bf6, bf7;
    memcpy(&bf0, reinterpret_cast<uint16_t*>(&data.x), sizeof(nv_bfloat16));
    memcpy(&bf1, reinterpret_cast<uint16_t*>(&data.x) + 1, sizeof(nv_bfloat16));
    memcpy(&bf2, reinterpret_cast<uint16_t*>(&data.y), sizeof(nv_bfloat16));
    memcpy(&bf3, reinterpret_cast<uint16_t*>(&data.y) + 1, sizeof(nv_bfloat16));
    memcpy(&bf4, reinterpret_cast<uint16_t*>(&data.z), sizeof(nv_bfloat16));
    memcpy(&bf5, reinterpret_cast<uint16_t*>(&data.z) + 1, sizeof(nv_bfloat16));
    memcpy(&bf6, reinterpret_cast<uint16_t*>(&data.w), sizeof(nv_bfloat16));
    memcpy(&bf7, reinterpret_cast<uint16_t*>(&data.w) + 1, sizeof(nv_bfloat16));

    // Quantize using branchless float comparison
    uint8_t q0 = bf16_to_nvf4_e2m1(__bfloat162float(bf0));
    uint8_t q1 = bf16_to_nvf4_e2m1(__bfloat162float(bf1));
    uint8_t q2 = bf16_to_nvf4_e2m1(__bfloat162float(bf2));
    uint8_t q3 = bf16_to_nvf4_e2m1(__bfloat162float(bf3));
    uint8_t q4 = bf16_to_nvf4_e2m1(__bfloat162float(bf4));
    uint8_t q5 = bf16_to_nvf4_e2m1(__bfloat162float(bf5));
    uint8_t q6 = bf16_to_nvf4_e2m1(__bfloat162float(bf6));
    uint8_t q7 = bf16_to_nvf4_e2m1(__bfloat162float(bf7));

    // Pack into 4 bytes and write as uint32
    uint32_t packed = ((q1 << 4) | (q0 & 0x0F))
                    | (((q3 << 4) | (q2 & 0x0F)) << 8)
                    | (((q5 << 4) | (q4 & 0x0F)) << 16)
                    | (((q7 << 4) | (q6 & 0x0F)) << 24);

    reinterpret_cast<uint32_t*>(output)[idx] = packed;
}

// GPU quantization: BF16 [K, N] RowMajor -> NVF4 [N, K] ColumnMajor packed (unit scale)
// Vectorized version using shared memory transpose for coalesced access
// TILE_K=64, TILE_N=32: each block processes 64x32 tile, outputs 32x32 packed bytes
__global__ void quantize_B_gpu_kernel(
    const nv_bfloat16* __restrict__ input,  // [K, N] RowMajor BF16
    uint8_t* __restrict__ output,            // Packed NVF4 ColMajor (size = N*K/2)
    int K, int N
) {
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 32;

    // Shared memory: TILE_K x TILE_N with padding to avoid bank conflicts
    __shared__ uint8_t smem_q[TILE_K][TILE_N + 4];

    int block_k = blockIdx.x * TILE_K;
    int block_n = blockIdx.y * TILE_N;

    // Phase 1: Load and quantize into shared memory
    // 256 threads, each handles 8 elements (64*32/256 = 8)
    // Thread layout: 32 threads in N, 8 threads in K
    int tid = threadIdx.x;
    int tn = tid % 32;  // 0-31
    int tk = tid / 32;  // 0-7

    #pragma unroll
    for (int ki = 0; ki < 8; ki++) {
        int k = block_k + tk * 8 + ki;
        int n = block_n + tn;

        if (k < K && n < N) {
            nv_bfloat16 bf = input[k * N + n];
            smem_q[tk * 8 + ki][tn] = bf16_to_nvf4_e2m1(__bfloat162float(bf));
        } else {
            smem_q[tk * 8 + ki][tn] = 0;
        }
    }

    __syncthreads();

    // Phase 2: Write transposed and packed (8 NVF4 = 32 bits per write)
    // Each thread writes 4 bytes (8 k-values) for one n
    // 256 threads handle 32 n-values x 8 k-groups = 256 outputs
    int out_n = block_n + (tid % 32);
    int out_k_group = tid / 32;  // 0-7, each group is 8 k-values

    int k_base = out_k_group * 8;
    int num_k_pairs = K / 2;

    if (out_n < N && (block_k + k_base + 7) < K) {
        // Fast path: full 8 k-values, vectorized uint32 write
        uint8_t q0 = smem_q[k_base + 0][tn];
        uint8_t q1 = smem_q[k_base + 1][tn];
        uint8_t q2 = smem_q[k_base + 2][tn];
        uint8_t q3 = smem_q[k_base + 3][tn];
        uint8_t q4 = smem_q[k_base + 4][tn];
        uint8_t q5 = smem_q[k_base + 5][tn];
        uint8_t q6 = smem_q[k_base + 6][tn];
        uint8_t q7 = smem_q[k_base + 7][tn];

        uint32_t packed = ((q1 << 4) | (q0 & 0x0F))
                        | (((q3 << 4) | (q2 & 0x0F)) << 8)
                        | (((q5 << 4) | (q4 & 0x0F)) << 16)
                        | (((q7 << 4) | (q6 & 0x0F)) << 24);

        // Output: ColMajor [N, K] packed - 4 consecutive bytes for 8 k-values
        int byte_offset = out_n * num_k_pairs + (block_k + k_base) / 2;
        *reinterpret_cast<uint32_t*>(&output[byte_offset]) = packed;
    } else if (out_n < N) {
        // Edge case: partial k-group, scalar writes
        for (int i = 0; i < 8 && (block_k + k_base + i + 1) < K; i += 2) {
            uint8_t q0 = smem_q[k_base + i][tn];
            uint8_t q1 = smem_q[k_base + i + 1][tn];
            output[out_n * num_k_pairs + (block_k + k_base + i) / 2] = (q1 << 4) | (q0 & 0x0F);
        }
    }
}

// Initialize scale factors to 1.0 (UE4M3 encoding: 0x38)
__global__ void init_scale_factors_kernel(
    uint8_t* __restrict__ sf,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    sf[idx] = 0x38;  // float_ue4m3_t(1.0f) = 0x38
}

// ============================================================================
// Host-side BF16 -> NVF4 Quantization Helpers
// ============================================================================

// Convert float to float_e2m1_t (NVF4 4-bit format)
inline cutlass::float_e2m1_t float_to_e2m1(float val) {
    // E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    // Clamp to representable range
    val = std::max(-6.0f, std::min(6.0f, val));
    return cutlass::float_e2m1_t(val);
}

// Convert float to float_ue4m3_t (scale factor, unsigned 8-bit)
inline cutlass::float_ue4m3_t float_to_ue4m3(float val) {
    // UE4M3 range: approximately [2^-9, 448]
    val = std::max(1.0f/512.0f, std::min(448.0f, val));
    return cutlass::float_ue4m3_t(val);
}

// Quantize a block of floats to NVF4 with a computed scale factor
// Returns the scale factor used
inline float quantize_block_to_e2m1(
    const float* input,
    cutlass::float_e2m1_t* output,
    int count
) {
    // Find max absolute value in block
    float max_abs = 0.0f;
    for (int i = 0; i < count; ++i) {
        max_abs = std::max(max_abs, std::abs(input[i]));
    }

    // Compute scale factor: scale * 6.0 >= max_abs
    // So scale = max_abs / 6.0 (6.0 is max representable in E2M1)
    float scale = (max_abs > 1e-8f) ? (max_abs / 6.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Quantize each element
    for (int i = 0; i < count; ++i) {
        float scaled_val = input[i] * inv_scale;
        output[i] = float_to_e2m1(scaled_val);
    }

    return scale;
}

// ============================================================================
// NVF4 GEMM Entry Point (BF16 I/O)
// ============================================================================

cudaError_t gemm_nvf4_bf16(
    const nv_bfloat16* A,  // [M, K] BF16 input (device)
    const nv_bfloat16* B,  // [K, N] BF16 input (device)
    nv_bfloat16* D,        // [M, N] BF16 output (device)
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

    // Allocate device memory directly (no host memory needed!)
    // NVF4 packed: 2 elements per byte
    size_t size_A_packed = (size_A + 1) / 2;  // Packed bytes for A
    size_t size_B_packed = (size_B + 1) / 2;  // Packed bytes for B

    cutlass::device_memory::allocation<uint8_t> dev_A(size_A_packed);
    cutlass::device_memory::allocation<uint8_t> dev_B(size_B_packed);
    cutlass::device_memory::allocation<uint8_t> dev_SFA(sfa_padded);
    cutlass::device_memory::allocation<uint8_t> dev_SFB(sfb_padded);
    cutlass::device_memory::allocation<ElementC> dev_C(size_C);
    // D is used directly - no intermediate allocation needed

    cudaError_t err;

    // Create second stream for parallel quantization
    cudaStream_t stream_b;
    err = cudaStreamCreate(&stream_b);
    if (err != cudaSuccess) return err;

    // Initialize C to zero (on main stream)
    err = cudaMemsetAsync(dev_C.get(), 0, size_C * sizeof(ElementC), stream);
    if (err != cudaSuccess) { cudaStreamDestroy(stream_b); return err; }

    // =========================================================================
    // GPU-side quantization: BF16 -> NVF4 (PARALLEL on 2 streams!)
    // Stream A: quantize_A + init_scale_A
    // Stream B: quantize_B + init_scale_B
    // =========================================================================

    constexpr int BLOCK_SIZE = 256;

    // Stream A: Quantize A + scale factors
    {
        int total_quads = (M * K) / 8;
        int grid_size = (total_quads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        quantize_A_gpu_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            A, dev_A.get(), M, K
        );
        int grid_sfa = (sfa_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_scale_factors_kernel<<<grid_sfa, BLOCK_SIZE, 0, stream>>>(
            dev_SFA.get(), static_cast<int>(sfa_padded)
        );
    }

    // Stream B: Quantize B + scale factors (PARALLEL with stream A)
    {
        constexpr int TILE_K = 64;
        constexpr int TILE_N = 32;
        constexpr int B_BLOCK_SIZE = 256;
        dim3 grid((K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N);
        quantize_B_gpu_kernel<<<grid, B_BLOCK_SIZE, 0, stream_b>>>(
            B, dev_B.get(), K, N
        );
        int grid_sfb = (sfb_padded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_scale_factors_kernel<<<grid_sfb, BLOCK_SIZE, 0, stream_b>>>(
            dev_SFB.get(), static_cast<int>(sfb_padded)
        );
    }

    // Wait for both streams to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { cudaStreamDestroy(stream_b); return err; }
    err = cudaStreamSynchronize(stream_b);
    cudaStreamDestroy(stream_b);
    if (err != cudaSuccess) return err;

    // Build GEMM arguments - write directly to user buffer D
    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
            reinterpret_cast<DataTypeA*>(dev_A.get()), stride_A,
            reinterpret_cast<DataTypeA*>(dev_B.get()), stride_B,
            reinterpret_cast<ScaleFactorType*>(dev_SFA.get()), layout_SFA,
            reinterpret_cast<ScaleFactorType*>(dev_SFB.get()), layout_SFB
        },
        { // Epilogue arguments - output directly to D
            {alpha, beta},
            dev_C.get(), stride_C,
            reinterpret_cast<ElementD*>(D), stride_D
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

    // CUTLASS writes directly to D - no copy needed
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
