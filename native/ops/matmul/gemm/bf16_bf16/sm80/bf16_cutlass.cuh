/**
 * CUTLASS-based GEMM kernels for PyGPUkit
 *
 * Provides high-performance matrix multiplication using NVIDIA CUTLASS library.
 * Multi-SM support with runtime dispatch for optimal performance across GPU architectures.
 *
 * Supported architectures (CUTLASS 2.x API):
 * - SM 80 (A100): 4-stage pipeline, 48KB shared memory (datacenter)
 * - SM 86 (RTX 30xx): 5-stage pipeline, 100KB shared memory (Ampere consumer)
 * - SM 89 (RTX 40xx): 6-stage pipeline, 128KB shared memory (Ada Lovelace)
 *
 * Future architectures (CUTLASS 3.x/4.x API):
 * - SM 90 (H100): Hopper with WGMMA/TMA (see matmul_cutlass_sm90.cuh)
 * - SM 100 (B200): Blackwell datacenter, 232KB smem, 2SM MMA (see matmul_cutlass_sm100.cuh)
 * - SM 120 (RTX 5090): Blackwell GeForce, 101KB smem, no cluster (see matmul_cutlass_sm120.cuh)
 *
 * NOT supported:
 * - SM < 80 (Turing and older)
 *
 * Supported dtypes:
 * - FP32 (with TF32 TensorCore acceleration)
 * - FP16 (native TensorCore)
 * - BF16 (native TensorCore)
 *
 * Epilogue variants:
 * - Default: D = alpha * A @ B + beta * C
 * - Bias: D = A @ B + bias (with broadcast)
 * - BiasGELU: D = gelu(A @ B + bias)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/util/device_memory.h"

// SM90+ kernels use CUTLASS 3.x/4.x API
// Conditionally included based on CUTLASS compile-time architecture support

// SM90 (Hopper) - CUTLASS 3.x with WGMMA/TMA
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#include "../sm90/bf16_cutlass.cuh"
#endif

// SM100 (Blackwell datacenter: B200) - CUTLASS 4.x with 2SM MMA
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#include "../sm100/bf16_cutlass.cuh"
#endif

// NOTE: SM120 CUTLASS 4.x kernels are DISABLED.
// CUTLASS 4.3.3's SM120 builder only supports F8F6F4 (FP8/FP6/FP4) MMA,
// NOT FP32/FP16/BF16. Will be re-enabled when FP8 support is added.
//
// #if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
// #include "../sm120/bf16_cutlass.cuh"
// #endif

namespace pygpukit {
namespace ops {
namespace cutlass_gemm {

// ============================================================================
// SM Version Detection
// ============================================================================

// Cached SM version (initialized on first use)
inline int get_cached_sm_version() {
    static int sm_version = -1;
    if (sm_version < 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);
        sm_version = props.major * 10 + props.minor;
    }
    return sm_version;
}

// Minimum supported SM version
constexpr int MIN_SM_VERSION = 80;

// Check if SM version is supported for CUTLASS kernels
// Note: SM 120 (Blackwell GeForce) can use CUTLASS 2.x kernels (SM80 ArchTag)
//       as a fallback since Blackwell supports all Ampere instructions.
//       CUTLASS 4.x native SM120 kernels only support FP8, so we use SM80 path.
inline bool is_sm_supported() {
    int sm = get_cached_sm_version();
    // SM 80+: CUTLASS 2.x/3.x kernels work
    // SM 120: Uses CUTLASS 2.x (SM80 ArchTag) as fallback
    return sm >= MIN_SM_VERSION;
}

// SM version classification for kernel selection
// Returns the "tier" for kernel dispatch:
//   120: SM120+ (Blackwell GeForce: RTX 5090/5080)
//   100: SM100-119 (Blackwell datacenter: B200)
//    90: SM90-99 (Hopper: H100)
//    89: SM89 (Ada Lovelace: RTX 40xx)
//    86: SM86-88 (Ampere consumer: RTX 30xx)
//    80: SM80-85 (Ampere datacenter: A100)
inline int get_sm_tier() {
    int sm = get_cached_sm_version();
    if (sm >= 120) return 120;  // Blackwell GeForce (RTX 5090)
    if (sm >= 100) return 100;  // Blackwell datacenter (B200)
    if (sm >= 90)  return 90;   // Hopper (H100)
    if (sm >= 89)  return 89;   // Ada Lovelace (RTX 40xx)
    if (sm >= 86)  return 86;   // Ampere (consumer)
    return 80;                   // Ampere (datacenter)
}

// Check if SM >= 86 for optimized 5-stage pipeline (legacy, for backward compat)
inline bool use_5stage_pipeline() {
    return get_cached_sm_version() >= 86;
}

// ============================================================================
// TF32 GEMM (FP32 input/output, TF32 TensorCore)
// ============================================================================

// TF32 GEMM: FP32 in -> TF32 TensorCore -> FP32 out
// For row-major inputs, use all-ColumnMajor with transpose trick:
//   C (MxN row) = A (MxK row) @ B (KxN row)
//   becomes: C^T (NxM col) = B^T (NxK col) @ A^T (KxM col)
// where row-major X = col-major X^T in memory

// SM80 (A100): 4-stage pipeline, optimized for data center
using TF32Gemm_Sm80 = cutlass::gemm::device::Gemm<
    float,                                      // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    float,                                      // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    float,                                      // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere)
    cutlass::gemm::GemmShape<128, 128, 16>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 16>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 8>,         // InstructionShape (mma.sync)
    cutlass::epilogue::thread::LinearCombination<
        float, 128 / cutlass::sizeof_bits<float>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                           // Stages (4-stage for SM80)
>;

// SM86 (RTX 30xx): 5-stage pipeline, 100KB shared memory
using TF32Gemm_Sm86 = cutlass::gemm::device::Gemm<
    float,                                      // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    float,                                      // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    float,                                      // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere TensorCore compatible)
    cutlass::gemm::GemmShape<128, 128, 16>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 16>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 8>,         // InstructionShape (mma.sync)
    cutlass::epilogue::thread::LinearCombination<
        float, 128 / cutlass::sizeof_bits<float>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5                                           // Stages (5-stage for SM86)
>;

// SM89 (RTX 40xx Ada): 6-stage pipeline, 128KB shared memory
// Note: Uses Sm80 arch tag for CUTLASS 2.x compatibility (runtime dispatch selects for SM89)
using TF32Gemm_Sm89 = cutlass::gemm::device::Gemm<
    float,                                      // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    float,                                      // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    float,                                      // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Sm80 for CUTLASS 2.x compat)
    cutlass::gemm::GemmShape<128, 256, 16>,     // ThreadBlockShape (larger N for Ada)
    cutlass::gemm::GemmShape<64, 64, 16>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 8>,         // InstructionShape (mma.sync)
    cutlass::epilogue::thread::LinearCombination<
        float, 128 / cutlass::sizeof_bits<float>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    6                                           // Stages (6-stage for SM89)
>;

// Default alias (SM80 for backward compatibility)
using TF32Gemm = TF32Gemm_Sm80;

// ============================================================================
// TF32 Batched GEMM (FP32 input/output, TF32 TensorCore for batch operations)
// ============================================================================

// SM86 (RTX 30xx): 5-stage pipeline for batched operations
using TF32GemmBatched_Sm86 = cutlass::gemm::device::GemmBatched<
    float,                                      // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    float,                                      // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    float,                                      // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere TensorCore compatible)
    cutlass::gemm::GemmShape<128, 128, 16>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 16>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 8>,         // InstructionShape (mma.sync)
    cutlass::epilogue::thread::LinearCombination<
        float, 128 / cutlass::sizeof_bits<float>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    5                                           // Stages (5-stage for SM86)
>;

// Default batched alias
using TF32GemmBatched = TF32GemmBatched_Sm86;

// ============================================================================
// FP16 GEMM (FP16 input/output, FP16 TensorCore)
// ============================================================================

// SM80 (A100): FP16 GEMM with 4-stage pipeline
using FP16Gemm_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                            // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::half_t,                            // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::half_t,                            // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape (mma.sync.m16n8k16)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                           // Stages (4-stage for SM80)
>;

// SM86 (RTX 30xx): FP16 GEMM with 5-stage pipeline
using FP16Gemm_Sm86 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                            // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::half_t,                            // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::half_t,                            // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere TensorCore compatible)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape (mma.sync.m16n8k16)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5                                           // Stages (5-stage for SM86)
>;

// SM89 (RTX 40xx Ada): FP16 GEMM with 6-stage pipeline
// Note: Uses Sm80 arch tag for CUTLASS 2.x compatibility (runtime dispatch selects for SM89)
using FP16Gemm_Sm89 = cutlass::gemm::device::Gemm<
    cutlass::half_t,                            // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::half_t,                            // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::half_t,                            // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Sm80 for CUTLASS 2.x compat)
    cutlass::gemm::GemmShape<128, 256, 32>,     // ThreadBlockShape (larger N for Ada)
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape (mma.sync.m16n8k16)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    6                                           // Stages (6-stage for SM89)
>;

// Default alias (SM80 for backward compatibility)
using FP16Gemm = FP16Gemm_Sm80;

// ============================================================================
// BF16 GEMM (BF16 input/output, BF16 TensorCore)
// ============================================================================

// SM80 (A100): BF16 GEMM with 4-stage pipeline
using BF16Gemm_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,                        // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::bfloat16_t,                        // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::bfloat16_t,                        // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4                                           // Stages (4-stage for SM80)
>;

// SM86 (RTX 30xx): BF16 GEMM with 5-stage pipeline
using BF16Gemm_Sm86 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,                        // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::bfloat16_t,                        // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::bfloat16_t,                        // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Ampere TensorCore compatible)
    cutlass::gemm::GemmShape<128, 128, 32>,     // ThreadBlockShape
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5                                           // Stages (5-stage for SM86)
>;

// SM89 (RTX 40xx Ada): BF16 GEMM with 6-stage pipeline
// Note: Uses Sm80 arch tag for CUTLASS 2.x compatibility (runtime dispatch selects for SM89)
using BF16Gemm_Sm89 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,                        // ElementA (will be B^T)
    cutlass::layout::ColumnMajor,               // LayoutA
    cutlass::bfloat16_t,                        // ElementB (will be A^T)
    cutlass::layout::ColumnMajor,               // LayoutB
    cutlass::bfloat16_t,                        // ElementC (will be C^T)
    cutlass::layout::ColumnMajor,               // LayoutC
    float,                                      // ElementAccumulator (FP32 for precision)
    cutlass::arch::OpClassTensorOp,             // OperatorClass (TensorCore)
    cutlass::arch::Sm80,                        // ArchTag (Sm80 for CUTLASS 2.x compat)
    cutlass::gemm::GemmShape<128, 256, 32>,     // ThreadBlockShape (larger N for Ada)
    cutlass::gemm::GemmShape<64, 64, 32>,       // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value,
        float, float>,                          // EpilogueOp (128-bit aligned)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    6                                           // Stages (6-stage for SM89)
>;

// Default alias (SM80 for backward compatibility)
using BF16Gemm = BF16Gemm_Sm80;

// ============================================================================
// BiasGELU GEMM Types (Epilogue Fusion: D = gelu(A @ B + bias))
// ============================================================================

// TF32 BiasGELU - SM80 (4-stage)
using TF32GemmBiasGELU_Sm80 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

// TF32 BiasGELU - SM86 (5-stage)
using TF32GemmBiasGELU_Sm86 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5
>;

// TF32 BiasGELU - SM89 (6-stage)
// Note: Uses Sm80 arch tag for CUTLASS 2.x compatibility
using TF32GemmBiasGELU_Sm89 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    6
>;

using TF32GemmBiasGELU = TF32GemmBiasGELU_Sm80;

// FP16 BiasGELU - SM80 (4-stage)
using FP16GemmBiasGELU_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

// FP16 BiasGELU - SM86+ (5-stage)
using FP16GemmBiasGELU_Sm86 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5
>;

using FP16GemmBiasGELU = FP16GemmBiasGELU_Sm80;

// BF16 BiasGELU - SM80 (4-stage)
using BF16GemmBiasGELU_Sm80 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

// BF16 BiasGELU - SM86+ (5-stage)
using BF16GemmBiasGELU_Sm86 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5
>;

// FP16 BiasGELU - SM89 (6-stage, Ada Lovelace)
// Note: Uses Sm80 arch tag for CUTLASS 2.x compatibility
using FP16GemmBiasGELU_Sm89 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    6
>;

// BF16 BiasGELU - SM89 (6-stage, Ada Lovelace)
// Note: Uses Sm80 arch tag for CUTLASS 2.x compatibility
using BF16GemmBiasGELU_Sm89 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationGELU<
        cutlass::bfloat16_t, 128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    6
>;

using FP16GemmBiasGELU = FP16GemmBiasGELU_Sm80;
using BF16GemmBiasGELU = BF16GemmBiasGELU_Sm80;

// ============================================================================
// Wrapper functions
// ============================================================================

/**
 * TF32 GEMM: C = alpha * A @ B + beta * C
 *
 * @param A Input matrix A (M x K), row-major, FP32
 * @param B Input matrix B (K x N), row-major, FP32
 * @param C Output matrix C (M x N), row-major, FP32
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A @ B
 * @param beta Scalar multiplier for C (set to 0 for C = A @ B)
 * @param stream CUDA stream
 * @return cudaError_t
 *
 * Layout trick for row-major inputs with RowMajorxColumnMajor kernel:
 * - CUTLASS kernel: D (MxN row) = A (MxK row) @ B (KxN col)
 * - Our inputs: C (MxN row) = A (MxK row) @ B (KxN row)
 *
 * Key insight: row-major B (KxN) = column-major B^T (NxK) in memory
 *
 * We compute: C^T (NxM row) = B^T (NxK row) @ A^T (KxM col)
 * Which is equivalent to: C (MxN row) = A (MxK row) @ B (KxN row)
 *
 * For the kernel:
 * - M' = N, N' = M, K' = K
 * - A' = B^T (NxK row-major), pointer = B, ld = N (stride between rows)
 * - B' = A^T (KxM col-major) = A (MxK row-major) in memory, pointer = A, ld = K
 * - C' = C^T (NxM row-major), pointer = C, ld = M (stride between rows)
 */
// Template helper for GEMM dispatch
template<typename GemmOp>
inline cudaError_t run_gemm(
    cutlass::gemm::GemmCoord problem_size,
    const void* A, int ldA,
    const void* B, int ldB,
    void* C, int ldC,
    void* D, int ldD,
    float alpha, float beta,
    cudaStream_t stream
) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementC = typename GemmOp::ElementC;

    typename GemmOp::Arguments arguments{
        problem_size,
        {static_cast<const ElementA*>(A), ldA},
        {static_cast<const ElementB*>(B), ldB},
        {static_cast<ElementC*>(C), ldC},
        {static_cast<ElementC*>(D), ldD},
        {alpha, beta}
    };

    GemmOp gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = GemmOp::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

inline cudaError_t gemm_tf32(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    // Runtime SM dispatch with tiered kernel selection
    int sm_tier = get_sm_tier();

    // SM120 (Blackwell GeForce): Use CUTLASS 2.x (SM86) as fallback
    // CUTLASS 4.x native SM120 kernels only support FP8, not FP32/FP16/BF16
    // SM100/SM90 kernels also don't work on SM120 (different tensor core gen)

    // SM100 (Blackwell datacenter: B200 only, NOT SM120)
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (sm_tier >= 100 && sm_tier < 120) {
        return cutlass_gemm_sm100::gemm_tf32_sm100(A, B, C, M, N, K, alpha, beta, stream);
    }
#endif

    // SM90-99 (Hopper: H100 only)
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    if (sm_tier >= 90 && sm_tier < 100) {
        return cutlass_gemm_sm90::gemm_tf32_sm90(A, B, C, M, N, K, alpha, beta, stream);
    }
#endif

    // CUTLASS 2.x API for SM80-89 AND SM120+ (Blackwell GeForce fallback)
    // Transpose trick: C^T (NxM col) = B^T (NxK col) @ A^T (KxM col)
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // SM120+ uses SM86 kernel (5-stage, works on Blackwell)
    if (sm_tier >= 120 || sm_tier == 89) {
        // SM120 (Blackwell GeForce) / SM89 (Ada): Use SM86 5-stage for stability
        return run_gemm<TF32Gemm_Sm86>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    } else if (sm_tier >= 86) {
        // SM86-88 (Ampere consumer): 5-stage pipeline
        return run_gemm<TF32Gemm_Sm86>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    } else {
        // SM80-85 (Ampere datacenter): 4-stage pipeline
        return run_gemm<TF32Gemm_Sm80>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    }
}

/**
 * FP16 GEMM: C = alpha * A @ B + beta * C (row-major inputs)
 * Uses same transpose trick as TF32
 */
inline cudaError_t gemm_fp16(
    const __half* A,
    const __half* B,
    __half* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    // Runtime SM dispatch with tiered kernel selection
    int sm_tier = get_sm_tier();

    // SM100 (Blackwell datacenter: B200 only, NOT SM120)
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (sm_tier >= 100 && sm_tier < 120) {
        return cutlass_gemm_sm100::gemm_fp16_sm100(A, B, C, M, N, K, alpha, beta, stream);
    }
#endif

    // SM90-99 (Hopper: H100 only)
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    if (sm_tier >= 90 && sm_tier < 100) {
        return cutlass_gemm_sm90::gemm_fp16_sm90(A, B, C, M, N, K, alpha, beta, stream);
    }
#endif

    // CUTLASS 2.x API for SM80-89 AND SM120+ (Blackwell GeForce fallback)
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    if (sm_tier >= 120 || sm_tier == 89) {
        // SM120 (Blackwell GeForce) / SM89 (Ada): Use SM86 5-stage
        return run_gemm<FP16Gemm_Sm86>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    } else if (sm_tier >= 86) {
        // SM86-88 (Ampere consumer): 5-stage pipeline
        return run_gemm<FP16Gemm_Sm86>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    } else {
        // SM80-85 (Ampere datacenter): 4-stage pipeline
        return run_gemm<FP16Gemm_Sm80>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    }
}

/**
 * BF16 GEMM: C = alpha * A @ B + beta * C (row-major inputs)
 * Uses same transpose trick as TF32
 */
inline cudaError_t gemm_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    // Runtime SM dispatch with tiered kernel selection
    int sm_tier = get_sm_tier();

    // SM100 (Blackwell datacenter: B200 only, NOT SM120)
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    if (sm_tier >= 100 && sm_tier < 120) {
        return cutlass_gemm_sm100::gemm_bf16_sm100(A, B, C, M, N, K, alpha, beta, stream);
    }
#endif

    // SM90-99 (Hopper: H100 only)
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    if (sm_tier >= 90 && sm_tier < 100) {
        return cutlass_gemm_sm90::gemm_bf16_sm90(A, B, C, M, N, K, alpha, beta, stream);
    }
#endif

    // CUTLASS 2.x API for SM80-89 AND SM120+ (Blackwell GeForce fallback)
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    if (sm_tier >= 120 || sm_tier == 89) {
        // SM120 (Blackwell GeForce) / SM89 (Ada): Use SM86 5-stage
        return run_gemm<BF16Gemm_Sm86>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    } else if (sm_tier >= 86) {
        // SM86-88 (Ampere consumer): 5-stage pipeline
        return run_gemm<BF16Gemm_Sm86>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    } else {
        // SM80-85 (Ampere datacenter): 4-stage pipeline
        return run_gemm<BF16Gemm_Sm80>(
            problem_size, B, N, A, K, C, N, C, N, alpha, beta, stream);
    }
}

// ============================================================================
// BiasGELU Wrapper functions
// ============================================================================

// Template helper for BiasGELU GEMM dispatch (with bias broadcast stride=0)
template<typename GemmOp>
inline cudaError_t run_gemm_bias_gelu(
    cutlass::gemm::GemmCoord problem_size,
    const void* A, int ldA,
    const void* B, int ldB,
    const void* bias,  // stride=0 for broadcast
    void* D, int ldD,
    cudaStream_t stream
) {
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementC = typename GemmOp::ElementC;

    typename GemmOp::Arguments arguments{
        problem_size,
        {static_cast<const ElementA*>(A), ldA},
        {static_cast<const ElementB*>(B), ldB},
        {static_cast<const ElementC*>(bias), 0},  // stride=0 for broadcast
        {static_cast<ElementC*>(D), ldD},
        {1.0f, 1.0f}  // alpha=1, beta=1
    };

    GemmOp gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = GemmOp::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

/**
 * TF32 GEMM with fused BiasGELU: D = gelu(A @ B + bias)
 * Uses transpose trick, bias broadcast with stride=0
 */
inline cudaError_t gemm_tf32_bias_gelu(
    const float* A,
    const float* B,
    const float* bias,
    float* D,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Runtime SM dispatch with tiered kernel selection
    int sm_tier = get_sm_tier();
    if (sm_tier >= 89) {
        return run_gemm_bias_gelu<TF32GemmBiasGELU_Sm89>(
            problem_size, B, N, A, K, bias, D, N, stream);
    } else if (sm_tier >= 86) {
        return run_gemm_bias_gelu<TF32GemmBiasGELU_Sm86>(
            problem_size, B, N, A, K, bias, D, N, stream);
    } else {
        return run_gemm_bias_gelu<TF32GemmBiasGELU_Sm80>(
            problem_size, B, N, A, K, bias, D, N, stream);
    }
}

/**
 * FP16 GEMM with fused BiasGELU: D = gelu(A @ B + bias)
 */
inline cudaError_t gemm_fp16_bias_gelu(
    const __half* A,
    const __half* B,
    const __half* bias,
    __half* D,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Runtime SM dispatch with tiered kernel selection
    int sm_tier = get_sm_tier();
    if (sm_tier >= 89) {
        return run_gemm_bias_gelu<FP16GemmBiasGELU_Sm89>(
            problem_size, B, N, A, K, bias, D, N, stream);
    } else if (sm_tier >= 86) {
        return run_gemm_bias_gelu<FP16GemmBiasGELU_Sm86>(
            problem_size, B, N, A, K, bias, D, N, stream);
    } else {
        return run_gemm_bias_gelu<FP16GemmBiasGELU_Sm80>(
            problem_size, B, N, A, K, bias, D, N, stream);
    }
}

/**
 * BF16 GEMM with fused BiasGELU: D = gelu(A @ B + bias)
 */
inline cudaError_t gemm_bf16_bias_gelu(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    const __nv_bfloat16* bias,
    __nv_bfloat16* D,
    int M, int N, int K,
    cudaStream_t stream = nullptr
) {
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Runtime SM dispatch with tiered kernel selection
    int sm_tier = get_sm_tier();
    if (sm_tier >= 89) {
        return run_gemm_bias_gelu<BF16GemmBiasGELU_Sm89>(
            problem_size, B, N, A, K, bias, D, N, stream);
    } else if (sm_tier >= 86) {
        return run_gemm_bias_gelu<BF16GemmBiasGELU_Sm86>(
            problem_size, B, N, A, K, bias, D, N, stream);
    } else {
        return run_gemm_bias_gelu<BF16GemmBiasGELU_Sm80>(
            problem_size, B, N, A, K, bias, D, N, stream);
    }
}

// ============================================================================
// Batched GEMM Implementation
// ============================================================================

/**
 * Template helper for batched GEMM dispatch
 *
 * Memory layout for strided batched GEMM:
 * - A[batch, M, K] row-major: stride_A = M * K
 * - B[batch, K, N] row-major: stride_B = K * N
 * - C[batch, M, N] row-major: stride_C = M * N
 *
 * Using the transpose trick for CUTLASS column-major kernels:
 * - C^T[batch, N, M] = B^T[batch, N, K] @ A^T[batch, K, M]
 */
template<typename GemmBatchedOp>
inline cudaError_t run_gemm_batched(
    cutlass::gemm::GemmCoord problem_size,
    const void* A, int ldA, int64_t strideA,
    const void* B, int ldB, int64_t strideB,
    void* C, int ldC, int64_t strideC,
    float alpha, float beta,
    int batch_count,
    cudaStream_t stream
) {
    using ElementA = typename GemmBatchedOp::ElementA;
    using ElementB = typename GemmBatchedOp::ElementB;
    using ElementC = typename GemmBatchedOp::ElementC;

    typename GemmBatchedOp::Arguments arguments{
        problem_size,
        {static_cast<const ElementA*>(A), ldA},
        strideA,
        {static_cast<const ElementB*>(B), ldB},
        strideB,
        {static_cast<ElementC*>(C), ldC},
        strideC,
        {static_cast<ElementC*>(C), ldC},
        strideC,
        {alpha, beta},
        batch_count
    };

    GemmBatchedOp gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t workspace_size = GemmBatchedOp::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

/**
 * FP32 Strided Batched GEMM using CUTLASS TensorCore (TF32)
 *
 * Computes: C[b] = A[b] @ B[b] for b in [0, batch_count)
 * Where A[batch, M, K], B[batch, K, N], C[batch, M, N] are row-major.
 */
inline cudaError_t gemm_batched_fp32(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int batch_count,
    int64_t strideA,
    int64_t strideB,
    int64_t strideC,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
) {
    // Transpose trick: C^T[N,M] = B^T[N,K] @ A^T[K,M]
    // For batched: each batch element uses the same transformation
    cutlass::gemm::GemmCoord problem_size(N, M, K);

    // Note: Strides remain the same (element count between batches)
    // but the roles of A/B are swapped for the transpose trick
    return run_gemm_batched<TF32GemmBatched_Sm86>(
        problem_size,
        B, N, strideB,   // B^T as first operand (ld = N)
        A, K, strideA,   // A^T as second operand (ld = K)
        C, N, strideC,   // C^T as output (ld = N)
        alpha, beta,
        batch_count,
        stream
    );
}

// ============================================================================
// Dispatch function for runtime dtype selection
// ============================================================================

enum class GemmDtype {
    FP32_TF32,  // FP32 input, TF32 TensorCore
    FP16,       // FP16 TensorCore
    BF16        // BF16 TensorCore
};

/**
 * Check if matrix dimensions are compatible with CUTLASS TensorCore kernels
 * TensorCore requires alignment to tile sizes
 */
inline bool is_cutlass_compatible(int M, int N, int K) {
    // Minimum alignment for TensorCore (based on ThreadBlockShape)
    // TF32: 128x128x16, FP16/BF16: 128x128x32
    // For simplicity, require 16-alignment on all dimensions
    return (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0);
}

}  // namespace cutlass_gemm
}  // namespace ops
}  // namespace pygpukit
