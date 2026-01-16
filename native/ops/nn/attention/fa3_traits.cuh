/**
 * Flash Attention 3 - Architecture Traits
 *
 * Defines architecture-specific types and constants for FA3.
 * Supports SM100 (Blackwell datacenter) and SM120 (Blackwell GeForce).
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pygpukit {
namespace ops {
namespace nn {
namespace fa3 {

// =============================================================================
// Architecture Detection
// =============================================================================

enum class Arch {
    SM90,   // Hopper (future)
    SM100,  // Blackwell datacenter - tcgen05
    SM120,  // Blackwell GeForce - mma.sync.kind::f8f6f4
    Unknown
};

__host__ __device__ constexpr Arch get_arch(int sm_version) {
    if (sm_version >= 120) return Arch::SM120;
    if (sm_version >= 100) return Arch::SM100;
    if (sm_version >= 90)  return Arch::SM90;
    return Arch::Unknown;
}

// =============================================================================
// Tile Configuration
// =============================================================================

template<Arch arch, typename Element>
struct TileConfig;

// SM120 BF16 configuration
template<>
struct TileConfig<Arch::SM120, __nv_bfloat16> {
    // MMA tile: m16n8k16 for BF16
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 16;

    // Block tile
    static constexpr int TILE_Q = 64;      // Q positions per block
    static constexpr int TILE_KV = 64;     // KV positions per iteration
    static constexpr int HEAD_DIM = 128;   // Head dimension

    // Warp configuration
    static constexpr int NUM_WARPS = 12;           // Total warps
    static constexpr int NUM_PRODUCER_WARPS = 4;   // TMA load warps
    static constexpr int NUM_CONSUMER_WARPS = 8;   // MMA warps (2 warpgroups)

    // Pipeline stages
    static constexpr int STAGES_KV = 4;    // KV double-buffer + prefetch
    static constexpr int STAGES_Q = 2;     // Q double-buffer

    // Shared memory
    static constexpr int SMEM_Q_SIZE = TILE_Q * HEAD_DIM * sizeof(__nv_bfloat16);
    static constexpr int SMEM_K_SIZE = TILE_KV * HEAD_DIM * sizeof(__nv_bfloat16) * STAGES_KV;
    static constexpr int SMEM_V_SIZE = TILE_KV * HEAD_DIM * sizeof(__nv_bfloat16) * STAGES_KV;
};

// SM120 FP16 configuration
template<>
struct TileConfig<Arch::SM120, __half> {
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 16;

    static constexpr int TILE_Q = 64;
    static constexpr int TILE_KV = 64;
    static constexpr int HEAD_DIM = 128;

    static constexpr int NUM_WARPS = 12;
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int NUM_CONSUMER_WARPS = 8;

    static constexpr int STAGES_KV = 4;
    static constexpr int STAGES_Q = 2;

    static constexpr int SMEM_Q_SIZE = TILE_Q * HEAD_DIM * sizeof(__half);
    static constexpr int SMEM_K_SIZE = TILE_KV * HEAD_DIM * sizeof(__half) * STAGES_KV;
    static constexpr int SMEM_V_SIZE = TILE_KV * HEAD_DIM * sizeof(__half) * STAGES_KV;
};

// SM100 placeholder (tcgen05 - to be implemented)
template<>
struct TileConfig<Arch::SM100, __nv_bfloat16> {
    // SM100 uses tcgen05.mma with different tile sizes
    static constexpr int MMA_M = 64;   // tcgen05 supports larger tiles
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 32;

    static constexpr int TILE_Q = 128;
    static constexpr int TILE_KV = 128;
    static constexpr int HEAD_DIM = 128;

    static constexpr int NUM_WARPS = 12;
    static constexpr int NUM_PRODUCER_WARPS = 4;
    static constexpr int NUM_CONSUMER_WARPS = 8;

    static constexpr int STAGES_KV = 5;
    static constexpr int STAGES_Q = 2;

    static constexpr int SMEM_Q_SIZE = TILE_Q * HEAD_DIM * sizeof(__nv_bfloat16);
    static constexpr int SMEM_K_SIZE = TILE_KV * HEAD_DIM * sizeof(__nv_bfloat16) * STAGES_KV;
    static constexpr int SMEM_V_SIZE = TILE_KV * HEAD_DIM * sizeof(__nv_bfloat16) * STAGES_KV;
};

// =============================================================================
// MMA Operation Traits
// =============================================================================

template<Arch arch, typename Element, typename Accumulator = float>
struct MmaTraits;

// SM120 BF16 MMA traits
template<>
struct MmaTraits<Arch::SM120, __nv_bfloat16, float> {
    using ElementA = __nv_bfloat16;
    using ElementB = __nv_bfloat16;
    using ElementC = float;

    // mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 16;

    // Fragment sizes (registers per thread)
    static constexpr int A_REGS = 4;  // 4 x uint32
    static constexpr int B_REGS = 2;  // 2 x uint32
    static constexpr int C_REGS = 4;  // 4 x float
};

// SM120 FP16 MMA traits
template<>
struct MmaTraits<Arch::SM120, __half, float> {
    using ElementA = __half;
    using ElementB = __half;
    using ElementC = float;

    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 16;

    static constexpr int A_REGS = 4;
    static constexpr int B_REGS = 2;
    static constexpr int C_REGS = 4;
};

// SM100 BF16 MMA traits (tcgen05)
template<>
struct MmaTraits<Arch::SM100, __nv_bfloat16, float> {
    using ElementA = __nv_bfloat16;
    using ElementB = __nv_bfloat16;
    using ElementC = float;

    // tcgen05.mma.cta_group::1.kind::f16 (larger tiles)
    static constexpr int M = 64;
    static constexpr int N = 8;
    static constexpr int K = 32;

    // Tensor memory based - different register model
    static constexpr int A_REGS = 0;  // Uses tensor memory
    static constexpr int B_REGS = 0;
    static constexpr int C_REGS = 0;
};

// =============================================================================
// Pipeline Configuration
// =============================================================================

template<Arch arch>
struct PipelineConfig {
    // Default: async pipeline with barriers
    static constexpr bool USE_TMA = true;
    static constexpr bool USE_WARP_SPECIALIZATION = true;
    static constexpr int PRODUCER_WARP_COUNT = 4;
    static constexpr int CONSUMER_WARP_COUNT = 8;
};

template<>
struct PipelineConfig<Arch::SM100> {
    static constexpr bool USE_TMA = true;
    static constexpr bool USE_WARP_SPECIALIZATION = true;
    static constexpr bool USE_TENSOR_MEMORY = true;  // SM100 has tensor memory
    static constexpr int PRODUCER_WARP_COUNT = 4;
    static constexpr int CONSUMER_WARP_COUNT = 8;
};

template<>
struct PipelineConfig<Arch::SM120> {
    static constexpr bool USE_TMA = true;
    static constexpr bool USE_WARP_SPECIALIZATION = true;
    static constexpr bool USE_TENSOR_MEMORY = false;  // SM120 no tensor memory
    static constexpr int PRODUCER_WARP_COUNT = 4;
    static constexpr int CONSUMER_WARP_COUNT = 8;
};

}  // namespace fa3
}  // namespace nn
}  // namespace ops
}  // namespace pygpukit
