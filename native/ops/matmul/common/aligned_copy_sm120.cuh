/**
 * Aligned Copy Operations for SM120 FP8 GEMM
 *
 * Workaround for CUTLASS Issue #2902:
 * - partition_S() drops alignment from 1024 to 8 bytes
 * - SM75_U32x4_LDSM_N requires 16-byte alignment
 *
 * This file provides:
 * 1. Inline PTX helpers for alignment-safe shared memory loads
 * 2. A macro to patch CUTLASS's LDSM operations post-include
 *
 * Usage:
 *   // Include this AFTER CUTLASS headers
 *   #include <cutlass/...>
 *   #include "aligned_copy_sm120.cuh"
 *
 *   // The CUTLASS kernel will use patched copy operations
 *   // if PYGPUKIT_PATCH_CUTLASS_LDSM_POST is defined
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Core PTX Helpers for Shared Memory Operations
// ============================================================================

namespace pygpukit {
namespace ops {
namespace aligned_copy {

/**
 * Convert shared memory pointer to generic address space (32-bit for PTX)
 */
__device__ __forceinline__
uint32_t smem_ptr_to_u32(const void* ptr) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
    return 0;
#endif
}

/**
 * Load 4x u32 (16 bytes) from shared memory with alignment check.
 *
 * IMPORTANT: ldmatrix.sync requires ALL threads in the warp to participate.
 * This function assumes it's called by the full warp (CUTLASS pattern).
 * For single-thread usage, use ld_shared_u32x4_scalar instead.
 *
 * Behavior:
 * - 16-byte aligned: uses ldmatrix.sync (fast, requires full warp)
 * - Misaligned: falls back to scalar loads (slower but always safe)
 */
__device__ __forceinline__
void ld_shared_u32x4_safe(
    uint32_t smem_addr,
    uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    if ((smem_addr & 0xF) == 0) {
        // 16-byte aligned: use ldmatrix (fast path)
        // NOTE: ldmatrix.sync requires all warp threads to execute this
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
            : "r"(smem_addr)
        );
    } else {
        // Misaligned: use scalar loads (slow but correct)
        asm volatile(
            "ld.shared.u32 %0, [%4];\n"
            "ld.shared.u32 %1, [%5];\n"
            "ld.shared.u32 %2, [%6];\n"
            "ld.shared.u32 %3, [%7];\n"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
            : "r"(smem_addr),
              "r"(smem_addr + 4u),
              "r"(smem_addr + 8u),
              "r"(smem_addr + 12u)
        );
    }
#endif
}

/**
 * Load 4x u32 with forced alignment (trust caller)
 */
__device__ __forceinline__
void ld_shared_u32x4_trusted(
    uint32_t smem_addr,
    uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        : "r"(smem_addr)
    );
#endif
}

/**
 * Load 4x u32 using scalar loads only (always safe)
 */
__device__ __forceinline__
void ld_shared_u32x4_scalar(
    uint32_t smem_addr,
    uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
{
#if defined(__CUDA_ARCH__)
    asm volatile(
        "ld.shared.u32 %0, [%4];\n"
        "ld.shared.u32 %1, [%5];\n"
        "ld.shared.u32 %2, [%6];\n"
        "ld.shared.u32 %3, [%7];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        : "r"(smem_addr),
          "r"(smem_addr + 4u),
          "r"(smem_addr + 8u),
          "r"(smem_addr + 12u)
    );
#endif
}

/**
 * Load 4x u32 with transpose and alignment check
 */
__device__ __forceinline__
void ld_shared_u32x4_trans_safe(
    uint32_t smem_addr,
    uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    if ((smem_addr & 0xF) == 0) {
        asm volatile(
            "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
            : "r"(smem_addr)
        );
    } else {
        // Scalar fallback (no transpose - caller must handle)
        asm volatile(
            "ld.shared.u32 %0, [%4];\n"
            "ld.shared.u32 %1, [%5];\n"
            "ld.shared.u32 %2, [%6];\n"
            "ld.shared.u32 %3, [%7];\n"
            : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
            : "r"(smem_addr),
              "r"(smem_addr + 4u),
              "r"(smem_addr + 8u),
              "r"(smem_addr + 12u)
        );
    }
#endif
}

/**
 * Load 2x u32 (8 bytes) with alignment check
 */
__device__ __forceinline__
void ld_shared_u32x2_safe(
    uint32_t smem_addr,
    uint32_t& dst0, uint32_t& dst1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    if ((smem_addr & 0x7) == 0) {
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(dst0), "=r"(dst1)
            : "r"(smem_addr)
        );
    } else {
        asm volatile(
            "ld.shared.u32 %0, [%2];\n"
            "ld.shared.u32 %1, [%3];\n"
            : "=r"(dst0), "=r"(dst1)
            : "r"(smem_addr),
              "r"(smem_addr + 4u)
        );
    }
#endif
}

/**
 * Load 1x u32 with ldmatrix
 */
__device__ __forceinline__
void ld_shared_u32x1(uint32_t smem_addr, uint32_t& dst0)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst0)
        : "r"(smem_addr)
    );
#endif
}

}  // namespace aligned_copy
}  // namespace ops
}  // namespace pygpukit

// ============================================================================
// CUTLASS Integration Macros
// ============================================================================

/**
 * Macro to wrap a shared memory load with alignment-safe version.
 * Use this in custom kernels or modified CUTLASS mainloops.
 *
 * Example:
 *   uint32_t r0, r1, r2, r3;
 *   PYGPUKIT_SAFE_LDSM_X4(smem_ptr, r0, r1, r2, r3);
 */
#define PYGPUKIT_SAFE_LDSM_X4(smem_ptr, r0, r1, r2, r3) \
    do { \
        uint32_t _addr = pygpukit::ops::aligned_copy::smem_ptr_to_u32(smem_ptr); \
        pygpukit::ops::aligned_copy::ld_shared_u32x4_safe(_addr, r0, r1, r2, r3); \
    } while(0)

#define PYGPUKIT_SAFE_LDSM_X4_TRANS(smem_ptr, r0, r1, r2, r3) \
    do { \
        uint32_t _addr = pygpukit::ops::aligned_copy::smem_ptr_to_u32(smem_ptr); \
        pygpukit::ops::aligned_copy::ld_shared_u32x4_trans_safe(_addr, r0, r1, r2, r3); \
    } while(0)

#define PYGPUKIT_SAFE_LDSM_X2(smem_ptr, r0, r1) \
    do { \
        uint32_t _addr = pygpukit::ops::aligned_copy::smem_ptr_to_u32(smem_ptr); \
        pygpukit::ops::aligned_copy::ld_shared_u32x2_safe(_addr, r0, r1); \
    } while(0)

// ============================================================================
// Post-Include Patch for CUTLASS SM75 LDSM Operations
// ============================================================================
//
// IMPORTANT: Include this AFTER cute/arch/copy_sm75.hpp
//
// This redefines the copy() function for SM75 LDSM structs using
// our alignment-safe implementations.
// ============================================================================

#if defined(PYGPUKIT_PATCH_CUTLASS_LDSM_POST) && defined(CUTE_ARCH_COPY_SM75_HPP)

// Ensure the original structs exist
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

namespace cute {

// Override SM75_U32x4_LDSM_N::copy with our safe version
// Note: This uses ADL to find our implementation
struct SM75_U32x4_LDSM_N_Safe : SM75_U32x4_LDSM_N {
    CUTE_HOST_DEVICE static void
    copy(uint128_t const& smem_src,
         uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
        uint32_t addr = pygpukit::ops::aligned_copy::smem_ptr_to_u32(&smem_src);
        pygpukit::ops::aligned_copy::ld_shared_u32x4_safe(addr, dst0, dst1, dst2, dst3);
#endif
    }
};

}  // namespace cute

#endif  // CUTE_ARCH_LDSM_SM75_ACTIVATED
#endif  // PYGPUKIT_PATCH_CUTLASS_LDSM_POST && CUTE_ARCH_COPY_SM75_HPP
