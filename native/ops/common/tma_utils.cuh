/**
 * TMA (Tensor Memory Accelerator) Utilities
 *
 * Provides TMA descriptor creation and async copy operations for SM90+.
 * Based on CUTLASS patterns from cute/arch/copy_sm90_tma.hpp
 *
 * Usage:
 *   - Flash Attention 3: Async Q/K/V tile loading
 *   - GEMM: Async A/B matrix tile loading
 *   - Any kernel needing efficient global->shared transfers
 *
 * Requirements:
 *   - CUDA 12.0+ for SM90 (Hopper)
 *   - CUDA 13.1+ for SM120 (Blackwell GeForce)
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace pygpukit {
namespace ops {
namespace tma {

// =============================================================================
// Architecture Detection
// =============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define PYGPUKIT_TMA_ENABLED 1
#else
#define PYGPUKIT_TMA_ENABLED 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define PYGPUKIT_TMA_SM120 1
#else
#define PYGPUKIT_TMA_SM120 0
#endif

// =============================================================================
// Shared Memory Pointer Utilities
// =============================================================================

__device__ __forceinline__
uint32_t smem_ptr_to_uint(void const* ptr) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
    return 0;
#endif
}

__device__ __forceinline__
uint32_t smem_ptr_to_uint(void* ptr) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
    return 0;
#endif
}

// =============================================================================
// Barrier Operations (mbarrier)
// =============================================================================

/**
 * Initialize a barrier in shared memory.
 * Must be called by a single thread per barrier before use.
 */
__device__ __forceinline__
void barrier_init(uint64_t& smem_barrier, int thread_count = 1) {
#if PYGPUKIT_TMA_ENABLED
    uint32_t smem_addr = smem_ptr_to_uint(&smem_barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(smem_addr), "r"(thread_count)
    );
#endif
}

/**
 * Set expected transaction bytes and arrive at barrier.
 * Called by producer threads before issuing TMA loads.
 */
__device__ __forceinline__
void barrier_arrive_expect_tx(uint64_t& smem_barrier, uint32_t tx_bytes) {
#if PYGPUKIT_TMA_ENABLED
    uint32_t smem_addr = smem_ptr_to_uint(&smem_barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(smem_addr), "r"(tx_bytes)
    );
#endif
}

/**
 * Arrive at barrier without transaction count.
 * Called by consumer threads.
 */
__device__ __forceinline__
void barrier_arrive(uint64_t& smem_barrier) {
#if PYGPUKIT_TMA_ENABLED
    uint32_t smem_addr = smem_ptr_to_uint(&smem_barrier);
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
        :: "r"(smem_addr)
    );
#endif
}

/**
 * Wait on barrier until phase bit flips.
 * Blocking wait - spins until barrier completes.
 */
__device__ __forceinline__
void barrier_wait(uint64_t& smem_barrier, int phase_bit) {
#if PYGPUKIT_TMA_ENABLED
    uint32_t smem_addr = smem_ptr_to_uint(&smem_barrier);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@!P1 bra LAB_WAIT;\n"
        "}\n"
        :: "r"(smem_addr), "r"(phase_bit)
        : "memory"
    );
#endif
}

/**
 * Non-blocking barrier test.
 * Returns true if barrier is complete.
 */
__device__ __forceinline__
bool barrier_try_wait(uint64_t& smem_barrier, int phase_bit) {
#if PYGPUKIT_TMA_ENABLED
    uint32_t smem_addr = smem_ptr_to_uint(&smem_barrier);
    uint32_t result;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
        "selp.u32 %0, 1, 0, P1;\n"
        "}\n"
        : "=r"(result)
        : "r"(smem_addr), "r"(phase_bit)
        : "memory"
    );
    return result != 0;
#else
    return true;
#endif
}

/**
 * Invalidate barrier (reset for next use).
 */
__device__ __forceinline__
void barrier_invalidate(uint64_t& smem_barrier) {
#if PYGPUKIT_TMA_ENABLED
    uint32_t smem_addr = smem_ptr_to_uint(&smem_barrier);
    asm volatile(
        "mbarrier.inval.shared::cta.b64 [%0];\n"
        :: "r"(smem_addr)
    );
#endif
}

// =============================================================================
// TMA Copy Operations
// =============================================================================

/**
 * TMA 2D load from global to shared memory.
 *
 * @param desc_ptr    Pointer to CUtensorMap descriptor
 * @param smem_ptr    Destination in shared memory
 * @param mbar_ptr    Barrier to signal on completion
 * @param crd0        First coordinate (innermost dimension)
 * @param crd1        Second coordinate (outer dimension)
 * @param cache_hint  L2 cache hint (0 for normal, 1 for streaming)
 */
__device__ __forceinline__
void tma_load_2d(
    void const* desc_ptr,
    void* smem_ptr,
    uint64_t* mbar_ptr,
    int32_t crd0,
    int32_t crd1,
    uint64_t cache_hint = 0
) {
#if PYGPUKIT_TMA_ENABLED
    uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_addr = smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = smem_ptr_to_uint(mbar_ptr);

#if PYGPUKIT_TMA_SM120
    // SM120: shared::cta (no cluster support on GeForce)
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;\n"
        :
        : "r"(smem_addr), "l"(gmem_desc), "r"(mbar_addr),
          "r"(crd0), "r"(crd1), "l"(cache_hint)
        : "memory"
    );
#else
    // SM90: shared::cluster
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;\n"
        :
        : "r"(smem_addr), "l"(gmem_desc), "r"(mbar_addr),
          "r"(crd0), "r"(crd1), "l"(cache_hint)
        : "memory"
    );
#endif
#endif
}

/**
 * TMA 3D load from global to shared memory.
 * Useful for loading with head dimension.
 */
__device__ __forceinline__
void tma_load_3d(
    void const* desc_ptr,
    void* smem_ptr,
    uint64_t* mbar_ptr,
    int32_t crd0,
    int32_t crd1,
    int32_t crd2,
    uint64_t cache_hint = 0
) {
#if PYGPUKIT_TMA_ENABLED
    uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_addr = smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = smem_ptr_to_uint(mbar_ptr);

#if PYGPUKIT_TMA_SM120
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;\n"
        :
        : "r"(smem_addr), "l"(gmem_desc), "r"(mbar_addr),
          "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
        : "memory"
    );
#else
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;\n"
        :
        : "r"(smem_addr), "l"(gmem_desc), "r"(mbar_addr),
          "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
        : "memory"
    );
#endif
#endif
}

/**
 * TMA prefetch to L2 cache (no shared memory destination).
 */
__device__ __forceinline__
void tma_prefetch_2d(
    void const* desc_ptr,
    int32_t crd0,
    int32_t crd1
) {
#if PYGPUKIT_TMA_ENABLED
    uint64_t gmem_desc = reinterpret_cast<uint64_t>(desc_ptr);
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];\n"
        :
        : "l"(gmem_desc), "r"(crd0), "r"(crd1)
        : "memory"
    );
#endif
}

// =============================================================================
// TMA Descriptor Creation (Host Side)
// =============================================================================

/**
 * TMA descriptor wrapper for attention tensors.
 * Stores the CUtensorMap and metadata.
 */
struct TmaDescriptor {
    CUtensorMap tensor_map;
    size_t tile_size_bytes;

    TmaDescriptor() : tile_size_bytes(0) {
        memset(&tensor_map, 0, sizeof(tensor_map));
    }
};

/**
 * Swizzle mode for TMA.
 * Higher swizzle = better bank conflict avoidance but stricter alignment.
 */
enum class SwizzleMode {
    None = 0,      // No swizzle
    Swizzle32B,    // 32-byte swizzle (256B alignment)
    Swizzle64B,    // 64-byte swizzle (512B alignment)
    Swizzle128B    // 128-byte swizzle (1024B alignment) - best for FA3
};

/**
 * Create a 2D TMA descriptor for attention tensor.
 *
 * @param desc         Output descriptor
 * @param base_ptr     Base pointer to tensor in global memory
 * @param dim0         Inner dimension size (e.g., head_dim)
 * @param dim1         Outer dimension size (e.g., seq_len)
 * @param stride0      Stride of inner dimension (usually 1)
 * @param stride1      Stride of outer dimension (usually dim0)
 * @param tile0        Tile size for inner dimension
 * @param tile1        Tile size for outer dimension
 * @param swizzle      Swizzle mode
 * @return             CUDA_SUCCESS on success
 */
inline CUresult create_tma_descriptor_2d_bf16(
    TmaDescriptor& desc,
    void* base_ptr,
    uint64_t dim0,
    uint64_t dim1,
    uint64_t stride0,
    uint64_t stride1,
    uint32_t tile0,
    uint32_t tile1,
    SwizzleMode swizzle = SwizzleMode::Swizzle128B
) {
    // Convert swizzle mode
    CUtensorMapSwizzle cu_swizzle;
    switch (swizzle) {
        case SwizzleMode::None:       cu_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
        case SwizzleMode::Swizzle32B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_32B; break;
        case SwizzleMode::Swizzle64B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_64B; break;
        case SwizzleMode::Swizzle128B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
        default: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
    }

    // Global dimensions (in elements)
    uint64_t global_dims[2] = {dim0, dim1};

    // Global strides (in bytes) - stride of each dimension
    // Note: stride[0] is always sizeof(element), stride[1] is stride between rows
    uint64_t global_strides[1] = {stride1 * sizeof(__nv_bfloat16)};  // Only need N-1 strides

    // Box dimensions (tile size in elements)
    uint32_t box_dims[2] = {tile0, tile1};

    // Element strides within box (usually 1)
    uint32_t element_strides[2] = {1, 1};

    // Calculate tile size in bytes
    desc.tile_size_bytes = tile0 * tile1 * sizeof(__nv_bfloat16);

    // Create the tensor map
    CUresult result = cuTensorMapEncodeTiled(
        &desc.tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,                                      // Rank (2D)
        base_ptr,
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu_swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    return result;
}

/**
 * Create a 3D TMA descriptor (for batched attention).
 */
inline CUresult create_tma_descriptor_3d_bf16(
    TmaDescriptor& desc,
    void* base_ptr,
    uint64_t dim0,      // head_dim
    uint64_t dim1,      // seq_len
    uint64_t dim2,      // num_heads
    uint64_t stride1,   // stride between sequence positions
    uint64_t stride2,   // stride between heads
    uint32_t tile0,     // tile for head_dim
    uint32_t tile1,     // tile for seq_len
    SwizzleMode swizzle = SwizzleMode::Swizzle128B
) {
    CUtensorMapSwizzle cu_swizzle;
    switch (swizzle) {
        case SwizzleMode::None:       cu_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
        case SwizzleMode::Swizzle32B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_32B; break;
        case SwizzleMode::Swizzle64B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_64B; break;
        case SwizzleMode::Swizzle128B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
        default: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
    }

    uint64_t global_dims[3] = {dim0, dim1, dim2};
    uint64_t global_strides[2] = {
        stride1 * sizeof(__nv_bfloat16),
        stride2 * sizeof(__nv_bfloat16)
    };
    uint32_t box_dims[3] = {tile0, tile1, 1};  // Load one head at a time
    uint32_t element_strides[3] = {1, 1, 1};

    desc.tile_size_bytes = tile0 * tile1 * sizeof(__nv_bfloat16);

    return cuTensorMapEncodeTiled(
        &desc.tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        base_ptr,
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu_swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

/**
 * Create a 2D TMA descriptor for FP16 tensor.
 */
inline CUresult create_tma_descriptor_2d_fp16(
    TmaDescriptor& desc,
    void* base_ptr,
    uint64_t dim0,
    uint64_t dim1,
    uint64_t stride0,
    uint64_t stride1,
    uint32_t tile0,
    uint32_t tile1,
    SwizzleMode swizzle = SwizzleMode::Swizzle128B
) {
    CUtensorMapSwizzle cu_swizzle;
    switch (swizzle) {
        case SwizzleMode::None:       cu_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
        case SwizzleMode::Swizzle32B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_32B; break;
        case SwizzleMode::Swizzle64B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_64B; break;
        case SwizzleMode::Swizzle128B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
        default: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
    }

    uint64_t global_dims[2] = {dim0, dim1};
    uint64_t global_strides[1] = {stride1 * sizeof(__half)};
    uint32_t box_dims[2] = {tile0, tile1};
    uint32_t element_strides[2] = {1, 1};

    desc.tile_size_bytes = tile0 * tile1 * sizeof(__half);

    return cuTensorMapEncodeTiled(
        &desc.tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        base_ptr,
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu_swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

/**
 * Create a 2D TMA descriptor for FP32 tensor.
 */
inline CUresult create_tma_descriptor_2d_f32(
    TmaDescriptor& desc,
    void* base_ptr,
    uint64_t dim0,
    uint64_t dim1,
    uint64_t stride0,
    uint64_t stride1,
    uint32_t tile0,
    uint32_t tile1,
    SwizzleMode swizzle = SwizzleMode::Swizzle128B
) {
    CUtensorMapSwizzle cu_swizzle;
    switch (swizzle) {
        case SwizzleMode::None:       cu_swizzle = CU_TENSOR_MAP_SWIZZLE_NONE; break;
        case SwizzleMode::Swizzle32B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_32B; break;
        case SwizzleMode::Swizzle64B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_64B; break;
        case SwizzleMode::Swizzle128B: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
        default: cu_swizzle = CU_TENSOR_MAP_SWIZZLE_128B; break;
    }

    uint64_t global_dims[2] = {dim0, dim1};
    uint64_t global_strides[1] = {stride1 * sizeof(float)};
    uint32_t box_dims[2] = {tile0, tile1};
    uint32_t element_strides[2] = {1, 1};

    desc.tile_size_bytes = tile0 * tile1 * sizeof(float);

    return cuTensorMapEncodeTiled(
        &desc.tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        base_ptr,
        global_dims,
        global_strides,
        box_dims,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu_swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

// =============================================================================
// Fence Operations
// =============================================================================

/**
 * Fence to ensure shared memory writes are visible to TMA.
 */
__device__ __forceinline__
void fence_proxy_async_shared() {
#if PYGPUKIT_TMA_ENABLED
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

/**
 * Commit group for async operations.
 */
__device__ __forceinline__
void cp_async_bulk_commit_group() {
#if PYGPUKIT_TMA_ENABLED
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#endif
}

/**
 * Wait for all async bulk operations to complete.
 */
__device__ __forceinline__
void cp_async_bulk_wait_group_read() {
#if PYGPUKIT_TMA_ENABLED
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
#endif
}

}  // namespace tma
}  // namespace ops
}  // namespace pygpukit
