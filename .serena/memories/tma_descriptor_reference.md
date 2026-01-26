# TMA (Tensor Memory Accelerator) Reference

## Overview

TMA is a hardware unit available on SM90+ (Hopper, Blackwell) that enables efficient bulk tensor copies between global and shared memory.

## Key Components

### CUtensorMap / TMA Descriptor

Host-side tensor description that encodes:
- Data type and dimensions
- Strides and tile sizes
- Swizzle mode for bank-conflict-free access

```cpp
// TMA Descriptor creation (host side)
CUtensorMap tensor_map;
cuTensorMapEncodeTiled(
    &tensor_map,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    2,                          // 2D tensor
    base_ptr,
    {HEAD_DIM, seq_len},        // Global dimensions
    {HEAD_DIM, 1},              // Global strides
    {TILE_KV, HEAD_DIM},        // Tile dimensions
    CU_TENSOR_MAP_SWIZZLE_128B  // Bank-conflict-free swizzle
);
```

### PTX Instructions

```cpp
// TMA load from global to shared
ptx::cp_async_bulk_tensor(
  ptx::space_shared, ptx::space_global,
  &smem_buffer, &tensor_map, tensor_coords,
  cuda::device::barrier_native_handle(bar));

// TMA store from shared to global  
ptx::cp_async_bulk_tensor(
  ptx::space_global, ptx::space_shared,
  &tensor_map, tensor_coords, &smem_buffer);
ptx::cp_async_bulk_commit_group();
```

### mbarrier (Async Barrier)

```cpp
// Initialize barrier
cuda::ptx::mbarrier_init(&bar, thread_count);

// Arrive with expected transaction count
uint64_t token = cuda::ptx::mbarrier_arrive_expect_tx(
    cuda::ptx::sem_release, cuda::ptx::scope_cluster,
    cuda::ptx::space_shared, &bar, tx_count, 0);

// Wait for completion
while (!cuda::ptx::mbarrier_try_wait(&bar, token)) {}
```

## Swizzle Modes

| Mode | Alignment | Use Case |
|------|-----------|----------|
| `CU_TENSOR_MAP_SWIZZLE_NONE` | - | Simple access |
| `CU_TENSOR_MAP_SWIZZLE_32B` | 256B | Small tiles |
| `CU_TENSOR_MAP_SWIZZLE_64B` | 512B | Medium tiles |
| `CU_TENSOR_MAP_SWIZZLE_128B` | 1024B | Large tiles, bank-conflict-free |

## Warp Specialization Model

Flash Attention 3 uses producer/consumer warp specialization:

- **Producer Warps (4)**: Issue TMA loads asynchronously
- **Consumer Warps (8)**: Compute MMA operations

```cpp
if (warp_id < NUM_PRODUCER_WARPS) {
    // Producer: issue TMA loads
    for (int stage = 0; stage < STAGES; ++stage) {
        cp_async_bulk_tensor(...);
        mbarrier_arrive(...);
    }
} else {
    // Consumer: compute MMA
    for (int iter = 0; iter < num_iters; ++iter) {
        mbarrier_wait(...);
        mma_sync(...);
    }
}
```

## CUTLASS Reference Files

```
third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp     # TMA copy operations
third_party/cutlass/include/cute/atom/copy_traits_sm90.hpp  # Copy traits
third_party/cutlass/include/cutlass/arch/memory_sm90.hpp    # Memory utilities
```

## PyGPUkit Existing Implementations

### Current State
- **No direct TMA wrapper exists** - TMA is used only through CUTLASS CollectiveBuilder API
- `native/ops/matmul/common/aligned_copy_sm120.cuh` - Only ldmatrix/shared memory utilities, NOT TMA
- SM90/SM100/SM120 GEMM kernels use CUTLASS's internal TMA abstraction

### CUTLASS TMA Usage Pattern (gemm/bf16_bf16/sm90/bf16_cutlass.cuh)
```cpp
// CollectiveBuilder automatically uses TMA for SM90+
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<...>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### For Custom TMA (e.g., FA3)
Need to implement our own TMA utilities:
1. Host-side `CUtensorMap` creation wrapper
2. Device-side async copy operations
3. Barrier management (mbarrier)

## CUTLASS TMA Reference Files

```
third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp   # TMA copy operations
third_party/cutlass/include/cute/arch/copy_sm90_desc.hpp  # Barrier utilities
third_party/cutlass/include/cute/arch/copy_sm100_tma.hpp  # SM100-specific TMA
```

### Key CUTLASS TMA Structures

```cpp
// From copy_sm90_tma.hpp
struct SM90_TMA_LOAD_2D {
  static void copy(
    void const* desc_ptr,     // CUtensorMap pointer
    uint64_t* mbar_ptr,       // Shared memory barrier
    uint64_t cache_hint,
    void* smem_ptr,
    int32_t crd0, int32_t crd1  // Tensor coordinates
  );
};

// PTX instruction used (SM120 variant)
// cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint
```

### Barrier Utilities (copy_sm90_desc.hpp)
```cpp
void initialize_barrier(uint64_t& smem_barrier, int thread_count);
void set_barrier_transaction_bytes(uint64_t& smem_barrier, uint32_t bytes);
void wait_barrier(uint64_t& smem_barrier, int phase_bit);
```

## Required Headers

```cpp
#include <cuda.h>                    // CUtensorMap, cuTensorMapEncode*
#include <cuda/barrier>              // cuda::barrier
#include <cuda/ptx>                  // PTX intrinsics
```

## Architecture Requirements

| Feature | SM Version |
|---------|------------|
| TMA Basic | SM90+ (Hopper) |
| TMA with tcgen05 | SM100 (Blackwell DC) |
| TMA with mma.sync | SM120 (Blackwell GeForce) |
