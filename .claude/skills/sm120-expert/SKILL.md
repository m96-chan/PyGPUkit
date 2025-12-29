---
name: sm120-expert
description: SM120 (Blackwell) CUDA expert. Use for wgmma/mma PTX inline assembly, TMA, narrow precision (FP4/FP6/FP8), and block-scaled GEMM development.
---

# SM120 Blackwell Expert

Expert knowledge for NVIDIA Blackwell (SM120/SM120a) GPU programming.

## Reference Files

```
third_party/cutlass/include/cute/arch/mma_sm120.hpp         # MMA PTX inline asm
third_party/cutlass/include/cute/arch/mma_sm120_sparse.hpp  # Sparse MMA
third_party/cutlass/include/cute/atom/mma_traits_sm120.hpp  # MMA traits
```

## SM120 MMA Instruction Format

### Basic F8F6F4 MMA (m16n8k32)

```cpp
asm volatile(
  "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
  "{%0,  %1,  %2,  %3},"   // D registers (output, f32 x4)
  "{%4,  %5,  %6,  %7},"   // A registers (input, u32 x4)
  "{%8,  %9},"             // B registers (input, u32 x2)
  "{%10, %11, %12, %13};\n" // C registers (accumulator, f32 x4)
  : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
  :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
     "r"(b0),  "r"(b1),
     "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
```

### Block-Scaled MXF8F6F4 MMA

```cpp
asm volatile(
  "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
  "{%0,  %1,  %2,  %3},"   // D registers
  "{%4,  %5,  %6,  %7},"   // A registers
  "{%8,  %9},"             // B registers
  "{%10, %11, %12, %13},"  // C registers
  "{%14},"                 // Scale factor A (ue8m0)
  "{%15, %16},"            // Block/Thread ID A
  "{%17},"                 // Scale factor B (ue8m0)
  "{%18, %19};\n"          // Block/Thread ID B
  :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
  :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
      "r"(b0),   "r"(b1),
      "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
      "r"(uint32_t(sfa)), "h"(bidA), "h"(tidA),
      "r"(uint32_t(sfb)), "h"(bidB), "h"(tidB));
```

## Supported Data Types

| Type | Notation | Bits | Description |
|------|----------|------|-------------|
| `e2m1` | NVF4 | 4 | NVIDIA FP4 (2-bit exp, 1-bit mantissa) |
| `e3m2` | MXF6 | 6 | MX FP6 (3-bit exp, 2-bit mantissa) |
| `e2m3` | MXF6 | 6 | MX FP6 (2-bit exp, 3-bit mantissa) |
| `e4m3` | FP8 | 8 | FP8 E4M3 (4-bit exp, 3-bit mantissa) |
| `e5m2` | FP8 | 8 | FP8 E5M2 (5-bit exp, 2-bit mantissa) |
| `ue8m0` | Scale | 8 | Block scale factor (unsigned exp only) |

## MMA Shape & Register Layout

### m16n8k32 (TN layout)

| Fragment | Registers | Type | Count |
|----------|-----------|------|-------|
| D (output) | float[4] | `"=f"` | 4 |
| A (input) | uint32_t[4] | `"r"` | 4 |
| B (input) | uint32_t[2] | `"r"` | 2 |
| C (accum) | float[4] | `"f"` | 4 |

### Block-Scaled Additional

| Fragment | Registers | Type | Description |
|----------|-----------|------|-------------|
| SFA | uint8_t | `"r"` (cast to u32) | Scale factor A |
| SFB | uint8_t | `"r"` (cast to u32) | Scale factor B |
| bidA/tidA | uint16_t | `"h"` | Block/Thread ID |

## Compile Flags

```cpp
// Required macro
#define CUTE_ARCH_F8F6F4_MMA_ENABLED     // Basic F8F6F4
#define CUTE_ARCH_MXF8F6F4_MMA_ENABLED   // Block-scaled MX
```

```cmake
# CMake settings
set(CMAKE_CUDA_ARCHITECTURES "120a")
# or
-gencode arch=compute_120a,code=sm_120a
```

## Valid Tile Shapes (CUTLASS)

| MMA Tile | Layout | Dispatch Policy |
|----------|--------|-----------------|
| 128x128x128 | TN | Pingpong / Cooperative |
| 256x128x128 | TN | Cooperative |
| 128x128x256 | TN | Pingpong / Cooperative |

## TMA (Tensor Memory Accelerator)

### cp.async.bulk.tensor

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

### Swizzle Modes

| Mode | Alignment | Use Case |
|------|-----------|----------|
| `CU_TENSOR_MAP_SWIZZLE_NONE` | - | Simple access |
| `CU_TENSOR_MAP_SWIZZLE_32B` | 256B | Small tiles |
| `CU_TENSOR_MAP_SWIZZLE_64B` | 512B | Medium tiles |
| `CU_TENSOR_MAP_SWIZZLE_128B` | 1024B | Large tiles, bank-conflict-free |

## wgmma (Warpgroup MMA)

SM120 uses wgmma for larger tile sizes (inherited from SM90 Hopper):

```cpp
// wgmma fence
asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

// wgmma commit group
asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");

// wgmma wait
asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
```

## File Locations in PyGPUkit

| Path | Description |
|------|-------------|
| `native/ops/matmul/gemm/fp8/bf16/sm120/` | FP8->BF16 GEMM |
| `native/ops/matmul/gemm/nvf4/bf16/sm120/` | NVF4->BF16 GEMM |
| `native/ops/matmul/gemm/fp8/fp8/sm120/` | FP8->FP8 GEMM |
| `native/ops/matmul/gemv/bf16/bf16/sm120/` | BF16 GEMV |
| `native/ops/matmul/common/aligned_copy_sm120.cuh` | TMA utilities |

## Usage

When asked about SM120/Blackwell:
1. Reference mma_sm120.hpp for PTX inline assembly
2. Check supported data type combinations
3. Verify tile shapes match dispatch policy
4. Use TMA for efficient global<->shared transfers
5. Apply 128B swizzle for bank-conflict-free access

## CUDA Version Requirement

- **CUDA 13.1+** required for SM120a (RTX 5090)
- **PTX ISA 8.7+** for all F8F6F4 instructions

---

## Context7 Reference (CUTLASS/CUDA Docs)

### CUTLASS SM120 Unit Tests

```cpp
// Tensor Core GEMM
#include "test/unit/gemm/device/sm120_tensorop_gemm/sm120_tensorop_gemm.cu"

// Block-Scaled GEMM (MXF4/NVF4)
#include "test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf4_mxf4_f32_f32.cu"
#include "test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32.cu"
#include "test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf6_mxf8_f32_f32.cu"
```

### MLA (Multi-Head Latent Attention) for Blackwell

```cpp
#include <cutlass/gemm/collective/fmha_inference.hpp>
#include <cutlass/gemm/collective/mla_kernel.hpp>

// Supports: fp16, bf16, fp8
// Uses 2x Blackwell tensor cores for large latent head dimensions
// TMA + cp.async loading, variable sequence length
```

### TMA with 128B Swizzle (Bank-Conflict-Free)

```cuda
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   // 128-byte swizzle requires 1024-byte alignment
   __shared__ alignas(1024) int4 smem_buffer[8][8];
   __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ barrier bar;

   if (threadIdx.x == 0) { init(&bar, blockDim.x); }
   __syncthreads();

   barrier::arrival_token token;
   if (is_elected()) {
     int32_t tensor_coords[2] = { 0, 0 };
     ptx::cp_async_bulk_tensor(
       ptx::space_shared, ptx::space_global,
       &smem_buffer, &tensor_map, tensor_coords,
       cuda::device::barrier_native_handle(bar));
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else {
     token = bar.arrive();
   }
   bar.wait(std::move(token));

   // XOR swizzle for bank-conflict-free transpose
   for(int j = threadIdx.x; j < 8; j += blockDim.x) {
      for(int i = 0; i < 8; ++i) {
         const int swiz_j = (i % 8) ^ j;
         const int swiz_i_tr = (j % 8) ^ i;
         smem_buffer_tr[j][swiz_i_tr] = smem_buffer[i][swiz_j];
      }
   }

   // Fence before TMA store
   ptx::fence_proxy_async(ptx::space_shared);
   __syncthreads();

   if (is_elected()) {
       ptx::cp_async_bulk_tensor(
         ptx::space_global, ptx::space_shared,
         &tensor_map, tensor_coords, &smem_buffer_tr);
       ptx::cp_async_bulk_commit_group();
   }
}
```

### mbarrier (Async Barrier) PTX

```cpp
#include <cuda/ptx>

// Initialize barrier
cuda::ptx::mbarrier_init(&bar, thread_count);

// Arrive with expected transaction count
uint64_t token = cuda::ptx::mbarrier_arrive_expect_tx(
    cuda::ptx::sem_release, cuda::ptx::scope_cluster,
    cuda::ptx::space_shared, &bar, tx_count, 0);

// Wait for completion
while (!cuda::ptx::mbarrier_try_wait(&bar, token)) {}
```

### fence_proxy_alias (Virtual Aliasing)

```cuda
// Required between multicast (mc) and unicast (uc) access to same memory
cuda::ptx::fence_proxy_alias();

// Example: after multimem reduction, before unicast read
cuda::ptx::multimem_red(cuda::ptx::release_t, cuda::ptx::scope_sys_t,
                        cuda::ptx::op_add_t, counter_mc, n);
cuda::ptx::fence_proxy_alias();
while (expected > atomic_ref(counter_uc).load(cuda::memory_order_acquire));
```

### multimem (Multi-GPU Memory) PTX

```cpp
// Atomic reduction to all replicas (requires SM90+)
cuda::ptx::multimem_red(cuda::ptx::release_t, cuda::ptx::scope_sys_t,
                        cuda::ptx::op_add_t, arrival_counter_mc, n);

// Load-reduce from all replicas
asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
             : "=f"(result) : "l"(partial_mc) : "memory");
```

### MMIO with PTX Inline Assembly

```cpp
// Write to MMIO register (strict memory access preservation)
int value = 13;
asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;"
    : : "l"(mmio_reg), "r"(value) : "memory");

// Read from MMIO register
asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];"
    : "=r"(value) : "l"(mmio_reg) : "memory");
```

## External References

- [CUTLASS Blackwell Functionality](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)
- [CUTLASS SM120 GEMM Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)
- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Programming Guide - TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#tensor-memory-access)
