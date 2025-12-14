# TF32 TensorCore GEMM Design Document

## Overview

PyGPUkit v0.2.3 introduces TF32 TensorCore acceleration for Ampere+ GPUs (SM >= 80).
This document describes the kernel architecture, PTX fragment mappings, and optimization strategies.

**Performance Achieved:**
- RTX 3090 Ti: **27.38 TFLOPS** (8192×8192×8192)
- Correctness: ~3-5% relative error (expected for TF32 precision)

**Reference Implementation:** [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)

---

## TF32 Precision

TF32 (TensorFloat-32) is an Ampere-specific format that:
- Uses 19-bit mantissa (vs FP32's 23-bit)
- Provides ~0.1% relative error per multiply-accumulate
- Accumulates in full FP32 precision
- Requires no code changes for input/output (still uses float32)

```
FP32:  1 sign + 8 exponent + 23 mantissa = 32 bits
TF32:  1 sign + 8 exponent + 10 mantissa = 19 bits (internal)
```

---

## Kernel Architecture

### Tiling Parameters

```cpp
constexpr int BM = 128;      // Block tile M
constexpr int BN = 128;      // Block tile N
constexpr int BK = 16;       // Block tile K

constexpr int WMMA_M = 16;   // MMA instruction M
constexpr int WMMA_N = 8;    // MMA instruction N
constexpr int WMMA_K = 8;    // MMA instruction K

constexpr int WARPS_M = 4;   // Warps in M direction
constexpr int WARPS_N = 2;   // Warps in N direction (8 warps total)

constexpr int WARP_TILES_M = 2;  // Tiles per warp in M
constexpr int WARP_TILES_N = 8;  // Tiles per warp in N
```

### Block Decomposition

```
Block (128×128 output):
┌───────────────────────────────────────────┐
│  Warp 0,1 (row 0)     │  Warp 2,3 (row 1) │
│  32×64 each           │  32×64 each       │
├───────────────────────┼───────────────────┤
│  Warp 4,5 (row 2)     │  Warp 6,7 (row 3) │
│  32×64 each           │  32×64 each       │
└───────────────────────┴───────────────────┘

Each warp computes:
- WARP_TILES_M × WARP_TILES_N = 2 × 8 = 16 MMA operations
- Output size: 32 × 64 elements
```

### Thread Block Configuration

- **Threads per block:** 256 (8 warps × 32 threads)
- **Shared memory:** ~20KB per block
  - A tile: 128 × 20 × 4 bytes = 10KB (with padding)
  - B tile: 16 × 132 × 4 bytes = 8.5KB (with padding)
- **Occupancy:** 2 blocks per SM

---

## PTX mma.sync Instruction

### Instruction Format

```
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    {d0, d1, d2, d3},    // C fragment (output)
    {a0, a1, a2, a3},    // A fragment (input)
    {b0, b1},            // B fragment (input)
    {c0, c1, c2, c3};    // C fragment (accumulator)
```

### Fragment Layouts (Empirically Verified)

**CRITICAL:** PTX `mma.sync` has DIFFERENT layouts than the WMMA API.
These mappings were verified using `dump_c_fragment.cu`.

#### A Fragment (16×8 matrix, row-major)

Each thread (lane 0-31) holds 4 registers:

```cpp
int a_row = lane / 4;      // 0-7
int a_col = lane % 4;      // 0-3

a[0] = A[a_row][a_col]           // rows 0-7,  cols 0-3
a[1] = A[a_row + 8][a_col]       // rows 8-15, cols 0-3
a[2] = A[a_row][a_col + 4]       // rows 0-7,  cols 4-7
a[3] = A[a_row + 8][a_col + 4]   // rows 8-15, cols 4-7
```

#### B Fragment (8×8 matrix, col-major)

Each thread holds 2 registers:

```cpp
int b_row = lane % 4;      // 0-3
int b_col = lane / 4;      // 0-7

b[0] = B[b_row][b_col]           // rows 0-3, cols 0-7
b[1] = B[b_row + 4][b_col]       // rows 4-7, cols 0-7
```

#### C Fragment (16×8 matrix) - KEY DIFFERENCE

Each thread holds 4 registers with **stride-2 column access**:

```cpp
int c_row = lane / 4;           // 0-7
int c_col = (lane % 4) * 2;     // 0, 2, 4, 6 (NOT 0-3!)

c[0] = C[c_row][c_col]           // rows 0-7,  cols 0,2,4,6
c[1] = C[c_row][c_col + 1]       // rows 0-7,  cols 1,3,5,7
c[2] = C[c_row + 8][c_col]       // rows 8-15, cols 0,2,4,6
c[3] = C[c_row + 8][c_col + 1]   // rows 8-15, cols 1,3,5,7
```

### Common Mistakes

1. **C fragment column stride:** PTX uses `(lane%4)*2`, NOT `lane%4`
2. **C fragment pairs:** c[0],c[1] are adjacent columns; c[2],c[3] are +8 rows
3. **WMMA vs PTX size:** PTX m16n8k8 uses half the B/C columns of WMMA 16×16×8

---

## cp.async Pipeline

### Double-Buffering Strategy

The kernel uses cp.async for asynchronous global→shared memory transfers:

```cpp
// Prologue: load first tile
load_A_async(0, 0);
load_B_async(0, 0);
cp_async_commit();
cp_async_wait_0();
__syncthreads();

// Main loop
for (int kt = 0; kt < num_k_tiles; ++kt) {
    int curr = kt & 1;      // Current stage
    int next = curr ^ 1;    // Next stage (OTHER buffer)

    // Prefetch next tile into OTHER buffer
    load_A_async(next, kt + 1);
    load_B_async(next, kt + 1);
    cp_async_commit();

    // Process current tile (compute)
    for (int kk = 0; kk < BK; kk += WMMA_K) {
        // MMA operations using smA[curr] and smB[curr]
    }

    // Wait for prefetch
    cp_async_wait_0();
    __syncthreads();
}
```

### Key Insight

**Always prefetch into the stage you're NOT currently reading from.**

Common bug:
```cpp
// WRONG - overwrites current buffer!
load_async((kt+2) & 1, kt + 2);  // On kt=0, this writes to stage 0!
```

Correct approach:
```cpp
// CORRECT - prefetch into OTHER stage
int next = curr ^ 1;
load_async(next, kt + 1);
```

---

## Shared Memory Layout

### Padding for Bank Conflict Avoidance

```cpp
constexpr int A_PAD = 4;  // Pad A columns
constexpr int B_PAD = 4;  // Pad B columns

__shared__ float smA[2][BM][BK + A_PAD];  // 128 × 20
__shared__ float smB[2][BK][BN + B_PAD];  // 16 × 132
```

### Load Pattern

A tile loading (256 threads loading 128×16 elements):
```cpp
const int a_row = tid / 4;      // 64 rows per iteration
const int a_col = (tid % 4) * 4; // 16-byte (float4) loads

for (int i = 0; i < 2; ++i) {
    int row = a_row + i * 64;
    cp_async_16(&smA[stage][row][a_col], &A[gm * K + gk]);
}
```

---

## Optimization Techniques

### 1. A Fragment Hoisting

A fragments are reused across all N-direction tiles:

```cpp
for (int wm = 0; wm < WARP_TILES_M; ++wm) {
    // Load A fragment ONCE (hoisted outside wn loop)
    float a0 = smA[curr][tile_m + a_row_base][kk + a_col_base];
    float a1 = smA[curr][tile_m + a_row_base + 8][kk + a_col_base];
    float a2 = smA[curr][tile_m + a_row_base][kk + a_col_base + 4];
    float a3 = smA[curr][tile_m + a_row_base + 8][kk + a_col_base + 4];

    for (int wn = 0; wn < WARP_TILES_N; ++wn) {
        // Only B changes per iteration
        float b0 = smB[curr][kk + b_row_base][tile_n + b_col];
        float b1 = smB[curr][kk + b_row_base + 4][tile_n + b_col];
        // MMA instruction
    }
}
```

**Impact:** +1.35 TFLOPS improvement

### 2. Launch Bounds

```cpp
__global__ void __launch_bounds__(256, 2)
sgemm_tf32_ampere_kernel(...)
```

- 256 threads per block
- Target 2 blocks per SM for optimal occupancy

### 3. Vectorized Loads

Using `cp.async.cg` with 16-byte (float4) transfers:

```cpp
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t addr = smem_u32(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(addr), "l"(gmem)
    );
}
```

---

## API Usage

### Python API

```python
import pygpukit as gp

a = gp.from_numpy(a_np)  # float32
b = gp.from_numpy(b_np)  # float32

# Explicit TF32 mode
c = gp.matmul(a, b, use_tf32=True)

# Environment variable control
# Set PYGPUKIT_ALLOW_TF32=1 for default TF32
c = gp.matmul(a, b)  # Uses env var
```

### Device Capabilities

```python
caps = gp.get_device_capabilities()
print(f"TensorCore: {caps.tensorcore}")      # True for SM >= 80
print(f"SM Version: {caps.sm_version}")      # e.g., 86
```

---

## Performance Results

### RTX 3090 Ti (SM 86)

| Matrix Size | FP32 (TFLOPS) | TF32 (TFLOPS) | Speedup |
|-------------|---------------|---------------|---------|
| 2048×2048   | 7.6           | 10.2          | 1.34×   |
| 4096×4096   | 13.2          | 19.5          | 1.48×   |
| 8192×8192   | 18.2          | 27.5          | 1.51×   |

### Comparison with cuBLAS

| Library | FP32 | TF32 |
|---------|------|------|
| cuBLAS  | ~21 TFLOPS | ~59 TFLOPS |
| PyGPUkit | 18 TFLOPS (86%) | 27 TFLOPS (46%) |

---

## File Locations

- **Kernel:** `native/ops/matmul_f32_tf32.cuh`
- **Dispatch:** `native/ops/basic.cu`
- **Python API:** `src/pygpukit/ops/basic.py`
- **Rust types:** `rust/pygpukit-core/src/device.rs`
- **Tests:** `tests/test_tf32_api.py`

---

## Future Work

Continued optimization with [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) as reference:

- [ ] Swizzled shared memory layout
- [ ] Multi-stage pipeline (4+ stages)
- [ ] Hopper TF32x3 mode (3× throughput)
- [ ] Epilogue fusion (bias + activation)
- [ ] Auto-tuner for tile sizes
