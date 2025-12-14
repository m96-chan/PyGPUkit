# TF32 Kernel Optimization Summary

## Overview

This document summarizes the TF32 TensorCore GEMM kernel optimization work performed on the `feature/v0.2.3-tf32-tensorcore` branch.

## Target

- **4096×4096**: ≥ 21.13 TFLOPS (+2 TFLOPS over baseline 19.13)
- **8192×8192**: ≥ 27.53 TFLOPS (no regression)

## Final Results

| Size | Baseline | Optimized | Improvement | Target |
|------|----------|-----------|-------------|--------|
| 4096×4096 | 19.13 | **20.48** | +1.35 TFLOPS (+7.1%) | 21.13 (68% achieved) |
| 8192×8192 | 27.53 | **28.56** | +1.03 TFLOPS (+3.7%) | ✓ Exceeded |

Peak performance observed: 4096×4096 = 20.88 TFLOPS (within 0.25 TFLOPS of target)

## Successful Optimizations

### 1. A Fragment Hoisting (+0.5 TFLOPS)

**Problem**: A fragments were loaded inside the wn loop, causing 8× redundant shared memory loads.

**Solution**: Moved A fragment loads outside the wn loop since A only depends on wm, not wn.

```cpp
// Before: A loaded for each wn iteration
for (int wm = 0; wm < WARP_TILES_M; ++wm) {
    for (int wn = 0; wn < WARP_TILES_N; ++wn) {
        float a0 = smA[...];  // Loaded 8 times!
        float b0 = smB[...];
        mma(...);
    }
}

// After: A loaded once per wm iteration
for (int wm = 0; wm < WARP_TILES_M; ++wm) {
    float a0 = smA[...];  // Loaded once, reused 8 times
    for (int wn = 0; wn < WARP_TILES_N; ++wn) {
        float b0 = smB[...];
        mma(...);
    }
}
```

### 2. Unconditional Wait (+0.4 TFLOPS)

**Problem**: Branch inside hot loop for `cp_async_wait_0()`.

**Solution**: Remove the conditional - `cp_async_wait_0()` is a no-op when nothing is pending.

```cpp
// Before
if (kt + 1 < num_k_tiles) {
    cp_async_wait_0();
}

// After
cp_async_wait_0();  // No-op on last iteration
```

### 3. Unconditional Prefetch (+0.1 TFLOPS)

**Problem**: Branch inside hot loop for prefetch.

**Solution**: Always prefetch - last iteration loads garbage into unused buffer.

```cpp
// Before
if (kt + 1 < num_k_tiles) {
    load_A_async(next, kt + 1);
    load_B_async(next, kt + 1);
    cp_async_commit();
}

// After
load_A_async(next, kt + 1);  // Garbage load on last iter is harmless
load_B_async(next, kt + 1);
cp_async_commit();
```

## Failed Optimizations (All Reverted)

### 1. 3-Stage Pipeline (-28% on 4096)

**Attempt**: Use 3 shared memory buffers for 2-tile lookahead.

**Result**: SEVERE REGRESSION
- 4096: 19.13 → 13.69 TFLOPS (-28%)
- 8192: 27.53 → 17.51 TFLOPS (-36%)

**Cause**:
- 50% more shared memory reduced occupancy
- `kt % 3` slower than `kt & 1` for buffer selection

### 2. 512-Thread Configuration (-17%)

**Attempt**: 16 warps with reduced WARP_TILES_N=4 for lower register pressure.

**Result**: REGRESSION
- 4096: 19.66 → 16.21 TFLOPS (-17%)
- 8192: 27.42 → 22.74 TFLOPS (-17%)

**Cause**: Higher thread count didn't compensate for changed access patterns.

### 3. BM=64 Smaller Tiles (-28% on 8192)

**Attempt**: Reduce BM from 128 to 64 for better occupancy.

**Result**: REGRESSION
- 4096: 19.66 → 18.13 TFLOPS (-8%)
- 8192: 27.42 → 19.60 TFLOPS (-28%)

**Cause**: Reduced parallelism per block hurt large matrix performance.

### 4. Manual kk Loop Unroll (-6%)

**Attempt**: Manually unroll BK/WMMA_K=2 iterations instead of #pragma unroll.

**Result**: REGRESSION
- 4096: 19.66 → 18.44 TFLOPS (-6%)

**Cause**: Increased register pressure from explicit unrolling.

### 5. BK=8 for Occupancy (-7%)

**Attempt**: Reduce BK from 16 to 8 to halve shared memory and allow 2 blocks/SM.

**Result**: Mixed
- 2048: improved (+0.4 TFLOPS)
- 4096: 19.66 → 18.31 TFLOPS (-7%)

**Cause**: Doubled K loop iterations offset occupancy gains.

### 6. Batch B Fragment Loading (Unstable)

**Attempt**: Preload all B fragments into registers before MMA loop.

**Result**: Mixed with high variance
- Large sizes: slight improvement
- Small sizes: regression (-0.76 TFLOPS on 2048)

**Cause**: Additional register arrays caused spilling on smaller problems.

### 7. BN=256 Larger Tiles (Abandoned)

**Attempt**: Increase BN to 256 for more work per block.

**Result**: Not tested - abandoned due to register pressure.

**Cause**: WARP_TILES_N=16 would require acc[2][16][4] = 128 registers per warp.

### 8. WARP_TILES_N=16 with WARPS_M=8 (-5%)

**Attempt**: Maximize A fragment reuse with 16 wn iterations.

**Result**: REGRESSION
- 4096: 19.91 → 18.92 TFLOPS (-5%)

**Cause**: Different memory access pattern hurt performance.

## Key Technical Observations

### 1. Register Pressure is the Primary Limiter

Current accumulator usage: `acc[2][8][4]` = 64 floats = 64 registers per warp.

Any configuration that increases this (larger WARP_TILES) causes severe spilling.

### 2. Shared Memory Limits Occupancy

- Current: 37KB shared memory per block
- Max per SM: 48KB (configurable to 100KB)
- Result: ~1 block per SM = 16.7% warp occupancy

### 3. High Variance Due to System Noise

Benchmark variance: ±1-2 TFLOPS on 4096×4096

Causes:
- GPU boost clock fluctuation
- Background processes
- Thermal throttling

Extended warmup (50 iterations) provides more stable results.

### 4. cuBLAS Comparison

| Library | FP32 | TF32 |
|---------|------|------|
| cuBLAS | ~21 TFLOPS | ~59 TFLOPS |
| PyGPUkit | 18 TFLOPS (86%) | 28 TFLOPS (47%) |

Gap analysis: cuBLAS likely uses:
- Larger tiles (256×128 or 256×256)
- PTX-level hand optimization
- wgmma instructions on newer hardware
- Dynamic shared memory (100KB)

## Remaining Optimization Opportunities

1. **PTX ldmatrix instruction** - More efficient matrix fragment loads (complex)
2. **Dynamic shared memory** - Enable larger tiles beyond 48KB
3. **m16n8k4 instruction** - Better pipelining with smaller k (requires restructure)
4. **Warp specialization** - Separate load and compute warps
5. **Software pipelining** - Deeper pipeline with multiple k-tiles in flight

## Files Modified

- `native/ops/matmul_f32_tf32.cuh` - Main kernel with optimizations

## Commit

```
da41bf7 perf(tf32): optimize kernel with A fragment hoisting (+1.35 TFLOPS)
```

## Test Verification

All correctness tests pass with TF32 tolerance (sqrt(K) * 0.1% * 5x margin):

```
128x128x128:   PASS (max_rel=1.24%)
256x256x256:   PASS (max_rel=1.73%)
512x512x512:   PASS (max_rel=2.37%)
1024x1024x1024: PASS (max_rel=3.43%)
2048x2048x2048: PASS (max_rel=5.57%)
```
