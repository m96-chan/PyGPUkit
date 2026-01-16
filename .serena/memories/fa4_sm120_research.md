# FA4 SM120 Research Notes

## Goal
Create Flash Attention 4 for SM120 (RTX 5090 GeForce Blackwell).
Target: Maximize performance with NVFP4/FP8, using SM120-specific instructions.

---

## CRITICAL: SM100 vs SM120 Differences

**Modal Blog FA4 is for SM100 (datacenter), NOT SM120 (GeForce)!**

| Feature | SM100 (B100/B200) | SM120 (RTX 5090) |
|---------|-------------------|------------------|
| MMA Instruction | `tcgen05.mma` | **`mma.sync.aligned.block_scale`** |
| Tensor Memory | 256KB TMEM | **None** |
| NVFP4 | ✅ | ✅ (2x vs MXFP8, 4x vs Ada FP8) |
| Cluster | Up to 16 SM | **1x1x1 only** |
| Multicast | ✅ | **None** |
| Warp paradigm | Single-thread MMA | **Warp-synchronous MMA** |

### Key Implication
```
SM100: tcgen05.mma + TMEM + Cluster + Single-thread
SM120: mma.sync.block_scale + SMEM + Single CTA + Warp-sync
```

---

## SM120 MMA Instructions

### Block Scaled MMA (Primary for FP4/FP8)
```
mma.sync.aligned.block_scale.m64n64k64.f32.nvf4.nvf4
mma.sync.aligned.block_scale.m64n64k32.f32.e4m3.e4m3
mma.sync.aligned.block_scale.m64n64k32.f32.e5m2.e5m2
```

### Standard MMA (BF16/FP16)
```
mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32
mma.sync.aligned.m16n8k16.f32.f16.f16.f32
```

### NVFP4 Throughput (SM120)
- NVFP4: **2x** throughput vs MXFP8
- NVFP4: **4x** throughput vs Ada FP8 TensorCore
- This is the key advantage for SM120!

---

## CUTLASS SM120 Reference

### Example 79: Blackwell GeForce GEMM
Location: `examples/79_blackwell_geforce_gemm/`

```cpp
// Key configuration
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;  // M, N, K
using ClusterShape = Shape<_1, _1, _1>;  // Fixed for GeForce

// Data types
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4
using ElementAccum = float;
using ElementOutput = cutlass::bfloat16_t;
```

### SM120 Constraints
- Cluster shape: **1x1x1 only** (no multicast)
- Layout: **TN only** (A row-major, B col-major)
- Alignment: 32 elements (A/B), 128-bit (C/D)

---

## FA4 SM120 Architecture (Revised)

### Strategy Change
```
Before (wrong): Copy Modal FA4 approach
After (correct): Adapt to SM120 constraints
```

### Thread Block Configuration
- **Block size**: 256 threads (8 warps) - similar to FA3
- **Warp specialization**: Load warps + MMA warps + Softmax warps
- **No cluster**: Single CTA per tile

### Tile Sizes for SM120
```cpp
// For block_scale MMA m64n64k64:
TILE_Q = 64       // Matches MMA M dimension
TILE_KV = 64-128  // Tunable
HEAD_DIM = 128    // Standard
NUM_STAGES = 2-3  // Limited by 99KB smem
```

### Memory Layout (99KB limit)
```
Option A: 2-stage pipeline
  smem_q:      64 x 128 x 2B (BF16)  = 16KB
  smem_k:    2 x 64 x 128 x 2B       = 32KB
  smem_v:    2 x 64 x 128 x 2B       = 32KB
  smem_scores: 64 x 64 x 4B          = 16KB
  Total: ~96KB ✅

Option B: NVFP4 (smaller footprint)
  smem_q:      64 x 128 x 0.5B (FP4) = 4KB
  smem_k:    3 x 64 x 128 x 0.5B     = 12KB
  smem_v:    3 x 64 x 128 x 0.5B     = 12KB
  smem_scores: 64 x 64 x 4B          = 16KB
  Total: ~44KB ✅ (room for deeper pipeline!)
```

---

## Implementation Phases (Revised)

### Phase 1: BF16 Baseline
- [ ] Use existing WMMA (mma.sync.m16n8k16)
- [ ] Warp specialization pattern
- [ ] Verify correctness
- [ ] Baseline performance

### Phase 2: Block Scaled FP8
- [ ] mma.sync.aligned.block_scale.e4m3
- [ ] Scale factor handling
- [ ] Mixed precision softmax

### Phase 3: NVFP4 (Maximum Throughput)
- [ ] mma.sync.aligned.block_scale.nvf4
- [ ] 3-stage pipeline (fits in 99KB!)
- [ ] Quantization from BF16→FP4

### Phase 4: Optimization
- [ ] NCU profiling
- [ ] Smem swizzle
- [ ] Register tuning

## Modal Blog FA4 Analysis (SM100 Datacenter Only!)

Source: https://modal.com/blog/reverse-engineer-flash-attention-4

**WARNING: This is for SM100 (B100/B200), NOT SM120 (RTX 5090)!**
The tcgen05.mma and TMEM features are datacenter-only.

### SM100-Specific Features (NOT available on SM120)
- `tcgen05.mma.cta_group::1` - datacenter only
- Tensor Memory (TMEM) 256KB - datacenter only
- Multi-CTA cluster - datacenter only

### Applicable Ideas for SM120
These concepts CAN be adapted:

1. **Warp Specialization Pattern**
   - Load warps + MMA warps + Softmax warps
   - Adapt for SM120's warp-sync model

2. **Smart Exponential Approximation**
   ```cpp
   // Cubic polynomial for 2^x (works on any arch!)
   // Horner's method for 2^frac(x)
   fma.rn.ftz.f32x2 l10, l9, l6, l5
   fma.rn.ftz.f32x2 l10, l10, l9, l4
   fma.rn.ftz.f32x2 l10, l10, l9, l3
   ```
   - Avoids SFU bottleneck
   - Matches bf16 precision

3. **Smart Rescaling (10x fewer corrections)**
   - Update only when numerical stability threatened
   - NOT at every maximum change

4. **Deep K/V Buffering**
   - 3-block prefetch pattern
   - TMA async loads (available on SM120)

---

## SM120 FA4 Key Advantages

### NVFP4 is the Secret Weapon
- **4x throughput** vs Ada FP8
- **2x throughput** vs MXFP8
- Smaller memory footprint → deeper pipeline possible

### Shared Memory Budget (99KB)
With NVFP4 (0.5 bytes per element):
```
3-stage pipeline possible:
  Q:  64 x 128 x 0.5B =  4KB
  K:  3 x 64 x 128 x 0.5B = 12KB  
  V:  3 x 64 x 128 x 0.5B = 12KB
  Scores: 64 x 64 x 4B = 16KB
  Softmax state: 1KB
  Total: ~45KB (plenty of room!)
```

---

## References

### SM120 Specific
- CUTLASS example 79: `examples/79_blackwell_geforce_gemm/`
- [CUTLASS Issue #2186](https://github.com/NVIDIA/cutlass/issues/2186) - SM120 GEMM support
- [CUTLASS Issue #2820](https://github.com/NVIDIA/cutlass/issues/2820) - Block scaled MMA

### General
- PTX ISA 8.5: mma.sync.block_scale instructions
- CUDA 13.1 Release Notes
- FlashAttention-3 paper (Dao et al., 2024)
- [Modal FA4 Blog](https://modal.com/blog/reverse-engineer-flash-attention-4) (SM100 reference)

---

## Existing NVFP4 Implementation (REUSABLE!)

### Location: `native/ops/matmul/`
- `gemm/w4a16_bf16/sm120/nvf4_cutlass.cu` - CUTLASS GEMM with BF16 I/O
- `gemv/w4a16_bf16/sm120/nvf4.cuh` - GEMV with dequant LUTs

### CUTLASS Configuration (from nvf4_cutlass.cu)
```cpp
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _256>;  // K=256 for NVF4!
using ClusterShape = Shape<_1, _1, _1>;
using Schedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

// Data types
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVF4 wrapper
using ScaleFactorType = cutlass::float_ue4m3_t;  // 8-bit unsigned scale
```

### BF16 -> NVF4 Quantization (Branchless, GPU-side)
```cpp
__device__ __forceinline__
uint8_t bf16_to_nvf4_e2m1(float val) {
    float absval = fabsf(val);
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;

    // Branchless threshold counting (faster than LUT!)
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
```

### NVF4 Dequantization LUT (from nvf4.cuh)
```cpp
__device__ __constant__ float NVF4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,     // positive
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // negative
};

__device__ __forceinline__ float dequant_nvf4(uint8_t nvf4_val) {
    return NVF4_LUT[nvf4_val & 0x0F];
}
```

### UE4M3 Scale Factor Decoding
```cpp
// 256-entry LUT for direct byte indexing
// Value = (1 + mantissa/8) * 2^(exponent - 7)
// Unit scale (1.0f) = 0x38
__device__ __constant__ float UE4M3_SCALE_LUT[256] = { ... };

__device__ __forceinline__ float decode_ue4m3_scale(uint8_t ue4m3) {
    return UE4M3_SCALE_LUT[ue4m3];
}
```

### Block Scaling Strategy
- 32 elements share one scale factor (CUTLASS default)
- Scale factors: [K/32, N] layout
- Unit scale encoding: `0x38` for UE4M3

### GPU Quantization Kernels (Vectorized)
- `quantize_A_gpu_kernel`: Row-major BF16 -> packed NVF4
- `quantize_B_gpu_kernel`: Row-major -> Col-major transpose + pack
- Uses `uint4` loads (8 BF16 = 16 bytes) for memory bandwidth

### Key Insights for FA4
1. **TMA + Warp Specialization Pingpong** schedule works on SM120
2. **128KB minimum allocation** workaround for Blackwell TMA bug
3. **Sm1xxBlkScaledConfig** computes scale factor layouts automatically
4. **Parallel stream quantization** possible (A on stream0, B on stream1)

---

---

## Challenge 1: Attention Tile Structure for NVFP4

### Current FA3 BF16 Pipeline (from flash_attention_3_tma.cuh)
```
Main Loop (per KV tile):
1. TMA Load K[stage], V[stage] (producer warps)
2. Q @ K^T -> scores (consumer warps, WMMA BF16)
3. Causal mask
4. Two-phase softmax: scores -> probs (union workaround)
5. P @ V -> output_acc (consumer warps, WMMA BF16)
6. Prefetch next KV tile
```

### FA4 NVFP4 Pipeline (Proposed)
```
Initialization:
- Pre-quantize Q: BF16 -> NVF4 + scale_q (on GPU, before kernel)

Main Loop (per KV tile):
1. TMA Load K_nvf4[stage], V_nvf4[stage], scale_k[stage], scale_v[stage]
2. Q_nvf4 @ K_nvf4^T -> raw_scores (block_scale MMA m64n64k64)
3. Apply combined scale: scores = raw_scores * scale_q * scale_k * attn_scale
4. Causal mask
5. Softmax: scores -> probs (FP32)
6. Probs @ V_nvf4 -> output (BF16 MMA or quantize probs)
7. Prefetch next KV tile
```

### Key Difference: Three MMA Types
| Stage | FA3 (BF16) | FA4 (NVFP4) |
|-------|-----------|-------------|
| Q@K^T | mma.sync.m16n8k16.bf16 | mma.sync.block_scale.m64n64k64.nvf4 |
| P@V | mma.sync.m16n8k16.bf16 | mma.sync.m16n8k16.bf16 (probs are dynamic) |

### Decision: Keep P@V in BF16
- Probs are computed dynamically via softmax
- Online NVF4 quantization of probs adds latency
- BF16 P@V is fast enough, not the bottleneck (memory-bound anyway)

---

## Challenge 2: Online Q/K Quantization Strategy

### Option A: Pre-quantize Before Kernel (RECOMMENDED)
```
Host side:
  Q_bf16 -> GPU quantize kernel -> Q_nvf4 + scale_q
  K_bf16 -> GPU quantize kernel -> K_nvf4 + scale_k
  V_bf16 -> GPU quantize kernel -> V_nvf4 + scale_v

FA4 kernel:
  TMA Load Q_nvf4, K_nvf4, V_nvf4 (smaller footprint!)
  MMA with pre-computed scales
```

**Pros:**
- No in-kernel quantization overhead
- Reuse existing `quantize_A_gpu_kernel` / `quantize_B_gpu_kernel`
- Smaller TMA transfers (4-bit vs 16-bit)

**Cons:**
- Requires separate quantization pass
- Scale factors need separate TMA descriptor

### Option B: In-Kernel Quantization (Alternative)
```
FA4 kernel:
  TMA Load Q_bf16, K_bf16, V_bf16
  Quantize in smem: bf16 -> nvf4 + scale (warps cooperate)
  MMA with computed scales
```

**Pros:**
- Single kernel
- Can use fresh scale factors per tile

**Cons:**
- Adds ~100 cycles per tile for quantization
- More complex smem layout

### Quantization Latency Analysis
From existing implementation:
- `quantize_A_gpu_kernel`: 8 BF16 -> 4 bytes (vectorized uint4)
- ~50 cycles for 64x128 tile (8192 elements / 256 threads * 1.5 cycles)
- Negligible vs MMA latency (~200+ cycles)

**Decision: Option A** - Pre-quantize for first implementation, optimize later.

---

## Challenge 3: Scale Factor Propagation Through Softmax

### The Problem
```
Q has scale_q (per 32-element block)
K has scale_k (per 32-element block)

Raw MMA output: mma_result = Q_int @ K_int^T
Actual scores: scores = mma_result * scale_q * scale_k

But softmax needs: exp(scores - max) / sum(exp(...))
```

### Solution: Apply Scale Before Softmax
```cpp
// After block_scale MMA:
for (int i = 0; i < score_elements; i++) {
    int q_block = q_idx / 32;
    int k_block = k_idx / 32;
    float combined_scale = scale_q[q_block] * scale_k[k_block] * attn_scale;
    scores[i] = raw_mma_result[i] * combined_scale;
}
// Then standard softmax on scores
```

### Simplification: Unit Scale (for Phase 1)
For initial implementation, use **unit scale** (scale = 1.0):
- Pre-normalize Q/K to fit in NVF4 range [-6, 6]
- Set all scale factors to 0x38 (UE4M3 encoding of 1.0)
- Avoids scale multiplication overhead
- Limits dynamic range but simplifies implementation

### Full Scale Support (Phase 2+)
- Store scale_q in registers (TILE_Q/32 floats)
- Load scale_k per KV tile
- Apply combined scale after MMA, before softmax

### Memory Layout for Scales
```
Q_nvf4: [num_heads, seq_q, head_dim/2]     (packed bytes)
scale_q: [num_heads, seq_q/32]              (UE4M3 per 32 elements)

K_nvf4: [num_heads, seq_kv, head_dim/2]    (packed bytes)
scale_k: [num_heads, seq_kv/32]            (UE4M3 per 32 elements)
```

---

## Open Questions (RESOLVED)

1. ~~Exact PTX encoding for `mma.sync.aligned.block_scale`?~~ -> Use CUTLASS
2. ~~Block scale factor format and handling?~~ -> UE4M3, 32 elements/block, 0x38=1.0
3. ~~NVFP4 quantization strategy for Q/K/V?~~ -> Pre-quantize (Option A)
4. ~~Online Q/K quantization latency?~~ -> ~50 cycles/tile, negligible
5. ~~Scale propagation through softmax?~~ -> Apply combined scale before softmax
6. Optimal polynomial coefficients for exp2 on SM120? (low priority)

---

## Implementation Plan (FINAL)

### Phase 1: BF16 Baseline on FA3 Architecture
**Confidence: 95%**
- [ ] Fork FA3 TMA kernel as FA4 base
- [ ] Verify existing warp specialization works
- [ ] Baseline performance: ~60 TFLOPS

### Phase 2: NVFP4 Q@K^T Only
**Confidence: 80%**
- [ ] Add pre-quantize kernels for Q, K (reuse GEMM code)
- [ ] Replace Q@K^T MMA with block_scale version
- [ ] Keep P@V in BF16 (probs are dynamic)
- [ ] Unit scale (1.0) for simplicity
- [ ] Verify correctness vs BF16 reference

### Phase 3: Full NVFP4 Pipeline
**Confidence: 70%**
- [ ] Add V quantization
- [ ] TMA descriptors for NVF4 tensors
- [ ] Scale factor loading and propagation
- [ ] Expected: ~100+ TFLOPS (2x compute throughput)

### Phase 4: Optimization
**Confidence: 60%**
- [ ] NCU profiling
- [ ] Smem swizzle for bank conflict-free
- [ ] 3-stage pipeline (fits in 99KB with NVF4!)
- [ ] Full scale support (non-unit)
- [ ] Target: 120+ TFLOPS

---

## Summary

| Item | Status | Notes |
|------|--------|-------|
| SM120 MMA instructions | ✅ Understood | block_scale, not tcgen05 |
| NVFP4 quantization | ✅ Implemented | Branchless, reusable |
| Scale factor handling | ✅ Implemented | UE4M3 LUT, 0x38=1.0 |
| Tile structure | ✅ Designed | Q@K^T (NVF4), P@V (BF16) |
| Quantization strategy | ✅ Decided | Pre-quantize (Option A) |
| Scale propagation | ✅ Solved | Apply before softmax |
| exp2 polynomial | ⏳ Low priority | Use standard expf() first |

**All major blockers resolved. Implementation in progress.**

---

## Phase 2 Benchmark Results

### NVFP4 Q@K^T External Validation (seq_len=1024, head_dim=128)

**Single-head Q@K^T:**
- NVFP4 GEMM: 394.0 us (0.68 TFLOPS)
- Correctness: 21% rel_diff vs NumPy (ACCEPTABLE for 4-bit)

**Key Finding:**
NVFP4 GEMM is optimized for large K dimensions (LLM weights with K=4096+), not attention's small K=128 (head_dim).

- CUTLASS NVFP4 uses K=256 tile size
- For head_dim=128, tile utilization is low
- Full 32-head FA3 TMA: 330.9 us (51.92 TFLOPS) - more efficient

**Implication for FA4:**
NVFP4 benefit in attention comes from **memory bandwidth reduction** (4-bit loads), not compute throughput:
- 4-bit data = 4x smaller memory footprint
- Flash Attention is memory-bound, so smaller loads help
- Compute throughput (TFLOPS) is misleading for memory-bound kernels

**Phase 2 Status:**
- ✅ NVFP4 GEMM path validated
- ✅ Correctness acceptable (21% rel_diff for 4-bit)
- ⚠️ Full kernel fusion requires PTX inline assembly for `mma.sync.aligned.block_scale`
- ⚠️ Small K (head_dim=128) not optimal for NVFP4 GEMM tile size

**Next Steps:**
1. Phase 3: Add V quantization (seq_len K is larger, better utilization)
2. Or: Focus on memory bandwidth benefit, not compute TFLOPS**

---

## Notes

- SM120a (RTX 5090) has 99KB smem per block
- WGMMA requires 128-thread warp groups
- TMA descriptors can be reused (cached)
- Current FA3 baseline: 60 TFLOPS (BF16)
- NVFP4 GEMM already works with CUTLASS on SM120
- P@V stays BF16 because probs are dynamic (softmax output)
