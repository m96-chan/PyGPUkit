# FA4 SM120 Implementation Report

**Date:** 2026-01-16
**Hardware:** RTX 5090 (SM 120a, Blackwell GeForce)
**CUDA:** 13.1

---

## Executive Summary

This report documents the investigation of Flash Attention 4 (FA4) for SM120 (RTX 5090 Blackwell GeForce). The goal was to evaluate NVFP4 (4-bit floating point) for attention computation using SM120's `mma.sync.aligned.block_scale` instructions.

### Key Findings

| Finding | Impact |
|---------|--------|
| SM120 uses `mma.sync.block_scale`, NOT `tcgen05.mma` | Architecture differs from datacenter SM100 |
| NVFP4 GEMM optimized for large K (K=256 tiles) | Poor utilization for attention's K=128 (head_dim) |
| P@V has K=seq_len (larger) = 3.7x better performance | NVFP4 better suited for P@V than Q@K^T |
| Softmax outputs (P) cannot use NVFP4 | Values ~0.001 << NVFP4 minimum 0.25 |
| FA3 TMA baseline: 51.97 TFLOPS @ seq=1024 | Already highly optimized |

### Recommendation

**Do not proceed with full FA4 NVFP4 implementation for GeForce SM120.**

The architectural constraints (no TMEM, limited cluster support, K=256 tile size mismatch) make NVFP4 attention less beneficial than expected. Focus optimization efforts on:
1. FA3 TMA pipeline improvements
2. W8A16 or W4A16 for LLM weight quantization (where NVFP4 shines)

---

## Phase 1: BF16 Baseline

### Objective
Establish FA4 kernel baseline with BF16 precision, verifying the kernel structure before adding NVFP4.

### Results

| Metric | Value |
|--------|-------|
| Kernel | FA4 Phase 1 (BF16 baseline) |
| Sequence Length | 1024 |
| Num Heads | 32 |
| Head Dim | 128 |
| Performance | 51.19 TFLOPS |
| Correctness | PASS (vs FA3 TMA reference) |

### Analysis
The BF16 baseline matches FA3 TMA performance, confirming the kernel structure is correct.

---

## Phase 2: NVFP4 Q@K^T Validation

### Objective
Validate NVFP4 GEMM for the Q@K^T attention score computation.

### Results

| Metric | Value |
|--------|-------|
| Operation | Q@K^T (single head) |
| Dimensions | [1024, 128] @ [128, 1024] |
| K dimension | 128 (head_dim) |
| NVFP4 Time | 353.3 us |
| NVFP4 TFLOPS | 0.76 |
| Correctness | 21.4% rel_diff (ACCEPTABLE for 4-bit) |

### Key Finding: K Dimension Mismatch

CUTLASS NVFP4 GEMM uses K=256 tile size, optimized for LLM weight matrices:
- LLM weights: K=4096+ (excellent tile utilization)
- Attention Q@K^T: K=128 (50% tile utilization)

This results in **suboptimal performance** for attention's small K dimension.

---

## Phase 3: Full NVFP4 Pipeline Validation

### Objective
Evaluate NVFP4 for P@V (attention output computation) where K=seq_len is larger.

### Results

| Metric | Q@K^T | P@V |
|--------|-------|-----|
| K dimension | 128 (head_dim) | 1024 (seq_len) |
| NVFP4 Time | 353.3 us | 94.7 us |
| NVFP4 TFLOPS | 0.76 | 2.84 |
| Speedup | baseline | **3.73x** |

### Key Finding: P (Softmax Output) Cannot Use NVFP4

**Critical limitation discovered:**

```
Softmax output values: ~1/seq_len = 0.000977
NVFP4 smallest positive: 0.25
Result: ALL P values quantize to 0 (100% error)
```

NVFP4's representable range `[-6, +6]` with smallest positive `0.25` cannot represent softmax probabilities. This **fundamentally prevents** using NVFP4 for P@V.

### Memory Footprint Analysis

| Format | P + V (single head) | Reduction |
|--------|---------------------|-----------|
| BF16 | 2304 KB | baseline |
| NVFP4 | 576 KB | **4x** |

While NVFP4 offers 4x memory reduction, the softmax output limitation makes this benefit unrealizable for attention.

---

## SM120 vs SM100 Architecture Comparison

### Why Modal Blog FA4 Doesn't Apply to GeForce

| Feature | SM100 (B100/B200) | SM120 (RTX 5090) |
|---------|-------------------|------------------|
| MMA Instruction | `tcgen05.mma` | `mma.sync.block_scale` |
| Tensor Memory | 256KB TMEM | **None** |
| Cluster Size | Up to 16 SM | **1x1x1 only** |
| Multicast | Yes | **None** |
| Warp Paradigm | Single-thread | Warp-synchronous |

The Modal Blog reverse-engineered FA4 for **datacenter** Blackwell (SM100), which has significantly different hardware capabilities than GeForce Blackwell (SM120).

---

## Recommended FA4 Architecture for SM120

If proceeding with FA4 despite limitations:

### Hybrid Precision Strategy
```
Q: Pre-quantize to NVFP4 (static, can clip to [-6, 6])
K: Pre-quantize to NVFP4 (static, can clip to [-6, 6])
V: Pre-quantize to NVFP4 (static, can clip to [-6, 6])
P: Keep in BF16 (dynamic softmax output, small values)
```

### Pipeline Structure
```
Q@K^T: mma.sync.block_scale.m64n64k64.nvf4  (4-bit MMA)
Softmax: FP32 accumulation
P@V: mma.sync.m16n8k16.bf16 (standard BF16 MMA)
```

### Expected Gains vs Complexity
| Optimization | Expected Gain | Complexity |
|--------------|---------------|------------|
| NVFP4 Q@K^T only | ~10-15% | High (PTX inline asm) |
| Memory bandwidth | ~15-20% | Medium (smaller loads) |
| Full FA4 (both GEMMs) | Not possible | N/A (P precision issue) |

---

## Benchmark Summary

### Full Attention (Fused, 32 Heads)

| Implementation | Time (us) | TFLOPS |
|----------------|-----------|--------|
| FA3 TMA | 330.5 | 51.97 |
| FA4 Phase 1 (BF16) | ~330 | ~52 |

### Component Benchmarks (Single Head, Unfused)

| Operation | K Dimension | NVFP4 Time (us) | TFLOPS |
|-----------|-------------|-----------------|--------|
| Q@K^T | 128 | 353.3 | 0.76 |
| P@V | 1024 | 94.7 | 2.84 |

---

## Conclusions

1. **NVFP4 is not suitable for full attention computation on SM120**
   - Softmax outputs are too small for NVFP4 range
   - K=128 (head_dim) causes poor tile utilization for Q@K^T

2. **FA3 TMA is already highly optimized**
   - 51.97 TFLOPS on RTX 5090
   - Further optimization should focus on TMA pipeline, not precision changes

3. **NVFP4 benefits are limited to:**
   - LLM weight quantization (W4A16 GEMM with large K)
   - Memory bandwidth reduction in memory-bound kernels

4. **SM120 (GeForce) differs significantly from SM100 (datacenter)**
   - No TMEM, limited cluster support
   - Modal Blog FA4 techniques don't directly apply

---

## Files

| File | Description |
|------|-------------|
| `native/ops/nn/attention/flash_attention_4_sm120.cuh` | FA4 kernel (Phase 1 baseline) |
| `benchmark_fa4_sm120.py` | Benchmark script (all phases) |
| `.serena/memories/fa4_sm120_research.md` | Research notes |
| `docs/fa4_sm120_implementation_report.md` | This report |

---

## References

- CUTLASS Example 79: Blackwell GeForce GEMM
- PTX ISA 8.5: `mma.sync.aligned.block_scale` instructions
- FlashAttention-3 Paper (Dao et al., 2024)
- Modal Blog: Reverse Engineering Flash Attention 4 (SM100 only)
