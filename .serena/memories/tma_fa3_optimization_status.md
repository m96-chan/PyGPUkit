# TMA FA3 Optimization Status

## Current Implementation Status (2026-01-16)

### Files Created/Modified
- `native/ops/common/tma_utils.cuh` - TMA descriptor creation, barrier ops, TMA loads
- `native/ops/common/warp_scheduler.cuh` - Producer/consumer warp specialization
- `native/ops/common/pipeline.cuh` - Multi-stage async pipeline management
- `native/ops/nn/attention/flash_attention_3_tma.cuh` - TMA-enabled FA3 kernel
- `native/ops/nn/attention/sdpa_causal.inl` - Integration with SDPA dispatch
- `examples/benchmark_fa3_tma.py` - Benchmark script

### Environment Variables
- `PYGPUKIT_FA3_TMA`: 0=off, 1=on, -1=auto (default: on for SM90+, seq>512)

### Current Configuration (v3 - Bug Fixed)
```
Stage count:        2
Producer warps:     4
Consumer warps:     8
Total threads:      384 (12 warps)
TILE_Q:             32  (reduced to fit 99KB smem limit)
TILE_KV:            64
HEAD_DIM:           128
Shared memory:      ~96KB
```

### Benchmark Results (RTX 5090, SM 120a)

**v3 Results (2026-01-16, after __syncthreads fix):**
| Config | FA2 (us) | FA3 (us) | TMA (us) | FA3 TFLOPS | TMA TFLOPS |
|--------|----------|----------|----------|------------|------------|
| [32, 512, 128] | 7245 | 6827 | 6611 | 0.63 | 0.65 |
| [32, 1024, 128] | 25136 | 25939 | 25845 | 0.66 | 0.66 |
| [32, 2048, 128] | 96982 | 97640 | 99221 | 0.70 | 0.69 |
| [32, 4096, 128] | 388437 | 394461 | 387519 | 0.70 | 0.71 |

**Key Observations:**
- TMA and baseline FA3 have similar performance (~0.65-0.71 TFLOPS)
- Performance is severely compute-underutilized (RTX 5090 BF16 theoretical: ~1800 TFLOPS)
- Even at 1% utilization should be ~18 TFLOPS, current is ~4% of that

**Correctness: PASS** (all implementations match)

## v3 Changes (2026-01-16) - __syncthreads() Divergence Fix

### Bug Fixed
**Problem:** Kernel hung at 256+ blocks due to `__syncthreads()` inside `consumer_compute_output()` which was only called by consumer warps (`if(is_consumer)`). Producer warps never reached the sync points.

**Debug Process:**
- Added per-thread debug (tid=0 producer, tid=128/383 consumer)
- Found producer reached "after compute_output" but consumers didn't
- Root cause: lines 278 and 288 had `__syncthreads()` in consumer-only function

**Fix:** Split `consumer_compute_output()` into two functions:
1. `convert_scores_to_probs()` - Called by ALL threads (contains sync points)
2. `consumer_compute_output_matmul()` - Called only by consumers (no syncs)

### Code Change
```cpp
// OLD (broken): __syncthreads inside consumer-only function
if (is_consumer) {
    consumer_compute_output(smem, stage, tid, num_threads);  // Had syncs inside!
}

// NEW (fixed): Separate sync-containing code from consumer-only code
convert_scores_to_probs<Config>(smem, tid, num_threads);  // ALL threads
if (is_consumer) {
    consumer_compute_output_matmul<Config>(smem, stage, tid, num_threads);  // No syncs
}
```

## Shared Memory Optimization Progress

### v1 -> v2 Changes (2026-01-16)
1. **Union for scores/probs** - Merged `smem_scores` and `smem_probs_bf16` into single union
   - Safe in-place conversion: read to registers -> syncthreads -> write BF16
   - Savings: 8 KB
2. **Reduced stages from 4 to 3**
   - K/V buffers: 128 KB -> 96 KB
   - Savings: 32 KB

### Current Shared Memory Layout (~161 KB)
```
smem_q:           64 x 128 x 2 =  16 KB
smem_k:       3 x 64 x 128 x 2 =  48 KB  (3 stages)
smem_v:       3 x 64 x 128 x 2 =  48 KB  (3 stages)
smem_scores/probs (union):       16 KB  (float for softmax, BF16 for P@V)
output_acc:    64 x 128 x 4    =  32 KB
softmax_max/sum:                  ~0.5 KB
barriers:                         ~24 B
---------------------------------------
Total:                          ~161 KB (was ~201 KB, saved 40 KB)
```
- SM120 max shared memory: 228 KB
- **Occupancy: still 1 block/SM** (need ~114 KB for 2 blocks)

## Identified Performance Bottlenecks

### Critical Issue: Only ~0.7 TFLOPS (Expected ~18+ TFLOPS)

The kernel is severely compute-underutilized. Main bottlenecks identified from code analysis:

### 1. Sequential Softmax Computation (lines 470-509)
**Impact: VERY HIGH**
```cpp
for (int q = 0; q < q_len; ++q) {  // Sequential over query positions!
    // Only lane-level parallelism (32 threads)
    for (int kv = lane_id; kv < kv_len; kv += 32) { ... }
}
```
- Processing one query at a time
- Only 32 threads active per query row
- With q_len=32 and 8 consumer warps, only 1/8 of consumer threads useful at a time

### 2. Consumer-Only Q@K Computation (line 451)
**Impact: HIGH**
```cpp
if (is_consumer) {
    consumer_compute_scores<Config>(smem, ...);  // 4/12 warps idle!
}
```
- Producer warps (4 out of 12) completely idle during compute phases
- 33% of threads wasted

### 3. Excessive Synchronization
**Impact: MEDIUM-HIGH**
- Multiple `__syncthreads()` per iteration:
  - Line 454 (after Q@K)
  - Line 466 (after causal mask)
  - Line 511 (after softmax)
  - Lines 262, 272 (in convert_scores_to_probs)
  - Line 540 (end of iteration)
- Each sync serializes all threads

### 4. Small TILE_Q (32) Due to Smem Limit
**Impact: MEDIUM**
- RTX 5090 reports max 101KB shared memory per block
- TILE_Q=64 would require ~160KB
- Limited parallelism per block

### 5. WMMA vs wgmma (Blackwell)
**Impact: MEDIUM**
- Using older WMMA API (16x16x16)
- SM120a has optimized wgmma instructions
- Missing FP8/FP4 narrow precision opportunities

## Next Steps (Priority Order)

### Priority 1: Parallelize Softmax Across Query Positions (CRITICAL)
**Expected Impact: 8-32x speedup potential**

Current implementation processes one query row at a time. Need to:
1. Have each consumer warp handle a different query row
2. Parallelize across all 8 consumer warps (256 threads) instead of 32

```cpp
// Target: Each warp handles different query row
int warp_q = warp_id - NUM_PRODUCER_WARPS;  // 0-7 for consumers
for (int q = warp_q; q < q_len; q += NUM_CONSUMER_WARPS) {
    // Process query row q with full warp parallelism
}
```

### Priority 2: Use All Warps for Compute
**Expected Impact: 1.3x speedup**

Producer warps (4 out of 12) are idle during compute phases:
- Option A: Let producers also do Q@K (after TMA loads complete)
- Option B: Reduce to 2 producer warps, increase to 10 consumer warps

### Priority 3: Reduce Synchronization Points
**Expected Impact: 1.2-1.5x speedup**

Merge consecutive syncs, use warp-level sync where possible:
- Combine causal mask and softmax phases
- Use `__syncwarp()` within warp-local operations

### Priority 4: Upgrade to wgmma (SM120a)
**Expected Impact: 1.5-2x for matmul portions**

Replace WMMA with Blackwell-native wgmma instructions:
- Larger tile sizes (64x128x16 or similar)
- Better register utilization
- FP8 precision path available

### Profiling Scripts
- Profile: `examples/ncu_fa3_profile.py`
- Benchmark: `examples/benchmark_fa3_tma.py`
- NCU batch: `run_ncu_tma.bat`

## Bottleneck Analysis (2026-01-16)

### NCU Profiling Attempt
NCU requires administrator privileges on Windows. Analysis performed via code structure review.

### Verdict: Type A (Compute Inefficiency) is Dominant

**Primary Bottleneck: WMMA API on SM120**
- WMMA 16×16×16 designed for Volta/Ampere
- SM120 has wgmma with 64×64×16+ tiles
- Blackwell tensor cores significantly underutilized by WMMA
- All implementations (FA2, FA3, TMA) show ~0.7 TFLOPS → common bottleneck

**Secondary Issues:**
- 4 producer warps idle during compute (33% waste)
- 5-6 __syncthreads() per KV tile iteration
- Small TILE_Q=32 limits parallelism

### Recommendation
**Proceed with wgmma refactor** (expected 5-10x improvement)

### Implementation Plan for wgmma
1. Replace WMMA with `wgmma.mma_async` PTX inline assembly
2. Larger tile sizes (64×128 or 64×64 per warp group)
3. Async warpgroup execution pattern
4. Reduce producer warps to 2 (increase consumers to 10)

## Optimization Analysis (2026-01-17)

### 1. NSYS Barrier Stall Analysis
- **Sync overhead: ~0.2%** (NOT the bottleneck)
- Kernel stats: ~1.09ms average, 23 invocations
- CUDA API: 46.7% sync, 24.7% H2D, 22.3% D2H
- GPU metrics require admin on Blackwell (RTX 5090)

### 2. Swizzle Optimization Analysis
- All SMEM buffers have 256-byte stride = 64 banks (wraps every row)
- Potential 2-way bank conflicts for BF16
- No explicit swizzle implemented
- WMMA may hide latency through hardware scheduling
- **Recommendation**: Swizzle could help but uncertain ROI at 64.6 TFLOPS

### 3. Warp Ratio Tuning Results

| Version | Producer | Consumer | Total | Avg TFLOPS | Peak TFLOPS |
|---------|----------|----------|-------|------------|-------------|
| V0 | 4 | 8 | 12 | 62.81 | 64.58 |
| V3 | 2 | 10 | 12 | 62.69 | 64.58 |
| V4 | 4 | 12 | 16 | 63.49 | 64.56 |
| V5 | 2 | 14 | 16 | 63.15 | 64.60 |
| **V6** | **4** | **16** | **20** | **63.57** | **64.60** |

**Key Findings:**
- All versions hit same ~64.6 TFLOPS ceiling → **common bottleneck**
- V6 (4+16 warps) has best average TFLOPS
- 4 producers better than 2 (more TMA issuing warps helps)
- More consumer warps marginally helps
- **Ceiling is NOT warp-ratio dependent**

### Bottleneck Conclusion
Performance ceiling (~64.6 TFLOPS) is constrained by:
1. **WMMA instruction throughput** - WMMA 16×16×16 underutilizes SM120 tensor cores
2. **6 syncs per KV tile** - Required for correctness (see sync analysis)
3. Potential solution: **wgmma upgrade** for larger tiles and async execution

## wgmma Investigation (2026-01-17)

### Critical Finding: SM120 Has NO wgmma

**Architecture Comparison:**
| Feature | SM90 (Hopper) | SM100 (Blackwell DC) | SM120 (Blackwell GeForce) |
|---------|---------------|----------------------|---------------------------|
| wgmma.mma_async | YES | NO | **NO** |
| tcgen05.mma + TMEM | NO | YES | NO |
| mma.sync.aligned | YES | YES | YES |
| Block-scaled MMA | NO | YES | YES |

**SM120 Available Instructions:**
- `mma.sync.aligned.m16n8k16` for BF16 (current WMMA)
- `mma.sync.aligned.block_scale.m64n64k32.f32.e4m3.e4m3` for FP8
- `mma.sync.aligned.block_scale.m64n64k64.f32.nvf4.nvf4` for NVFP4

**Conclusion:** BF16 at 64.6 TFLOPS is the **architectural ceiling** for SM120.
Only FP8/NVFP4 can achieve larger tile sizes (64×64 vs 16×8).

## FP8 Attention Re-evaluation (2026-01-17)

### Test Summary
Comprehensive CPU simulation of FP8 E4M3 attention with various scaling strategies.

### Key Findings

**1. Per-element FP8 Quantization Error:**
- Mean relative error: ~2.3% per element
- Max relative error: ~6% per element
- This is acceptable for individual values

**2. Matmul Accumulation Error (Q@K^T):**
- **20-45% relative error** even with optimal scaling
- Error compounds over head_dim=128 accumulations
- Scaling (absmax, per-row, per-head) provides minimal improvement

**3. Softmax Sensitivity:**
- Softmax is NOT the problem
- Score errors of 0.1 cause only ~0.0003 probability error
- Almost no error amplification from softmax

**4. Full Attention Results:**
| Strategy | Relative Error |
|----------|---------------|
| Naive FP8 | 25-35% |
| Scaled FP8 | 24-32% |
| Per-head scaled | 27-28% |

**5. Input Value Distribution:**
- Real attention inputs use <1% of FP8 dynamic range
- Even scaling to 100 doesn't help
- Best scale factor found: 5.0 with 24.75% error

### Root Cause Analysis
FP8 E4M3 has only 3 mantissa bits:
- ~12.5% relative precision per element
- Matmul over N=128 elements accumulates correlated rounding errors
- Result: 20-35% error is **fundamental limitation**, not implementation issue

### Conclusion (Scoped to Test Conditions)

**テスト条件での結論:**
- FP8（per-head / 単一スケール）で Q@K^T を FP8 演算 → **20-45% 誤差で不適**
- BF16 FA3 TMA (SM120) で **~65 TFLOPS が現実的な生産目標**

**一般化してはいけない点:**
- FP8 の量子化そのもの（要素単体 ~2.3%）は許容域
- Q@K^T が壊れる理由 = スケール粒度が粗すぎる
- Block-scaled FP8 (MXFP8等) なら再評価余地あり

**Q@K^T が特に壊れやすい理由:**
- Softmax前のlogitは、小さな相対誤差でも「順位（どこが最大か）」が崩れる
- その後のsoftmaxで「注意の向きが別方向」になりやすい
- GEMMとしての誤差ではなく、attention semantics の破壊

### FP8 の実用パス

**Path A: KV cache FP8 (推奨・低コスト)**
- Q@K^T はBF16のまま（精度問題を踏まない）
- KV cache のみ FP8 で格納 → dequant → BF16 GEMM
- 帯域削減の旨み（特に長文で効く）
- 実装: 「dequant → BF16 GEMM」なら現実的

**Path B: Block-scaled FP8 (中〜高コスト)**
- per-head ではなく per-(head, token_block) 粒度
- CUTLASS の blockscaled MMA パスを利用
- MXFP8 等の規格に準拠
- 工数: 中〜高

### Recommendation
1. **BF16 FA3 TMA (SM120) ~65 TFLOPS** を生産目標として確定
2. FP8 は「危険な主役」ではなく「脇役の帯域削減」から入る
3. 次の一手: **KV cache FP8** で帯域メリットだけ取る

## FP8 Selective Quantization 発見 (2026-01-17)

### 決定的実験結果

| Quantization | Relative Error |
|--------------|----------------|
| Q=FP8, K=FP32, V=FP32 | **0.30%** |
| Q=FP32, K=FP8, V=FP32 | **0.22%** |
| Q=FP32, K=FP32, V=FP8 | **32.98%** |
| Q=FP8, K=FP8, V=FP32 | **0.40%** |
| Q=FP32, K=FP8, V=FP8 | **32.97%** |
| Q=FP8, K=FP8, V=FP8 | **32.89%** |

### Key Insight

**エラーの99%は V の量子化から来ている**

理由:
- Q @ K^T → softmax で正規化 → 小さな誤差は吸収される
- P @ V → V の量子化誤差がそのまま出力に反映

### 実用的結論

1. **Q, K は FP8 で安全** (~0.3-0.4% error)
2. **V は BF16 のままにする**
3. 帯域削減 (Q, K の 50%) + 精度維持の両立が可能

### 推奨実装パス

```
Q: FP8 storage → dequant → BF16 GEMM
K: FP8 storage (KV cache) → dequant → BF16 GEMM
V: BF16 storage (KV cache) → BF16 GEMM
```

帯域削減: Q=50%, K=50%, V=0% → 全体で ~33% 削減

## GPU 検証結果 (2026-01-17)

### CPU vs GPU 比較

| 条件 | CPU (FP32 Sim) | GPU (BF16 FA3) |
|------|----------------|----------------|
| Q,K=FP8, V=BF16 | **0.4%** | **21-49%** |
| Q,K=BF16, V=FP8 | 33% | **131%** |
| All=FP8 | 33% | 131% |

### 発見
**CPU シミュレーションは楽観的すぎた**

原因:
1. CPU: FP32 中間精度で計算
2. GPU: BF16 GEMM + BF16 中間値

FP8 量子化誤差 + BF16 丸め誤差が複合して大きくなる

### 結論の修正
- **FP8/NVFP4 Q,K は BF16 FA3 では実用不可** (21-49% error)
- CPU simulation の「Q,K は安全」は **BF16 環境では成立しない**
- native FP8 block-scale MMA (FP32 accumulator) でないと精度が出ない

## FP8 Block-Scale PoC Results (2026-01-17)

### Key Finding: Q@K^T Score Error != Attention Output Error

| Measurement | FP8 Q,K Error |
|-------------|---------------|
| Q@K^T scores (raw) | ~29% (high, small values near zero) |
| Attention output (after softmax) | **0.25%** (softmax normalizes error) |

### Selective Quantization Results (CPU FP32 Accum)

| Strategy | Error |
|----------|-------|
| Q=FP8 only | 0.18% |
| K=FP8 only | 0.19% |
| **Q,K=FP8** | **0.25%** |
| V=FP8 | 18.54% |
| All=FP8 | 18.54% |

### Root Cause of GPU vs CPU Discrepancy

| Path | Accumulator | Q,K=FP8 Error |
|------|-------------|---------------|
| CPU simulation | FP32 | 0.25% |
| GPU dequant-to-BF16 | BF16 | 21-49% |
| Native block_scale MMA | FP32 | Expected ~0.25% |

**The difference is accumulator precision, NOT quantization itself!**

### Recommended SM120 FA4 Implementation

1. **Phase 2: FP8 Q@K^T**
   - mma.sync.aligned.block_scale.m64n64k32.f32.e4m3.e4m3
   - Q, K = FP8 E4M3 (block-scaled, 32 elements per scale)
   - Accumulator = FP32
   - **VIABLE** (0.25% error expected)

2. **P@V remains BF16 WMMA**
   - V quantization error not absorbed by softmax
   - V=FP8 gives ~18% error (unacceptable)

## Commit History
- `dcd1f8e` - feat(attention): integrate TMA FA3 into SDPA dispatch
- `adee44b` - wip(fa3): add TMA-enabled Flash Attention 3 kernel
- `028af30` - refactor(fa3): parallelize softmax and fix consumer warp indexing
