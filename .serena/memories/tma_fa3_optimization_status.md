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

## Commit History
- `dcd1f8e` - feat(attention): integrate TMA FA3 into SDPA dispatch
- `adee44b` - wip(fa3): add TMA-enabled Flash Attention 3 kernel
- (current) - fix(fa3): __syncthreads divergence bug, kernel now stable at scale
