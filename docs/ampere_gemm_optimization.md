# Ampere-Optimized FP32 GEMM Kernel Design

## Executive Summary

Target: 22-32 TFLOPS FP32 GEMM on RTX 3090 Ti (SM 8.6)
Current: ~10 TFLOPS (28% efficiency)
Peak: 35.6 TFLOPS theoretical

## 1. RTX 3090 Ti Architecture Analysis

### Hardware Specifications
- **SMs**: 84
- **CUDA Cores**: 10,752 (128 per SM)
- **Clock**: ~1.86 GHz (boost)
- **Theoretical FP32**: 35.6 TFLOPS
- **Memory Bandwidth**: 1,008 GB/s (GDDR6X)
- **L2 Cache**: 6 MB
- **Shared Memory**: 100 KB per SM (max)
- **Registers**: 65,536 per SM (256 per thread max)

### Ampere-Specific Features
1. **cp.async**: Asynchronous global→shared copy bypassing registers
2. **Async barriers**: Fine-grained pipeline synchronization
3. **Large shared memory**: Up to 100KB with L1 carveout
4. **Improved L2**: 6MB with better hit rates

## 2. Current Kernel Bottleneck Analysis

### Estimated Roofline Position
```
Arithmetic Intensity (AI) = 2*M*N*K / (M*K + K*N + M*N) * 4 bytes
For 4096×4096: AI = 2*4096^3 / (3*4096^2 * 4) ≈ 682 FLOPS/byte

Ridge point = 35.6 TFLOPS / 1.008 TB/s = 35.3 FLOPS/byte
AI >> Ridge point → Should be compute-bound
```

### Why Current Kernel Underperforms

1. **Synchronous Memory Loads**:
   - Each tile load requires register staging
   - Memory latency not hidden
   - ~400 cycles latency per global load

2. **Insufficient Pipelining**:
   - Single buffer: compute waits for load
   - No overlap between load and compute phases

3. **Warp Stalls**:
   - Barrier stalls (waiting for __syncthreads)
   - Scoreboard stalls (register dependencies)
   - Memory dependency stalls

4. **Suboptimal Register Usage**:
   - Not maximizing ILP within thread
   - Potential register spilling

## 3. Optimization Strategy

### 3.1 Asynchronous Copy Pipeline (cp.async)

Replace synchronous loads:
```cuda
// OLD: Synchronous (2 instructions, uses registers)
float val = A[global_idx];
As[shared_idx] = val;

// NEW: Asynchronous (1 instruction, bypasses registers)
asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 16;\n"
    :: "r"(smem_ptr), "l"(gmem_ptr)
);
```

Benefits:
- Saves registers (no intermediate storage)
- Enables true async operation
- Better memory coalescing

### 3.2 Multi-Stage Software Pipeline

```
Stage Layout (4 stages, BK=8):
┌──────────┬──────────┬──────────┬──────────┐
│ Stage 0  │ Stage 1  │ Stage 2  │ Stage 3  │
│ Load K+3 │ Load K+2 │ Load K+1 │ Compute K│
└──────────┴──────────┴──────────┴──────────┘

Timeline:
Iter 0: Load[0]
Iter 1: Load[1], Wait[0]
Iter 2: Load[2], Compute[0], Wait[1]
Iter 3: Load[3], Compute[1], Wait[2]
Iter 4: Load[4], Compute[2], Wait[3]
...
```

This hides ~300-400 cycles of memory latency.

### 3.3 Tile Configuration

```
CTA Tile:     128 × 128 (output elements per block)
Warp Tile:    32 × 64   (each warp handles this)
Thread Tile:  8 × 8     (each thread computes 64 outputs)
BK:           8         (small for more pipeline stages)
Stages:       4         (hide memory latency)

Block: 256 threads (16×16)
Grid:  (N/128) × (M/128)
```

### 3.4 Register Blocking

Per thread register allocation:
- Accumulators: 8×8 = 64 floats = 64 registers
- A fragment: 8 floats = 8 registers
- B fragment: 8 floats = 8 registers
- Indexing: ~8 registers
- **Total: ~88 registers/thread**

With 256 threads: 256 × 88 = 22,528 registers/block
SM has 65,536 → Can run 2-3 blocks/SM → 50-75% occupancy

### 3.5 Shared Memory Layout

```
4-stage pipeline shared memory:
As[4][BK][BM+PAD] = 4 × 8 × (128+4) × 4 bytes = 16,896 bytes
Bs[4][BK][BN+PAD] = 4 × 8 × (128+4) × 4 bytes = 16,896 bytes
Total: ~34 KB (fits in 48KB default)
```

Padding eliminates bank conflicts.

### 3.6 Instruction-Level Parallelism

Within each thread's inner loop:
```cuda
// Interleave loads and computes for ILP
float a0 = As[k][ty*TM + 0];
float b0 = Bs[k][tx*TN + 0];
acc[0][0] = fmaf(a0, b0, acc[0][0]);

float a1 = As[k][ty*TM + 1];
float b1 = Bs[k][tx*TN + 1];
acc[0][1] = fmaf(a0, b1, acc[0][1]);
acc[1][0] = fmaf(a1, b0, acc[1][0]);
// ... etc
```

## 4. Expected Performance

### Theoretical Analysis

```
Operations per CTA per K-tile:
  = 2 × BM × BN × BK = 2 × 128 × 128 × 8 = 262,144 FLOPs

Memory loads per K-tile:
  = (BM × BK + BK × BN) × 4 = (1024 + 1024) × 4 = 8,192 bytes

Arithmetic Intensity = 262,144 / 8,192 = 32 FLOPS/byte
```

With 1 TB/s effective bandwidth: 32 × 1000 = 32 TFLOPS theoretical max

### Realistic Targets

- **Minimum**: 22 TFLOPS (62% efficiency)
- **Target**: 28 TFLOPS (79% efficiency)
- **Stretch**: 32 TFLOPS (90% efficiency)

## 5. Nsight Compute Metrics to Monitor

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| SM Throughput | >80% | Compute utilization |
| Memory Throughput | >70% | DRAM bandwidth |
| Shared Memory Throughput | >80% | SMEM bandwidth |
| Occupancy | 50-75% | Active warps/max |
| Warp Stall (No Instruction) | <5% | Instruction cache |
| Warp Stall (Barrier) | <10% | Sync overhead |
| Warp Stall (Memory) | <15% | Latency hiding |
| Register Spilling | 0 | No LMEM usage |

### Expected Stall Breakdown (optimized)
```
Not Selected:    40%  (normal scheduling)
Wait:            25%  (barrier/fence)
Math Pipe:       20%  (FMA throughput)
Memory:          10%  (unavoidable)
Other:            5%
```

## 6. Implementation Checklist

- [ ] cp.async infrastructure with proper barriers
- [ ] 4-stage pipeline state machine
- [ ] Efficient shared memory indexing
- [ ] Bank-conflict-free access patterns
- [ ] Vectorized loads where possible (float4)
- [ ] Optimal thread-to-data mapping
- [ ] Epilogue handling for non-multiple sizes
- [ ] Integration with existing dispatch
