---
name: benchmark
description: Run matmul performance benchmarks. Use when user wants to measure TFLOPS, compare kernel performance, or verify correctness after code changes.
---

# Benchmark PyGPUkit

Run comprehensive matmul benchmarks for all supported dtypes.

## Usage

```bash
# Full benchmark
python scripts/benchmark.py

# Quick mode (fewer iterations)
python scripts/benchmark.py --quick

# Specific sizes
python scripts/benchmark.py --sizes 4096,8192

# TF32 kernel version
python scripts/benchmark.py --tf32-version v2
```

## Options

- `--sizes`: Comma-separated matrix sizes (default: 2048,4096,8192)
- `--quick`: Fewer warmup/iterations for faster results
- `--dtypes`: Which dtypes to test (default: fp32,tf32,fp16,bf16)
- `--tf32-version`: v1 (WMMA) or v2 (PTX mma.sync, default)

## Instructions

1. Ensure the project is built before benchmarking
2. Run `python scripts/benchmark.py [options]`
3. Report the performance results including:
   - TFLOPS for each dtype and size
   - Correctness verification (PASS/FAIL)
   - Comparison with theoretical peak

## Expected Results (RTX 5090)

| Dtype | Target TFLOPS |
|-------|---------------|
| FP32  | ~18           |
| TF32  | ~27           |
| FP16  | ~15           |
| BF16  | ~15           |

## Environment Variables

- `PYGPUKIT_ALLOW_TF32=1`: Enable TF32 TensorCore
- `PYGPUKIT_TF32_V2=1`: Use PTX mma.sync kernel
