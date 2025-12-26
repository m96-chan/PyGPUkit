---
name: kernel-dev
description: CUDA kernel development workflow. Use when writing, testing, or optimizing GPU kernels. Follows the Edit-Build-Validate-Benchmark-Commit cycle.
---

# CUDA Kernel Development

Workflow for developing and optimizing CUDA kernels.

## Development Cycle

```
Edit -> Build -> Validate -> Benchmark -> Commit
```

**ALWAYS commit after validation/benchmark, regardless of results.**

## Commands

```bash
# 1. Build (from Git Bash)
./build.sh 86       # RTX 3090 Ti
./build.sh 120a     # RTX 5090

# 2. Validate correctness
python -c "
import numpy as np
import _pygpukit_native as native
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)
C = native.matmul(native.from_numpy(A), native.from_numpy(B)).to_numpy()
expected = A @ B
error = np.max(np.abs(C - expected)) / np.max(np.abs(expected))
print(f'Relative error: {error:.2e}')
print('PASS' if error < 1e-3 else 'FAIL')
"

# 3. Benchmark
python scripts/benchmark.py --quick

# 4. Commit (MANDATORY)
git add -A && git commit -m 'wip(kernel): description'
```

## Commit Message Format

```
wip(tf32): <summary of changes>

Benchmark results (RTX 3090 Ti):
- 2048x2048: XX.XX TFLOPS
- 4096x4096: XX.XX TFLOPS
- 8192x8192: XX.XX TFLOPS

Correctness: <PASS/FAIL>
```

## Instructions

1. Make kernel code changes
2. Build the project
3. Run correctness validation
4. Run benchmark
5. Commit with results
6. If regression, revert to previous commit

## File Locations

- `native/ops/matmul/` - MatMul kernels
- `native/ops/gemv/` - GEMV kernels
- `native/ops/matmul/gemm/` - GEMM implementations
- `native/core/` - Core CUDA utilities

## Performance Targets (RTX 3090 Ti)

| Kernel | Target TFLOPS |
|--------|---------------|
| FP32 naive | ~18 |
| TF32 TensorCore | ~35 |
| cuBLAS | ~59 |

## Notes

- Never overwrite working kernel without commit
- Always include benchmark results in commit
- Regression = immediate revert
