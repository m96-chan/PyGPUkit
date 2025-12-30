# Benchmark Skill

Run unified benchmark suite for GEMM, GEMV, and attention kernels.

## Commands

```bash
# Quick benchmark (default: GEMM + GEMV)
python -m pygpukit.benchmark --quick

# Full benchmark with all sizes
python -m pygpukit.benchmark

# Save results to JSON
python -m pygpukit.benchmark --quick --save results.json

# Compare with baseline
python -m pygpukit.benchmark --compare baseline.json

# Fail on regression (for CI)
python -m pygpukit.benchmark --compare baseline.json --fail-on-regression

# Specific benchmarks
python -m pygpukit.benchmark --gemm --sizes 4096,8192
python -m pygpukit.benchmark --gemv --dtypes bf16,fp8
python -m pygpukit.benchmark --attention --seq-lens 512,1024,2048

# All benchmarks including FP8 (SM120+)
python -m pygpukit.benchmark --all --fp8

# Markdown output for README
python -m pygpukit.benchmark --quick --markdown
```

## Output

- Time in microseconds (us)
- TFLOPS for compute benchmarks
- Correctness verification
- JSON export for regression tracking

## Usage in Code

```python
from pygpukit.benchmark import BenchmarkSuite

suite = BenchmarkSuite(quick=True)
suite.add_gemm(sizes=[(4096, 4096, 4096)])
suite.add_gemv(dtypes=["bf16", "fp8"])
report = suite.run()
report.save("baseline.json")

# Compare
comparison = suite.compare("baseline.json")
if comparison.has_regression(threshold=0.05):
    print("Regression detected!")
```
