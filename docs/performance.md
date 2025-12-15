# Performance Tuning Guide

This guide covers how to maximize GPU performance with PyGPUkit.

## Matrix Multiplication Performance

### TensorCore Acceleration

PyGPUkit uses NVIDIA CUTLASS for TensorCore-accelerated matrix multiplication.

| Precision | Peak TFLOPS (RTX 3090 Ti) | When to Use |
|-----------|---------------------------|-------------|
| FP32 (NO_TF32) | ~18 TFLOPS | Full precision required |
| TF32 | ~31 TFLOPS | Default for FP32 inputs |
| FP16 | ~63 TFLOPS | Memory-bound workloads |
| BF16 | ~63 TFLOPS | Training, better dynamic range |

### TF32 Mode (Default)

TF32 (TensorFloat-32) uses TensorCore with reduced mantissa precision:
- **Input:** FP32 (truncated to 19-bit mantissa)
- **Accumulator:** FP32
- **Error:** ~0.1% per operation

```python
import pygpukit as gpk
import numpy as np

# TF32 is automatic for FP32 matmul (default)
a = gpk.from_numpy(np.random.randn(4096, 4096).astype(np.float32))
b = gpk.from_numpy(np.random.randn(4096, 4096).astype(np.float32))
c = a @ b  # Uses TF32 TensorCore (~30 TFLOPS)
```

### Disabling TF32

For applications requiring full FP32 precision:

```python
# Method 1: Per-operation
c = gpk.matmul(a, b, use_tf32=False)

# Method 2: Environment variable (affects all matmuls)
# PYGPUKIT_NO_TF32=1 python my_script.py
```

### FP16/BF16 for Maximum Throughput

```python
import pygpukit as gpk
import numpy as np

# FP16 matmul (~63 TFLOPS)
a = gpk.from_numpy(np.random.randn(4096, 4096).astype(np.float16))
b = gpk.from_numpy(np.random.randn(4096, 4096).astype(np.float16))
c = a @ b

# BF16 matmul (~63 TFLOPS)
a32 = gpk.from_numpy(np.random.randn(4096, 4096).astype(np.float32))
a_bf16 = a32.astype(gpk.bfloat16)
b_bf16 = b32.astype(gpk.bfloat16)
c = a_bf16 @ b_bf16
```

---

## Dimension Alignment

CUTLASS TensorCore requires dimensions divisible by 16 for optimal performance.

### Optimal Dimensions

```python
# Good: All dimensions are multiples of 16
a = gpk.zeros((4096, 4096))  # 4096 = 256 * 16
b = gpk.zeros((4096, 4096))
c = a @ b  # Uses CUTLASS TensorCore

# Suboptimal: Falls back to slower kernel
a = gpk.zeros((4000, 4000))  # 4000 is not a multiple of 16
b = gpk.zeros((4000, 4000))
c = a @ b  # Uses fallback kernel
```

### Common Aligned Sizes

| Dimension | Use Case |
|-----------|----------|
| 768 | BERT/GPT-2 hidden size |
| 1024 | GPT-2 Medium |
| 2048 | GPT-2 Large |
| 3072 | BERT MLP intermediate |
| 4096 | LLaMA hidden size |
| 8192 | Large batch/context |

---

## Fused Operations

Fused operations reduce memory bandwidth by combining multiple operations into a single GPU kernel.

### linear_bias_gelu

Instead of separate matmul + bias + gelu:

```python
import pygpukit as gpk

# Unfused (3 GPU kernels, 3 memory round-trips)
y = gpk.matmul(x, gpk.transpose(weight))
y = y + bias
y = gpk.gelu(y)

# Fused (1 GPU kernel, 1 memory round-trip)
y = gpk.linear_bias_gelu(x, weight, bias)
```

**Performance benefit:** Up to 2-3x faster for memory-bound workloads.

### bias_add_inplace

Avoid allocating new arrays for bias addition:

```python
# Creates new array (slower)
output = output + bias

# In-place modification (faster, no allocation)
gpk.bias_add_inplace(output, bias)
```

---

## Multi-SM Optimization

PyGPUkit automatically selects optimized kernels based on GPU architecture:

| GPU | SM Version | Pipeline Stages | Shared Memory |
|-----|------------|-----------------|---------------|
| A100 | SM80 | 4-stage | 48KB |
| RTX 3090 | SM86 | 5-stage | 100KB |
| RTX 4090 | SM89 | 5-stage | 100KB |
| H100 | SM90 | 5-stage | 100KB |

This is automatic - no configuration needed.

---

## Memory Management

### Minimize CPU-GPU Transfers

```python
import pygpukit as gpk
import numpy as np

# Bad: Frequent transfers
for i in range(1000):
    a_np = np.random.randn(1024, 1024).astype(np.float32)
    a = gpk.from_numpy(a_np)  # CPU -> GPU transfer
    result = gpk.relu(a)
    r_np = result.to_numpy()  # GPU -> CPU transfer

# Good: Keep data on GPU
a = gpk.from_numpy(np.random.randn(1024, 1024).astype(np.float32))
for i in range(1000):
    a = gpk.relu(a)  # All operations on GPU
result = a.to_numpy()  # Single transfer at the end
```

### Pre-transpose Weights

For repeated linear operations:

```python
from pygpukit.llm import Linear

# Linear layer caches transposed weights
linear = Linear(weight, bias)

# First call transposes weight (cached)
y1 = linear(x1)

# Subsequent calls reuse cached transpose
y2 = linear(x2)  # No transpose overhead
```

---

## Benchmarking

### Using benchmark.py

```bash
# Full benchmark
python benchmark.py

# Quick benchmark
python benchmark.py --quick

# Specific sizes
python benchmark.py --sizes 4096 8192

# Specific dtypes
python benchmark.py --dtypes float32 float16
```

### Manual Benchmarking

```python
import pygpukit as gpk
import numpy as np
import time

def benchmark_matmul(M, N, K, dtype=np.float32, warmup=10, iterations=100):
    a = gpk.from_numpy(np.random.randn(M, K).astype(dtype))
    b = gpk.from_numpy(np.random.randn(K, N).astype(dtype))

    # Warmup
    for _ in range(warmup):
        c = a @ b

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        c = a @ b
    elapsed = time.perf_counter() - start

    # Calculate TFLOPS
    flops = 2 * M * N * K * iterations
    tflops = flops / elapsed / 1e12

    print(f"{M}x{N}x{K}: {tflops:.1f} TFLOPS")
    return tflops

# Run benchmarks
benchmark_matmul(4096, 4096, 4096)
benchmark_matmul(8192, 8192, 8192)
```

---

## Performance Checklist

1. **Use aligned dimensions** (multiples of 16)
2. **Use FP16/BF16** for maximum throughput when precision allows
3. **Use fused operations** (`linear_bias_gelu`, `bias_add_inplace`)
4. **Minimize CPU-GPU transfers** - keep data on GPU
5. **Batch operations** - larger matrices are more efficient
6. **Pre-transpose weights** for repeated linear operations

---

## Expected Performance (RTX 3090 Ti)

| Operation | Size | Performance |
|-----------|------|-------------|
| matmul (TF32) | 8192x8192 | ~31 TFLOPS |
| matmul (FP16) | 8192x8192 | ~63 TFLOPS |
| matmul (BF16) | 8192x8192 | ~63 TFLOPS |
| matmul (FP32) | 8192x8192 | ~18 TFLOPS |
| linear_bias_gelu | 512x768x3072 | ~25 TFLOPS |

---

## Troubleshooting

### Low Performance

1. **Check dimension alignment:**
   ```python
   M, K = a.shape
   K2, N = b.shape
   if M % 16 != 0 or N % 16 != 0 or K % 16 != 0:
       print("Warning: Non-aligned dimensions, using fallback kernel")
   ```

2. **Verify TensorCore is being used:**
   ```python
   caps = gpk.get_device_capabilities()
   print(f"TensorCore available: {caps.has_tensor_core}")
   ```

3. **Check GPU utilization:**
   ```bash
   nvidia-smi dmon -s u -d 1
   ```

### Memory Issues

1. **Monitor VRAM usage:**
   ```python
   info = gpk.get_device_info()
   print(f"Total VRAM: {info.total_memory / 1e9:.1f} GB")
   ```

2. **Use smaller batches** if running out of memory

3. **Delete unused arrays** to free GPU memory:
   ```python
   del large_array
   ```
