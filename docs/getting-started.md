# Getting Started with PyGPUkit

PyGPUkit is a lightweight GPU runtime for Python that provides NumPy-like array operations with CUDA acceleration.

## Installation

```bash
pip install pygpukit
```

### Requirements

- Python 3.10+
- NVIDIA GPU with drivers installed (RTX 30XX or newer)
- **Optional:** CUDA Toolkit (for JIT compilation of custom kernels)

### Supported GPUs

| GPU Series | Support |
|------------|---------|
| RTX 40XX (Ada) | Full |
| RTX 30XX (Ampere) | Full |
| A100, H100 | Full |
| RTX 20XX, GTX 10XX | Not supported (SM < 80) |

## Quick Start

### Basic Array Operations

```python
import pygpukit as gpk
import numpy as np

# Create GPU arrays
a = gpk.zeros((1024, 1024), dtype="float32")
b = gpk.ones((1024, 1024), dtype="float32")

# Element-wise operations
c = gpk.add(a, b)
d = gpk.mul(a, b)

# Or use operator overloads
c = a + b
d = a * b
e = a - b
f = a / b

# Matrix multiplication
g = a @ b  # or gpk.matmul(a, b)

# Transfer to/from NumPy
np_array = c.to_numpy()
gpu_array = gpk.from_numpy(np_array)
```

### Data Types

```python
import pygpukit as gpk

# Available data types
gpk.float32   # 32-bit float (default)
gpk.float64   # 64-bit float
gpk.float16   # 16-bit float (half precision)
gpk.bfloat16  # Brain float16
gpk.int32     # 32-bit integer
gpk.int64     # 64-bit integer

# Create arrays with specific dtype
a = gpk.zeros((1024, 1024), dtype=gpk.float16)
b = gpk.ones((1024, 1024), dtype=gpk.bfloat16)

# Convert between types
c = a.astype(gpk.float32)
```

### Math Operations

```python
import pygpukit as gpk
import numpy as np

a = gpk.from_numpy(np.random.randn(1024, 1024).astype(np.float32))

# Unary operations
b = gpk.exp(a)
c = gpk.log(a)

# Activation functions
d = gpk.relu(a)
e = gpk.gelu(a)

# Reductions
total = gpk.sum(a)      # Sum of all elements
avg = gpk.mean(a)       # Mean of all elements
maximum = gpk.max(a)    # Maximum element
```

### Matrix Operations

```python
import pygpukit as gpk
import numpy as np

# Matrix multiplication (uses CUTLASS TensorCore)
a = gpk.from_numpy(np.random.randn(4096, 4096).astype(np.float32))
b = gpk.from_numpy(np.random.randn(4096, 4096).astype(np.float32))
c = a @ b  # ~30 TFLOPS on RTX 3090 Ti

# Transpose
d = gpk.transpose(a)
```

### Neural Network Operations

```python
import pygpukit as gpk
import numpy as np

# LayerNorm
batch, features = 32, 768
x = gpk.from_numpy(np.random.randn(batch, features).astype(np.float32))
gamma = gpk.ones(features)
beta = gpk.zeros(features)
normalized = gpk.layernorm(x, gamma, beta, eps=1e-5)

# Fused Linear + Bias + GELU (uses CUTLASS epilogue fusion)
input = gpk.from_numpy(np.random.randn(512, 768).astype(np.float32))
weight = gpk.from_numpy(np.random.randn(3072, 768).astype(np.float32))
bias = gpk.from_numpy(np.random.randn(3072).astype(np.float32))
output = gpk.linear_bias_gelu(input, weight, bias)
```

## Checking GPU Status

```python
import pygpukit as gpk

# Check if CUDA is available
print(f"CUDA available: {gpk.is_cuda_available()}")

# Check if JIT compilation is available
print(f"NVRTC available: {gpk.is_nvrtc_available()}")

# Get device info
info = gpk.get_device_info()
print(f"Device: {info.name}")
print(f"Compute capability: {info.compute_capability}")
print(f"Total memory: {info.total_memory / 1e9:.1f} GB")
```

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [LLM Guide](llm.md) - Loading and running LLM models
- [Performance Tuning](performance.md) - Optimizing for maximum throughput
- [Scheduler Guide](scheduler.md) - Multi-LLM concurrent execution
