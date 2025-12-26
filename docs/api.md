# PyGPUkit API Reference

## Core Types

### GPUArray

The main array type for GPU computation.

```python
class GPUArray:
    """N-dimensional array stored in GPU memory."""

    # Properties
    shape: tuple[int, ...]  # Array dimensions
    dtype: DataType         # Data type
    ndim: int               # Number of dimensions
    size: int               # Total number of elements
    nbytes: int             # Total bytes
    itemsize: int           # Bytes per element

    # Methods
    def to_numpy(self) -> np.ndarray:
        """Copy array data to NumPy array."""

    def astype(self, dtype: DataType) -> GPUArray:
        """Convert to different data type."""
```

**Example:**
```python
import pygpukit as gpk
import numpy as np

a = gpk.from_numpy(np.random.randn(100, 100).astype(np.float32))
print(f"Shape: {a.shape}")       # (100, 100)
print(f"Dtype: {a.dtype}")       # float32
print(f"Size: {a.size}")         # 10000
print(f"Bytes: {a.nbytes}")      # 40000

# Convert to numpy
np_arr = a.to_numpy()

# Convert dtype
b = a.astype(gpk.float16)
```

### DataType

Data type enumeration.

| Type | Description | Size |
|------|-------------|------|
| `gpk.float32` | 32-bit float | 4 bytes |
| `gpk.float64` | 64-bit float | 8 bytes |
| `gpk.float16` | 16-bit float | 2 bytes |
| `gpk.bfloat16` | Brain float16 | 2 bytes |
| `gpk.int32` | 32-bit integer | 4 bytes |
| `gpk.int64` | 64-bit integer | 8 bytes |

---

## Factory Functions

### zeros

```python
def zeros(
    shape: tuple[int, ...] | int,
    dtype: str | DataType = "float32",
) -> GPUArray:
    """Create array filled with zeros."""
```

**Example:**
```python
a = gpk.zeros((1024, 1024))
b = gpk.zeros((512,), dtype=gpk.float16)
c = gpk.zeros(100)  # 1D array
```

### ones

```python
def ones(
    shape: tuple[int, ...] | int,
    dtype: str | DataType = "float32",
) -> GPUArray:
    """Create array filled with ones."""
```

**Example:**
```python
a = gpk.ones((1024, 1024))
b = gpk.ones((512,), dtype="float64")
```

### empty

```python
def empty(
    shape: tuple[int, ...] | int,
    dtype: str | DataType = "float32",
) -> GPUArray:
    """Create uninitialized array (faster than zeros)."""
```

**Example:**
```python
# Use when you'll overwrite all values
a = gpk.empty((1024, 1024))
```

### from_numpy

```python
def from_numpy(array: np.ndarray) -> GPUArray:
    """Create GPUArray from NumPy array (copies data to GPU)."""
```

**Example:**
```python
np_arr = np.random.randn(100, 100).astype(np.float32)
gpu_arr = gpk.from_numpy(np_arr)
```

---

## Elementwise Operations

All elementwise operations require arrays of the same shape and dtype.

### add

```python
def add(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise addition: c = a + b"""
```

### sub

```python
def sub(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise subtraction: c = a - b"""
```

### mul

```python
def mul(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise multiplication: c = a * b"""
```

### div

```python
def div(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise division: c = a / b"""
```

**Example:**
```python
a = gpk.ones((1024, 1024))
b = gpk.ones((1024, 1024)) * 2

c = gpk.add(a, b)   # or a + b
d = gpk.sub(a, b)   # or a - b
e = gpk.mul(a, b)   # or a * b
f = gpk.div(a, b)   # or a / b
```

---

## Math Operations

### exp

```python
def exp(a: GPUArray) -> GPUArray:
    """Element-wise exponential: e^x"""
```

### log

```python
def log(a: GPUArray) -> GPUArray:
    """Element-wise natural logarithm: ln(x)"""
```

### sin

```python
def sin(a: GPUArray) -> GPUArray:
    """Element-wise sine: sin(x)"""
```

### cos

```python
def cos(a: GPUArray) -> GPUArray:
    """Element-wise cosine: cos(x)"""
```

### sqrt

```python
def sqrt(a: GPUArray) -> GPUArray:
    """Element-wise square root: sqrt(x)"""
```

### rsqrt

```python
def rsqrt(a: GPUArray) -> GPUArray:
    """Element-wise reciprocal square root: 1/sqrt(x)"""
```

### abs

```python
def abs(a: GPUArray) -> GPUArray:
    """Element-wise absolute value: |x|"""
```

### neg

```python
def neg(a: GPUArray) -> GPUArray:
    """Element-wise negation: -x"""
```

**Example:**
```python
a = gpk.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
b = gpk.exp(a)   # [e^1, e^2, e^3]
c = gpk.log(a)   # [0, ln(2), ln(3)]
d = gpk.sin(a)   # [sin(1), sin(2), sin(3)]
e = gpk.cos(a)   # [cos(1), cos(2), cos(3)]
f = gpk.sqrt(a)  # [1, 1.414, 1.732]
g = gpk.rsqrt(a) # [1, 0.707, 0.577]
```

---

## Comparison Operations

### clamp

```python
def clamp(a: GPUArray, min_val: float, max_val: float) -> GPUArray:
    """Clamp values to range [min_val, max_val]."""
```

### where

```python
def where(cond: GPUArray, x: GPUArray, y: GPUArray) -> GPUArray:
    """Element-wise conditional: cond ? x : y"""
```

**Example:**
```python
x = gpk.from_numpy(np.array([-2.0, 0.5, 3.0], dtype=np.float32))

# Clamp to [-1, 1]
y = gpk.clamp(x, -1.0, 1.0)  # [-1.0, 0.5, 1.0]

# Conditional selection
cond = gpk.from_numpy(np.array([1.0, 0.0, 1.0], dtype=np.float32))
a = gpk.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
b = gpk.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float32))
result = gpk.where(cond, a, b)  # [1.0, 5.0, 3.0]
```

---

## Activation Functions

### relu

```python
def relu(a: GPUArray) -> GPUArray:
    """ReLU activation: max(0, x)"""
```

### gelu

```python
def gelu(a: GPUArray) -> GPUArray:
    """GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
```

### sigmoid

```python
def sigmoid(a: GPUArray) -> GPUArray:
    """Sigmoid activation: 1 / (1 + exp(-x))"""
```

### tanh

```python
def tanh(a: GPUArray) -> GPUArray:
    """Hyperbolic tangent activation: tanh(x)"""
```

**Example:**
```python
x = gpk.from_numpy(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))
y_relu = gpk.relu(x)     # [0, 0, 1, 2]
y_gelu = gpk.gelu(x)     # [-0.159, 0, 0.841, 1.955]
y_sigmoid = gpk.sigmoid(x)  # [0.269, 0.5, 0.731, 0.881]
y_tanh = gpk.tanh(x)     # [-0.762, 0, 0.762, 0.964]
```

---

## Matrix Operations

### matmul

```python
def matmul(
    a: GPUArray,
    b: GPUArray,
    *,
    use_tf32: bool | None = None,
) -> GPUArray:
    """Matrix multiplication: C = A @ B

    Args:
        a: First matrix [M, K]
        b: Second matrix [K, N]
        use_tf32: TF32 TensorCore mode (None=env var, True=force, False=disable)

    Returns:
        Result matrix [M, N]
    """
```

**Example:**
```python
a = gpk.from_numpy(np.random.randn(4096, 4096).astype(np.float32))
b = gpk.from_numpy(np.random.randn(4096, 4096).astype(np.float32))

# Default: uses CUTLASS TF32 TensorCore (~30 TFLOPS)
c = a @ b

# Force TF32 mode
c = gpk.matmul(a, b, use_tf32=True)

# Force FP32 mode (full precision)
c = gpk.matmul(a, b, use_tf32=False)
```

### transpose

```python
def transpose(a: GPUArray) -> GPUArray:
    """Matrix transpose: B = A.T

    Args:
        a: Input matrix [rows, cols]

    Returns:
        Transposed matrix [cols, rows]
    """
```

**Example:**
```python
a = gpk.from_numpy(np.random.randn(100, 200).astype(np.float32))
b = gpk.transpose(a)  # shape: (200, 100)
```

---

## Reduction Operations

Reductions return a scalar GPUArray of shape `[1]`.

### sum

```python
def sum(a: GPUArray) -> GPUArray:
    """Sum of all elements."""
```

### mean

```python
def mean(a: GPUArray) -> GPUArray:
    """Mean of all elements."""
```

### max

```python
def max(a: GPUArray) -> GPUArray:
    """Maximum element."""
```

### min

```python
def min(a: GPUArray) -> GPUArray:
    """Minimum element."""
```

### argmax

```python
def argmax(a: GPUArray) -> GPUArray:
    """Index of maximum element."""
```

### sum_axis

```python
def sum_axis(a: GPUArray, axis: int) -> GPUArray:
    """Sum along specified axis.

    Args:
        a: Input array
        axis: Axis to reduce (0 for rows, 1 for columns)

    Returns:
        Reduced array with axis removed
    """
```

**Example:**
```python
a = gpk.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

total = gpk.sum(a)      # [10.0]
avg = gpk.mean(a)       # [2.5]
maximum = gpk.max(a)    # [4.0]
minimum = gpk.min(a)    # [1.0]
max_idx = gpk.argmax(a) # [3] (index of 4.0)

# Get scalar value
print(total.to_numpy()[0])  # 10.0

# Sum along axis
mat = gpk.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
row_sum = gpk.sum_axis(mat, axis=1)  # [3, 7]
col_sum = gpk.sum_axis(mat, axis=0)  # [4, 6]
```

---

## Neural Network Operations

### layernorm

```python
def layernorm(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float = 1e-5,
) -> GPUArray:
    """Layer normalization.

    Computes: (x - mean) / sqrt(var + eps) * gamma + beta

    Args:
        input: Input tensor [batch, features]
        gamma: Scale parameter [features]
        beta: Bias parameter [features]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor [batch, features]
    """
```

**Example:**
```python
batch, features = 32, 768
x = gpk.from_numpy(np.random.randn(batch, features).astype(np.float32))
gamma = gpk.ones(features)
beta = gpk.zeros(features)

y = gpk.layernorm(x, gamma, beta, eps=1e-5)
```

### bias_add_inplace

```python
def bias_add_inplace(output: GPUArray, bias: GPUArray) -> None:
    """Add bias to output in-place.

    Computes: output[batch, features] += bias[features]

    Args:
        output: Output tensor [batch, features] (modified in-place)
        bias: Bias tensor [features]
    """
```

**Example:**
```python
output = gpk.from_numpy(np.random.randn(32, 768).astype(np.float32))
bias = gpk.from_numpy(np.random.randn(768).astype(np.float32))

gpk.bias_add_inplace(output, bias)  # output modified in-place
```

### linear_bias_gelu

```python
def linear_bias_gelu(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray,
) -> GPUArray:
    """Fused Linear + Bias + GELU operation.

    Computes: output = gelu(input @ weight.T + bias)

    Uses CUTLASS epilogue fusion when dimensions are multiples of 16.

    Args:
        input: Input tensor [batch, in_features]
        weight: Weight tensor [out_features, in_features]
        bias: Bias tensor [out_features]

    Returns:
        Output tensor [batch, out_features]
    """
```

**Example:**
```python
batch, in_feat, out_feat = 512, 768, 3072

input = gpk.from_numpy(np.random.randn(batch, in_feat).astype(np.float32))
weight = gpk.from_numpy(np.random.randn(out_feat, in_feat).astype(np.float32))
bias = gpk.from_numpy(np.random.randn(out_feat).astype(np.float32))

# Fused operation (single GPU kernel)
output = gpk.linear_bias_gelu(input, weight, bias)

# Equivalent unfused operations (multiple GPU kernels)
# output = gpk.gelu(gpk.matmul(input, gpk.transpose(weight)) + bias)
```

---

## FP8 Operations (SM120+)

FP8 E4M3 GEMM operations for Blackwell GPUs (RTX 5090, B100, B200).

### fp8_fp8_sm120_available

```python
def fp8_fp8_sm120_available() -> bool:
    """Check if FP8 I/O GEMM is available (requires SM120+)."""
```

### fp8_fp8_get_scale_sizes

```python
def fp8_fp8_get_scale_sizes(M: int, N: int, K: int) -> tuple[int, int]:
    """Get required scale factor sizes for blockwise FP8 GEMM.

    Args:
        M: Number of rows in A
        N: Number of columns in B
        K: Inner dimension

    Returns:
        Tuple of (scale_A_size, scale_B_size)
    """
```

### matmul_fp8_fp8_sm120

```python
def matmul_fp8_fp8_sm120(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 E4M3 GEMM with unity scaling.

    Args:
        a: FP8 E4M3 matrix [M, K] (stored as uint8)
        b: FP8 E4M3 matrix [K, N] (stored as uint8)
        out: Optional output buffer [M, N]

    Returns:
        FP8 E4M3 result [M, N] (stored as uint8)
    """
```

### matmul_fp8_fp8_blockwise_sm120

```python
def matmul_fp8_fp8_blockwise_sm120(
    a: GPUArray,
    b: GPUArray,
    scale_a: GPUArray,
    scale_b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 E4M3 GEMM with blockwise scaling.

    For FP8 models (Llama 3.1 FP8, Qwen FP8, etc.) that store
    per-block scale factors alongside quantized weights.

    Args:
        a: FP8 E4M3 matrix [M, K] (stored as uint8)
        b: FP8 E4M3 matrix [K, N] (stored as uint8)
        scale_a: Scale factors for A (size from fp8_fp8_get_scale_sizes)
        scale_b: Scale factors for B (size from fp8_fp8_get_scale_sizes)
        out: Optional output buffer [M, N]

    Returns:
        FP8 E4M3 result [M, N] (stored as uint8)

    Note:
        Minimum matrix size is 128x128x128 due to CUTLASS tile requirements.
    """
```

**Example:**
```python
import pygpukit as gpk
import numpy as np

if gpk.fp8_fp8_sm120_available():
    M, N, K = 4096, 4096, 4096

    # Create FP8 data (stored as uint8)
    A = gpk.from_numpy(np.random.randint(0, 255, (M, K), dtype=np.uint8))
    B = gpk.from_numpy(np.random.randint(0, 255, (K, N), dtype=np.uint8))

    # Get scale sizes and create scale factors
    sfa_size, sfb_size = gpk.fp8_fp8_get_scale_sizes(M, N, K)
    scale_A = gpk.from_numpy(np.ones(sfa_size, dtype=np.float32))
    scale_B = gpk.from_numpy(np.ones(sfb_size, dtype=np.float32))

    # Blockwise scaled FP8 GEMM
    C = gpk.matmul_fp8_fp8_blockwise_sm120(A, B, scale_A, scale_B)
```

---

## Device Information

### is_cuda_available

```python
def is_cuda_available() -> bool:
    """Check if CUDA is available."""
```

### get_device_info

```python
def get_device_info() -> DeviceInfo:
    """Get current GPU device information."""
```

**DeviceInfo properties:**
- `name: str` - Device name
- `compute_capability: tuple[int, int]` - SM version (e.g., (8, 6))
- `total_memory: int` - Total VRAM in bytes
- `multiprocessor_count: int` - Number of SMs

### get_device_capabilities

```python
def get_device_capabilities() -> DeviceCapabilities:
    """Get device capability details."""
```

**Example:**
```python
if gpk.is_cuda_available():
    info = gpk.get_device_info()
    print(f"GPU: {info.name}")
    print(f"SM: {info.compute_capability}")
    print(f"VRAM: {info.total_memory / 1e9:.1f} GB")

    caps = gpk.get_device_capabilities()
    print(f"TensorCore: {caps.has_tensor_core}")
```

---

## JIT Compilation

### is_nvrtc_available

```python
def is_nvrtc_available() -> bool:
    """Check if NVRTC (JIT compiler) is available."""
```

### jit

```python
def jit(
    source: str,
    func: str,
    options: list[str] | None = None,
) -> JITKernel:
    """Compile CUDA kernel from source.

    Args:
        source: CUDA C++ source code
        func: Kernel function name
        options: Compiler options (e.g., ["-arch=sm_86"])

    Returns:
        Compiled kernel object
    """
```

**Example:**
```python
source = '''
extern "C" __global__
void scale(float* x, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= factor;
}
'''

if gpk.is_nvrtc_available():
    kernel = gpk.jit(source, func="scale")

    x = gpk.ones(1024)
    kernel(x, factor=2.0, n=1024)
    # x is now all 2.0
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYGPUKIT_NO_TF32` | Disable TF32 TensorCore for matmul | Not set (TF32 enabled) |
| `PYGPUKIT_NO_CUTLASS` | Disable CUTLASS, use fallback kernels | Not set |

**Example:**
```bash
# Force full FP32 precision (disable TF32)
PYGPUKIT_NO_TF32=1 python my_script.py
```
