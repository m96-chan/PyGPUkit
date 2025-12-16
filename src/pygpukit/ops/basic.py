"""Basic operations for GPUArrays."""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy


def _validate_same_shape(a: GPUArray, b: GPUArray, op_name: str) -> None:
    """Validate that two arrays have the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"{op_name} requires arrays of same shape, got {a.shape} and {b.shape}")


def _validate_same_dtype(a: GPUArray, b: GPUArray, op_name: str) -> None:
    """Validate that two arrays have the same dtype."""
    if a.dtype != b.dtype:
        raise ValueError(f"{op_name} requires arrays of same dtype, got {a.dtype} and {b.dtype}")


def _validate_float_dtype(a: GPUArray, op_name: str) -> None:
    """Validate that array has float dtype."""
    from pygpukit.core.dtypes import bfloat16, float16, float32, float64

    if a.dtype not in (float32, float64, float16, bfloat16):
        raise ValueError(f"{op_name} requires float dtype, got {a.dtype}")


def add(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise addition of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        A new GPUArray containing the element-wise sum.

    Raises:
        ValueError: If shapes don't match.
    """
    _validate_same_shape(a, b, "add")
    _validate_same_dtype(a, b, "add")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        # Fast path: use native operations with zero-copy
        return _add_native(a, b)
    else:
        # CPU simulation
        return _add_cpu(a, b)


def _add_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of add."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = a_np + b_np
    return from_numpy(result_np)


def _add_native(a: GPUArray, b: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of add (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays (zero-copy if already native)
    a_native = a._get_native()
    b_native = b._get_native()

    # Perform operation on GPU
    c_native = native.add(a_native, b_native)

    # Wrap result (zero-copy)
    return GPUArray._wrap_native(c_native)


def mul(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise multiplication of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        A new GPUArray containing the element-wise product.

    Raises:
        ValueError: If shapes don't match.
    """
    _validate_same_shape(a, b, "mul")
    _validate_same_dtype(a, b, "mul")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _mul_native(a, b)
    else:
        return _mul_cpu(a, b)


def _mul_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of mul."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = a_np * b_np
    return from_numpy(result_np)


def _mul_native(a: GPUArray, b: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of mul (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays (zero-copy if already native)
    a_native = a._get_native()
    b_native = b._get_native()

    # Perform operation on GPU
    c_native = native.mul(a_native, b_native)

    # Wrap result (zero-copy)
    return GPUArray._wrap_native(c_native)


def sub(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise subtraction of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        A new GPUArray containing the element-wise difference.

    Raises:
        ValueError: If shapes don't match.
    """
    _validate_same_shape(a, b, "sub")
    _validate_same_dtype(a, b, "sub")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _sub_native(a, b)
    else:
        return _sub_cpu(a, b)


def _sub_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of sub."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = a_np - b_np
    return from_numpy(result_np)


def _sub_native(a: GPUArray, b: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of sub (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    b_native = b._get_native()
    c_native = native.sub(a_native, b_native)
    return GPUArray._wrap_native(c_native)


def div(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise division of two arrays.

    Args:
        a: First input array (dividend).
        b: Second input array (divisor).

    Returns:
        A new GPUArray containing the element-wise quotient.

    Raises:
        ValueError: If shapes don't match.
    """
    _validate_same_shape(a, b, "div")
    _validate_same_dtype(a, b, "div")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _div_native(a, b)
    else:
        return _div_cpu(a, b)


def _div_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of div."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = a_np / b_np
    return from_numpy(result_np)


def _div_native(a: GPUArray, b: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of div (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    b_native = b._get_native()
    c_native = native.div(a_native, b_native)
    return GPUArray._wrap_native(c_native)


def exp(a: GPUArray) -> GPUArray:
    """Element-wise exponential.

    Args:
        a: Input array (float32 or float64).

    Returns:
        A new GPUArray containing exp(a).

    Raises:
        ValueError: If dtype is not float32 or float64.
    """
    _validate_float_dtype(a, "exp")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _exp_native(a)
    else:
        return _exp_cpu(a)


def _exp_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of exp."""
    a_np = a.to_numpy()
    result_np = np.exp(a_np)
    return from_numpy(result_np)


def _exp_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of exp (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.exp(a_native)
    return GPUArray._wrap_native(c_native)


def log(a: GPUArray) -> GPUArray:
    """Element-wise natural logarithm.

    Args:
        a: Input array (float32 or float64).

    Returns:
        A new GPUArray containing log(a).

    Raises:
        ValueError: If dtype is not float32 or float64.
    """
    _validate_float_dtype(a, "log")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _log_native(a)
    else:
        return _log_cpu(a)


def _log_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of log."""
    a_np = a.to_numpy()
    result_np = np.log(a_np)
    return from_numpy(result_np)


def _log_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of log (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.log(a_native)
    return GPUArray._wrap_native(c_native)


def relu(a: GPUArray) -> GPUArray:
    """Element-wise ReLU (Rectified Linear Unit).

    Computes max(0, x) for each element.

    Args:
        a: Input array (float32 or float64).

    Returns:
        A new GPUArray containing relu(a).

    Raises:
        ValueError: If dtype is not float32 or float64.
    """
    _validate_float_dtype(a, "relu")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _relu_native(a)
    else:
        return _relu_cpu(a)


def _relu_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of relu."""
    a_np = a.to_numpy()
    result_np = np.maximum(0, a_np)
    return from_numpy(result_np)


def _relu_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of relu (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.relu(a_native)
    return GPUArray._wrap_native(c_native)


def matmul(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
    use_tf32: bool | None = None,
) -> GPUArray:
    """Matrix multiplication of two 2D arrays.

    Args:
        a: First input array (M x K).
        b: Second input array (K x N).
        out: Optional output array (M x N). If provided, result is written to this
            array instead of allocating a new one. This enables CUDA Graph capture
            since no memory allocation occurs during the operation.
        use_tf32: Whether to use TF32 TensorCore acceleration (Ampere+ only).
            - None (default): Use PYGPUKIT_ALLOW_TF32 environment variable
            - True: Force TF32 mode (requires SM >= 80 and float32)
            - False: Force FP32 mode

    Returns:
        The result GPUArray (M x N). If out is provided, returns out.

    Raises:
        ValueError: If arrays are not 2D or dimensions don't match.
        RuntimeError: If use_tf32=True but GPU doesn't support it or dtype is not float32.

    Example:
        # Allocate new output
        y = pk.matmul(x, W)

        # Write to existing buffer (for CUDA Graph capture)
        pk.matmul(x, W, out=y)
    """
    if a.ndim != 2:
        raise ValueError(f"matmul requires 2D arrays, got {a.ndim}D for first argument")
    if b.ndim != 2:
        raise ValueError(f"matmul requires 2D arrays, got {b.ndim}D for second argument")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul dimension mismatch: {a.shape} @ {b.shape} "
            f"(inner dimensions {a.shape[1]} and {b.shape[0]} must match)"
        )

    _validate_same_dtype(a, b, "matmul")

    # Validate out array if provided
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(f"out shape {out.shape} does not match expected {expected_shape}")
        if out.dtype != a.dtype:
            raise ValueError(f"out dtype {out.dtype} does not match input dtype {a.dtype}")

    # Check TF32 dtype requirement early (before backend dispatch)
    if use_tf32 is True:
        from pygpukit.core.dtypes import float32

        if a.dtype != float32:
            raise RuntimeError("TF32 matmul requires float32 dtype")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_native(a, b, out=out, use_tf32=use_tf32)
    else:
        return _matmul_cpu(a, b, out=out)


def _matmul_cpu(a: GPUArray, b: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """CPU implementation of matmul."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    if out is not None:
        out_np = out.to_numpy()
        np.matmul(a_np, b_np, out=out_np)
        # Copy back to GPU - this is inefficient but CPU backend is for fallback only
        out._data = from_numpy(out_np)._data
        return out
    else:
        result_np = np.matmul(a_np, b_np)
        return from_numpy(result_np)


def _matmul_native(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
    use_tf32: bool | None = None,
) -> GPUArray:
    """Native C++ CUDA implementation of matmul (zero-copy).

    Args:
        a: First input array.
        b: Second input array.
        out: Optional output array. If provided, result is written in-place.
        use_tf32: Whether to use TF32 TensorCore acceleration.
            None means use environment variable PYGPUKIT_ALLOW_TF32.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays (zero-copy if already native)
    a_native = a._get_native()
    b_native = b._get_native()

    if out is not None:
        # In-place operation - write to existing buffer
        out_native = out._get_native()
        if use_tf32 is not None:
            native.matmul_tf32_(a_native, b_native, out_native, use_tf32)
        else:
            native.matmul_(a_native, b_native, out_native)
        return out
    else:
        # Allocate new output
        if use_tf32 is not None:
            c_native = native.matmul_tf32(a_native, b_native, use_tf32)
        else:
            c_native = native.matmul(a_native, b_native)
        return GPUArray._wrap_native(c_native)


# ============================================================================
# Reduction Operations
# ============================================================================


def sum(a: GPUArray) -> GPUArray:
    """Sum of all elements.

    Args:
        a: Input array (float32 or float64).

    Returns:
        A scalar GPUArray (shape [1]) containing the sum.

    Raises:
        ValueError: If dtype is not float32 or float64.
    """
    _validate_float_dtype(a, "sum")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _sum_native(a)
    else:
        return _sum_cpu(a)


def _sum_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of sum."""
    a_np = a.to_numpy()
    result_np = np.array([np.sum(a_np)], dtype=a_np.dtype)
    return from_numpy(result_np)


def _sum_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of sum (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.sum(a_native)
    return GPUArray._wrap_native(c_native)


def mean(a: GPUArray) -> GPUArray:
    """Mean of all elements.

    Args:
        a: Input array (float32 or float64).

    Returns:
        A scalar GPUArray (shape [1]) containing the mean.

    Raises:
        ValueError: If dtype is not float32 or float64.
    """
    _validate_float_dtype(a, "mean")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _mean_native(a)
    else:
        return _mean_cpu(a)


def _mean_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of mean."""
    a_np = a.to_numpy()
    result_np = np.array([np.mean(a_np)], dtype=a_np.dtype)
    return from_numpy(result_np)


def _mean_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of mean (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.mean(a_native)
    return GPUArray._wrap_native(c_native)


def max(a: GPUArray) -> GPUArray:
    """Max of all elements.

    Args:
        a: Input array (float32 or float64).

    Returns:
        A scalar GPUArray (shape [1]) containing the maximum value.

    Raises:
        ValueError: If dtype is not float32 or float64.
    """
    _validate_float_dtype(a, "max")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _max_native(a)
    else:
        return _max_cpu(a)


def _max_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of max."""
    a_np = a.to_numpy()
    result_np = np.array([np.max(a_np)], dtype=a_np.dtype)
    return from_numpy(result_np)


def _max_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of max (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.max(a_native)
    return GPUArray._wrap_native(c_native)


# ============================================================================
# Neural Network Operations
# ============================================================================


def gelu(a: GPUArray) -> GPUArray:
    """GELU (Gaussian Error Linear Unit) activation.

    Computes: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        a: Input array (float32, float64, float16, or bfloat16).

    Returns:
        A new GPUArray containing gelu(a).

    Raises:
        ValueError: If dtype is not a float type.
    """
    _validate_float_dtype(a, "gelu")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _gelu_native(a)
    else:
        return _gelu_cpu(a)


def _gelu_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of gelu."""
    a_np = a.to_numpy()
    # GELU approximation
    x = a_np.astype(np.float32) if a_np.dtype in [np.float16] else a_np
    c1 = 0.7978845608  # sqrt(2/pi)
    c2 = 0.044715
    result = x * 0.5 * (1 + np.tanh(c1 * (x + c2 * x**3)))
    return from_numpy(result.astype(a_np.dtype))


def _gelu_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of gelu (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.gelu(a_native)
    return GPUArray._wrap_native(c_native)


def layernorm(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float = 1e-5,
) -> GPUArray:
    """Layer normalization.

    Computes: (x - mean) / sqrt(var + eps) * gamma + beta

    Args:
        input: Input array of shape [batch, features].
        gamma: Scale parameter of shape [features].
        beta: Bias parameter of shape [features].
        eps: Small epsilon for numerical stability.

    Returns:
        A new GPUArray containing the normalized output.

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(input, "layernorm")

    if input.ndim != 2:
        raise ValueError(f"layernorm expects 2D input [batch, features], got {input.ndim}D")
    if gamma.ndim != 1 or beta.ndim != 1:
        raise ValueError("layernorm expects 1D gamma and beta")
    if input.dtype != gamma.dtype or input.dtype != beta.dtype:
        raise ValueError("layernorm: all inputs must have same dtype")

    features = input.shape[1]
    if gamma.shape[0] != features or beta.shape[0] != features:
        raise ValueError(
            f"layernorm: gamma/beta size {gamma.shape[0]} must match features {features}"
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _layernorm_native(input, gamma, beta, eps)
    else:
        return _layernorm_cpu(input, gamma, beta, eps)


def _layernorm_cpu(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float,
) -> GPUArray:
    """CPU implementation of layernorm."""
    x = input.to_numpy()
    g = gamma.to_numpy()
    b = beta.to_numpy()

    # Compute mean and variance along features axis
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)

    # Normalize
    normalized = (x - mean) / np.sqrt(var + eps)

    # Apply affine transform
    result = normalized * g + b
    return from_numpy(result)


def _layernorm_native(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float,
) -> GPUArray:
    """Native C++ CUDA implementation of layernorm (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    gamma_native = gamma._get_native()
    beta_native = beta._get_native()
    c_native = native.layernorm(input_native, gamma_native, beta_native, eps)
    return GPUArray._wrap_native(c_native)


def softmax(input: GPUArray) -> GPUArray:
    """Softmax activation applied row-wise.

    Computes: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))

    Args:
        input: Input array of shape [batch, features].

    Returns:
        A new GPUArray containing the softmax output.

    Raises:
        ValueError: If input is not 2D or dtype is not a float type.
    """
    _validate_float_dtype(input, "softmax")

    if input.ndim != 2:
        raise ValueError(f"softmax expects 2D input [batch, features], got {input.ndim}D")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _softmax_native(input)
    else:
        return _softmax_cpu(input)


def _softmax_cpu(input: GPUArray) -> GPUArray:
    """CPU implementation of softmax."""
    x = input.to_numpy()
    # Numerical stability: subtract max
    x_max = x.max(axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return from_numpy(exp_x / exp_x.sum(axis=1, keepdims=True))


def _softmax_native(input: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of softmax (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    c_native = native.softmax(input_native)
    return GPUArray._wrap_native(c_native)


def transpose(a: GPUArray) -> GPUArray:
    """Matrix transpose.

    Args:
        a: Input array of shape [rows, cols].

    Returns:
        A new GPUArray of shape [cols, rows] containing a.T.

    Raises:
        ValueError: If input is not 2D or dtype is not a float type.
    """
    _validate_float_dtype(a, "transpose")

    if a.ndim != 2:
        raise ValueError(f"transpose expects 2D input [rows, cols], got {a.ndim}D")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _transpose_native(a)
    else:
        return _transpose_cpu(a)


def _transpose_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of transpose."""
    a_np = a.to_numpy()
    return from_numpy(a_np.T.copy())


def _transpose_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of transpose (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.transpose(a_native)
    return GPUArray._wrap_native(c_native)


def bias_add_inplace(output: GPUArray, bias: GPUArray) -> None:
    """Add bias to output in-place.

    Computes: output[batch, features] += bias[features]

    Args:
        output: Output array of shape [batch, features] (modified in-place).
        bias: Bias array of shape [features].

    Raises:
        ValueError: If shapes don't match or dtypes don't match.
    """
    _validate_float_dtype(output, "bias_add_inplace")

    if output.ndim != 2:
        raise ValueError(
            f"bias_add_inplace expects 2D output [batch, features], got {output.ndim}D"
        )
    if bias.ndim != 1:
        raise ValueError(f"bias_add_inplace expects 1D bias [features], got {bias.ndim}D")
    if output.dtype != bias.dtype:
        raise ValueError("bias_add_inplace: output and bias must have same dtype")

    features = output.shape[1]
    if bias.shape[0] != features:
        raise ValueError(
            f"bias_add_inplace: bias size {bias.shape[0]} must match features {features}"
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        _bias_add_inplace_native(output, bias)
    else:
        _bias_add_inplace_cpu(output, bias)


def _bias_add_inplace_cpu(output: GPUArray, bias: GPUArray) -> None:
    """CPU implementation of bias_add_inplace."""
    # For CPU backend, we need to get numpy arrays, modify, and update
    output_np = output.to_numpy()
    bias_np = bias.to_numpy()
    output_np += bias_np
    # Note: This creates a new array - for CPU backend, in-place is not truly in-place
    # The native backend does true in-place modification
    output._data = from_numpy(output_np)._data


def _bias_add_inplace_native(output: GPUArray, bias: GPUArray) -> None:
    """Native C++ CUDA implementation of bias_add_inplace (true in-place)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    output_native = output._get_native()
    bias_native = bias._get_native()
    native.bias_add_inplace(output_native, bias_native)


# ============================================================================
# Fused Operations (CUTLASS Epilogue Fusion)
# ============================================================================


def rmsnorm(
    input: GPUArray,
    gamma: GPUArray,
    eps: float = 1e-5,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """RMS Normalization (Root Mean Square Normalization).

    Computes: x / sqrt(mean(x^2) + eps) * gamma

    Simpler than LayerNorm (no mean subtraction, no beta).
    Used in Llama and other modern LLMs.

    Args:
        input: Input array of shape [batch, features].
        gamma: Scale parameter of shape [features].
        eps: Small epsilon for numerical stability.
        out: Optional output buffer. If provided, result is written in-place
            (for CUDA Graph capture).

    Returns:
        A new GPUArray containing the normalized output (or out if provided).

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(input, "rmsnorm")

    if input.ndim != 2:
        raise ValueError(f"rmsnorm expects 2D input [batch, features], got {input.ndim}D")
    if gamma.ndim != 1:
        raise ValueError("rmsnorm expects 1D gamma")
    if input.dtype != gamma.dtype:
        raise ValueError("rmsnorm: all inputs must have same dtype")

    features = input.shape[1]
    if gamma.shape[0] != features:
        raise ValueError(f"rmsnorm: gamma size {gamma.shape[0]} must match features {features}")

    # Validate out array if provided
    if out is not None:
        if out.shape != input.shape:
            raise ValueError(f"out shape {out.shape} does not match input shape {input.shape}")
        if out.dtype != input.dtype:
            raise ValueError(f"out dtype {out.dtype} does not match input dtype {input.dtype}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _rmsnorm_native(input, gamma, eps, out=out)
    else:
        return _rmsnorm_cpu(input, gamma, eps, out=out)


def _rmsnorm_cpu(
    input: GPUArray,
    gamma: GPUArray,
    eps: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """CPU implementation of rmsnorm."""
    x = input.to_numpy()
    g = gamma.to_numpy()

    # RMS = sqrt(mean(x^2) + eps)
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + eps)

    # Normalize and scale
    result = (x / rms) * g

    if out is not None:
        out_np = out.to_numpy()
        np.copyto(out_np, result)
        out._data = from_numpy(out_np)._data
        return out
    return from_numpy(result)


def _rmsnorm_native(
    input: GPUArray,
    gamma: GPUArray,
    eps: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ CUDA implementation of rmsnorm (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    gamma_native = gamma._get_native()

    if out is not None:
        out_native = out._get_native()
        native.rmsnorm_(input_native, gamma_native, out_native, eps)
        return out
    else:
        c_native = native.rmsnorm(input_native, gamma_native, eps)
        return GPUArray._wrap_native(c_native)


def linear_bias_gelu(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray,
) -> GPUArray:
    """Fused linear + bias + GELU operation.

    Computes: output = gelu(input @ weight^T + bias)

    When dimensions are multiples of 16, this uses CUTLASS TensorCore
    epilogue fusion for efficiency. Otherwise, falls back to separate
    matmul + bias_add + gelu operations.

    Args:
        input: Input array of shape [batch, in_features].
        weight: Weight array of shape [out_features, in_features].
        bias: Bias array of shape [out_features].

    Returns:
        A new GPUArray of shape [batch, out_features].

    Raises:
        ValueError: If shapes or dtypes don't match.

    Note:
        Best performance when dimensions are multiples of 16 (uses TensorCore).
        Non-aligned dimensions use native fallback path.
    """
    _validate_float_dtype(input, "linear_bias_gelu")

    if input.ndim != 2:
        raise ValueError(
            f"linear_bias_gelu expects 2D input [batch, in_features], got {input.ndim}D"
        )
    if weight.ndim != 2:
        raise ValueError(
            f"linear_bias_gelu expects 2D weight [out_features, in_features], got {weight.ndim}D"
        )
    if bias.ndim != 1:
        raise ValueError(f"linear_bias_gelu expects 1D bias [out_features], got {bias.ndim}D")

    if input.dtype != weight.dtype or input.dtype != bias.dtype:
        raise ValueError("linear_bias_gelu: all inputs must have same dtype")

    in_features = input.shape[1]
    out_features = weight.shape[0]

    if weight.shape[1] != in_features:
        raise ValueError(
            f"linear_bias_gelu: weight.shape[1]={weight.shape[1]} must match "
            f"input.shape[1]={in_features}"
        )
    if bias.shape[0] != out_features:
        raise ValueError(
            f"linear_bias_gelu: bias.shape[0]={bias.shape[0]} must match "
            f"weight.shape[0]={out_features}"
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _linear_bias_gelu_native(input, weight, bias)
    else:
        return _linear_bias_gelu_cpu(input, weight, bias)


def _linear_bias_gelu_cpu(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray,
) -> GPUArray:
    """CPU implementation of linear_bias_gelu."""
    x = input.to_numpy()
    w = weight.to_numpy()
    b = bias.to_numpy()

    # Linear: y = x @ w.T + b
    y = x @ w.T + b

    # GELU approximation (same as GPU kernel)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    result = y * 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (y + 0.044715 * y**3)))

    return from_numpy(result.astype(x.dtype))


def _linear_bias_gelu_native(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray,
) -> GPUArray:
    """Native C++ CUDA implementation of linear_bias_gelu (CUTLASS fused kernel)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    weight_native = weight._get_native()
    bias_native = bias._get_native()
    c_native = native.linear_bias_gelu(input_native, weight_native, bias_native)
    return GPUArray._wrap_native(c_native)


# ============================================================================
# Additional Neural Network Operations
# ============================================================================


def silu(a: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """SiLU (Swish) activation: y = x * sigmoid(x).

    Used in Llama and other modern LLMs as the activation in MLP layers.

    Args:
        a: Input array.
        out: Optional pre-allocated output array. If provided, the result
            is written to this array (for CUDA Graph capture support).

    Returns:
        A new GPUArray containing the SiLU-activated values, or the out array if provided.

    Raises:
        ValueError: If dtype is not a float type.
    """
    _validate_float_dtype(a, "silu")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _silu_native(a, out=out)
    else:
        return _silu_cpu(a)


def _silu_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of SiLU."""
    x = a.to_numpy()
    # SiLU = x * sigmoid(x) = x / (1 + exp(-x))
    result = x / (1.0 + np.exp(-x))
    return from_numpy(result)


def _silu_native(a: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """Native C++ CUDA implementation of SiLU (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()

    if out is not None:
        out_native = out._get_native()
        native.silu_(a_native, out_native)
        return out
    else:
        c_native = native.silu(a_native)
        return GPUArray._wrap_native(c_native)


def sdpa_causal(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    scale: float = 0.0,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Scaled Dot-Product Attention with causal mask.

    Computes attention with automatic causal masking for autoregressive
    sequence generation. This is the core attention operation used in
    transformer models.

    Algorithm:
        scores = Q @ K^T / scale
        scores = apply_causal_mask(scores)
        weights = softmax(scores)
        output = weights @ V

    Args:
        Q: Query tensor of shape [n_heads, q_len, head_dim].
        K: Key tensor of shape [n_heads, kv_len, head_dim].
        V: Value tensor of shape [n_heads, kv_len, head_dim].
        scale: Scaling factor (typically 1/sqrt(head_dim)).
               If <= 0, computed automatically from head_dim.
        out: Optional output buffer [n_heads, q_len, head_dim].
             If provided, result is written in-place (for CUDA Graph capture).

    Returns:
        Output tensor of shape [n_heads, q_len, head_dim].

    Raises:
        ValueError: If shapes or dtypes don't match.

    Note:
        For KV cache usage during inference, kv_len >= q_len.
        The causal mask ensures query at position i can only attend
        to key positions 0 to (kv_len - q_len + i).
    """
    _validate_float_dtype(Q, "sdpa_causal")

    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("sdpa_causal expects 3D inputs [n_heads, seq_len, head_dim]")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError("sdpa_causal: Q, K, V must have same dtype")

    n_heads, q_len, head_dim = Q.shape

    if K.shape[0] != n_heads or V.shape[0] != n_heads:
        raise ValueError("sdpa_causal: n_heads mismatch")
    if K.shape[2] != head_dim or V.shape[2] != head_dim:
        raise ValueError("sdpa_causal: head_dim mismatch")
    if K.shape[1] != V.shape[1]:
        raise ValueError("sdpa_causal: K and V seq_len mismatch")

    # Validate out array if provided
    if out is not None:
        if out.shape != (n_heads, q_len, head_dim):
            raise ValueError(
                f"out shape {out.shape} does not match expected {(n_heads, q_len, head_dim)}"
            )
        if out.dtype != Q.dtype:
            raise ValueError(f"out dtype {out.dtype} does not match Q dtype {Q.dtype}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _sdpa_causal_native(Q, K, V, scale, out=out)
    else:
        return _sdpa_causal_cpu(Q, K, V, scale, out=out)


def _sdpa_causal_cpu(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    scale: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """CPU implementation of SDPA with causal mask."""
    q = Q.to_numpy()
    k = K.to_numpy()
    v = V.to_numpy()

    n_heads, q_len, head_dim = q.shape
    kv_len = k.shape[1]

    if scale <= 0:
        scale = 1.0 / np.sqrt(head_dim)

    # scores: [n_heads, q_len, kv_len]
    scores = np.matmul(q, k.transpose(0, 2, 1)) * scale

    # Create causal mask
    causal_offset = kv_len - q_len
    for i in range(q_len):
        max_attend = causal_offset + i + 1
        if max_attend < kv_len:
            scores[:, i, max_attend:] = -np.inf

    # Softmax over last dimension
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # output: [n_heads, q_len, head_dim]
    output = np.matmul(weights, v)

    if out is not None:
        out_np = out.to_numpy()
        np.copyto(out_np, output.astype(q.dtype))
        out._data = from_numpy(out_np)._data
        return out
    return from_numpy(output.astype(q.dtype))


def _sdpa_causal_native(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    scale: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ CUDA implementation of SDPA with causal mask."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    q_native = Q._get_native()
    k_native = K._get_native()
    v_native = V._get_native()

    if out is not None:
        out_native = out._get_native()
        native.sdpa_causal_(q_native, k_native, v_native, out_native, scale)
        return out
    else:
        c_native = native.sdpa_causal(q_native, k_native, v_native, scale)
        return GPUArray._wrap_native(c_native)


def sdpa_causal_fixed_cache(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    out: GPUArray,
    context_len: int,
    scale: float = 0.0,
) -> None:
    """SDPA with fixed-length KV cache for CUDA Graph capture.

    This variant is designed for use with pre-allocated KV caches where
    the buffer size (max_seq_len) is larger than the actual context length.

    Args:
        Q: Query tensor of shape [n_heads, q_len, head_dim].
        K: Key cache of shape [n_heads, max_seq_len, head_dim].
        V: Value cache of shape [n_heads, max_seq_len, head_dim].
        out: Pre-allocated output buffer [n_heads, q_len, head_dim].
        context_len: Actual number of valid tokens in KV cache.
        scale: Scaling factor (typically 1/sqrt(head_dim)).
               If <= 0, computed automatically from head_dim.

    Raises:
        ValueError: If shapes or dtypes don't match, or context_len is invalid.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    q_native = Q._get_native()
    k_native = K._get_native()
    v_native = V._get_native()
    out_native = out._get_native()

    native.sdpa_causal_fixed_cache(q_native, k_native, v_native, out_native, context_len, scale)


def rope_inplace(
    q: GPUArray,
    k: GPUArray,
    cos: GPUArray,
    sin: GPUArray,
) -> None:
    """Apply Rotary Position Embedding (RoPE) to Q and K tensors in-place.

    Args:
        q: Query tensor of shape [seq_len, n_heads_q, head_dim] (modified in-place).
        k: Key tensor of shape [seq_len, n_heads_k, head_dim] (modified in-place).
        cos: Precomputed cosine of shape [seq_len, head_dim].
        sin: Precomputed sine of shape [seq_len, head_dim].

    Note:
        This operation modifies q and k in-place.
        Works with GQA (n_heads_k can be different from n_heads_q).
    """
    _validate_float_dtype(q, "rope_inplace")

    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("rope_inplace expects 3D q, k [seq_len, n_heads, head_dim]")
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("rope_inplace expects 2D cos, sin [seq_len, head_dim]")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        _rope_inplace_native(q, k, cos, sin)
    else:
        _rope_inplace_cpu(q, k, cos, sin)


def _rope_inplace_cpu(
    q: GPUArray,
    k: GPUArray,
    cos: GPUArray,
    sin: GPUArray,
) -> None:
    """CPU implementation of rope_inplace."""

    q_np = q.to_numpy()
    k_np = k.to_numpy()
    cos_np = cos.to_numpy()
    sin_np = sin.to_numpy()

    seq_len, n_heads_q, head_dim = q_np.shape
    n_heads_k = k_np.shape[1]
    half_dim = head_dim // 2

    # Apply RoPE to Q
    for s in range(seq_len):
        c = cos_np[s, :half_dim]
        sn = sin_np[s, :half_dim]
        for h in range(n_heads_q):
            q0 = q_np[s, h, :half_dim].copy()
            q1 = q_np[s, h, half_dim:].copy()
            q_np[s, h, :half_dim] = q0 * c - q1 * sn
            q_np[s, h, half_dim:] = q1 * c + q0 * sn

    # Apply RoPE to K
    for s in range(seq_len):
        c = cos_np[s, :half_dim]
        sn = sin_np[s, :half_dim]
        for h in range(n_heads_k):
            k0 = k_np[s, h, :half_dim].copy()
            k1 = k_np[s, h, half_dim:].copy()
            k_np[s, h, :half_dim] = k0 * c - k1 * sn
            k_np[s, h, half_dim:] = k1 * c + k0 * sn

    # Update the GPUArray data in-place
    q._data = from_numpy(q_np)._data
    k._data = from_numpy(k_np)._data


def _rope_inplace_native(
    q: GPUArray,
    k: GPUArray,
    cos: GPUArray,
    sin: GPUArray,
) -> None:
    """Native C++ CUDA implementation of rope_inplace."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    q_native = q._get_native()
    k_native = k._get_native()
    cos_native = cos._get_native()
    sin_native = sin._get_native()
    native.rope_inplace(q_native, k_native, cos_native, sin_native)


# ============================================================================
# Tensor Manipulation Operations
# ============================================================================


def concat_axis0(a: GPUArray, b: GPUArray) -> GPUArray:
    """Concatenate two tensors along axis 0.

    Args:
        a: First tensor of shape [dim0_a, ...].
        b: Second tensor of shape [dim0_b, ...].

    Returns:
        Concatenated tensor of shape [dim0_a + dim0_b, ...].

    Raises:
        ValueError: If shapes don't match along non-concatenation axes.
    """
    _validate_same_dtype(a, b, "concat_axis0")

    if a.ndim != b.ndim:
        raise ValueError(f"concat_axis0: dimension mismatch ({a.ndim}D vs {b.ndim}D)")

    for i in range(1, a.ndim):
        if a.shape[i] != b.shape[i]:
            raise ValueError(
                f"concat_axis0: shape mismatch at axis {i} ({a.shape[i]} vs {b.shape[i]})"
            )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _concat_axis0_native(a, b)
    else:
        return _concat_axis0_cpu(a, b)


def _concat_axis0_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of concat_axis0."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result = np.concatenate([a_np, b_np], axis=0)
    return from_numpy(result)


def _concat_axis0_native(a: GPUArray, b: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of concat_axis0."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    b_native = b._get_native()
    c_native = native.concat_axis0(a_native, b_native)
    return GPUArray._wrap_native(c_native)


def repeat_interleave_axis1(input: GPUArray, repeats: int) -> GPUArray:
    """Repeat tensor elements along axis 1 (interleaved).

    For GQA: expands [n_heads_kv, seq_len, head_dim] to [n_heads, seq_len, head_dim]
    by repeating each KV head `repeats` times.

    Args:
        input: Input tensor of shape [dim0, dim1, dim2].
        repeats: Number of times to repeat each element along axis 1.

    Returns:
        Tensor of shape [dim0, dim1 * repeats, dim2].
    """
    _validate_float_dtype(input, "repeat_interleave_axis1")

    if input.ndim != 3:
        raise ValueError(
            f"repeat_interleave_axis1 expects 3D input [d0, d1, d2], got {input.ndim}D"
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _repeat_interleave_axis1_native(input, repeats)
    else:
        return _repeat_interleave_axis1_cpu(input, repeats)


def _repeat_interleave_axis1_cpu(input: GPUArray, repeats: int) -> GPUArray:
    """CPU implementation of repeat_interleave_axis1."""
    x = input.to_numpy()
    # np.repeat with axis=1 gives interleaved repeat
    result = np.repeat(x, repeats, axis=1)
    return from_numpy(result)


def _repeat_interleave_axis1_native(input: GPUArray, repeats: int) -> GPUArray:
    """Native C++ CUDA implementation of repeat_interleave_axis1."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    c_native = native.repeat_interleave_axis1(input_native, repeats)
    return GPUArray._wrap_native(c_native)


def transpose_3d_021(input: GPUArray) -> GPUArray:
    """Transpose 3D tensor: [d0, d1, d2] -> [d1, d0, d2].

    Swaps axes 0 and 1 while keeping axis 2 in place.
    Useful for converting [seq_len, n_heads, head_dim] to [n_heads, seq_len, head_dim].

    Args:
        input: 3D tensor to transpose.

    Returns:
        Transposed tensor with axes 0 and 1 swapped.
    """
    _validate_float_dtype(input, "transpose_3d_021")

    if input.ndim != 3:
        raise ValueError(f"transpose_3d_021 expects 3D input, got {input.ndim}D")

    backend = get_backend()

    # Native transpose_3d_021 supports float32/float16/bfloat16
    if isinstance(backend, NativeBackend) and backend.is_available():
        dtype_str = str(input.dtype)
        if dtype_str in ("float32", "float16", "bfloat16"):
            return _transpose_3d_021_native(input)
        else:
            return _transpose_3d_021_cpu(input)
    else:
        return _transpose_3d_021_cpu(input)


def _transpose_3d_021_cpu(input: GPUArray) -> GPUArray:
    """CPU implementation of transpose_3d_021."""
    x = input.to_numpy()
    result = np.transpose(x, (1, 0, 2)).copy()
    return from_numpy(result)


def _transpose_3d_021_native(input: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of transpose_3d_021."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    c_native = native.transpose_3d_021(input_native)
    return GPUArray._wrap_native(c_native)


def reshape_copy(input: GPUArray, new_shape: tuple[int, ...]) -> GPUArray:
    """Reshape tensor with copy (ensures contiguous output).

    Args:
        input: Input tensor to reshape.
        new_shape: Target shape (total elements must match).

    Returns:
        Reshaped tensor with new shape.

    Raises:
        ValueError: If total element count doesn't match.
    """
    _validate_float_dtype(input, "reshape_copy")

    # Verify total size
    input_size = 1
    for dim in input.shape:
        input_size *= dim

    output_size = 1
    for dim in new_shape:
        output_size *= dim

    if input_size != output_size:
        raise ValueError(f"reshape_copy: total size mismatch ({input_size} vs {output_size})")

    backend = get_backend()

    # Native reshape_copy supports float32/float16/bfloat16
    if isinstance(backend, NativeBackend) and backend.is_available():
        dtype_str = str(input.dtype)
        if dtype_str in ("float32", "float16", "bfloat16"):
            return _reshape_copy_native(input, new_shape)
        else:
            return _reshape_copy_cpu(input, new_shape)
    else:
        return _reshape_copy_cpu(input, new_shape)


def _reshape_copy_cpu(input: GPUArray, new_shape: tuple[int, ...]) -> GPUArray:
    """CPU implementation of reshape_copy."""
    x = input.to_numpy()
    result = x.reshape(new_shape).copy()
    return from_numpy(result)


def _reshape_copy_native(input: GPUArray, new_shape: tuple[int, ...]) -> GPUArray:
    """Native C++ CUDA implementation of reshape_copy."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    c_native = native.reshape_copy(input_native, list(new_shape))
    return GPUArray._wrap_native(c_native)


# ============================================================================
# Fixed-Length KV Cache Operations (CUDA Graph Support)
# ============================================================================


def kv_cache_update(new_kv: GPUArray, cache: GPUArray, position: int) -> None:
    """Update KV cache at a single position (decode step).

    Used for fixed-length KV cache with CUDA Graph support.
    Copies new K or V values to a specific position in the pre-allocated cache.

    Args:
        new_kv: New K or V tensor of shape [1, num_kv_heads, head_dim].
        cache: Pre-allocated cache tensor of shape [max_seq_len, num_kv_heads, head_dim].
        position: Position index in cache where to write (0-indexed).

    Raises:
        ValueError: If shapes are incompatible.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    new_kv_native = new_kv._get_native()
    cache_native = cache._get_native()
    native.kv_cache_update(new_kv_native, cache_native, position)


def kv_cache_prefill(new_kv: GPUArray, cache: GPUArray, start_pos: int = 0) -> None:
    """Prefill KV cache from sequence (prefill step).

    Used for fixed-length KV cache with CUDA Graph support.
    Copies K or V values from prefill to the pre-allocated cache.

    Args:
        new_kv: K or V tensor from prefill of shape [seq_len, num_kv_heads, head_dim].
        cache: Pre-allocated cache tensor of shape [max_seq_len, num_kv_heads, head_dim].
        start_pos: Starting position in cache (default 0).

    Raises:
        ValueError: If shapes are incompatible.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    new_kv_native = new_kv._get_native()
    cache_native = cache._get_native()
    native.kv_cache_prefill(new_kv_native, cache_native, start_pos)
