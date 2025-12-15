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


def matmul(a: GPUArray, b: GPUArray, *, use_tf32: bool | None = None) -> GPUArray:
    """Matrix multiplication of two 2D arrays.

    Args:
        a: First input array (M x K).
        b: Second input array (K x N).
        use_tf32: Whether to use TF32 TensorCore acceleration (Ampere+ only).
            - None (default): Use PYGPUKIT_ALLOW_TF32 environment variable
            - True: Force TF32 mode (requires SM >= 80 and float32)
            - False: Force FP32 mode

    Returns:
        A new GPUArray containing the matrix product (M x N).

    Raises:
        ValueError: If arrays are not 2D or dimensions don't match.
        RuntimeError: If use_tf32=True but GPU doesn't support it or dtype is not float32.
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

    # Check TF32 dtype requirement early (before backend dispatch)
    if use_tf32 is True:
        from pygpukit.core.dtypes import float32

        if a.dtype != float32:
            raise RuntimeError("TF32 matmul requires float32 dtype")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_native(a, b, use_tf32=use_tf32)
    else:
        return _matmul_cpu(a, b)


def _matmul_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of matmul."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = np.matmul(a_np, b_np)
    return from_numpy(result_np)


def _matmul_native(a: GPUArray, b: GPUArray, *, use_tf32: bool | None = None) -> GPUArray:
    """Native C++ CUDA implementation of matmul (zero-copy).

    Args:
        a: First input array.
        b: Second input array.
        use_tf32: Whether to use TF32 TensorCore acceleration.
            None means use environment variable PYGPUKIT_ALLOW_TF32.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays (zero-copy if already native)
    a_native = a._get_native()
    b_native = b._get_native()

    # Perform operation on GPU
    if use_tf32 is not None:
        # Use explicit TF32 control
        c_native = native.matmul_tf32(a_native, b_native, use_tf32)
    else:
        # Use environment variable for TF32 control
        c_native = native.matmul(a_native, b_native)

    # Wrap result (zero-copy)
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
