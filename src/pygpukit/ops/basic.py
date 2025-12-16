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
