"""Unary operations for GPUArrays.

Corresponds to native/ops/unary/.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype


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


def sin(a: GPUArray) -> GPUArray:
    """Element-wise sine.

    Args:
        a: Input array (float types).

    Returns:
        A new GPUArray containing sin(a).
    """
    _validate_float_dtype(a, "sin")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.sin(a._get_native()))
    else:
        return from_numpy(np.sin(a.to_numpy()))


def cos(a: GPUArray) -> GPUArray:
    """Element-wise cosine.

    Args:
        a: Input array (float types).

    Returns:
        A new GPUArray containing cos(a).
    """
    _validate_float_dtype(a, "cos")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.cos(a._get_native()))
    else:
        return from_numpy(np.cos(a.to_numpy()))


def sqrt(a: GPUArray) -> GPUArray:
    """Element-wise square root.

    Args:
        a: Input array (float types).

    Returns:
        A new GPUArray containing sqrt(a).
    """
    _validate_float_dtype(a, "sqrt")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.sqrt(a._get_native()))
    else:
        return from_numpy(np.sqrt(a.to_numpy()))


def rsqrt(a: GPUArray) -> GPUArray:
    """Element-wise reciprocal square root: 1/sqrt(x).

    Args:
        a: Input array (float types).

    Returns:
        A new GPUArray containing 1/sqrt(a).
    """
    _validate_float_dtype(a, "rsqrt")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.rsqrt(a._get_native()))
    else:
        return from_numpy(1.0 / np.sqrt(a.to_numpy()))


def abs(a: GPUArray) -> GPUArray:
    """Element-wise absolute value.

    Args:
        a: Input array (float types).

    Returns:
        A new GPUArray containing |a|.
    """
    _validate_float_dtype(a, "abs")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.abs(a._get_native()))
    else:
        return from_numpy(np.abs(a.to_numpy()))


def neg(a: GPUArray) -> GPUArray:
    """Element-wise negation: -x.

    Args:
        a: Input array (float types).

    Returns:
        A new GPUArray containing -a.
    """
    _validate_float_dtype(a, "neg")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.neg(a._get_native()))
    else:
        return from_numpy(-a.to_numpy())
