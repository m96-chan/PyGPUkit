"""Reduction operations for GPUArrays.

Corresponds to native/ops/reduction/.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype


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


def softmax(input: GPUArray, axis: int = -1) -> GPUArray:
    """Softmax activation along the specified axis.

    Computes: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))

    Args:
        input: Input array of shape [..., features].
            Supports 2D, 3D, and 4D tensors.
        axis: The axis along which to compute softmax (default: -1, last axis).

    Returns:
        A new GPUArray containing the softmax output, same shape as input.

    Raises:
        ValueError: If dtype is not a float type or axis is invalid.
    """
    _validate_float_dtype(input, "softmax")

    if input.ndim < 2:
        raise ValueError(f"softmax expects at least 2D input, got {input.ndim}D")
    if input.ndim > 4:
        raise ValueError(f"softmax supports up to 4D input, got {input.ndim}D")

    # Normalize axis
    if axis < 0:
        axis = input.ndim + axis
    if axis != input.ndim - 1:
        raise ValueError(f"softmax currently only supports axis=-1 (last axis), got axis={axis}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _softmax_native_nd(input)
    else:
        return _softmax_cpu_nd(input)


def _softmax_cpu(input: GPUArray) -> GPUArray:
    """CPU implementation of softmax for 2D tensors."""
    x = input.to_numpy()
    # Numerical stability: subtract max
    x_max = x.max(axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return from_numpy(exp_x / exp_x.sum(axis=1, keepdims=True))


def _softmax_cpu_nd(input: GPUArray) -> GPUArray:
    """CPU implementation of softmax for N-D tensors (axis=-1)."""
    x = input.to_numpy()
    # Numerical stability: subtract max along last axis
    x_max = x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return from_numpy(exp_x / exp_x.sum(axis=-1, keepdims=True))


def _softmax_native(input: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of softmax (zero-copy) for 2D tensors."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    c_native = native.softmax(input_native)
    return GPUArray._wrap_native(c_native)


def _softmax_native_nd(input: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of softmax for N-D tensors.

    Flattens leading dimensions into a single batch dimension,
    applies softmax along the last axis, then reshapes back.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    original_shape = input.shape

    # Flatten all but last dimension into batch
    features = original_shape[-1]
    batch_size = 1
    for dim in original_shape[:-1]:
        batch_size *= dim

    # Reshape to 2D [batch, features]
    input_2d = input.reshape((batch_size, features))
    input_native = input_2d._get_native()

    # Apply softmax
    c_native = native.softmax(input_native)
    result_2d = GPUArray._wrap_native(c_native)

    # Reshape back to original shape
    return result_2d.reshape(original_shape)


def min(a: GPUArray) -> GPUArray:
    """Min of all elements.

    Args:
        a: Input array (float types).

    Returns:
        A scalar GPUArray (shape [1]) containing the minimum value.
    """
    _validate_float_dtype(a, "min")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.min(a._get_native()))
    else:
        a_np = a.to_numpy()
        return from_numpy(np.array([np.min(a_np)], dtype=a_np.dtype))


def argmax(a: GPUArray) -> GPUArray:
    """Index of maximum element.

    Args:
        a: Input array (float types).

    Returns:
        A scalar GPUArray (shape [1], dtype int64) containing the index of the maximum value.
    """
    _validate_float_dtype(a, "argmax")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.argmax(a._get_native()))
    else:
        a_np = a.to_numpy()
        return from_numpy(np.array([np.argmax(a_np)], dtype=np.int64))


def sum_axis(a: GPUArray, axis: int) -> GPUArray:
    """Sum along specified axis for 2D tensors.

    Args:
        a: Input 2D array [M, N] (float types).
        axis: Axis to sum along (0 or 1).
            axis=0: sum rows -> output [N]
            axis=1: sum columns -> output [M]

    Returns:
        A GPUArray with the sum along the specified axis.

    Raises:
        ValueError: If input is not 2D or axis is not 0 or 1.
    """
    _validate_float_dtype(a, "sum_axis")
    if a.ndim != 2:
        raise ValueError(f"sum_axis requires 2D input, got {a.ndim}D")
    if axis not in (0, 1):
        raise ValueError(f"sum_axis: axis must be 0 or 1, got {axis}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.sum_axis(a._get_native(), axis))
    else:
        a_np = a.to_numpy()
        return from_numpy(np.sum(a_np, axis=axis))
