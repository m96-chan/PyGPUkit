"""Basic operations for GPUArrays."""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend, has_native_module
from pygpukit.core.factory import from_numpy


def _validate_same_shape(a: GPUArray, b: GPUArray, op_name: str) -> None:
    """Validate that two arrays have the same shape."""
    if a.shape != b.shape:
        raise ValueError(
            f"{op_name} requires arrays of same shape, got {a.shape} and {b.shape}"
        )


def _validate_same_dtype(a: GPUArray, b: GPUArray, op_name: str) -> None:
    """Validate that two arrays have the same dtype."""
    if a.dtype != b.dtype:
        raise ValueError(
            f"{op_name} requires arrays of same dtype, got {a.dtype} and {b.dtype}"
        )


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
        # Use native C++ CUDA kernel for real GPU
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
    """Native C++ CUDA implementation of add."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Convert Python GPUArray to native GPUArray and perform operation
    a_native = native.from_numpy(a.to_numpy())
    b_native = native.from_numpy(b.to_numpy())
    c_native = native.add(a_native, b_native)

    # Convert result back to Python GPUArray
    return from_numpy(c_native.to_numpy())


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
    """Native C++ CUDA implementation of mul."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Convert Python GPUArray to native GPUArray and perform operation
    a_native = native.from_numpy(a.to_numpy())
    b_native = native.from_numpy(b.to_numpy())
    c_native = native.mul(a_native, b_native)

    # Convert result back to Python GPUArray
    return from_numpy(c_native.to_numpy())


def matmul(a: GPUArray, b: GPUArray) -> GPUArray:
    """Matrix multiplication of two 2D arrays.

    Args:
        a: First input array (M x K).
        b: Second input array (K x N).

    Returns:
        A new GPUArray containing the matrix product (M x N).

    Raises:
        ValueError: If arrays are not 2D or dimensions don't match.
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

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_native(a, b)
    else:
        return _matmul_cpu(a, b)


def _matmul_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of matmul."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = np.matmul(a_np, b_np)
    return from_numpy(result_np)


def _matmul_native(a: GPUArray, b: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of matmul."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Convert Python GPUArray to native GPUArray and perform operation
    a_native = native.from_numpy(a.to_numpy())
    b_native = native.from_numpy(b.to_numpy())
    c_native = native.matmul(a_native, b_native)

    # Convert result back to Python GPUArray
    return from_numpy(c_native.to_numpy())
