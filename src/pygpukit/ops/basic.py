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
