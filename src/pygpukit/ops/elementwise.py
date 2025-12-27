"""Elementwise operations for GPUArrays.

Corresponds to native/ops/elementwise/.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_same_dtype, _validate_same_shape

# =============================================================================
# Binary Operations (allocating)
# =============================================================================


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
        return _add_native(a, b)
    else:
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
    a_native = a._get_native()
    b_native = b._get_native()
    c_native = native.add(a_native, b_native)
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


def mul(a: GPUArray, b: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """Element-wise multiplication of two arrays.

    Args:
        a: First input array.
        b: Second input array.
        out: Optional pre-allocated output array. If provided, the result
            is written to this array (for CUDA Graph capture support).

    Returns:
        A new GPUArray containing the element-wise product, or the out array if provided.

    Raises:
        ValueError: If shapes don't match.
    """
    _validate_same_shape(a, b, "mul")
    _validate_same_dtype(a, b, "mul")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _mul_native(a, b, out=out)
    else:
        return _mul_cpu(a, b)


def _mul_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of mul."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = a_np * b_np
    return from_numpy(result_np)


def _mul_native(a: GPUArray, b: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """Native C++ CUDA implementation of mul (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    b_native = b._get_native()

    if out is not None:
        out_native = out._get_native()
        native.mul_(a_native, b_native, out_native)
        return out
    else:
        c_native = native.mul(a_native, b_native)
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


# =============================================================================
# In-place Operations (non-allocating, CUDA Graph compatible)
# =============================================================================


def add_inplace(a: GPUArray, b: GPUArray) -> None:
    """In-place addition: a += b.

    For CUDA Graph: no allocation.

    Args:
        a: Tensor to add to (modified in-place).
        b: Tensor to add.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    b_native = b._get_native()
    native.add_inplace(a_native, b_native)


def mul_inplace(a: GPUArray, b: GPUArray) -> None:
    """In-place multiplication: a *= b.

    For CUDA Graph: no allocation.

    Args:
        a: Tensor to multiply (modified in-place).
        b: Tensor to multiply by.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    b_native = b._get_native()
    native.mul_inplace(a_native, b_native)


def copy_to(src: GPUArray, dst: GPUArray) -> None:
    """GPU-to-GPU copy.

    For CUDA Graph: no allocation.

    Args:
        src: Source tensor.
        dst: Destination tensor (must be same size).
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    src_native = src._get_native()
    dst_native = dst._get_native()
    native.copy_to(src_native, dst_native)


def clamp(a: GPUArray, min_val: float, max_val: float) -> GPUArray:
    """Element-wise clamp: clamp(x, min, max).

    Args:
        a: Input array (float types).
        min_val: Minimum value.
        max_val: Maximum value.

    Returns:
        A new GPUArray with values clamped to [min_val, max_val].
    """
    import numpy as np

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(native.clamp(a._get_native(), min_val, max_val))
    else:
        a_np = a.to_numpy()
        return from_numpy(np.clip(a_np, min_val, max_val))


def where(cond: GPUArray, a: GPUArray, b: GPUArray) -> GPUArray:
    """Conditional select: where(cond, a, b) = cond ? a : b.

    Args:
        cond: Boolean condition array (uint8 or int8, 0=False, nonzero=True).
        a: Values to use where condition is True.
        b: Values to use where condition is False.

    Returns:
        A new GPUArray with values selected from a or b based on cond.
    """
    import numpy as np

    _validate_same_shape(a, b, "where")
    _validate_same_dtype(a, b, "where")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return GPUArray._wrap_native(
            native.where(cond._get_native(), a._get_native(), b._get_native())
        )
    else:
        cond_np: np.ndarray = cond.to_numpy().astype(bool)
        a_np = a.to_numpy()
        b_np = b.to_numpy()
        return from_numpy(np.where(cond_np, a_np, b_np))
