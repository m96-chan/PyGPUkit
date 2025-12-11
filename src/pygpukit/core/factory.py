"""Factory functions for creating GPUArrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.dtypes import DataType

if TYPE_CHECKING:
    pass


def zeros(
    shape: tuple[int, ...] | int,
    dtype: str | DataType = "float32",
) -> GPUArray:
    """Create a GPUArray filled with zeros.

    Args:
        shape: Shape of the array. Can be an integer for 1D arrays.
        dtype: Data type of the array. Can be string or DataType.

    Returns:
        A GPUArray filled with zeros.
    """
    if isinstance(shape, int):
        shape = (shape,)

    if isinstance(dtype, str):
        dtype = DataType.from_string(dtype)

    backend = get_backend()

    # Fast path: native backend
    if isinstance(backend, NativeBackend) and backend.is_available():
        return _zeros_native(shape, dtype)

    # Slow path: CPU simulation
    size = 1
    for dim in shape:
        size *= dim
    nbytes = size * dtype.itemsize

    device_ptr = backend.allocate(nbytes)
    backend.memset(device_ptr, 0, nbytes)

    return GPUArray(shape, dtype, device_ptr)


def _zeros_native(shape: tuple[int, ...], dtype: DataType) -> GPUArray:
    """Create zeros array using native backend."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Map Python DataType to native DataType
    native_dtype = _to_native_dtype(dtype, native)

    # Create native array
    native_array = native.zeros(list(shape), native_dtype)
    return GPUArray._wrap_native(native_array)


def ones(
    shape: tuple[int, ...] | int,
    dtype: str | DataType = "float32",
) -> GPUArray:
    """Create a GPUArray filled with ones.

    Args:
        shape: Shape of the array. Can be an integer for 1D arrays.
        dtype: Data type of the array. Can be string or DataType.

    Returns:
        A GPUArray filled with ones.
    """
    if isinstance(shape, int):
        shape = (shape,)

    if isinstance(dtype, str):
        dtype = DataType.from_string(dtype)

    backend = get_backend()

    # Fast path: native backend
    if isinstance(backend, NativeBackend) and backend.is_available():
        return _ones_native(shape, dtype)

    # Slow path: CPU simulation
    np_dtype = dtype.to_numpy_dtype()
    size = 1
    for dim in shape:
        size *= dim
    host_data = np.ones(size, dtype=np_dtype)

    device_ptr = backend.allocate(host_data.nbytes)
    backend.copy_host_to_device(host_data, device_ptr)

    return GPUArray(shape, dtype, device_ptr)


def _ones_native(shape: tuple[int, ...], dtype: DataType) -> GPUArray:
    """Create ones array using native backend."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    native_dtype = _to_native_dtype(dtype, native)
    native_array = native.ones(list(shape), native_dtype)
    return GPUArray._wrap_native(native_array)


def empty(
    shape: tuple[int, ...] | int,
    dtype: str | DataType = "float32",
) -> GPUArray:
    """Create an uninitialized GPUArray.

    Args:
        shape: Shape of the array. Can be an integer for 1D arrays.
        dtype: Data type of the array. Can be string or DataType.

    Returns:
        An uninitialized GPUArray.

    Note:
        The contents of the array are undefined and may contain
        garbage values.
    """
    if isinstance(shape, int):
        shape = (shape,)

    if isinstance(dtype, str):
        dtype = DataType.from_string(dtype)

    backend = get_backend()

    # Fast path: native backend
    if isinstance(backend, NativeBackend) and backend.is_available():
        return _empty_native(shape, dtype)

    # Slow path: CPU simulation
    size = 1
    for dim in shape:
        size *= dim
    nbytes = size * dtype.itemsize

    device_ptr = backend.allocate(nbytes)

    return GPUArray(shape, dtype, device_ptr)


def _empty_native(shape: tuple[int, ...], dtype: DataType) -> GPUArray:
    """Create empty array using native backend."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    native_dtype = _to_native_dtype(dtype, native)
    native_array = native.empty(list(shape), native_dtype)
    return GPUArray._wrap_native(native_array)


def from_numpy(array: np.ndarray) -> GPUArray:
    """Create a GPUArray from a NumPy array.

    Args:
        array: A NumPy array to copy to GPU.

    Returns:
        A GPUArray containing a copy of the data.
    """
    # Ensure array is contiguous
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    dtype = DataType.from_numpy_dtype(array.dtype)
    shape = array.shape

    backend = get_backend()

    # Fast path: native backend
    if isinstance(backend, NativeBackend) and backend.is_available():
        return _from_numpy_native(array)

    # Slow path: CPU simulation
    device_ptr = backend.allocate(array.nbytes)
    backend.copy_host_to_device(array, device_ptr)

    return GPUArray(shape, dtype, device_ptr)


def _from_numpy_native(array: np.ndarray) -> GPUArray:
    """Create GPUArray from numpy using native backend."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    native_array = native.from_numpy(array)
    return GPUArray._wrap_native(native_array)


def _to_native_dtype(dtype: DataType, native: Any) -> Any:
    """Convert Python DataType to native DataType."""
    from pygpukit.core.dtypes import float32, float64, int32, int64

    if dtype == float32:
        return native.DataType.Float32
    elif dtype == float64:
        return native.DataType.Float64
    elif dtype == int32:
        return native.DataType.Int32
    elif dtype == int64:
        return native.DataType.Int64
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
