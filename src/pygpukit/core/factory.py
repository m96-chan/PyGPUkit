"""Factory functions for creating GPUArrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import get_backend
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

    size = 1
    for dim in shape:
        size *= dim
    nbytes = size * dtype.itemsize

    backend = get_backend()
    device_ptr = backend.allocate(nbytes)
    backend.memset(device_ptr, 0, nbytes)

    return GPUArray(shape, dtype, device_ptr)


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

    # Create ones array on CPU and copy to GPU
    np_dtype = dtype.to_numpy_dtype()
    size = 1
    for dim in shape:
        size *= dim
    host_data = np.ones(size, dtype=np_dtype)

    backend = get_backend()
    device_ptr = backend.allocate(host_data.nbytes)
    backend.copy_host_to_device(host_data, device_ptr)

    return GPUArray(shape, dtype, device_ptr)


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

    size = 1
    for dim in shape:
        size *= dim
    nbytes = size * dtype.itemsize

    backend = get_backend()
    device_ptr = backend.allocate(nbytes)

    return GPUArray(shape, dtype, device_ptr)


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
    device_ptr = backend.allocate(array.nbytes)
    backend.copy_host_to_device(array, device_ptr)

    return GPUArray(shape, dtype, device_ptr)
