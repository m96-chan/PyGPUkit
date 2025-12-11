"""GPUArray implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pygpukit.core.backend import get_backend
from pygpukit.core.dtypes import DataType

if TYPE_CHECKING:
    import numpy as np


class GPUArray:
    """A NumPy-like array stored on GPU memory.

    Attributes:
        shape: Shape of the array.
        dtype: Data type of the array elements.
        size: Total number of elements.
        ndim: Number of dimensions.
        nbytes: Total bytes consumed by the array.
        itemsize: Size of each element in bytes.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: DataType,
        device_ptr: Any,
        owns_memory: bool = True,
    ) -> None:
        """Initialize a GPUArray.

        Args:
            shape: Shape of the array.
            dtype: Data type of elements.
            device_ptr: Pointer to device memory.
            owns_memory: Whether this array owns its memory.
        """
        self._shape = shape
        self._dtype = dtype
        self._device_ptr = device_ptr
        self._owns_memory = owns_memory
        self._last_access = time.time()
        self._on_gpu = True

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array."""
        return self._shape

    @property
    def dtype(self) -> DataType:
        """Return the data type of the array."""
        return self._dtype

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self._shape)

    @property
    def nbytes(self) -> int:
        """Return the total bytes consumed by the array."""
        return self.size * self._dtype.itemsize

    @property
    def itemsize(self) -> int:
        """Return the size of each element in bytes."""
        return self._dtype.itemsize

    @property
    def device_ptr(self) -> Any:
        """Return the device pointer."""
        self._last_access = time.time()
        return self._device_ptr

    @property
    def on_gpu(self) -> bool:
        """Return whether the data is on GPU."""
        return self._on_gpu

    @property
    def last_access(self) -> float:
        """Return the timestamp of last access."""
        return self._last_access

    def to_numpy(self) -> np.ndarray:
        """Copy array data to CPU and return as NumPy array.

        Returns:
            A NumPy array containing a copy of the data.
        """
        self._last_access = time.time()
        backend = get_backend()
        flat_array = backend.copy_device_to_host(
            self._device_ptr, self.nbytes, self._dtype
        )
        return flat_array.reshape(self._shape)

    def __repr__(self) -> str:
        return f"GPUArray(shape={self._shape}, dtype={self._dtype.name})"

    def __str__(self) -> str:
        return self.__repr__()

    def __del__(self) -> None:
        """Release GPU memory when array is deleted."""
        if self._owns_memory and self._device_ptr is not None:
            try:
                backend = get_backend()
                backend.free(self._device_ptr)
            except Exception:
                pass  # Ignore errors during cleanup
            self._device_ptr = None
