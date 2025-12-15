"""GPUArray implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from pygpukit.core.backend import get_backend, has_native_module
from pygpukit.core.dtypes import DataType

if TYPE_CHECKING:
    pass


class GPUArray:
    """A NumPy-like array stored on GPU memory.

    When the native C++ backend is available, this class wraps a native
    GPUArray for optimal performance (no Python↔C++ data copies during
    GPU operations).

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
        device_ptr: Any = None,
        owns_memory: bool = True,
        _native: Any = None,
    ) -> None:
        """Initialize a GPUArray.

        Args:
            shape: Shape of the array.
            dtype: Data type of elements.
            device_ptr: Pointer to device memory (for CPU simulation backend).
            owns_memory: Whether this array owns its memory.
            _native: Native GPUArray object (for native backend).
        """
        self._shape = shape
        self._dtype = dtype
        self._device_ptr = device_ptr
        self._owns_memory = owns_memory
        self._last_access = time.time()
        self._on_gpu = True
        self._native = _native  # Native GPUArray for zero-copy operations

    @classmethod
    def _wrap_native(cls, native_array: Any) -> GPUArray:
        """Wrap a native GPUArray.

        This is the fast path for GPU operations - no data copying.
        """
        from pygpukit.core.backend import get_native_module
        from pygpukit.core.dtypes import bfloat16, float16, float32, float64, int32, int64

        native = get_native_module()

        # Map native DataType to Python DataType
        native_dtype = native_array.dtype
        if native_dtype == native.DataType.Float32:
            dtype = float32
        elif native_dtype == native.DataType.Float64:
            dtype = float64
        elif native_dtype == native.DataType.Float16:
            dtype = float16
        elif native_dtype == native.DataType.BFloat16:
            dtype = bfloat16
        elif native_dtype == native.DataType.Int32:
            dtype = int32
        elif native_dtype == native.DataType.Int64:
            dtype = int64
        else:
            raise ValueError(f"Unknown native dtype: {native_dtype}")

        return cls(
            shape=tuple(native_array.shape),
            dtype=dtype,
            device_ptr=None,
            owns_memory=False,  # Native handles memory
            _native=native_array,
        )

    def _get_native(self) -> Any:
        """Get the native GPUArray, creating one if needed.

        This converts a CPU-simulation GPUArray to a native one on demand.
        """
        if self._native is not None:
            return self._native

        if not has_native_module():
            raise RuntimeError("Native module not available")

        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        # Convert to native GPUArray
        np_data = self.to_numpy()
        self._native = native.from_numpy(np_data)
        return self._native

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

        # Fast path: use native array directly
        if self._native is not None:
            result: np.ndarray = self._native.to_numpy()
            return result

        # Slow path: CPU simulation backend
        backend = get_backend()
        flat_array = backend.copy_device_to_host(self._device_ptr, self.nbytes, self._dtype)
        return flat_array.reshape(self._shape)

    def is_contiguous(self) -> bool:
        """Check if the array is contiguous in memory.

        Returns:
            Always True, as PyGPUkit arrays are always contiguous (no stride support).
        """
        return True

    def contiguous(self) -> GPUArray:
        """Return a contiguous array.

        If the array is already contiguous, returns self.
        Otherwise, returns a contiguous copy.

        Returns:
            A contiguous GPUArray (always self, since arrays are always contiguous).
        """
        # All PyGPUkit arrays are contiguous (no stride support yet)
        return self

    def clone(self) -> GPUArray:
        """Create a deep copy of the array.

        Returns:
            A new GPUArray with copied data.
        """
        from pygpukit.core.factory import from_numpy

        # Copy via NumPy (simple and reliable)
        np_data = self.to_numpy().copy()
        return from_numpy(np_data)

    def __repr__(self) -> str:
        backend_type = "native" if self._native is not None else "simulation"
        return f"GPUArray(shape={self._shape}, dtype={self._dtype.name}, backend={backend_type})"

    def __str__(self) -> str:
        return self.__repr__()

    def __del__(self) -> None:
        """Release GPU memory when array is deleted."""
        # Native arrays handle their own cleanup via RAII
        if self._native is not None:
            self._native = None
            return

        # CPU simulation cleanup
        if self._owns_memory and self._device_ptr is not None:
            try:
                backend = get_backend()
                backend.free(self._device_ptr)
            except Exception:
                pass  # Ignore errors during cleanup
            self._device_ptr = None

    # ========================================================================
    # Arithmetic operators
    # ========================================================================

    def __add__(self, other: GPUArray) -> GPUArray:
        """Element-wise addition."""
        from pygpukit.ops.basic import add

        return add(self, other)

    def __sub__(self, other: GPUArray) -> GPUArray:
        """Element-wise subtraction."""
        from pygpukit.ops.basic import sub

        return sub(self, other)

    def __mul__(self, other: GPUArray) -> GPUArray:
        """Element-wise multiplication."""
        from pygpukit.ops.basic import mul

        return mul(self, other)

    def __truediv__(self, other: GPUArray) -> GPUArray:
        """Element-wise division."""
        from pygpukit.ops.basic import div

        return div(self, other)

    def __matmul__(self, other: GPUArray) -> GPUArray:
        """Matrix multiplication."""
        from pygpukit.ops.basic import matmul

        return matmul(self, other)

    # ========================================================================
    # Type conversion
    # ========================================================================

    def astype(self, dtype: DataType) -> GPUArray:
        """Convert array to a different data type.

        Args:
            dtype: Target data type.

        Returns:
            A new GPUArray with the specified dtype.
        """
        if self._dtype == dtype:
            return self

        from pygpukit.core.dtypes import bfloat16, float16, float32
        from pygpukit.core.factory import from_numpy

        # Get numpy array
        np_data = self.to_numpy()

        # Handle BF16 source (stored as uint16)
        if self._dtype == bfloat16:
            # Convert BF16 (uint16) to FP32: shift left by 16 bits
            bf16_as_uint32 = np_data.astype(np.uint32) << 16
            fp32_data = bf16_as_uint32.view(np.float32)

            if dtype == float32:
                return from_numpy(fp32_data)
            elif dtype == float16:
                return from_numpy(fp32_data.astype(np.float16))
            else:
                return from_numpy(fp32_data.astype(dtype.to_numpy_dtype()))

        # Convert to BF16
        if dtype == bfloat16:
            # BF16: convert via float32, store as uint16
            fp32_data = np_data.astype(np.float32)
            # Round to nearest even
            uint32_view = fp32_data.view(np.uint32)
            bf16_data = ((uint32_view + 0x7FFF + ((uint32_view >> 16) & 1)) >> 16).astype(np.uint16)
            result = from_numpy(bf16_data)
            # Override dtype to bfloat16
            result._dtype = dtype
            return result
        else:
            target_np_dtype = dtype.to_numpy_dtype()
            converted: np.ndarray = np_data.astype(target_np_dtype)
            return from_numpy(converted)
