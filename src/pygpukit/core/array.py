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
        from pygpukit.core.dtypes import (
            bfloat16,
            float16,
            float32,
            float64,
            int8,
            int16,
            int32,
            int64,
            uint8,
        )

        native = get_native_module()

        # Map native DataType to Python DataType
        native_dtype = native_array.dtype
        if native_dtype == native.DataType.Float64:
            dtype = float64
        elif native_dtype == native.DataType.Float32:
            dtype = float32
        elif native_dtype == native.DataType.Float16:
            dtype = float16
        elif native_dtype == native.DataType.BFloat16:
            dtype = bfloat16
        elif native_dtype == native.DataType.Int64:
            dtype = int64
        elif native_dtype == native.DataType.Int32:
            dtype = int32
        elif native_dtype == native.DataType.Int16:
            dtype = int16
        elif native_dtype == native.DataType.Int8:
            dtype = int8
        elif native_dtype == native.DataType.UInt8:
            dtype = uint8
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

    def __add__(self, other: GPUArray | int | float) -> GPUArray:
        """Element-wise addition.

        Supports both GPUArray and scalar (int/float) operands.
        Broadcasting is supported for compatible shapes.
        """
        if isinstance(other, (int, float)):
            return self._scalar_op(other, lambda a, b: a + b)

        # Check if broadcasting is needed
        if self.shape != other.shape:
            # Use numpy broadcasting
            from pygpukit.core.factory import from_numpy

            a_np = self.to_numpy()
            b_np = other.to_numpy()
            result = a_np + b_np
            return from_numpy(result.astype(a_np.dtype))

        from pygpukit.ops.basic import add

        return add(self, other)

    def __radd__(self, other: int | float) -> GPUArray:
        """Right-hand addition for scalar + GPUArray."""
        return self._scalar_op(other, lambda a, b: b + a)

    def __sub__(self, other: GPUArray | int | float) -> GPUArray:
        """Element-wise subtraction.

        Supports both GPUArray and scalar (int/float) operands.
        """
        if isinstance(other, (int, float)):
            return self._scalar_op(other, lambda a, b: a - b)
        from pygpukit.ops.basic import sub

        return sub(self, other)

    def __rsub__(self, other: int | float) -> GPUArray:
        """Right-hand subtraction for scalar - GPUArray."""
        return self._scalar_op(other, lambda a, b: b - a)

    def __mul__(self, other: GPUArray | int | float) -> GPUArray:
        """Element-wise multiplication.

        Supports both GPUArray and scalar (int/float) operands.
        """
        if isinstance(other, (int, float)):
            return self._scalar_op(other, lambda a, b: a * b)
        from pygpukit.ops.basic import mul

        return mul(self, other)

    def __rmul__(self, other: int | float) -> GPUArray:
        """Right-hand multiplication for scalar * GPUArray."""
        return self._scalar_op(other, lambda a, b: b * a)

    def __truediv__(self, other: GPUArray | int | float) -> GPUArray:
        """Element-wise division.

        Supports both GPUArray and scalar (int/float) operands.
        """
        if isinstance(other, (int, float)):
            return self._scalar_op(other, lambda a, b: a / b)
        from pygpukit.ops.basic import div

        return div(self, other)

    def __rtruediv__(self, other: int | float) -> GPUArray:
        """Right-hand division for scalar / GPUArray."""
        return self._scalar_op(other, lambda a, b: b / a)

    def _scalar_op(self, scalar: int | float, op) -> GPUArray:
        """Apply a scalar operation using NumPy.

        Args:
            scalar: The scalar operand.
            op: A callable that takes (array, scalar) and returns the result.

        Returns:
            A new GPUArray with the result.
        """
        from pygpukit.core.factory import from_numpy

        np_data = self.to_numpy()
        result = op(np_data, scalar)
        return from_numpy(result.astype(np_data.dtype))

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

    def narrow(self, offset: int, length: int) -> GPUArray:
        """Create a zero-copy view into this array (1D slice along last axis).

        For a 2D array [batch, features], returns a view of [batch, length]
        starting at feature index `offset`.

        Args:
            offset: Starting index along the last axis (in elements).
            length: Number of elements to include in the view.

        Returns:
            A non-owning GPUArray view. Does not allocate memory.

        Note:
            The source array must outlive the view. The view shares memory
            with the source and does not own it.

        Example:
            # Split fused QKV output into Q, K, V views
            qkv = matmul(x, W_qkv)  # [1, q_dim + k_dim + v_dim]
            q = qkv.narrow(0, q_dim)  # [1, q_dim]
            k = qkv.narrow(q_dim, k_dim)  # [1, k_dim]
            v = qkv.narrow(q_dim + k_dim, v_dim)  # [1, v_dim]
        """
        if not has_native_module():
            raise RuntimeError("narrow() requires native backend")

        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        # Get source native array
        src_native = self._get_native()

        # For 2D [batch, features], the view shape is [batch, length]
        # For 1D [features], the view shape is [length]
        if self.ndim == 2:
            new_shape = [self.shape[0], length]
            # Offset is per-row, so for batch=1, offset_elements = offset
            offset_elements = offset
        elif self.ndim == 1:
            new_shape = [length]
            offset_elements = offset
        else:
            raise ValueError(f"narrow() only supports 1D or 2D arrays, got {self.ndim}D")

        # Call native narrow
        view_native = native.GPUArray.narrow(src_native, offset_elements, new_shape)

        # Wrap the view and keep reference to source to prevent memory from being freed
        view_arr = GPUArray._wrap_native(view_native)
        view_arr._source_ref = self
        return view_arr

    def view(self, new_shape: tuple[int, ...]) -> GPUArray:
        """Create a zero-copy view with a different shape (same total elements).

        This is a reshape operation that does not copy data. The new shape
        must have the same total number of elements as the original.

        Args:
            new_shape: The desired shape for the view.

        Returns:
            A non-owning GPUArray view with the new shape.

        Raises:
            ValueError: If new_shape has different total elements than original.
            RuntimeError: If native backend is not available.

        Example:
            # Reshape [1, 4096] to [1, 32, 128] for multi-head attention
            q = q_flat.view((1, num_heads, head_dim))
        """
        if not has_native_module():
            raise RuntimeError("view() requires native backend")

        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        # Validate element count
        new_size = 1
        for dim in new_shape:
            new_size *= dim

        if new_size != self.size:
            raise ValueError(
                f"Cannot view array of size {self.size} as shape {new_shape} (size {new_size})"
            )

        # Get source native array
        src_native = self._get_native()

        # Use narrow with offset=0 to create view with new shape
        view_native = native.GPUArray.narrow(src_native, 0, list(new_shape))

        # Wrap the view and keep reference to source to prevent memory from being freed
        view_arr = GPUArray._wrap_native(view_native)
        view_arr._source_ref = self  # Keep source alive while view exists
        return view_arr

    def slice_rows(self, num_rows: int) -> GPUArray:
        """Create a zero-copy view of the first N rows (batch dimension).

        For a 2D array [batch, features], returns a view of shape [num_rows, features].
        This is useful for working with pre-allocated buffers that may be larger
        than the actual batch size being processed.

        Args:
            num_rows: Number of rows to include in the view.

        Returns:
            A non-owning GPUArray view with shape [num_rows, features].

        Raises:
            ValueError: If num_rows exceeds the batch dimension.
            RuntimeError: If native backend is not available.

        Example:
            # Pre-allocated buffer for max_batch_size=8
            buffer = zeros((8, 4096), dtype="float16")
            # Get view for actual batch of 2
            batch_view = buffer.slice_rows(2)  # shape [2, 4096]
        """
        if not has_native_module():
            raise RuntimeError("slice_rows() requires native backend")

        if self.ndim != 2:
            raise ValueError(f"slice_rows() requires 2D array, got {self.ndim}D")

        if num_rows > self.shape[0]:
            raise ValueError(f"num_rows ({num_rows}) exceeds batch dimension ({self.shape[0]})")

        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        src_native = self._get_native()
        new_shape = [num_rows, self.shape[1]]

        # Use narrow with offset=0 to get first num_rows rows
        view_native = native.GPUArray.narrow(src_native, 0, new_shape)

        # Keep reference to source to prevent memory from being freed
        view_arr = GPUArray._wrap_native(view_native)
        view_arr._source_ref = self
        return view_arr

    def transpose(self, *axes: int) -> GPUArray:
        """Transpose the array by permuting its axes.

        Uses native GPU kernels when available for common patterns:
        - 2D (1,0): Native matmul.transpose()
        - 3D (1,0,2): Native tensor.transpose_3d_021()
        - 3D (0,2,1): Native tensor.transpose_3d_012()
        - 4D (0,2,1,3): Native tensor.transpose_4d_0213()
        - 4D (0,1,3,2): Native tensor.transpose_4d_0132()
        - Other patterns: CPU fallback

        Args:
            *axes: The new order of axes. If not provided, reverses all axes.
                   For a 3D array, transpose(0, 2, 1) swaps the last two axes.

        Returns:
            A new GPUArray with transposed data.

        Example:
            # Transpose 2D matrix
            a = from_numpy(np.array([[1, 2], [3, 4]]))
            b = a.transpose()  # or a.T

            # Permute 3D tensor axes
            x = from_numpy(np.zeros((2, 3, 4)))
            y = x.transpose(0, 2, 1)  # shape (2, 4, 3)
        """
        from pygpukit.core.backend import NativeBackend, get_backend
        from pygpukit.core.factory import from_numpy

        # Normalize axes
        if len(axes) == 0:
            # Reverse all axes
            axes = tuple(range(self.ndim - 1, -1, -1))

        # Check if we can use native implementations
        backend = get_backend()
        dtype_str = str(self.dtype)
        use_native = (
            isinstance(backend, NativeBackend)
            and backend.is_available()
            and dtype_str in ("float32", "float16", "bfloat16")
        )

        if use_native:
            # 2D transpose: (1, 0)
            if self.ndim == 2 and axes == (1, 0):
                from pygpukit.ops.matmul import transpose as matmul_transpose

                return matmul_transpose(self)

            # 3D transpose (1, 0, 2): [d0, d1, d2] -> [d1, d0, d2]
            if self.ndim == 3 and axes == (1, 0, 2):
                from pygpukit.ops.tensor import transpose_3d_021

                result = transpose_3d_021(self)
                return result if result is not None else self

            # 3D transpose (0, 2, 1): [d0, d1, d2] -> [d0, d2, d1]
            if self.ndim == 3 and axes == (0, 2, 1):
                from pygpukit.ops.tensor import transpose_3d_012

                result = transpose_3d_012(self)
                return result if result is not None else self

            # 4D transpose (0, 2, 1, 3): [d0, d1, d2, d3] -> [d0, d2, d1, d3]
            if self.ndim == 4 and axes == (0, 2, 1, 3):
                from pygpukit.ops.tensor import transpose_4d_0213

                result = transpose_4d_0213(self)
                return result if result is not None else self

            # 4D transpose (0, 1, 3, 2): [d0, d1, d2, d3] -> [d0, d1, d3, d2]
            if self.ndim == 4 and axes == (0, 1, 3, 2):
                from pygpukit.ops.tensor import transpose_4d_0132

                result = transpose_4d_0132(self)
                return result if result is not None else self

        # CPU fallback for unsupported patterns
        np_data = self.to_numpy()
        result = np_data.transpose(*axes)
        return from_numpy(result.copy())

    @property
    def T(self) -> GPUArray:
        """Return transposed array (reverses all axes)."""
        return self.transpose()

    def reshape(self, *shape: int) -> GPUArray:
        """Reshape the array to a new shape.

        Args:
            *shape: The new shape. Can be passed as separate args or as a tuple.
                    One dimension can be -1 to infer from the total size.

        Returns:
            A new GPUArray with the specified shape.

        Example:
            x = from_numpy(np.zeros((2, 3, 4)))
            y = x.reshape(6, 4)  # or x.reshape((6, 4))
            z = x.reshape(-1, 4)  # infer first dimension
        """
        from pygpukit.core.backend import NativeBackend, get_backend

        # Handle both reshape(2, 3) and reshape((2, 3))
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        # Handle -1 dimension inference
        shape = list(shape)
        total_size = 1
        for dim in self.shape:
            total_size *= dim

        neg_idx = -1
        known_size = 1
        for i, dim in enumerate(shape):
            if dim == -1:
                if neg_idx >= 0:
                    raise ValueError("reshape: only one dimension can be -1")
                neg_idx = i
            else:
                known_size *= dim

        if neg_idx >= 0:
            if total_size % known_size != 0:
                raise ValueError(
                    f"reshape: cannot infer dimension, total size {total_size} "
                    f"not divisible by {known_size}"
                )
            shape[neg_idx] = total_size // known_size

        shape = tuple(shape)

        # Verify total size
        output_size = 1
        for dim in shape:
            output_size *= dim
        if output_size != total_size:
            raise ValueError(
                f"reshape: cannot reshape array of size {total_size} into shape {shape}"
            )

        # Use native reshape_copy if available (keeps data on GPU)
        backend = get_backend()
        if isinstance(backend, NativeBackend) and backend.is_available():
            dtype_str = str(self.dtype)
            if dtype_str in ("float32", "float16", "bfloat16"):
                from pygpukit.core.backend import get_native_module

                native = get_native_module()
                input_native = self._get_native()
                c_native = native.reshape_copy(input_native, list(shape))
                return GPUArray._wrap_native(c_native)

        # CPU fallback
        from pygpukit.core.factory import from_numpy

        np_data = self.to_numpy()
        result = np_data.reshape(shape)
        return from_numpy(result.copy())

    def __getitem__(self, key) -> GPUArray:
        """Index or slice the array.

        Supports NumPy-style indexing including:
        - Integer indexing: arr[0]
        - Slicing: arr[:10], arr[1:5], arr[::2]
        - Multi-dimensional: arr[0, :, 1:3]

        Args:
            key: Index, slice, or tuple of indices/slices.

        Returns:
            A new GPUArray containing the selected elements.

        Example:
            x = from_numpy(np.arange(100).reshape(10, 10))
            row = x[0]        # First row
            col = x[:, 0]     # First column
            sub = x[:5, :5]   # 5x5 subarray
        """
        from pygpukit.core.factory import from_numpy

        np_data = self.to_numpy()
        result = np_data[key]
        # Handle scalar result
        if not isinstance(result, np.ndarray):
            result = np.array(result)
        return from_numpy(result.copy())
