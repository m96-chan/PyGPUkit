"""
Wrapper for PyGPUkit GPUArray to work with Triton.

Triton expects objects with:
- data_ptr() method returning CUDA device pointer
- dtype attribute returning a string like "float32", "bfloat16", etc.
"""

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import numpy as np

    import pygpukit._pygpukit_native as native


# Mapping from PyGPUkit DataType to Triton-compatible string
# Keys can be PascalCase (native) or lowercase (Python wrapper)
_DTYPE_MAP = {
    # Lowercase (Python GPUArray wrapper)
    "float64": "float64",
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "int64": "int64",
    "int32": "int32",
    "int16": "int16",
    "int8": "int8",
    "uint8": "uint8",
    # PascalCase (native DataType enum)
    "Float64": "float64",
    "Float32": "float32",
    "Float16": "float16",
    "BFloat16": "bfloat16",
    "Int64": "int64",
    "Int32": "int32",
    "Int16": "int16",
    "Int8": "int8",
    "UInt8": "uint8",
    "Int4": "uint8",  # Int4 packed as uint8
}


class TritonArray:
    """
    Wrapper around PyGPUkit GPUArray for Triton compatibility.

    Triton's JIT system requires objects with:
    - data_ptr() method
    - dtype attribute (string like "float32")
    - shape attribute (for strided tensors)

    This wrapper provides the correct interface without PyTorch dependency.
    """

    __slots__ = ("_arr", "_dtype_str", "_shape")

    def __init__(self, arr: "native.GPUArray"):
        """
        Wrap a PyGPUkit GPUArray for Triton.

        Args:
            arr: Native PyGPUkit GPUArray
        """
        self._arr = arr
        # Convert DataType enum to string
        dtype_name = str(arr.dtype).split(".")[-1]
        self._dtype_str = _DTYPE_MAP.get(dtype_name, "float32")
        self._shape = tuple(arr.shape)

    def data_ptr(self) -> int:
        """Return CUDA device pointer."""
        # Get native GPUArray and call data_ptr()
        native_arr = self._arr._get_native() if hasattr(self._arr, "_get_native") else self._arr
        return int(native_arr.data_ptr())

    @property
    def dtype(self) -> str:
        """Return Triton-compatible dtype string."""
        return self._dtype_str

    @property
    def shape(self) -> tuple:
        """Return tensor shape."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def numel(self) -> int:
        """Return total number of elements."""
        result = 1
        for s in self._shape:
            result *= s
        return result

    def stride(self, dim: Optional[int] = None) -> Union[int, tuple[int, ...]]:
        """Return strides (C-contiguous assumed)."""
        strides: list[int] = []
        acc = 1
        for s in reversed(self._shape):
            strides.append(acc)
            acc *= s
        strides_tuple = tuple(reversed(strides))
        if dim is not None:
            # Handle negative indices
            if dim < 0:
                dim = len(self._shape) + dim
            return strides_tuple[dim]
        return strides_tuple

    @property
    def __cuda_array_interface__(self) -> dict[str, object]:
        """Return CUDA Array Interface for compatibility."""
        cai: dict[str, object] = self._arr.__cuda_array_interface__
        return cai

    def __repr__(self) -> str:
        return f"TritonArray(shape={self._shape}, dtype={self._dtype_str})"


def from_gpuarray(arr: "native.GPUArray") -> TritonArray:
    """
    Convert a PyGPUkit GPUArray to TritonArray.

    Args:
        arr: PyGPUkit native GPUArray

    Returns:
        TritonArray that can be used with Triton kernels

    Example:
        >>> import pygpukit._pygpukit_native as native
        >>> from pygpukit.triton import from_gpuarray
        >>>
        >>> x = native.from_numpy(np.zeros((4, 4), dtype=np.float32))
        >>> tx = from_gpuarray(x)  # Now usable with Triton kernels
    """
    return TritonArray(arr)


def from_numpy(arr: "np.ndarray") -> TritonArray:
    """
    Convert a NumPy array to TritonArray (transfers to GPU).

    Args:
        arr: NumPy array

    Returns:
        TritonArray on GPU

    Example:
        >>> from pygpukit.triton import from_numpy
        >>>
        >>> tx = from_numpy(np.zeros((4, 4), dtype=np.float32))
    """
    import pygpukit._pygpukit_native as native

    gpu_arr = native.from_numpy(arr)
    return TritonArray(gpu_arr)
