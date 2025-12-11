"""Core module for PyGPUkit."""

from pygpukit.core.array import GPUArray
from pygpukit.core.device import DeviceInfo, get_device_info, is_cuda_available
from pygpukit.core.dtypes import DataType, float32, float64, int32, int64
from pygpukit.core.factory import empty, from_numpy, ones, zeros
from pygpukit.core.stream import Stream, StreamManager, default_stream

__all__ = [
    "GPUArray",
    "DeviceInfo",
    "get_device_info",
    "is_cuda_available",
    "DataType",
    "float32",
    "float64",
    "int32",
    "int64",
    "zeros",
    "ones",
    "empty",
    "from_numpy",
    "Stream",
    "StreamManager",
    "default_stream",
]
