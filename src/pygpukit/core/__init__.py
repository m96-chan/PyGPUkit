"""Core module for PyGPUkit."""

from pygpukit.core.array import GPUArray
from pygpukit.core.device import DeviceInfo, get_device_info, is_cuda_available
from pygpukit.core.dtypes import DataType, float32, float64, int32, int64
from pygpukit.core.factory import empty, from_numpy, ones, zeros
from pygpukit.core.stream import Stream, StreamManager, default_stream

# Import CUDA Event for GPU-side timing
try:
    from pygpukit._pygpukit_native import (
        CudaEvent,
        event_elapsed_ms,
        event_elapsed_us,
    )
except ImportError:
    CudaEvent = None  # type: ignore[misc, assignment]
    event_elapsed_ms = None  # type: ignore[assignment]
    event_elapsed_us = None  # type: ignore[assignment]

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
    "CudaEvent",
    "event_elapsed_ms",
    "event_elapsed_us",
]
