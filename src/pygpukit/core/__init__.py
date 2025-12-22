"""Core module for PyGPUkit."""

from pygpukit.core.array import GPUArray
from pygpukit.core.device import DeviceInfo, get_device_info, is_cuda_available
from pygpukit.core.dtypes import DataType, float32, float64, int16, int32, int64
from pygpukit.core.factory import empty, from_numpy, ones, zeros
from pygpukit.core.stream import Stream, StreamManager, default_stream

# Import CUDA Event for GPU-side timing (via auto-selecting loader)
try:
    from pygpukit._native_loader import get_native_module as _get_native

    _native = _get_native()
    CudaEvent = getattr(_native, "CudaEvent", None)
    event_elapsed_ms = getattr(_native, "event_elapsed_ms", None)
    event_elapsed_us = getattr(_native, "event_elapsed_us", None)
except (ImportError, AttributeError):
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
    "float64",
    "float32",
    "int64",
    "int32",
    "int16",
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
