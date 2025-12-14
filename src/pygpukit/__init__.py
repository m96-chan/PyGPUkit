"""PyGPUkit - A lightweight GPU runtime for Python."""

__version__ = "0.2.0"

from pygpukit.core.array import GPUArray
from pygpukit.core.device import (
    DeviceInfo,
    get_device_capabilities,
    get_device_info,
    is_cuda_available,
)
from pygpukit.core.dtypes import DataType, float32, float64, int32, int64
from pygpukit.core.factory import empty, from_numpy, ones, zeros
from pygpukit.core.stream import Stream, StreamManager, default_stream
from pygpukit.jit.compiler import JITKernel, jit
from pygpukit.ops.basic import add, matmul, mul

# Try to import Rust types
try:
    from pygpukit._pygpukit_rust import DeviceCapabilities, KernelType
except ImportError:
    DeviceCapabilities = None
    KernelType = None

__all__ = [
    # Version
    "__version__",
    # Array
    "GPUArray",
    # Device
    "DeviceInfo",
    "DeviceCapabilities",
    "KernelType",
    "get_device_info",
    "get_device_capabilities",
    "is_cuda_available",
    # Data types
    "DataType",
    "float32",
    "float64",
    "int32",
    "int64",
    # Factory functions
    "zeros",
    "ones",
    "empty",
    "from_numpy",
    # Stream
    "Stream",
    "StreamManager",
    "default_stream",
    # JIT
    "jit",
    "JITKernel",
    # Operations
    "add",
    "mul",
    "matmul",
]
