"""PyGPUkit - A lightweight GPU runtime for Python."""

__version__ = "0.2.6"

# LLM support (safetensors loader)
from pygpukit import llm
from pygpukit.core.array import GPUArray
from pygpukit.core.device import (
    DeviceInfo,
    FallbackDeviceCapabilities,
    get_device_capabilities,
    get_device_info,
    is_cuda_available,
)
from pygpukit.core.dtypes import DataType, bfloat16, float16, float32, float64, int32, int64
from pygpukit.core.factory import empty, from_numpy, ones, zeros
from pygpukit.core.stream import Stream, StreamManager, default_stream
from pygpukit.jit.compiler import (
    JITKernel,
    NvrtcError,
    NvrtcErrorCode,
    check_driver_compatibility,
    get_driver_requirements,
    get_nvrtc_path,
    get_nvrtc_version,
    get_warmup_error,
    is_nvrtc_available,
    is_warmup_done,
    jit,
    warmup,
)
from pygpukit.ops.basic import (
    add,
    div,
    exp,
    gelu,
    layernorm,
    log,
    matmul,
    max,
    mean,
    mul,
    relu,
    sub,
    sum,
)

# Try to import Rust types, fallback to Python implementations
try:
    from pygpukit._pygpukit_rust import DeviceCapabilities, KernelType
except ImportError:
    # Use Python fallback when Rust module is not available
    DeviceCapabilities = FallbackDeviceCapabilities
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
    "float16",
    "bfloat16",
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
    "NvrtcError",
    "NvrtcErrorCode",
    "is_nvrtc_available",
    "get_nvrtc_version",
    "get_nvrtc_path",
    "warmup",
    "is_warmup_done",
    "get_warmup_error",
    "get_driver_requirements",
    "check_driver_compatibility",
    # Operations
    "add",
    "sub",
    "mul",
    "div",
    "exp",
    "log",
    "relu",
    "gelu",
    "layernorm",
    "matmul",
    # Reductions
    "sum",
    "mean",
    "max",
    # LLM support
    "llm",
]
