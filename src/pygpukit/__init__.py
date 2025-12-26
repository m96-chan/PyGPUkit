"""PyGPUkit - A lightweight GPU runtime for Python."""

__version__ = "0.2.15"

# LLM support (safetensors loader)
from pygpukit import llm, ops
from pygpukit.core.array import GPUArray
from pygpukit.core.device import (
    DeviceInfo,
    FallbackDeviceCapabilities,
    get_device_capabilities,
    get_device_info,
    is_cuda_available,
)
from pygpukit.core.dtypes import (
    DataType,
    bfloat16,
    float16,
    float32,
    float64,
    int4,
    int8,
    int32,
    int64,
    uint8,
)
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
    abs,
    add,
    argmax,
    bias_add_inplace,
    clamp,
    cos,
    div,
    exp,
    gelu,
    layernorm,
    linear_bias_gelu,
    log,
    matmul,
    max,
    mean,
    min,
    mul,
    neg,
    relu,
    rsqrt,
    sigmoid,
    sin,
    softmax,
    sqrt,
    sub,
    sum,
    sum_axis,
    tanh,
    transpose,
    where,
)

# Try to import Rust types, fallback to Python implementations
try:
    from pygpukit._pygpukit_rust import DeviceCapabilities, KernelType
except ImportError:
    # Use Python fallback when Rust module is not available
    DeviceCapabilities = FallbackDeviceCapabilities
    KernelType = None

# Import CUDA Graph from native module (via auto-selecting loader)
try:
    from pygpukit._native_loader import get_native_module as _get_native

    _native = _get_native()
    CudaGraph = getattr(_native, "CudaGraph", None)
except (ImportError, AttributeError):
    try:
        from pygpukit._pygpukit_native import CudaGraph
    except ImportError:
        CudaGraph = None

# Import CUDA Event for GPU-side timing (via auto-selecting loader)
try:
    _native = _get_native()
    CudaEvent = getattr(_native, "CudaEvent", None)
    event_elapsed_ms = getattr(_native, "event_elapsed_ms", None)
    event_elapsed_us = getattr(_native, "event_elapsed_us", None)
except (ImportError, AttributeError, NameError):
    try:
        from pygpukit._pygpukit_native import CudaEvent, event_elapsed_ms, event_elapsed_us
    except ImportError:
        CudaEvent = None
        event_elapsed_ms = None
        event_elapsed_us = None

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
    "int8",
    "uint8",
    "int4",
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
    "ops",  # ops module for advanced usage
    "abs",
    "add",
    "argmax",
    "clamp",
    "cos",
    "div",
    "exp",
    "gelu",
    "layernorm",
    "log",
    "matmul",
    "mul",
    "neg",
    "relu",
    "rsqrt",
    "sigmoid",
    "sin",
    "softmax",
    "sqrt",
    "sub",
    "tanh",
    "transpose",
    "where",
    # Fused operations
    "bias_add_inplace",
    "linear_bias_gelu",
    # Reductions
    "argmax",
    "max",
    "mean",
    "min",
    "sum",
    "sum_axis",
    # LLM support
    "llm",
    # CUDA Graph
    "CudaGraph",
    # CUDA Event
    "CudaEvent",
    "event_elapsed_ms",
    "event_elapsed_us",
]
