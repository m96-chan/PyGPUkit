"""Backend abstraction for CUDA operations.

This module provides an abstraction layer that allows PyGPUkit to work
with real CUDA hardware when available, or fall back to a CPU simulation
for testing and development without GPU.

IMPORTANT: PyGPUkit does NOT use cuda-python.
GPU backend is C++ using CUDA Runtime/Driver API + NVRTC, exposed via pybind11.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pygpukit.core.dtypes import DataType

# Try to import native module
_native_module: Any = None

# Track NVRTC discovery status for warning
_nvrtc_search_performed: bool = False
_nvrtc_dll_found: str | None = None


def _find_nvrtc_dll() -> str | None:
    """Find NVRTC DLL in a version-agnostic way.

    Searches for nvrtc64_*.dll (Windows) or libnvrtc.so* (Linux) in:
    1. PATH directories
    2. CUDA_PATH/bin
    3. Common CUDA installation paths

    Returns:
        Path to NVRTC DLL if found, None otherwise.
    """
    global _nvrtc_search_performed, _nvrtc_dll_found

    if _nvrtc_search_performed:
        return _nvrtc_dll_found

    _nvrtc_search_performed = True

    if sys.platform == "win32":
        patterns = ["nvrtc64_*.dll", "nvrtc*.dll"]
    else:
        patterns = ["libnvrtc.so*", "libnvrtc*.so*"]

    search_paths: list[str] = []

    # 1. PATH directories
    path_env = os.environ.get("PATH", "")
    search_paths.extend(path_env.split(os.pathsep))

    # 2. CUDA_PATH/bin (Windows) or CUDA_PATH/lib64 (Linux)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        if sys.platform == "win32":
            search_paths.append(os.path.join(cuda_path, "bin"))
        else:
            search_paths.append(os.path.join(cuda_path, "lib64"))
            search_paths.append(os.path.join(cuda_path, "lib"))

    # 3. Common CUDA installation paths
    if sys.platform == "win32":
        # Windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin
        nvidia_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(nvidia_base):
            for version_dir in glob.glob(os.path.join(nvidia_base, "v*")):
                search_paths.append(os.path.join(version_dir, "bin"))
    else:
        # Linux: /usr/local/cuda*/lib64
        for cuda_dir in glob.glob("/usr/local/cuda*"):
            search_paths.append(os.path.join(cuda_dir, "lib64"))
            search_paths.append(os.path.join(cuda_dir, "lib"))
        # Also check standard library paths
        search_paths.extend(["/usr/lib64", "/usr/lib", "/usr/local/lib"])

    # Search for NVRTC DLL
    for search_dir in search_paths:
        if not search_dir or not os.path.isdir(search_dir):
            continue
        for pattern in patterns:
            matches = glob.glob(os.path.join(search_dir, pattern))
            if matches:
                # Return the first match (prefer newest version by sorting)
                matches.sort(reverse=True)
                _nvrtc_dll_found = matches[0]
                return _nvrtc_dll_found

    _nvrtc_dll_found = None
    return None


def _add_cuda_dll_directories() -> list[str]:
    """Add CUDA DLL directories on Windows for version-agnostic loading.

    Searches for NVRTC in multiple locations and adds all found CUDA
    directories to the DLL search path.

    Returns:
        List of directories added to DLL search path.
    """
    added_dirs: list[str] = []

    if sys.platform != "win32":
        return added_dirs

    search_paths: list[str] = []

    # 1. CUDA_PATH/bin
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        bin_path = os.path.join(cuda_path, "bin")
        if os.path.isdir(bin_path):
            search_paths.append(bin_path)

    # 2. PATH directories that contain CUDA DLLs
    path_env = os.environ.get("PATH", "")
    for path_dir in path_env.split(os.pathsep):
        if path_dir and os.path.isdir(path_dir):
            # Check if this directory has any nvrtc DLL
            if glob.glob(os.path.join(path_dir, "nvrtc*.dll")):
                search_paths.append(path_dir)

    # 3. Common CUDA installation paths
    nvidia_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(nvidia_base):
        for version_dir in sorted(glob.glob(os.path.join(nvidia_base, "v*")), reverse=True):
            bin_dir = os.path.join(version_dir, "bin")
            if os.path.isdir(bin_dir):
                search_paths.append(bin_dir)

    # Add unique directories
    seen: set[str] = set()
    for path in search_paths:
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized not in seen:
            seen.add(normalized)
            try:
                os.add_dll_directory(path)
                added_dirs.append(path)
            except (AttributeError, OSError):
                pass

    return added_dirs


def _emit_nvrtc_warning() -> None:
    """Emit a warning if NVRTC is not available but GPU is."""
    nvrtc_path = _find_nvrtc_dll()

    if nvrtc_path is None:
        warnings.warn(
            "NVRTC (NVIDIA Runtime Compiler) not found. "
            "JIT compilation of custom kernels is disabled.\n"
            "Pre-compiled GPU operations (matmul, add, etc.) will still work.\n\n"
            "NVRTC is optional. To enable JIT compilation:\n"
            "  https://developer.nvidia.com/cuda-downloads\n\n"
            "Check availability: pygpukit.is_nvrtc_available()",
            UserWarning,
            stacklevel=3,
        )


try:
    _add_cuda_dll_directories()
    from pygpukit import _pygpukit_native  # type: ignore[attr-defined]

    _native_module = _pygpukit_native

    # Check NVRTC availability and warn if not found (deferred to first use)
except ImportError:
    pass


@dataclass
class DeviceProperties:
    """Properties of a compute device."""

    name: str
    total_memory: int
    compute_capability: tuple[int, int] | None = None
    multiprocessor_count: int = 0
    max_threads_per_block: int = 1024
    warp_size: int = 32


class Backend(ABC):
    """Abstract base class for compute backends."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        ...

    @abstractmethod
    def get_device_count(self) -> int:
        """Get number of available devices."""
        ...

    @abstractmethod
    def get_device_properties(self, device_id: int = 0) -> DeviceProperties:
        """Get properties of a device."""
        ...

    @abstractmethod
    def allocate(self, size_bytes: int) -> Any:
        """Allocate memory on the device."""
        ...

    @abstractmethod
    def free(self, ptr: Any) -> None:
        """Free device memory."""
        ...

    @abstractmethod
    def copy_host_to_device(self, host_data: np.ndarray, device_ptr: Any) -> None:
        """Copy data from host to device."""
        ...

    @abstractmethod
    def copy_device_to_host(self, device_ptr: Any, size_bytes: int, dtype: DataType) -> np.ndarray:
        """Copy data from device to host."""
        ...

    @abstractmethod
    def memset(self, device_ptr: Any, value: int, size_bytes: int) -> None:
        """Set device memory to a value."""
        ...

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize the device."""
        ...

    @abstractmethod
    def create_stream(self, priority: int = 0) -> Any:
        """Create a compute stream."""
        ...

    @abstractmethod
    def destroy_stream(self, stream: Any) -> None:
        """Destroy a compute stream."""
        ...

    @abstractmethod
    def stream_synchronize(self, stream: Any) -> None:
        """Synchronize a stream."""
        ...


class CPUSimulationBackend(Backend):
    """CPU-based simulation backend for testing without GPU."""

    def __init__(self) -> None:
        self._allocations: dict[int, np.ndarray] = {}
        self._next_id = 0
        self._streams: dict[int, dict[str, Any]] = {}
        self._next_stream_id = 0

    def is_available(self) -> bool:
        return True

    def get_device_count(self) -> int:
        return 1

    def get_device_properties(self, device_id: int = 0) -> DeviceProperties:
        import psutil

        return DeviceProperties(
            name="CPU Simulation",
            total_memory=psutil.virtual_memory().total
            if hasattr(psutil, "virtual_memory")
            else 8 * 1024**3,
            compute_capability=None,
            multiprocessor_count=os.cpu_count() or 1,
            max_threads_per_block=1024,
            warp_size=32,
        )

    def allocate(self, size_bytes: int) -> int:
        import numpy as np

        buffer = np.zeros(size_bytes, dtype=np.uint8)
        ptr_id = self._next_id
        self._next_id += 1
        self._allocations[ptr_id] = buffer
        return ptr_id

    def free(self, ptr: int) -> None:
        if ptr in self._allocations:
            del self._allocations[ptr]

    def copy_host_to_device(self, host_data: np.ndarray, device_ptr: int) -> None:
        if device_ptr not in self._allocations:
            raise RuntimeError(f"Invalid device pointer: {device_ptr}")
        buffer = self._allocations[device_ptr]
        host_bytes = host_data.tobytes()
        buffer[: len(host_bytes)] = list(host_bytes)

    def copy_device_to_host(self, device_ptr: int, size_bytes: int, dtype: DataType) -> np.ndarray:
        if device_ptr not in self._allocations:
            raise RuntimeError(f"Invalid device pointer: {device_ptr}")
        buffer = self._allocations[device_ptr]
        np_dtype = dtype.to_numpy_dtype()
        result: np.ndarray = np.frombuffer(buffer[:size_bytes].tobytes(), dtype=np_dtype).copy()
        return result

    def memset(self, device_ptr: int, value: int, size_bytes: int) -> None:
        if device_ptr not in self._allocations:
            raise RuntimeError(f"Invalid device pointer: {device_ptr}")
        buffer = self._allocations[device_ptr]
        buffer[:size_bytes] = value

    def synchronize(self) -> None:
        pass

    def create_stream(self, priority: int = 0) -> int:
        stream_id = self._next_stream_id
        self._next_stream_id += 1
        self._streams[stream_id] = {"priority": priority}
        return stream_id

    def destroy_stream(self, stream: int) -> None:
        if stream in self._streams:
            del self._streams[stream]

    def stream_synchronize(self, stream: int) -> None:
        pass


class NativeBackend(Backend):
    """Real CUDA backend using native C++ module (pybind11).

    This backend uses the C++ native module (_pygpukit_native) which
    interfaces with CUDA Runtime/Driver API and NVRTC.
    """

    def __init__(self) -> None:
        self._native = _native_module
        self._cuda_available = False
        self._init_cuda()

    def _init_cuda(self) -> None:
        """Initialize CUDA via native module."""
        if self._native is None:
            self._cuda_available = False
            return
        try:
            self._cuda_available = self._native.is_cuda_available()
        except Exception:
            self._cuda_available = False

    def is_available(self) -> bool:
        return self._cuda_available

    def get_device_count(self) -> int:
        if not self._cuda_available or self._native is None:
            return 0
        count: int = self._native.get_device_count()
        return count

    def get_device_properties(self, device_id: int = 0) -> DeviceProperties:
        if not self._cuda_available or self._native is None:
            raise RuntimeError("CUDA is not available")

        props = self._native.get_device_properties(device_id)
        return DeviceProperties(
            name=props.name,
            total_memory=props.total_memory,
            compute_capability=(
                props.compute_capability_major,
                props.compute_capability_minor,
            ),
            multiprocessor_count=props.multiprocessor_count,
            max_threads_per_block=props.max_threads_per_block,
            warp_size=props.warp_size,
        )

    def allocate(self, size_bytes: int) -> Any:
        if not self._cuda_available or self._native is None:
            raise RuntimeError("CUDA is not available")
        # Use GPUArray internally for memory management
        raise NotImplementedError("Use GPUArray from native module directly")

    def free(self, ptr: Any) -> None:
        if not self._cuda_available or self._native is None:
            return
        # GPUArray handles its own memory via RAII
        pass

    def copy_host_to_device(self, host_data: np.ndarray, device_ptr: Any) -> None:
        if not self._cuda_available or self._native is None:
            raise RuntimeError("CUDA is not available")
        # Use GPUArray.copy_from_numpy() instead
        raise NotImplementedError("Use GPUArray.copy_from_numpy() instead")

    def copy_device_to_host(self, device_ptr: Any, size_bytes: int, dtype: DataType) -> np.ndarray:
        if not self._cuda_available or self._native is None:
            raise RuntimeError("CUDA is not available")
        # Use GPUArray.to_numpy() instead
        raise NotImplementedError("Use GPUArray.to_numpy() instead")

    def memset(self, device_ptr: Any, value: int, size_bytes: int) -> None:
        if not self._cuda_available or self._native is None:
            raise RuntimeError("CUDA is not available")
        # Use GPUArray.fill_zeros() for zero initialization
        raise NotImplementedError("Use GPUArray.fill_zeros() instead")

    def synchronize(self) -> None:
        if not self._cuda_available or self._native is None:
            return
        self._native.device_synchronize()

    def create_stream(self, priority: int = 0) -> Any:
        if not self._cuda_available or self._native is None:
            raise RuntimeError("CUDA is not available")
        stream_priority = (
            self._native.StreamPriority.High if priority < 0 else self._native.StreamPriority.Low
        )
        return self._native.Stream(stream_priority)

    def destroy_stream(self, stream: Any) -> None:
        # Stream handles its own destruction via RAII
        pass

    def stream_synchronize(self, stream: Any) -> None:
        if not self._cuda_available or self._native is None:
            return
        stream.synchronize()


# Global backend instance
_backend: Backend | None = None


def get_backend() -> Backend:
    """Get the current backend instance."""
    global _backend
    if _backend is None:
        # Try native C++ backend first, fall back to CPU simulation
        native_backend = NativeBackend()
        if native_backend.is_available():
            _backend = native_backend
        else:
            _backend = CPUSimulationBackend()
    return _backend


def has_native_module() -> bool:
    """Check if the native C++ module is available."""
    return _native_module is not None


def get_native_module() -> Any:
    """Get the native C++ module (for direct access)."""
    if _native_module is None:
        raise RuntimeError(
            "Native module not available. Build with CMake or install pygpukit with CUDA support."
        )
    return _native_module


def set_backend(backend: Backend) -> None:
    """Set the backend instance (useful for testing)."""
    global _backend
    _backend = backend


def reset_backend() -> None:
    """Reset the backend to auto-detection."""
    global _backend
    _backend = None
