"""Backend abstraction for CUDA operations.

This module provides an abstraction layer that allows PyGPUkit to work
with real CUDA hardware when available, or fall back to a CPU simulation
for testing and development without GPU.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pygpukit.core.dtypes import DataType


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
    def copy_device_to_host(
        self, device_ptr: Any, size_bytes: int, dtype: DataType
    ) -> np.ndarray:
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
            total_memory=psutil.virtual_memory().total if hasattr(psutil, 'virtual_memory') else 8 * 1024**3,
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

    def copy_device_to_host(
        self, device_ptr: int, size_bytes: int, dtype: DataType
    ) -> np.ndarray:
        if device_ptr not in self._allocations:
            raise RuntimeError(f"Invalid device pointer: {device_ptr}")
        buffer = self._allocations[device_ptr]
        np_dtype = dtype.to_numpy_dtype()
        result: np.ndarray = np.frombuffer(
            buffer[:size_bytes].tobytes(), dtype=np_dtype
        ).copy()
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


class CUDABackend(Backend):
    """Real CUDA backend using cuda-python."""

    def __init__(self) -> None:
        self._cuda_available = False
        self._cuda = None
        self._cudart = None
        self._init_cuda()

    def _init_cuda(self) -> None:
        """Initialize CUDA runtime."""
        try:
            from cuda import cuda, cudart

            err = cuda.cuInit(0)
            if err[0] == cuda.CUresult.CUDA_SUCCESS:
                self._cuda_available = True
                self._cuda = cuda
                self._cudart = cudart
        except ImportError:
            self._cuda_available = False
        except Exception:
            self._cuda_available = False

    def _check_error(self, result: tuple) -> Any:
        """Check CUDA error and raise exception if failed."""
        if result[0] != 0:
            raise RuntimeError(f"CUDA error: {result[0]}")
        return result[1] if len(result) > 1 else None

    def is_available(self) -> bool:
        return self._cuda_available

    def get_device_count(self) -> int:
        if not self._cuda_available or self._cudart is None:
            return 0
        result = self._cudart.cudaGetDeviceCount()
        count: int = self._check_error(result)
        return count

    def get_device_properties(self, device_id: int = 0) -> DeviceProperties:
        if not self._cuda_available or self._cudart is None:
            raise RuntimeError("CUDA is not available")

        props = self._check_error(self._cudart.cudaGetDeviceProperties(device_id))
        return DeviceProperties(
            name=props.name.decode() if isinstance(props.name, bytes) else str(props.name),
            total_memory=props.totalGlobalMem,
            compute_capability=(props.major, props.minor),
            multiprocessor_count=props.multiProcessorCount,
            max_threads_per_block=props.maxThreadsPerBlock,
            warp_size=props.warpSize,
        )

    def allocate(self, size_bytes: int) -> Any:
        if not self._cuda_available or self._cudart is None:
            raise RuntimeError("CUDA is not available")
        return self._check_error(self._cudart.cudaMalloc(size_bytes))

    def free(self, ptr: Any) -> None:
        if not self._cuda_available or self._cudart is None:
            return
        self._cudart.cudaFree(ptr)

    def copy_host_to_device(self, host_data: np.ndarray, device_ptr: Any) -> None:
        if not self._cuda_available or self._cudart is None:
            raise RuntimeError("CUDA is not available")
        size = host_data.nbytes
        self._check_error(
            self._cudart.cudaMemcpy(
                device_ptr,
                host_data.ctypes.data,
                size,
                self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
        )

    def copy_device_to_host(
        self, device_ptr: Any, size_bytes: int, dtype: DataType
    ) -> np.ndarray:
        if not self._cuda_available or self._cudart is None:
            raise RuntimeError("CUDA is not available")
        np_dtype = dtype.to_numpy_dtype()
        host_array: np.ndarray = np.empty(size_bytes // np_dtype.itemsize, dtype=np_dtype)
        self._check_error(
            self._cudart.cudaMemcpy(
                host_array.ctypes.data,
                device_ptr,
                size_bytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
        )
        return host_array

    def memset(self, device_ptr: Any, value: int, size_bytes: int) -> None:
        if not self._cuda_available or self._cudart is None:
            raise RuntimeError("CUDA is not available")
        self._check_error(self._cudart.cudaMemset(device_ptr, value, size_bytes))

    def synchronize(self) -> None:
        if not self._cuda_available or self._cudart is None:
            return
        self._check_error(self._cudart.cudaDeviceSynchronize())

    def create_stream(self, priority: int = 0) -> Any:
        if not self._cuda_available or self._cudart is None:
            raise RuntimeError("CUDA is not available")
        return self._check_error(
            self._cudart.cudaStreamCreateWithPriority(
                self._cudart.cudaStreamNonBlocking, priority
            )
        )

    def destroy_stream(self, stream: Any) -> None:
        if not self._cuda_available or self._cudart is None:
            return
        self._cudart.cudaStreamDestroy(stream)

    def stream_synchronize(self, stream: Any) -> None:
        if not self._cuda_available or self._cudart is None:
            return
        self._check_error(self._cudart.cudaStreamSynchronize(stream))


# Global backend instance
_backend: Backend | None = None


def get_backend() -> Backend:
    """Get the current backend instance."""
    global _backend
    if _backend is None:
        # Try CUDA first, fall back to CPU simulation
        cuda_backend = CUDABackend()
        if cuda_backend.is_available():
            _backend = cuda_backend
        else:
            _backend = CPUSimulationBackend()
    return _backend


def set_backend(backend: Backend) -> None:
    """Set the backend instance (useful for testing)."""
    global _backend
    _backend = backend


def reset_backend() -> None:
    """Reset the backend to auto-detection."""
    global _backend
    _backend = None
