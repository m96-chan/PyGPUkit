"""Device information and management."""

from __future__ import annotations

from dataclasses import dataclass

from pygpukit.core.backend import get_backend


@dataclass
class DeviceInfo:
    """Information about a GPU device.

    Attributes:
        name: Name of the device.
        total_memory: Total memory in bytes.
        compute_capability: CUDA compute capability (major, minor) or None.
        multiprocessor_count: Number of multiprocessors.
        max_threads_per_block: Maximum threads per block.
        warp_size: Warp size.
    """

    name: str
    total_memory: int
    compute_capability: tuple[int, int] | None
    multiprocessor_count: int
    max_threads_per_block: int
    warp_size: int


def is_cuda_available() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise.
    """
    from pygpukit.core.backend import NativeBackend

    backend = NativeBackend()
    return backend.is_available()


def get_device_info(device_id: int = 0) -> DeviceInfo:
    """Get information about a GPU device.

    Args:
        device_id: Device index (default 0).

    Returns:
        DeviceInfo containing device properties.
    """
    backend = get_backend()
    props = backend.get_device_properties(device_id)

    return DeviceInfo(
        name=props.name,
        total_memory=props.total_memory,
        compute_capability=props.compute_capability,
        multiprocessor_count=props.multiprocessor_count,
        max_threads_per_block=props.max_threads_per_block,
        warp_size=props.warp_size,
    )
