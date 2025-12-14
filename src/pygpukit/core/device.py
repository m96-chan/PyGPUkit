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


def get_device_capabilities(device_id: int = 0):
    """Get device capabilities from Rust backend.

    Returns a DeviceCapabilities object with:
    - sm_version: SM version (e.g., 86 for SM 8.6)
    - tensorcore: Whether TF32 TensorCores are available
    - tensorcore_fp16: Whether FP16 TensorCores are available
    - tensorcore_bf16: Whether BF16 TensorCores are available
    - async_copy: Whether cp.async is supported

    Args:
        device_id: Device index (default 0).

    Returns:
        DeviceCapabilities from Rust backend.
    """
    try:
        from pygpukit._pygpukit_rust import DeviceCapabilities
    except ImportError:
        # Rust module not available - create from device info
        info = get_device_info(device_id)
        if info.compute_capability:
            sm_version = info.compute_capability[0] * 10 + info.compute_capability[1]
        else:
            sm_version = 0

        # Create a simple capabilities object
        @dataclass
        class _DeviceCapabilities:
            device_id: int
            name: str
            sm_version: int
            compute_capability: int
            tensorcore: bool
            tensorcore_fp16: bool
            tensorcore_bf16: bool
            async_copy: bool

        return _DeviceCapabilities(
            device_id=device_id,
            name=info.name,
            sm_version=sm_version,
            compute_capability=sm_version,
            tensorcore=sm_version >= 80,
            tensorcore_fp16=sm_version >= 70,
            tensorcore_bf16=sm_version >= 80,
            async_copy=sm_version >= 80,
        )

    # Get actual device info and create capabilities
    info = get_device_info(device_id)
    if info.compute_capability:
        sm_version = info.compute_capability[0] * 10 + info.compute_capability[1]
    else:
        sm_version = 0

    # Create Rust DeviceCapabilities with actual SM version
    caps = DeviceCapabilities(sm_version)
    return caps
