"""Memory management utilities for GPU arrays.

Provides Python wrappers for native memory operations:
- Memory info (free/total)
- Async copy operations
- Device synchronization
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray
    from pygpukit.core.stream import Stream


def get_memory_info() -> tuple[int, int]:
    """Get GPU memory information.

    Returns:
        Tuple of (free_bytes, total_bytes).

    Example:
        free, total = get_memory_info()
        print(f"Free: {free / 1e9:.2f} GB / Total: {total / 1e9:.2f} GB")
    """
    from pygpukit.core.backend import get_backend, has_native_module

    if not has_native_module():
        # CPU simulation - return dummy values
        return (8 * 1024**3, 8 * 1024**3)  # 8 GB

    backend = get_backend()
    if not backend.is_available():
        return (0, 0)

    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    props = native.get_device_properties()
    # Native returns total_memory; free requires cudaMemGetInfo
    # For now return (total - some_estimate, total)
    return (props.total_memory, props.total_memory)


def copy_to_device_async(
    dst: GPUArray,
    src_ptr: int,
    size_bytes: int,
    stream: Stream,
) -> None:
    """Async copy from host pointer to GPUArray.

    Args:
        dst: Destination GPUArray.
        src_ptr: Source host memory pointer (as integer).
        size_bytes: Number of bytes to copy.
        stream: CUDA stream for async operation.

    Note:
        For true async behavior, src_ptr should point to pinned memory.
        Otherwise the copy may block.
    """
    from pygpukit.core.backend import get_native_module, has_native_module

    if not has_native_module():
        raise RuntimeError("copy_to_device_async requires native backend")

    native = get_native_module()
    native.memcpy_ptr_to_device_async(
        dst._get_native(),
        src_ptr,
        size_bytes,
        stream._get_native(),
    )


def copy_to_device_async_raw_stream(
    dst: GPUArray,
    src_ptr: int,
    size_bytes: int,
    stream_handle: int,
) -> None:
    """Async copy using raw stream handle (for CUDA Graph).

    Args:
        dst: Destination GPUArray.
        src_ptr: Source host memory pointer (as integer).
        size_bytes: Number of bytes to copy.
        stream_handle: Raw CUDA stream handle (cudaStream_t as int).

    Note:
        Used during CUDA Graph capture where Stream object may not be available.
    """
    from pygpukit.core.backend import get_native_module, has_native_module

    if not has_native_module():
        raise RuntimeError("copy_to_device_async_raw_stream requires native backend")

    native = get_native_module()
    native.memcpy_ptr_to_device_async_raw_stream(
        dst._get_native(),
        src_ptr,
        size_bytes,
        stream_handle,
    )


def copy_to_device(
    dst: GPUArray,
    src_ptr: int,
    size_bytes: int,
) -> None:
    """Synchronous copy from host pointer to GPUArray.

    Args:
        dst: Destination GPUArray.
        src_ptr: Source host memory pointer (as integer).
        size_bytes: Number of bytes to copy.

    Note:
        This is a blocking operation. Use copy_to_device_async for
        non-blocking copies.
    """
    from pygpukit.core.backend import get_native_module, has_native_module

    if not has_native_module():
        raise RuntimeError("copy_to_device requires native backend")

    native = get_native_module()
    native.memcpy_ptr_to_device(
        dst._get_native(),
        src_ptr,
        size_bytes,
    )


def copy_device_to_device_async(
    dst: GPUArray,
    src: GPUArray,
    stream: Stream,
) -> None:
    """Async copy between GPUArrays on device.

    Args:
        dst: Destination GPUArray.
        src: Source GPUArray.
        stream: CUDA stream for async operation.

    Note:
        Both arrays must have the same size in bytes.
    """
    from pygpukit.core.backend import get_native_module, has_native_module

    if not has_native_module():
        raise RuntimeError("copy_device_to_device_async requires native backend")

    if dst.nbytes != src.nbytes:
        raise ValueError(f"Size mismatch: dst.nbytes={dst.nbytes}, src.nbytes={src.nbytes}")

    native = get_native_module()
    native.memcpy_device_to_device_async(
        dst._get_native(),
        src._get_native(),
        stream._get_native(),
    )


def copy_device_to_device_offset(
    dst: GPUArray,
    dst_offset_bytes: int,
    src: GPUArray,
    src_offset_bytes: int,
    size_bytes: int,
) -> None:
    """Copy between GPUArrays with byte offsets.

    Args:
        dst: Destination GPUArray.
        dst_offset_bytes: Byte offset in destination.
        src: Source GPUArray.
        src_offset_bytes: Byte offset in source.
        size_bytes: Number of bytes to copy.
    """
    from pygpukit.core.backend import get_native_module, has_native_module

    if not has_native_module():
        raise RuntimeError("copy_device_to_device_offset requires native backend")

    native = get_native_module()
    native.memcpy_device_to_device_offset(
        dst._get_native(),
        dst_offset_bytes,
        src._get_native(),
        src_offset_bytes,
        size_bytes,
    )


def synchronize() -> None:
    """Synchronize all GPU operations.

    Blocks until all previously issued GPU operations complete.
    """
    from pygpukit.core.backend import get_native_module, has_native_module

    if not has_native_module():
        return  # No-op for CPU simulation

    native = get_native_module()
    native.synchronize()


__all__ = [
    "get_memory_info",
    "copy_to_device_async",
    "copy_to_device_async_raw_stream",
    "copy_to_device",
    "copy_device_to_device_async",
    "copy_device_to_device_offset",
    "synchronize",
]
