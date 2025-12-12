"""Memory Pool implementation for PyGPUkit.

This module provides a memory pool to reduce cudaMalloc/cudaFree overhead.
v0.2+ uses Rust backend for high-performance allocation tracking.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

# Check if Rust backend should be used
_USE_RUST = os.environ.get("PYGPUKIT_USE_RUST", "1").lower() in ("1", "true", "yes")

# Try to import Rust backend
_RUST_AVAILABLE = False
_rust_module: Any = None
try:
    import _pygpukit_rust._pygpukit_rust as _rust_module  # noqa: F811

    _RUST_AVAILABLE = True
except ImportError:
    pass


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""

    id: int
    size: int
    device_ptr: Any = None  # CUdeviceptr or native ptr
    host_data: np.ndarray | None = None  # For evicted data
    on_gpu: bool = True
    on_host: bool = False
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last access time."""
        self.last_access = time.time()


# Global default pool
_default_pool: MemoryPool | None = None


def set_default_pool(pool: MemoryPool | None) -> None:
    """Set the default memory pool."""
    global _default_pool
    _default_pool = pool


def get_default_pool() -> MemoryPool | None:
    """Get the default memory pool."""
    return _default_pool


class MemoryPool:
    """Memory pool for efficient GPU memory management.

    Features:
    - Pre-allocated memory blocks with size classes
    - LRU eviction policy for memory reuse
    - Thread-safe allocation/deallocation
    - Optional eviction to host memory
    - Rust backend for high-performance allocation tracking (v0.2+)

    Attributes:
        quota: Maximum memory this pool can use (bytes)
        enable_eviction: Whether to enable eviction to host memory

    Environment Variables:
        PYGPUKIT_USE_RUST: Set to "0" to disable Rust backend (default: "1")
    """

    # Size classes for block allocation (powers of 2)
    SIZE_CLASSES = [
        256,  # 256 B
        1024,  # 1 KB
        4096,  # 4 KB
        16384,  # 16 KB
        65536,  # 64 KB
        262144,  # 256 KB
        1048576,  # 1 MB
        4194304,  # 4 MB
        16777216,  # 16 MB
        67108864,  # 64 MB
        268435456,  # 256 MB
    ]

    def __init__(self, quota: int, enable_eviction: bool = False):
        """Initialize memory pool.

        Args:
            quota: Maximum memory this pool can use (bytes)
            enable_eviction: Enable eviction to host memory when pool is full
        """
        self._quota = quota
        self._enable_eviction = enable_eviction
        self._lock = threading.RLock()

        # Use Rust backend if available and enabled
        self._use_rust = _USE_RUST and _RUST_AVAILABLE
        self._rust_pool = None
        if self._use_rust:
            self._rust_pool = _rust_module.MemoryPool(quota, enable_eviction)

        # Python-side storage for MemoryBlock objects (needed for device_ptr, host_data)
        # Maps block_id -> MemoryBlock
        self._blocks: dict[int, MemoryBlock] = {}

        # Pure Python fallback storage (only used when Rust is disabled)
        if not self._use_rust:
            # Active allocations: block_id -> MemoryBlock
            self._active: dict[int, MemoryBlock] = {}

            # Free lists by size class: size -> [MemoryBlock, ...]
            self._free_lists: dict[int, list[MemoryBlock]] = {
                size: [] for size in self.SIZE_CLASSES
            }

            # LRU tracking: block_id -> MemoryBlock (ordered by access time)
            self._lru: OrderedDict[int, MemoryBlock] = OrderedDict()

            # Statistics
            self._next_id = 0
            self._used = 0
            self._cached = 0  # Memory in free lists
            self._allocation_count = 0
            self._reuse_count = 0
            self._eviction_count = 0
            self._cudamalloc_count = 0

        # Backend reference
        self._backend: Any = None

    def _get_backend(self) -> Any:
        """Get the backend for CUDA operations."""
        if self._backend is None:
            from pygpukit.core.backend import get_backend

            self._backend = get_backend()
        return self._backend

    @property
    def quota(self) -> int:
        """Maximum memory this pool can use."""
        if self._use_rust:
            return self._rust_pool.quota
        return self._quota

    @property
    def used(self) -> int:
        """Currently used memory (active allocations)."""
        with self._lock:
            if self._use_rust:
                return self._rust_pool.used
            return self._used

    @property
    def cached(self) -> int:
        """Memory in free lists (available for reuse)."""
        with self._lock:
            if self._use_rust:
                return self._rust_pool.cached
            return self._cached

    @property
    def available(self) -> int:
        """Available memory (quota - used)."""
        if self._use_rust:
            return self._rust_pool.available
        return self._quota - self._used

    def _get_size_class(self, size: int) -> int:
        """Get the appropriate size class for a given size."""
        for sc in self.SIZE_CLASSES:
            if size <= sc:
                return sc
        # Larger than any size class - use exact size rounded to 1MB
        return ((size + 1048575) // 1048576) * 1048576

    def allocate(self, size: int) -> MemoryBlock:
        """Allocate a memory block.

        Args:
            size: Size in bytes to allocate

        Returns:
            MemoryBlock representing the allocation

        Raises:
            MemoryError: If allocation exceeds quota and eviction is disabled
        """
        if self._use_rust:
            return self._allocate_rust(size)
        return self._allocate_python(size)

    def _allocate_rust(self, size: int) -> MemoryBlock:
        """Allocate using Rust backend."""
        with self._lock:
            try:
                block_id = self._rust_pool.allocate(size)
            except RuntimeError as e:
                raise MemoryError(str(e)) from e

            # Get block info from Rust
            rust_block = self._rust_pool.get_block(block_id)
            actual_size = rust_block.size if rust_block else size

            # Allocate actual CUDA memory
            backend = self._get_backend()
            try:
                device_ptr = backend.allocate(actual_size)
            except Exception:
                device_ptr = block_id  # CPU simulation fallback

            # Update Rust with device pointer
            self._rust_pool.set_device_ptr(
                block_id, device_ptr if isinstance(device_ptr, int) else block_id
            )

            # Create Python MemoryBlock
            block = MemoryBlock(
                id=block_id,
                size=actual_size,
                device_ptr=device_ptr,
                on_gpu=True,
                on_host=False,
            )
            self._blocks[block_id] = block
            return block

    def _allocate_python(self, size: int) -> MemoryBlock:
        """Allocate using pure Python backend."""
        size_class = self._get_size_class(size)

        with self._lock:
            # Try to reuse from free list
            if self._free_lists.get(size_class):
                block = self._free_lists[size_class].pop()
                block.touch()
                self._active[block.id] = block
                self._lru[block.id] = block
                self._lru.move_to_end(block.id)
                self._used += block.size
                self._cached -= block.size
                self._reuse_count += 1
                self._allocation_count += 1
                return block

            # Check quota
            if self._used + size_class > self._quota:
                if self._enable_eviction:
                    self._evict_lru(size_class)
                else:
                    raise MemoryError(
                        f"Memory pool quota exceeded: "
                        f"requested {size_class}, used {self._used}, quota {self._quota}"
                    )

            # Allocate new block
            block = self._allocate_new(size_class)
            self._active[block.id] = block
            self._lru[block.id] = block
            self._used += block.size
            self._allocation_count += 1
            self._cudamalloc_count += 1
            return block

    def _allocate_new(self, size: int) -> MemoryBlock:
        """Allocate a new memory block from CUDA."""
        backend = self._get_backend()

        block_id = self._next_id
        self._next_id += 1

        # Allocate device memory
        try:
            device_ptr = backend.allocate(size)
        except Exception:
            # Fallback for CPU simulation mode
            device_ptr = block_id  # Use ID as pseudo-pointer

        block = MemoryBlock(
            id=block_id,
            size=size,
            device_ptr=device_ptr,
            on_gpu=True,
            on_host=False,
        )
        return block

    def free(self, block: MemoryBlock) -> None:
        """Free a memory block (return to free list).

        Args:
            block: The block to free
        """
        if self._use_rust:
            self._free_rust(block)
        else:
            self._free_python(block)

    def _free_rust(self, block: MemoryBlock) -> None:
        """Free using Rust backend."""
        with self._lock:
            # Tell Rust to return block to free list
            self._rust_pool.free(block.id)
            # Keep Python block for potential reuse (device_ptr still valid)

    def _free_python(self, block: MemoryBlock) -> None:
        """Free using pure Python backend."""
        with self._lock:
            if block.id not in self._active:
                return

            del self._active[block.id]
            if block.id in self._lru:
                del self._lru[block.id]

            self._used -= block.size

            # Add to free list for reuse
            size_class = self._get_size_class(block.size)
            if size_class not in self._free_lists:
                self._free_lists[size_class] = []
            self._free_lists[size_class].append(block)
            self._cached += block.size

    def touch(self, block: MemoryBlock) -> None:
        """Mark a block as recently used (update LRU).

        Args:
            block: The block to touch
        """
        with self._lock:
            block.touch()
            if self._use_rust:
                self._rust_pool.touch(block.id)
            elif block.id in self._lru:
                self._lru.move_to_end(block.id)

    def _evict_lru(self, needed: int) -> None:
        """Evict least recently used blocks to make room.

        Args:
            needed: Amount of memory needed
        """
        freed = 0
        to_evict = []

        for _block_id, block in self._lru.items():
            if freed >= needed:
                break
            to_evict.append(block)
            freed += block.size

        for block in to_evict:
            self.evict(block)

    def evict(self, block: MemoryBlock) -> None:
        """Evict a block to host memory.

        Args:
            block: The block to evict
        """
        if not block.on_gpu:
            return

        with self._lock:
            # Copy data to host
            backend = self._get_backend()
            try:
                # Read data from GPU
                from pygpukit.core.dtypes import float32

                host_data = backend.copy_device_to_host(block.device_ptr, block.size, float32)
                block.host_data = host_data
            except Exception:
                # For CPU simulation, data is already on host
                block.host_data = np.zeros(block.size, dtype=np.uint8)

            # Free GPU memory
            try:
                backend.free(block.device_ptr)
            except Exception:
                pass

            block.on_gpu = False
            block.on_host = True
            block.device_ptr = None

            # Update tracking
            if self._use_rust:
                self._rust_pool.evict(block.id)
            else:
                self._eviction_count += 1
                # Update memory tracking
                if block.id in self._active:
                    self._used -= block.size

    def restore(self, block: MemoryBlock) -> None:
        """Restore an evicted block to GPU memory.

        Args:
            block: The block to restore
        """
        if block.on_gpu:
            return

        with self._lock:
            backend = self._get_backend()

            # Allocate GPU memory
            try:
                device_ptr = backend.allocate(block.size)
            except Exception:
                device_ptr = block.id

            # Copy data back to GPU
            if block.host_data is not None:
                try:
                    backend.copy_host_to_device(block.host_data, device_ptr)
                except Exception:
                    pass

            block.device_ptr = device_ptr
            block.on_gpu = True
            block.on_host = False
            block.host_data = None

            # Update tracking
            if self._use_rust:
                self._rust_pool.restore(block.id)
                self._rust_pool.set_device_ptr(
                    block.id, device_ptr if isinstance(device_ptr, int) else block.id
                )
            elif block.id in self._active:
                self._used += block.size

    def write(self, block: MemoryBlock, data: np.ndarray) -> None:
        """Write data to a block.

        Args:
            block: The block to write to
            data: NumPy array with data
        """
        if not block.on_gpu:
            self.restore(block)

        backend = self._get_backend()
        try:
            backend.copy_host_to_device(data, block.device_ptr)
        except Exception:
            # CPU simulation - store in host_data
            block.host_data = data.copy()

        self.touch(block)

    def read(self, block: MemoryBlock, dtype: np.dtype) -> np.ndarray:
        """Read data from a block.

        Args:
            block: The block to read from
            dtype: Data type for the result

        Returns:
            NumPy array with the data
        """
        if not block.on_gpu:
            if block.host_data is not None:
                result: np.ndarray = block.host_data.view(dtype)
                return result
            zeros_result: np.ndarray = np.zeros(block.size // np.dtype(dtype).itemsize, dtype=dtype)
            return zeros_result

        backend = self._get_backend()
        try:
            # Convert numpy dtype to DataType
            if dtype == np.float32:
                from pygpukit.core.dtypes import float32 as dt
            elif dtype == np.float64:
                from pygpukit.core.dtypes import float64 as dt
            elif dtype == np.int32:
                from pygpukit.core.dtypes import int32 as dt
            elif dtype == np.int64:
                from pygpukit.core.dtypes import int64 as dt
            else:
                from pygpukit.core.dtypes import float32 as dt

            result_data: np.ndarray = backend.copy_device_to_host(block.device_ptr, block.size, dt)
            return result_data
        except Exception:
            # CPU simulation
            if block.host_data is not None:
                fallback: np.ndarray = block.host_data.view(dtype)
                return fallback
            fallback_zeros: np.ndarray = np.zeros(
                block.size // np.dtype(dtype).itemsize, dtype=dtype
            )
            return fallback_zeros

    def stats(self) -> dict[str, Any]:
        """Get memory pool statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if self._use_rust:
                rust_stats = self._rust_pool.stats()
                return {
                    "quota": rust_stats.quota,
                    "used": rust_stats.used,
                    "cached": rust_stats.cached,
                    "available": rust_stats.available,
                    "allocation_count": rust_stats.allocation_count,
                    "reuse_count": rust_stats.reuse_count,
                    "eviction_count": rust_stats.eviction_count,
                    "cudamalloc_count": rust_stats.cudamalloc_count,
                    "active_blocks": rust_stats.active_blocks,
                    "free_blocks": rust_stats.free_blocks,
                }
            return {
                "quota": self._quota,
                "used": self._used,
                "cached": self._cached,
                "available": self.available,
                "allocation_count": self._allocation_count,
                "reuse_count": self._reuse_count,
                "eviction_count": self._eviction_count,
                "cudamalloc_count": self._cudamalloc_count,
                "active_blocks": len(self._active),
                "free_blocks": sum(len(fl) for fl in self._free_lists.values()),
            }

    def clear(self) -> None:
        """Clear all allocations and free lists."""
        with self._lock:
            backend = self._get_backend()

            if self._use_rust:
                # Free all Python-side blocks
                for block in self._blocks.values():
                    if block.on_gpu and block.device_ptr is not None:
                        try:
                            backend.free(block.device_ptr)
                        except Exception:
                            pass
                self._blocks.clear()
                self._rust_pool.clear()
            else:
                # Free all active blocks
                for block in self._active.values():
                    if block.on_gpu and block.device_ptr is not None:
                        try:
                            backend.free(block.device_ptr)
                        except Exception:
                            pass

                # Free all cached blocks
                for free_list in self._free_lists.values():
                    for block in free_list:
                        if block.on_gpu and block.device_ptr is not None:
                            try:
                                backend.free(block.device_ptr)
                            except Exception:
                                pass

                self._active.clear()
                self._lru.clear()
                for fl in self._free_lists.values():
                    fl.clear()
                self._used = 0
                self._cached = 0
