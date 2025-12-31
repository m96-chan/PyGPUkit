"""SafeTensors file loading for PyGPUkit LLM.

Provides:
- Dtype: Tensor data type enumeration
- TensorInfo: Metadata for a single tensor
- SafeTensorsFile: Memory-mapped single SafeTensors file
- ShardedSafeTensorsFile: Sharded model loader with lazy shard loading
- LazyModelLoader: Lazy GPU loading with LRU eviction for large models
- TensorState: State of a lazy tensor
- PoolStats: Memory pool statistics
- load_safetensors: Unified loader function
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pygpukit.core.backend import get_rust_module

if TYPE_CHECKING:
    from collections.abc import Sequence

# Get the Rust llm module
_rust = get_rust_module()
_llm = _rust.llm if _rust else None


class Dtype:
    """Tensor data type enumeration."""

    Float32 = 0
    Float16 = 1
    BFloat16 = 2
    Float64 = 3
    Float8E4M3 = 4  # FP8 E4M3 (1 sign, 4 exponent, 3 mantissa)
    Float8E5M2 = 5  # FP8 E5M2 (1 sign, 5 exponent, 2 mantissa)
    Int32 = 6
    Int64 = 7
    Int16 = 8
    Int8 = 9
    UInt8 = 10
    Bool = 11

    _NAMES = {
        0: "float32",
        1: "float16",
        2: "bfloat16",
        3: "float64",
        4: "float8_e4m3",
        5: "float8_e5m2",
        6: "int32",
        7: "int64",
        8: "int16",
        9: "int8",
        10: "uint8",
        11: "bool",
    }

    _SIZES = {
        0: 4,  # float32
        1: 2,  # float16
        2: 2,  # bfloat16
        3: 8,  # float64
        4: 1,  # float8_e4m3
        5: 1,  # float8_e5m2
        6: 4,  # int32
        7: 8,  # int64
        8: 2,  # int16
        9: 1,  # int8
        10: 1,  # uint8
        11: 1,  # bool
    }

    @classmethod
    def element_size(cls, dtype: int) -> int:
        """Get the size in bytes of a single element."""
        return cls._SIZES.get(dtype, 0)

    @classmethod
    def name(cls, dtype: int) -> str:
        """Get the string name of a dtype."""
        return cls._NAMES.get(dtype, "unknown")


class TensorInfo:
    """Metadata for a single tensor in a safetensors file."""

    def __init__(
        self,
        name: str,
        dtype: int,
        shape: Sequence[int],
        offset: int,
        size_bytes: int,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = list(shape)
        self.offset = offset
        self.size_bytes = size_bytes

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def dtype_name(self) -> str:
        """String name of the dtype."""
        return Dtype.name(self.dtype)

    def __repr__(self) -> str:
        return (
            f"TensorInfo(name='{self.name}', dtype={self.dtype_name}, "
            f"shape={self.shape}, size_bytes={self.size_bytes})"
        )


class SafeTensorsFile:
    """Memory-mapped SafeTensors file.

    Provides efficient access to tensor metadata and data from a .safetensors file
    using memory mapping for zero-copy data access.

    Example:
        >>> st = SafeTensorsFile("model.safetensors")
        >>> print(st.tensor_names)
        ['weight', 'bias']
        >>> info = st.tensor_info('weight')
        >>> print(info.shape, info.dtype_name)
        [768, 768] float16
        >>> data = st.tensor_bytes('weight')
    """

    def __init__(self, path: str):
        """Open a safetensors file.

        Args:
            path: Path to the .safetensors file
        """
        if _llm is None:
            raise RuntimeError("Rust LLM module not available")
        self._inner = _llm.SafeTensorsFile(path)

    @property
    def tensor_names(self) -> list[str]:
        """Get list of all tensor names."""
        return self._inner.tensor_names

    @property
    def file_size(self) -> int:
        """Total file size in bytes."""
        return self._inner.file_size

    @property
    def num_tensors(self) -> int:
        """Number of tensors in the file."""
        return self._inner.num_tensors

    def tensor_info(self, name: str) -> TensorInfo:
        """Get metadata for a tensor by name.

        Args:
            name: Tensor name

        Returns:
            TensorInfo with dtype, shape, offset, and size

        Raises:
            KeyError: If tensor name not found
        """
        info = self._inner.tensor_info(name)
        return TensorInfo(
            name=info.name,
            dtype=int(info.dtype),
            shape=info.shape,
            offset=info.offset,
            size_bytes=info.size_bytes,
        )

    def tensor_bytes(self, name: str) -> bytes:
        """Get raw tensor data as bytes.

        Args:
            name: Tensor name

        Returns:
            Raw bytes of the tensor data

        Raises:
            KeyError: If tensor name not found
        """
        return bytes(self._inner.tensor_bytes(name))

    def tensor_as_f32(self, name: str):
        """Get tensor data as numpy float32 array.

        Args:
            name: Tensor name

        Returns:
            1D numpy array of float32 values

        Raises:
            KeyError: If tensor name not found
            ValueError: If tensor dtype is not Float32
        """
        return self._inner.tensor_as_f32(name)

    def tensor_data_ptr(self, name: str) -> tuple[int, int]:
        """Get raw mmap pointer for direct GPU transfer.

        Args:
            name: Tensor name

        Returns:
            Tuple of (ptr, size_bytes) where ptr is the raw mmap address

        Raises:
            KeyError: If tensor name not found
        """
        return self._inner.tensor_data_ptr(name)

    def __len__(self) -> int:
        return self.num_tensors

    def __contains__(self, name: str) -> bool:
        return name in self._inner

    def __repr__(self) -> str:
        return f"SafeTensorsFile(num_tensors={self.num_tensors}, file_size={self.file_size})"


class ShardedSafeTensorsFile:
    """Sharded SafeTensors file loader.

    Handles models split across multiple .safetensors files with an index.json.
    Lazily opens shards on demand to minimize memory usage.

    Example:
        >>> st = ShardedSafeTensorsFile("model.safetensors.index.json")
        >>> print(st.tensor_names[:5])
        ['lm_head.weight', 'model.embed_tokens.weight', ...]
        >>> info = st.tensor_info('model.embed_tokens.weight')
        >>> data = st.tensor_bytes('model.embed_tokens.weight')
    """

    def __init__(self, index_json_path: str):
        """Open a sharded safetensors model.

        Args:
            index_json_path: Path to model.safetensors.index.json
        """
        import json
        from pathlib import Path

        self._index_path = Path(index_json_path)
        self._base_dir = self._index_path.parent

        with open(index_json_path, encoding="utf-8") as f:
            index = json.load(f)

        # weight_map: { tensor_name: shard_filename }
        self._weight_map: dict[str, str] = index.get("weight_map", {})
        self._metadata = index.get("metadata", {})

        # Lazy-loaded shard files
        self._shards: dict[str, SafeTensorsFile] = {}

        # Unique shard files
        self._shard_files = list(set(self._weight_map.values()))

    def _get_shard(self, shard_file: str) -> SafeTensorsFile:
        """Lazily open a shard file."""
        if shard_file not in self._shards:
            shard_path = self._base_dir / shard_file
            self._shards[shard_file] = SafeTensorsFile(str(shard_path))
        return self._shards[shard_file]

    @property
    def tensor_names(self) -> list[str]:
        """Get list of all tensor names across all shards."""
        return list(self._weight_map.keys())

    @property
    def file_size(self) -> int:
        """Total file size across all shards (lazy, opens all shards)."""
        total = 0
        for shard_file in self._shard_files:
            total += self._get_shard(shard_file).file_size
        return total

    @property
    def num_tensors(self) -> int:
        """Number of tensors across all shards."""
        return len(self._weight_map)

    def tensor_info(self, name: str) -> TensorInfo:
        """Get metadata for a tensor by name.

        Args:
            name: Tensor name

        Returns:
            TensorInfo with dtype, shape, offset, and size

        Raises:
            KeyError: If tensor name not found
        """
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found")
        shard_file = self._weight_map[name]
        return self._get_shard(shard_file).tensor_info(name)

    def tensor_bytes(self, name: str) -> bytes:
        """Get raw tensor data as bytes.

        Args:
            name: Tensor name

        Returns:
            Raw bytes of the tensor data

        Raises:
            KeyError: If tensor name not found
        """
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found")
        shard_file = self._weight_map[name]
        return self._get_shard(shard_file).tensor_bytes(name)

    def tensor_as_f32(self, name: str):
        """Get tensor data as numpy float32 array.

        Args:
            name: Tensor name

        Returns:
            1D numpy array of float32 values

        Raises:
            KeyError: If tensor name not found
            ValueError: If tensor dtype is not Float32
        """
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found")
        shard_file = self._weight_map[name]
        return self._get_shard(shard_file).tensor_as_f32(name)

    def tensor_data_ptr(self, name: str) -> tuple[int, int]:
        """Get raw mmap pointer for direct GPU transfer.

        Args:
            name: Tensor name

        Returns:
            Tuple of (ptr, size_bytes) where ptr is the raw mmap address

        Raises:
            KeyError: If tensor name not found
        """
        if name not in self._weight_map:
            raise KeyError(f"Tensor '{name}' not found")
        shard_file = self._weight_map[name]
        return self._get_shard(shard_file).tensor_data_ptr(name)

    def __len__(self) -> int:
        return self.num_tensors

    def __contains__(self, name: str) -> bool:
        return name in self._weight_map

    def __repr__(self) -> str:
        return (
            f"ShardedSafeTensorsFile(num_tensors={self.num_tensors}, "
            f"num_shards={len(self._shard_files)})"
        )


def load_safetensors(path: str) -> SafeTensorsFile | ShardedSafeTensorsFile:
    """Load a safetensors file (single or sharded).

    Automatically detects sharded models by .index.json extension.

    Args:
        path: Path to .safetensors file or .safetensors.index.json

    Returns:
        SafeTensorsFile or ShardedSafeTensorsFile for accessing tensor data

    Example:
        # Single file
        st = load_safetensors("model.safetensors")

        # Sharded model
        st = load_safetensors("model.safetensors.index.json")
    """
    if path.endswith(".index.json"):
        return ShardedSafeTensorsFile(path)
    else:
        return SafeTensorsFile(path)


class TensorState:
    """State of a lazy tensor.

    Attributes:
        OnDisk: Tensor is on disk only (mmap, not loaded to GPU)
        Loading: Tensor is currently loading to GPU
        OnGpu: Tensor is resident on GPU
        Evicted: Tensor was evicted from GPU (mmap still valid)
    """

    OnDisk = 0
    Loading = 1
    OnGpu = 2
    Evicted = 3

    _NAMES = {
        0: "OnDisk",
        1: "Loading",
        2: "OnGpu",
        3: "Evicted",
    }

    @classmethod
    def name(cls, state: int) -> str:
        """Get the string name of a state."""
        return cls._NAMES.get(state, "Unknown")


class PoolStats:
    """Memory pool statistics.

    Attributes:
        quota: Maximum memory allowed (bytes)
        used: Currently used memory (active allocations)
        cached: Memory in free lists (cached for reuse)
        available: Available memory (quota - used)
        allocation_count: Total number of allocations
        reuse_count: Number of blocks reused from free list
        eviction_count: Number of blocks evicted
        cudamalloc_count: Number of new CUDA allocations
        active_blocks: Number of active blocks
        free_blocks: Number of blocks in free lists
    """

    def __init__(
        self,
        quota: int,
        used: int,
        cached: int,
        available: int,
        allocation_count: int,
        reuse_count: int,
        eviction_count: int,
        cudamalloc_count: int,
        active_blocks: int,
        free_blocks: int,
    ):
        self.quota = quota
        self.used = used
        self.cached = cached
        self.available = available
        self.allocation_count = allocation_count
        self.reuse_count = reuse_count
        self.eviction_count = eviction_count
        self.cudamalloc_count = cudamalloc_count
        self.active_blocks = active_blocks
        self.free_blocks = free_blocks

    @property
    def utilization(self) -> float:
        """Utilization percentage (used / quota * 100)."""
        if self.quota == 0:
            return 0.0
        return (self.used / self.quota) * 100.0

    @property
    def total_blocks(self) -> int:
        """Total number of blocks (active + free)."""
        return self.active_blocks + self.free_blocks

    def __repr__(self) -> str:
        return (
            f"PoolStats(quota={self.quota}, used={self.used}, cached={self.cached}, "
            f"available={self.available}, active_blocks={self.active_blocks}, "
            f"free_blocks={self.free_blocks})"
        )


class LazyModelLoader:
    """Lazy model loader for large models (70B+).

    Memory-maps SafeTensors files and loads tensors to GPU on demand.
    When VRAM budget is exceeded, least-recently-used tensors are evicted.

    This is useful for models that exceed available VRAM, allowing you to
    load tensors on-demand and automatically manage GPU memory.

    Example:
        >>> loader = LazyModelLoader(memory_budget=8 * 1024**3)  # 8GB
        >>> loader.load_file("model-00001-of-00004.safetensors")
        >>> loader.load_file("model-00002-of-00004.safetensors")
        >>> print(loader.total_size)  # Total model size
        >>> print(loader.loaded_size)  # Currently on GPU
        >>> loader.unload_layer("model.layers.0.")  # Free layer 0
    """

    def __init__(self, memory_budget: int, enable_eviction: bool = True):
        """Create a new lazy model loader.

        Args:
            memory_budget: Maximum GPU memory to use in bytes
            enable_eviction: Whether to auto-evict when budget exceeded
        """
        if _llm is None:
            raise RuntimeError("Rust LLM module not available")
        self._inner = _llm.LazyModelLoader(memory_budget, enable_eviction)

    def load_file(self, path: str) -> None:
        """Load a SafeTensors file (mmap only, no GPU transfer yet).

        Args:
            path: Path to the SafeTensors file
        """
        self._inner.load_file(path)

    def get(self, name: str) -> TensorInfo | None:
        """Get tensor info by name.

        Args:
            name: Tensor name

        Returns:
            TensorInfo or None if not found
        """
        info = self._inner.get(name)
        if info is None:
            return None
        return TensorInfo(
            name=info.name,
            dtype=int(info.dtype),
            shape=info.shape,
            offset=info.offset,
            size_bytes=info.size_bytes,
        )

    @property
    def tensor_names(self) -> list[str]:
        """Get all tensor names."""
        return self._inner.tensor_names

    @property
    def total_size(self) -> int:
        """Get total model size in bytes (all files)."""
        return self._inner.total_size

    @property
    def loaded_size(self) -> int:
        """Get currently loaded size on GPU."""
        return self._inner.loaded_size

    @property
    def pool_stats(self) -> PoolStats:
        """Get memory pool statistics."""
        stats = self._inner.pool_stats
        return PoolStats(
            quota=stats.quota,
            used=stats.used,
            cached=stats.cached,
            available=stats.available,
            allocation_count=stats.allocation_count,
            reuse_count=stats.reuse_count,
            eviction_count=stats.eviction_count,
            cudamalloc_count=stats.cudamalloc_count,
            active_blocks=stats.active_blocks,
            free_blocks=stats.free_blocks,
        )

    @property
    def num_tensors(self) -> int:
        """Number of tensors in all files."""
        return self._inner.num_tensors

    @property
    def num_files(self) -> int:
        """Number of files loaded."""
        return self._inner.num_files

    def loaded_tensors(self) -> list[str]:
        """Get list of tensor names currently on GPU."""
        return self._inner.loaded_tensors()

    def num_loaded(self) -> int:
        """Get number of tensors currently on GPU."""
        return self._inner.num_loaded()

    def unload_model(self) -> int:
        """Unload entire model from GPU.

        Releases all GPU memory but keeps mmap references.
        Tensors can be reloaded by accessing them again.

        Returns:
            Number of bytes freed
        """
        return self._inner.unload_model()

    def unload_layer(self, prefix: str) -> tuple[int, int]:
        """Unload tensors matching a prefix.

        Useful for unloading specific transformer layers.
        E.g., prefix "model.layers.0." unloads all tensors in layer 0.

        Args:
            prefix: Tensor name prefix to match

        Returns:
            Tuple of (num_tensors_unloaded, bytes_freed)
        """
        return self._inner.unload_layer(prefix)

    def unload_tensors(self, names: list[str]) -> tuple[int, int]:
        """Unload specific tensors by name.

        Args:
            names: List of tensor names to unload

        Returns:
            Tuple of (num_tensors_unloaded, bytes_freed)
        """
        return self._inner.unload_tensors(names)

    def clear(self) -> int:
        """Clear all data (unload tensors + close mmaps).

        After this, the loader cannot be used until new files are loaded.

        Returns:
            Number of bytes freed from GPU
        """
        return self._inner.clear()

    def get_layer_tensors(self, prefix: str) -> list[str]:
        """Get tensor names matching a prefix.

        Args:
            prefix: Tensor name prefix to match (e.g., "model.layers.0.")

        Returns:
            List of tensor names matching the prefix
        """
        return self._inner.get_layer_tensors(prefix)

    def layer_size(self, prefix: str) -> int:
        """Get total size of tensors matching a prefix.

        Args:
            prefix: Tensor name prefix to match

        Returns:
            Total size in bytes
        """
        return self._inner.layer_size(prefix)

    def is_layer_loaded(self, prefix: str) -> bool:
        """Check if a layer is fully loaded on GPU.

        Args:
            prefix: Tensor name prefix to match

        Returns:
            True if all tensors in the layer are on GPU
        """
        return self._inner.is_layer_loaded(prefix)

    def layer_state(self, prefix: str) -> tuple[int, int, int, int]:
        """Get layer loading state.

        Args:
            prefix: Tensor name prefix to match

        Returns:
            Tuple of (total_tensors, loaded_tensors, total_bytes, loaded_bytes)
        """
        return self._inner.layer_state(prefix)

    def __len__(self) -> int:
        return self.num_tensors

    def __contains__(self, name: str) -> bool:
        return name in self._inner

    def __repr__(self) -> str:
        return (
            f"LazyModelLoader(files={self.num_files}, tensors={self.num_tensors}, "
            f"loaded={self.num_loaded()}/{self.num_tensors})"
        )


__all__ = [
    "Dtype",
    "TensorInfo",
    "SafeTensorsFile",
    "ShardedSafeTensorsFile",
    "LazyModelLoader",
    "TensorState",
    "PoolStats",
    "load_safetensors",
]
