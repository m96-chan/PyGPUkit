"""Memory management module for PyGPUkit."""

from pygpukit.memory.pool import (
    MemoryBlock,
    MemoryPool,
    get_default_pool,
    set_default_pool,
)

# Rust memory pool (v0.2+)
# Import Rust implementation if available
try:
    import _pygpukit_rust._pygpukit_rust as _rust

    RustMemoryPool = _rust.MemoryPool
    RustMemoryBlock = _rust.MemoryBlock
    RustPoolStats = _rust.PoolStats
    HAS_RUST_BACKEND = True
except ImportError:
    RustMemoryPool = None  # type: ignore
    RustMemoryBlock = None  # type: ignore
    RustPoolStats = None  # type: ignore
    HAS_RUST_BACKEND = False

__all__ = [
    "MemoryBlock",
    "MemoryPool",
    "get_default_pool",
    "set_default_pool",
    # Rust backend (v0.2+)
    "RustMemoryPool",
    "RustMemoryBlock",
    "RustPoolStats",
    "HAS_RUST_BACKEND",
]
