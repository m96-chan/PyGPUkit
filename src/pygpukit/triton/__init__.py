"""
PyGPUkit Triton Backend.

Provides Triton-based GPU kernels for rapid prototyping without PyTorch dependency.

Usage:
    import numpy as np
    from pygpukit.triton import from_gpuarray, from_numpy, kernels
    import pygpukit._pygpukit_native as native

    # Method 1: From GPUArray
    x = native.from_numpy(np.random.randn(4, 128).astype(np.float32))
    w = native.from_numpy(np.random.randn(128).astype(np.float32))
    out = native.empty([4, 128], native.Float32)

    tx, tw, tout = from_gpuarray(x), from_gpuarray(w), from_gpuarray(out)
    kernels.rmsnorm(tx, tw, tout)

    # Method 2: Direct from NumPy
    tx = from_numpy(np.random.randn(4, 128).astype(np.float32))
"""

from . import kernels
from .backend import triton_available, triton_version, use_triton_backend
from .wrapper import TritonArray, from_gpuarray, from_numpy

__all__ = [
    "TritonArray",
    "from_gpuarray",
    "from_numpy",
    "triton_available",
    "triton_version",
    "use_triton_backend",
    "kernels",
]
