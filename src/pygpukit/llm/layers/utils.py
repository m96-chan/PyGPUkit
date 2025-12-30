"""Weight repacking utilities for PyGPUkit LLM.

Provides:
- repack_weight: Repack weight tensor into contiguous GPU buffer
- repack_linear: Repack LinearBF16 layer weights in-place
- repack_norm: Repack Norm layer weights in-place
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy

from .linear import LinearBF16
from .norm import Norm


def repack_weight(weight: GPUArray) -> GPUArray:
    """Repack a weight tensor into a new contiguous GPU buffer.

    This fixes performance issues caused by fragmented GPU memory allocation.
    Weights allocated later during model loading may end up in suboptimal
    memory regions, causing 7x slower matmul performance.

    Args:
        weight: Original weight tensor on GPU

    Returns:
        New GPUArray with same data in freshly allocated contiguous memory
    """
    # Copy to CPU, then back to GPU to get fresh allocation
    # This ensures the new buffer is allocated contiguously
    weight_np = weight.to_numpy()
    return from_numpy(weight_np)


def repack_linear(linear: LinearBF16) -> None:
    """Repack a LinearBF16 layer's weight in-place.

    Args:
        linear: LinearBF16 layer to repack
    """
    linear.weight = repack_weight(linear.weight)
    # Clear transpose cache - will be regenerated on first use
    linear._weight_t = None
    if linear.bias is not None:
        linear.bias = repack_weight(linear.bias)


def repack_norm(norm: Norm) -> None:
    """Repack a Norm layer's weight in-place.

    Args:
        norm: Norm layer to repack
    """
    norm.weight = repack_weight(norm.weight)
    if norm.bias is not None:
        norm.bias = repack_weight(norm.bias)


__all__ = [
    "repack_weight",
    "repack_linear",
    "repack_norm",
]
