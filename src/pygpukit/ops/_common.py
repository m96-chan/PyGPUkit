"""Common utilities for ops modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray


def _validate_same_shape(a: GPUArray, b: GPUArray, op_name: str) -> None:
    """Validate that two arrays have the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"{op_name} requires arrays of same shape, got {a.shape} and {b.shape}")


def _validate_same_dtype(a: GPUArray, b: GPUArray, op_name: str) -> None:
    """Validate that two arrays have the same dtype."""
    if a.dtype != b.dtype:
        raise ValueError(f"{op_name} requires arrays of same dtype, got {a.dtype} and {b.dtype}")


def _validate_float_dtype(a: GPUArray, op_name: str) -> None:
    """Validate that array has float dtype."""
    from pygpukit.core.dtypes import bfloat16, float16, float32, float64

    if a.dtype not in (float32, float64, float16, bfloat16):
        raise ValueError(f"{op_name} requires float dtype, got {a.dtype}")
