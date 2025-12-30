"""Normalization layer implementations for PyGPUkit LLM.

Provides:
- Norm: Unified RMSNorm and LayerNorm
"""

from __future__ import annotations

from typing import Literal

from pygpukit.core.array import GPUArray
from pygpukit.ops.basic import (
    layernorm,
    rmsnorm,
)


class Norm:
    """Unified normalization layer supporting RMSNorm and LayerNorm."""

    def __init__(
        self,
        weight: GPUArray,
        bias: GPUArray | None = None,
        norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
        eps: float = 1e-5,
    ):
        self.weight = weight
        self.bias = bias
        self.norm_type = norm_type
        self.eps = eps

    def __call__(self, x: GPUArray) -> GPUArray:
        if self.norm_type == "rmsnorm":
            return rmsnorm(x, self.weight, self.eps)
        else:
            if self.bias is None:
                raise ValueError("LayerNorm requires bias")
            return layernorm(x, self.weight, self.bias, self.eps)


__all__ = [
    "Norm",
]
