"""MLP layer implementation for PyGPUkit LLM.

Provides:
- MLP: Unified MLP supporting GELU and SwiGLU activations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pygpukit.core.array import GPUArray
from pygpukit.ops.basic import (
    concat_axis0,
    gelu,
    mul,
    silu,
)

from .linear import LinearBF16, LinearFP8

if TYPE_CHECKING:
    from pygpukit.llm.config import TransformerConfig


class MLP:
    """Unified MLP supporting GELU and SwiGLU activations.

    GELU (GPT-2 style):
        fc1 -> GELU -> fc2

    SwiGLU (LLaMA style):
        gate_proj -> SiLU -> * up_proj -> down_proj

    Supports FP8 quantized weights via LinearFP8.
    """

    def __init__(
        self,
        config: TransformerConfig,
        # GELU path weights (GPUArray or LinearBF16/LinearFP8)
        fc1_weight: GPUArray | LinearBF16 | LinearFP8 | None = None,
        fc1_bias: GPUArray | None = None,
        fc2_weight: GPUArray | LinearBF16 | LinearFP8 | None = None,
        fc2_bias: GPUArray | None = None,
        # SwiGLU path weights (GPUArray or LinearBF16/LinearFP8)
        gate_proj: GPUArray | LinearBF16 | LinearFP8 | None = None,
        up_proj: GPUArray | LinearBF16 | LinearFP8 | None = None,
        down_proj: GPUArray | LinearBF16 | LinearFP8 | None = None,
    ):
        self.config = config
        self.activation = config.activation

        # Helper to wrap GPUArray in LinearBF16, or use pre-built LinearBF16/LinearFP8
        def wrap_linear(
            proj: GPUArray | LinearBF16 | LinearFP8 | None, bias: GPUArray | None = None
        ) -> LinearBF16 | LinearFP8 | None:
            if proj is None:
                return None
            if isinstance(proj, (LinearBF16, LinearFP8)):
                return proj
            return LinearBF16(proj, bias)

        if config.activation == "gelu":
            if fc1_weight is None or fc2_weight is None:
                raise ValueError("GELU MLP requires fc1_weight and fc2_weight")
            self.fc1 = wrap_linear(fc1_weight, fc1_bias)
            self.fc2 = wrap_linear(fc2_weight, fc2_bias)
        else:  # silu (SwiGLU)
            if gate_proj is None or up_proj is None or down_proj is None:
                raise ValueError("SwiGLU MLP requires gate_proj, up_proj, down_proj")

            self.gate_proj = wrap_linear(gate_proj)
            self.up_proj = wrap_linear(up_proj)
            self.down_proj = wrap_linear(down_proj)

            # Get intermediate size from the projection
            if isinstance(gate_proj, (LinearBF16, LinearFP8)):
                self.intermediate_size = gate_proj.out_features
            else:
                self.intermediate_size = gate_proj.shape[0]

            # Fused gate_up projection only for non-FP8 (GPUArray) weights
            # FP8 weights can't be concatenated trivially
            if isinstance(gate_proj, GPUArray) and isinstance(up_proj, GPUArray):
                gate_up_weight = concat_axis0(gate_proj, up_proj)
                self.gate_up_proj: LinearBF16 | None = LinearBF16(gate_up_weight, None)
            else:
                self.gate_up_proj = None

    def __call__(self, x: GPUArray) -> GPUArray:
        if self.activation == "gelu":
            h = self.fc1(x)
            h = gelu(h)
            return self.fc2(h)
        else:
            gate = silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(mul(gate, up))


__all__ = [
    "MLP",
]
