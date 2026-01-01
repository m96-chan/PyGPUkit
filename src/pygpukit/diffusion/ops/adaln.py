"""Adaptive Layer Normalization for DiT models.

AdaLN and AdaLN-Zero are key components of Diffusion Transformers (DiT),
providing timestep-conditioned normalization.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy


def adaln(
    x: GPUArray,
    scale: GPUArray,
    shift: GPUArray,
    eps: float = 1e-5,
) -> GPUArray:
    """Adaptive Layer Normalization.

    Applies layer normalization with learned scale and shift from conditioning:
        y = (1 + scale) * LayerNorm(x) + shift

    Args:
        x: Input tensor [B, N, D].
        scale: Scale parameter [B, D] from conditioning MLP.
        shift: Shift parameter [B, D] from conditioning MLP.
        eps: Epsilon for numerical stability.

    Returns:
        Output tensor [B, N, D].
    """
    if x.ndim != 3:
        raise ValueError(f"adaln expects 3D input [B, N, D], got {x.ndim}D")
    if scale.ndim != 2 or shift.ndim != 2:
        raise ValueError("scale and shift must be 2D [B, D]")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _adaln_native(x, scale, shift, eps)
    else:
        return _adaln_cpu(x, scale, shift, eps)


def _adaln_cpu(
    x: GPUArray,
    scale: GPUArray,
    shift: GPUArray,
    eps: float,
) -> GPUArray:
    """CPU implementation of AdaLN."""
    x_np = x.to_numpy()
    scale_np = scale.to_numpy()
    shift_np = shift.to_numpy()

    B, N, D = x_np.shape

    # Layer normalization
    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    x_norm = (x_np - mean) / np.sqrt(var + eps)

    # Apply adaptive scale and shift
    # scale, shift: [B, D] -> [B, 1, D]
    scale_np = scale_np[:, np.newaxis, :]
    shift_np = shift_np[:, np.newaxis, :]

    output = (1.0 + scale_np) * x_norm + shift_np

    return from_numpy(output.astype(x_np.dtype))


def _adaln_native(
    x: GPUArray,
    scale: GPUArray,
    shift: GPUArray,
    eps: float,
) -> GPUArray:
    """Native CUDA implementation of AdaLN."""
    # TODO: Implement native CUDA kernel
    return _adaln_cpu(x, scale, shift, eps)


def adaln_zero(
    x: GPUArray,
    scale: GPUArray,
    shift: GPUArray,
    gate: GPUArray,
    residual: GPUArray,
    eps: float = 1e-5,
) -> GPUArray:
    """Adaptive Layer Normalization with Zero-Init Gating.

    Used in DiT for gated residual connections:
        y = residual + gate * f(adaln(x))

    Where f is the attention/mlp output and this function computes
    the adaln part with gating applied.

    Args:
        x: Input tensor [B, N, D] (e.g., attention output).
        scale: Scale parameter [B, D] from conditioning MLP.
        shift: Shift parameter [B, D] from conditioning MLP.
        gate: Gate parameter [B, D] (initialized to zero).
        residual: Residual input [B, N, D].
        eps: Epsilon for numerical stability.

    Returns:
        Output tensor [B, N, D].
    """
    if x.ndim != 3:
        raise ValueError(f"adaln_zero expects 3D input [B, N, D], got {x.ndim}D")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _adaln_zero_native(x, scale, shift, gate, residual, eps)
    else:
        return _adaln_zero_cpu(x, scale, shift, gate, residual, eps)


def _adaln_zero_cpu(
    x: GPUArray,
    scale: GPUArray,
    shift: GPUArray,
    gate: GPUArray,
    residual: GPUArray,
    eps: float,
) -> GPUArray:
    """CPU implementation of AdaLN-Zero."""
    x_np = x.to_numpy()
    scale_np = scale.to_numpy()
    shift_np = shift.to_numpy()
    gate_np = gate.to_numpy()
    residual_np = residual.to_numpy()

    B, N, D = x_np.shape

    # Layer normalization
    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    x_norm = (x_np - mean) / np.sqrt(var + eps)

    # Apply adaptive scale and shift
    scale_np = scale_np[:, np.newaxis, :]
    shift_np = shift_np[:, np.newaxis, :]
    gate_np = gate_np[:, np.newaxis, :]

    adaln_out = (1.0 + scale_np) * x_norm + shift_np

    # Apply gate and add residual
    output = residual_np + gate_np * adaln_out

    return from_numpy(output.astype(x_np.dtype))


def _adaln_zero_native(
    x: GPUArray,
    scale: GPUArray,
    shift: GPUArray,
    gate: GPUArray,
    residual: GPUArray,
    eps: float,
) -> GPUArray:
    """Native CUDA implementation of AdaLN-Zero."""
    # TODO: Implement native CUDA kernel
    return _adaln_zero_cpu(x, scale, shift, gate, residual, eps)


def modulation(
    conditioning: GPUArray,
    linear_weight: GPUArray,
    linear_bias: GPUArray,
    num_outputs: int = 6,
) -> list[GPUArray]:
    """Compute modulation parameters from conditioning.

    Common pattern in DiT: project conditioning to multiple modulation params.

    Args:
        conditioning: Conditioning vector [B, D].
        linear_weight: Projection weight [num_outputs * D, D].
        linear_bias: Projection bias [num_outputs * D].
        num_outputs: Number of modulation parameters (typically 6).

    Returns:
        List of num_outputs tensors, each [B, D].
    """
    c = conditioning.to_numpy()
    w = linear_weight.to_numpy()
    b = linear_bias.to_numpy()

    # Linear projection
    out = np.dot(c, w.T) + b

    d_per_output = out.shape[1] // num_outputs

    # Split into num_outputs parts
    outputs = []
    for i in range(num_outputs):
        part = out[:, i * d_per_output : (i + 1) * d_per_output]
        outputs.append(from_numpy(part.astype(c.dtype)))

    return outputs


__all__ = [
    "adaln",
    "adaln_zero",
    "modulation",
]
