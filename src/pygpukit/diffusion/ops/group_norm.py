"""Group Normalization for diffusion models.

GroupNorm is essential for VAE and UNet architectures where BatchNorm
is not suitable due to small batch sizes during inference.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy


def group_norm(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    num_groups: int,
    eps: float = 1e-5,
) -> GPUArray:
    """Group Normalization.

    Divides channels into groups and normalizes within each group.
    Used extensively in VAE and UNet architectures.

    Args:
        input: Input tensor of shape [N, C, H, W] (NCHW format).
        gamma: Scale parameter of shape [C].
        beta: Bias parameter of shape [C].
        num_groups: Number of groups to divide channels into.
        eps: Small epsilon for numerical stability.

    Returns:
        Normalized tensor of shape [N, C, H, W].

    Raises:
        ValueError: If C is not divisible by num_groups.
    """
    if input.ndim != 4:
        raise ValueError(f"group_norm expects 4D input [N, C, H, W], got {input.ndim}D")

    N, C, H, W = input.shape

    if C % num_groups != 0:
        raise ValueError(f"Channels {C} must be divisible by num_groups {num_groups}")

    if gamma.shape != (C,) or beta.shape != (C,):
        raise ValueError(f"gamma/beta must have shape [{C}]")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _group_norm_native(input, gamma, beta, num_groups, eps)
    else:
        return _group_norm_cpu(input, gamma, beta, num_groups, eps)


def _group_norm_cpu(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    num_groups: int,
    eps: float,
) -> GPUArray:
    """CPU implementation of GroupNorm."""
    x = input.to_numpy()
    g = gamma.to_numpy()
    b = beta.to_numpy()

    N, C, H, W = x.shape
    channels_per_group = C // num_groups

    # Reshape to [N, num_groups, channels_per_group, H, W]
    x_reshaped = x.reshape(N, num_groups, channels_per_group, H, W)

    # Compute mean and variance over (channels_per_group, H, W)
    mean = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)
    var = x_reshaped.var(axis=(2, 3, 4), keepdims=True)

    # Normalize
    x_norm = (x_reshaped - mean) / np.sqrt(var + eps)

    # Reshape back to [N, C, H, W]
    x_norm = x_norm.reshape(N, C, H, W)

    # Apply affine transform (broadcast over spatial dimensions)
    result = x_norm * g.reshape(1, C, 1, 1) + b.reshape(1, C, 1, 1)

    return from_numpy(result.astype(x.dtype))


def _group_norm_native(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    num_groups: int,
    eps: float,
) -> GPUArray:
    """Native CUDA implementation of GroupNorm."""
    # TODO: Implement native CUDA kernel for GroupNorm
    # For now, fall back to CPU implementation
    return _group_norm_cpu(input, gamma, beta, num_groups, eps)


def group_norm_silu(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    num_groups: int,
    eps: float = 1e-5,
) -> GPUArray:
    """Fused GroupNorm + SiLU activation.

    Combines GroupNorm with SiLU activation for better performance.
    Common pattern in VAE decoder blocks.

    Args:
        input: Input tensor of shape [N, C, H, W].
        gamma: Scale parameter of shape [C].
        beta: Bias parameter of shape [C].
        num_groups: Number of groups.
        eps: Epsilon for numerical stability.

    Returns:
        GroupNorm(x) * sigmoid(GroupNorm(x))
    """
    normalized = group_norm(input, gamma, beta, num_groups, eps)

    # Apply SiLU: x * sigmoid(x)
    x = normalized.to_numpy()
    result = x * (1.0 / (1.0 + np.exp(-x)))
    return from_numpy(result.astype(x.dtype))


__all__ = [
    "group_norm",
    "group_norm_silu",
]
