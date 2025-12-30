"""Normalization layers for GPUArrays.

Corresponds to native/ops/nn/norm/.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype


def layernorm(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float = 1e-5,
) -> GPUArray:
    """Layer normalization.

    Computes: (x - mean) / sqrt(var + eps) * gamma + beta

    Args:
        input: Input array of shape [batch, features] or [batch, seq_len, features].
        gamma: Scale parameter of shape [features].
        beta: Bias parameter of shape [features].
        eps: Small epsilon for numerical stability.

    Returns:
        A new GPUArray containing the normalized output.

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(input, "layernorm")

    if input.ndim not in (2, 3):
        raise ValueError(f"layernorm expects 2D or 3D input, got {input.ndim}D")
    if gamma.ndim != 1 or beta.ndim != 1:
        raise ValueError("layernorm expects 1D gamma and beta")
    if input.dtype != gamma.dtype or input.dtype != beta.dtype:
        raise ValueError("layernorm: all inputs must have same dtype")

    features = input.shape[-1]  # Last dimension is features
    if gamma.shape[0] != features or beta.shape[0] != features:
        raise ValueError(
            f"layernorm: gamma/beta size {gamma.shape[0]} must match features {features}"
        )

    # Handle 3D input by reshaping to 2D, processing, and reshaping back
    if input.ndim == 3:
        batch, seq_len, feat = input.shape
        input_2d = input.reshape(batch * seq_len, feat)
        result_2d = _layernorm_dispatch(input_2d, gamma, beta, eps)
        return result_2d.reshape(batch, seq_len, feat)
    else:
        return _layernorm_dispatch(input, gamma, beta, eps)


def _layernorm_dispatch(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float,
) -> GPUArray:
    """Dispatch layernorm to native or CPU implementation."""
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _layernorm_native(input, gamma, beta, eps)
    else:
        return _layernorm_cpu(input, gamma, beta, eps)


def _layernorm_cpu(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float,
) -> GPUArray:
    """CPU implementation of layernorm."""
    x = input.to_numpy()
    g = gamma.to_numpy()
    b = beta.to_numpy()

    # Compute mean and variance along features axis
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)

    # Normalize
    normalized = (x - mean) / np.sqrt(var + eps)

    # Apply affine transform
    result = normalized * g + b
    return from_numpy(result)


def _layernorm_native(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float,
) -> GPUArray:
    """Native C++ CUDA implementation of layernorm (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    gamma_native = gamma._get_native()
    beta_native = beta._get_native()
    c_native = native.layernorm(input_native, gamma_native, beta_native, eps)
    return GPUArray._wrap_native(c_native)


def rmsnorm(
    input: GPUArray,
    gamma: GPUArray,
    eps: float = 1e-5,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """RMS Normalization (Root Mean Square Normalization).

    Computes: x / sqrt(mean(x^2) + eps) * gamma

    Simpler than LayerNorm (no mean subtraction, no beta).
    Used in Llama and other modern LLMs.

    Args:
        input: Input array of shape [batch, features].
        gamma: Scale parameter of shape [features].
        eps: Small epsilon for numerical stability.
        out: Optional output buffer. If provided, result is written in-place
            (for CUDA Graph capture).

    Returns:
        A new GPUArray containing the normalized output (or out if provided).

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(input, "rmsnorm")

    if input.ndim != 2:
        raise ValueError(f"rmsnorm expects 2D input [batch, features], got {input.ndim}D")
    if gamma.ndim != 1:
        raise ValueError("rmsnorm expects 1D gamma")
    if input.dtype != gamma.dtype:
        raise ValueError("rmsnorm: all inputs must have same dtype")

    features = input.shape[1]
    if gamma.shape[0] != features:
        raise ValueError(f"rmsnorm: gamma size {gamma.shape[0]} must match features {features}")

    # Validate out array if provided
    if out is not None:
        if out.shape != input.shape:
            raise ValueError(f"out shape {out.shape} does not match input shape {input.shape}")
        if out.dtype != input.dtype:
            raise ValueError(f"out dtype {out.dtype} does not match input dtype {input.dtype}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _rmsnorm_native(input, gamma, eps, out=out)
    else:
        return _rmsnorm_cpu(input, gamma, eps, out=out)


def _rmsnorm_cpu(
    input: GPUArray,
    gamma: GPUArray,
    eps: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """CPU implementation of rmsnorm."""
    x = input.to_numpy()
    g = gamma.to_numpy()

    # RMS = sqrt(mean(x^2) + eps)
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + eps)

    # Normalize and scale
    result = (x / rms) * g

    if out is not None:
        out_np = out.to_numpy()
        np.copyto(out_np, result)
        out._data = from_numpy(out_np)._data
        return out
    return from_numpy(result)


def _rmsnorm_native(
    input: GPUArray,
    gamma: GPUArray,
    eps: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ CUDA implementation of rmsnorm (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    gamma_native = gamma._get_native()

    if out is not None:
        out_native = out._get_native()
        native.rmsnorm_(input_native, gamma_native, out_native, eps)
        return out
    else:
        c_native = native.rmsnorm(input_native, gamma_native, eps)
        return GPUArray._wrap_native(c_native)


__all__ = [
    "layernorm",
    "rmsnorm",
]
