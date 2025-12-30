"""Activation functions for GPUArrays.

Corresponds to native/ops/nn/activation/.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype


def gelu(a: GPUArray) -> GPUArray:
    """GELU (Gaussian Error Linear Unit) activation.

    Computes: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        a: Input array (float32, float64, float16, or bfloat16).

    Returns:
        A new GPUArray containing gelu(a).

    Raises:
        ValueError: If dtype is not a float type.
    """
    _validate_float_dtype(a, "gelu")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _gelu_native(a)
    else:
        return _gelu_cpu(a)


def _gelu_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of gelu."""
    a_np = a.to_numpy()
    # GELU approximation
    x = a_np.astype(np.float32) if a_np.dtype in [np.float16] else a_np
    c1 = 0.7978845608  # sqrt(2/pi)
    c2 = 0.044715
    result = x * 0.5 * (1 + np.tanh(c1 * (x + c2 * x**3)))
    return from_numpy(result.astype(a_np.dtype))


def _gelu_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of gelu (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.gelu(a_native)
    return GPUArray._wrap_native(c_native)


def silu(a: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """SiLU (Swish) activation: y = x * sigmoid(x).

    Used in Llama and other modern LLMs as the activation in MLP layers.

    Args:
        a: Input array.
        out: Optional pre-allocated output array. If provided, the result
            is written to this array (for CUDA Graph capture support).

    Returns:
        A new GPUArray containing the SiLU-activated values, or the out array if provided.

    Raises:
        ValueError: If dtype is not a float type.
    """
    _validate_float_dtype(a, "silu")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _silu_native(a, out=out)
    else:
        return _silu_cpu(a)


def _silu_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of SiLU."""
    x = a.to_numpy()
    # SiLU = x * sigmoid(x) = x / (1 + exp(-x))
    result = x / (1.0 + np.exp(-x))
    return from_numpy(result)


def _silu_native(a: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """Native C++ CUDA implementation of SiLU (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()

    if out is not None:
        out_native = out._get_native()
        native.silu_(a_native, out_native)
        return out
    else:
        c_native = native.silu(a_native)
        return GPUArray._wrap_native(c_native)


def sigmoid(a: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """Sigmoid activation: y = 1 / (1 + exp(-x)).

    Args:
        a: Input array.
        out: Optional pre-allocated output array.

    Returns:
        A new GPUArray containing the sigmoid-activated values.
    """
    _validate_float_dtype(a, "sigmoid")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        a_native = a._get_native()

        if out is not None:
            out_native = out._get_native()
            native.sigmoid_(a_native, out_native)
            return out
        else:
            return GPUArray._wrap_native(native.sigmoid(a_native))
    else:
        x = a.to_numpy()
        result = 1.0 / (1.0 + np.exp(-x))
        return from_numpy(result)


def tanh(a: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """Tanh activation.

    Args:
        a: Input array.
        out: Optional pre-allocated output array.

    Returns:
        A new GPUArray containing the tanh-activated values.
    """
    _validate_float_dtype(a, "tanh")
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        a_native = a._get_native()

        if out is not None:
            out_native = out._get_native()
            native.tanh_(a_native, out_native)
            return out
        else:
            return GPUArray._wrap_native(native.tanh(a_native))
    else:
        x = a.to_numpy()
        return from_numpy(np.tanh(x))


__all__ = [
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
]
