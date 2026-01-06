"""Convolution operations.

Provides GPU-accelerated 1D convolution operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy

if TYPE_CHECKING:
    pass


def conv1d(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> GPUArray:
    """1D convolution.

    Args:
        input: Input tensor [batch, in_channels, length]
        weight: Weight tensor [out_channels, in_channels, kernel_size]
        bias: Optional bias tensor [out_channels]
        stride: Convolution stride (default: 1)
        padding: Input padding (default: 0)

    Returns:
        Output tensor [batch, out_channels, out_length]

    Example:
        >>> import pygpukit as pk
        >>> x = pk.GPUArray([1, 80, 3000], dtype='float32')  # [batch, mel_bins, time]
        >>> w = pk.GPUArray([256, 80, 3], dtype='float32')   # [out_ch, in_ch, kernel]
        >>> b = pk.GPUArray([256], dtype='float32')          # [out_ch]
        >>> y = pk.ops.conv1d(x, w, b, stride=1, padding=1)
        >>> print(y.shape)  # [1, 256, 3000]
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _conv1d_native(input, weight, bias, stride, padding)
    else:
        return _conv1d_cpu(input, weight, bias, stride, padding)


def _conv1d_native(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None,
    stride: int,
    padding: int,
) -> GPUArray:
    """Native CUDA conv1d implementation."""
    from pygpukit._native_loader import get_native_module

    native = get_native_module()

    input_native = input._get_native()
    weight_native = weight._get_native()

    if bias is not None:
        bias_native = bias._get_native()
        result_native = native.conv1d_bias(
            input_native, weight_native, bias_native, stride, padding
        )
    else:
        result_native = native.conv1d(input_native, weight_native, stride, padding)

    return GPUArray._wrap_native(result_native)


def _conv1d_cpu(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None,
    stride: int,
    padding: int,
) -> GPUArray:
    """CPU fallback using im2col + matmul."""
    x_np = input.to_numpy()
    w_np = weight.to_numpy()
    b_np = bias.to_numpy() if bias is not None else None

    batch, in_channels, length = x_np.shape
    out_channels, _, kernel_size = w_np.shape

    # Apply padding
    if padding > 0:
        x_np = np.pad(x_np, ((0, 0), (0, 0), (padding, padding)), mode="constant")

    # Compute output length
    out_length = (x_np.shape[2] - kernel_size) // stride + 1

    # im2col: extract patches
    col = np.zeros((batch, in_channels * kernel_size, out_length), dtype=x_np.dtype)
    for i in range(out_length):
        start = i * stride
        end = start + kernel_size
        col[:, :, i] = x_np[:, :, start:end].reshape(batch, -1)

    # matmul
    w_flat = w_np.reshape(out_channels, -1)
    out = np.zeros((batch, out_channels, out_length), dtype=x_np.dtype)
    for b_idx in range(batch):
        out[b_idx] = w_flat @ col[b_idx]

    # Add bias
    if b_np is not None:
        out = out + b_np.reshape(1, -1, 1)

    return from_numpy(out)
