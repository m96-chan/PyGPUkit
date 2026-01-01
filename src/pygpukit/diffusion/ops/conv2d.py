"""2D Convolution for diffusion models.

Provides conv2d and conv2d_transpose operations for VAE and UNet.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy


def conv2d(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> GPUArray:
    """2D Convolution.

    Args:
        input: Input tensor [N, C_in, H, W].
        weight: Filter weights [C_out, C_in/groups, K_h, K_w].
        bias: Optional bias [C_out].
        stride: Stride for convolution.
        padding: Padding for input.
        dilation: Dilation for filter.
        groups: Number of groups for grouped convolution.

    Returns:
        Output tensor [N, C_out, H_out, W_out].
    """
    if input.ndim != 4:
        raise ValueError(f"conv2d expects 4D input, got {input.ndim}D")
    if weight.ndim != 4:
        raise ValueError(f"conv2d expects 4D weight, got {weight.ndim}D")

    # Normalize stride, padding, dilation to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _conv2d_native(input, weight, bias, stride, padding, dilation, groups)
    else:
        return _conv2d_cpu(input, weight, bias, stride, padding, dilation, groups)


def _conv2d_cpu(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> GPUArray:
    """CPU implementation of conv2d using im2col."""
    x = input.to_numpy()
    w = weight.to_numpy()

    N, C_in, H, W = x.shape
    C_out, C_in_per_group, K_h, K_w = w.shape

    if C_in != C_in_per_group * groups:
        raise ValueError(
            f"Input channels {C_in} != weight channels {C_in_per_group} * groups {groups}"
        )

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # Calculate output dimensions
    H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    # Apply padding
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    # Im2col: extract patches
    # Output shape: [N, C_in, K_h, K_w, H_out, W_out]
    patches = np.zeros((N, C_in, K_h, K_w, H_out, W_out), dtype=x.dtype)

    for kh in range(K_h):
        for kw in range(K_w):
            h_start = kh * dil_h
            w_start = kw * dil_w
            patches[:, :, kh, kw, :, :] = x[
                :,
                :,
                h_start : h_start + H_out * stride_h : stride_h,
                w_start : w_start + W_out * stride_w : stride_w,
            ]

    # Reshape for matrix multiplication
    # patches: [N, C_in * K_h * K_w, H_out * W_out]
    patches = patches.reshape(N, C_in * K_h * K_w, H_out * W_out)

    # weight: [C_out, C_in * K_h * K_w]
    w_reshaped = w.reshape(C_out, C_in_per_group * K_h * K_w)

    if groups == 1:
        # Standard convolution
        output = np.matmul(w_reshaped, patches)  # [N, C_out, H_out * W_out]
    else:
        # Grouped convolution
        C_out_per_group = C_out // groups
        output = np.zeros((N, C_out, H_out * W_out), dtype=x.dtype)
        for g in range(groups):
            g_in_start = g * C_in_per_group
            g_in_end = (g + 1) * C_in_per_group
            g_out_start = g * C_out_per_group
            g_out_end = (g + 1) * C_out_per_group

            patches_g = patches[:, g_in_start * K_h * K_w : g_in_end * K_h * K_w, :]
            w_g = w_reshaped[g_out_start:g_out_end, :]
            output[:, g_out_start:g_out_end, :] = np.matmul(w_g, patches_g)

    # Reshape to [N, C_out, H_out, W_out]
    output = output.reshape(N, C_out, H_out, W_out)

    # Add bias
    if bias is not None:
        b = bias.to_numpy()
        output = output + b.reshape(1, C_out, 1, 1)

    return from_numpy(output.astype(x.dtype))


def _conv2d_native(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> GPUArray:
    """Native CUDA implementation of conv2d."""
    # Check if we can use optimized kernels
    _, C_in_per_group, K_h, K_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # Optimized path for 1x1 conv (no padding, no dilation, stride=1, groups=1)
    if K_h == 1 and K_w == 1 and groups == 1 and dil_h == 1 and dil_w == 1:
        if stride_h == 1 and stride_w == 1 and pad_h == 0 and pad_w == 0:
            try:
                from pygpukit._pygpukit_native import conv2d_1x1 as native_conv2d_1x1

                # Reshape weight from [C_out, C_in, 1, 1] to [C_out, C_in]
                w_np = weight.to_numpy().squeeze(-1).squeeze(-1)
                w_2d = from_numpy(w_np)

                if bias is not None:
                    result = native_conv2d_1x1(input._array, w_2d._array, bias._array)
                else:
                    result = native_conv2d_1x1(input._array, w_2d._array, None)
                return GPUArray._from_native(result)
            except (ImportError, AttributeError):
                pass

    # Optimized path for 3x3 conv (dilation=1, groups=1)
    if K_h == 3 and K_w == 3 and groups == 1 and dil_h == 1 and dil_w == 1:
        try:
            from pygpukit._pygpukit_native import conv2d_3x3 as native_conv2d_3x3

            if bias is not None:
                result = native_conv2d_3x3(
                    input._array, weight._array, bias._array, pad_h, pad_w, stride_h, stride_w
                )
            else:
                result = native_conv2d_3x3(
                    input._array, weight._array, None, pad_h, pad_w, stride_h, stride_w
                )
            return GPUArray._from_native(result)
        except (ImportError, AttributeError):
            pass

    # Fall back to CPU for other cases
    return _conv2d_cpu(input, weight, bias, stride, padding, dilation, groups)


def conv2d_transpose(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    groups: int = 1,
    dilation: int | tuple[int, int] = 1,
) -> GPUArray:
    """Transposed 2D Convolution (Deconvolution).

    Used for upsampling in VAE decoder and UNet.

    Args:
        input: Input tensor [N, C_in, H, W].
        weight: Filter weights [C_in, C_out/groups, K_h, K_w].
        bias: Optional bias [C_out].
        stride: Stride for convolution.
        padding: Padding for input.
        output_padding: Additional padding for output.
        groups: Number of groups.
        dilation: Dilation for filter.

    Returns:
        Output tensor [N, C_out, H_out, W_out].
    """
    if input.ndim != 4:
        raise ValueError(f"conv2d_transpose expects 4D input, got {input.ndim}D")

    # Normalize to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _conv2d_transpose_native(
            input, weight, bias, stride, padding, output_padding, groups, dilation
        )
    else:
        return _conv2d_transpose_cpu(
            input, weight, bias, stride, padding, output_padding, groups, dilation
        )


def _conv2d_transpose_cpu(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    groups: int,
    dilation: tuple[int, int],
) -> GPUArray:
    """CPU implementation of transposed conv2d."""
    x = input.to_numpy()
    w = weight.to_numpy()

    N, C_in, H, W = x.shape
    C_in_w, C_out_per_group, K_h, K_w = w.shape

    if C_in != C_in_w:
        raise ValueError(f"Input channels {C_in} != weight in_channels {C_in_w}")

    C_out = C_out_per_group * groups
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_pad_h, out_pad_w = output_padding
    dil_h, dil_w = dilation

    # Calculate output dimensions
    H_out = (H - 1) * stride_h - 2 * pad_h + dil_h * (K_h - 1) + 1 + out_pad_h
    W_out = (W - 1) * stride_w - 2 * pad_w + dil_w * (K_w - 1) + 1 + out_pad_w

    # Simple implementation: for each input location, add weighted contribution
    output = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    C_in_per_group = C_in // groups

    for n in range(N):
        for g in range(groups):
            for c_in in range(C_in_per_group):
                c_in_global = g * C_in_per_group + c_in
                for c_out in range(C_out_per_group):
                    c_out_global = g * C_out_per_group + c_out
                    for h in range(H):
                        for w_idx in range(W):
                            for kh in range(K_h):
                                for kw in range(K_w):
                                    h_out = h * stride_h - pad_h + kh * dil_h
                                    w_out = w_idx * stride_w - pad_w + kw * dil_w
                                    if 0 <= h_out < H_out and 0 <= w_out < W_out:
                                        output[n, c_out_global, h_out, w_out] += (
                                            x[n, c_in_global, h, w_idx]
                                            * w[c_in_global, c_out, kh, kw]
                                        )

    # Add bias
    if bias is not None:
        b = bias.to_numpy()
        output = output + b.reshape(1, C_out, 1, 1)

    return from_numpy(output.astype(x.dtype))


def _conv2d_transpose_native(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    groups: int,
    dilation: tuple[int, int],
) -> GPUArray:
    """Native CUDA implementation of transposed conv2d."""
    # TODO: Implement native CUDA kernel
    return _conv2d_transpose_cpu(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )


__all__ = [
    "conv2d",
    "conv2d_transpose",
]
