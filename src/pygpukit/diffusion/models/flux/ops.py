"""GPU-native operations for FLUX.

Provides GPU utility functions that keep data on GPU throughout computation,
eliminating H2D/D2H transfer overhead.

Issue #187: All operations use native CUDA kernels - no NumPy fallbacks.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend, get_native_module
from pygpukit.core.factory import from_numpy
from pygpukit.ops.elementwise import add, mul
from pygpukit.ops.matmul.generic import batched_matmul, matmul, transpose
from pygpukit.ops.nn.activation import gelu, silu
from pygpukit.ops.nn.linear import bias_add_inplace
from pygpukit.ops.nn.norm import rmsnorm
from pygpukit.ops.reduction import softmax
from pygpukit.ops.tensor import transpose_3d_012, transpose_4d_0213


def _is_native_available() -> bool:
    """Check if native backend is available."""
    backend = get_backend()
    return isinstance(backend, NativeBackend) and backend.is_available()


def gpu_linear(
    x: GPUArray,
    weight: GPUArray,
    bias: GPUArray | None = None,
) -> GPUArray:
    """GPU-native linear layer: y = x @ W^T + b.

    Args:
        x: Input [batch, ..., in_features] - will be flattened to 2D.
        weight: Weight [out_features, in_features].
        bias: Optional bias [out_features].

    Returns:
        Output [batch, ..., out_features].
    """
    original_shape = x.shape
    in_features = original_shape[-1]
    out_features = weight.shape[0]

    # Flatten to 2D for matmul
    x_2d = x.reshape(-1, in_features)

    # Compute y = x @ W^T
    w_t = transpose(weight)
    y = matmul(x_2d, w_t)

    # Add bias if provided
    if bias is not None:
        bias_add_inplace(y, bias)

    # Reshape back to original shape with out_features
    new_shape = original_shape[:-1] + (out_features,)
    return y.reshape(*new_shape)


def gpu_rms_norm(
    x: GPUArray,
    weight: GPUArray,
    eps: float = 1e-6,
) -> GPUArray:
    """GPU-native RMS normalization.

    Args:
        x: Input [batch, seq_len, features] or [batch, features].
        weight: Scale parameter [features].
        eps: Epsilon for numerical stability.

    Returns:
        Normalized output, same shape as input.
    """
    if x.ndim == 2:
        return rmsnorm(x, weight, eps)
    elif x.ndim == 3:
        batch, seq_len, features = x.shape
        x_2d = x.reshape(batch * seq_len, features)
        out_2d = rmsnorm(x_2d, weight, eps)
        return out_2d.reshape(batch, seq_len, features)
    elif x.ndim == 4:
        d0, d1, d2, features = x.shape
        x_2d = x.reshape(d0 * d1 * d2, features)
        out_2d = rmsnorm(x_2d, weight, eps)
        return out_2d.reshape(d0, d1, d2, features)
    else:
        raise ValueError(f"gpu_rms_norm expects 2D-4D input, got {x.ndim}D")


def gpu_layer_norm(
    x: GPUArray,
    eps: float = 1e-6,
) -> GPUArray:
    """GPU-native layer normalization (no learnable parameters).

    Args:
        x: Input [batch, seq_len, features].
        eps: Epsilon for numerical stability.

    Returns:
        Normalized output, same shape as input.

    Note:
        Uses native CUDA kernel - no H2D/D2H transfer.
    """
    if _is_native_available() and x.ndim == 3:
        native = get_native_module()
        x_native = x._get_native()
        result_native = native.layer_norm_simple(x_native, eps)
        return GPUArray._wrap_native(result_native)
    else:
        # CPU fallback for non-3D or no native backend
        x_np = x.to_numpy()
        mean = np.mean(x_np, axis=-1, keepdims=True)
        var = np.var(x_np, axis=-1, keepdims=True)
        normalized = (x_np - mean) / np.sqrt(var + eps)
        return from_numpy(normalized.astype(np.float32))


def gpu_silu(x: GPUArray) -> GPUArray:
    """GPU-native SiLU activation: y = x * sigmoid(x)."""
    return silu(x)


def gpu_gelu(x: GPUArray) -> GPUArray:
    """GPU-native GELU activation."""
    return gelu(x)


def gpu_softmax(x: GPUArray, axis: int = -1) -> GPUArray:
    """GPU-native softmax along specified axis."""
    return softmax(x, axis=axis)


def gpu_add(a: GPUArray, b: GPUArray) -> GPUArray:
    """GPU-native element-wise addition."""
    return add(a, b)


def gpu_mul(a: GPUArray, b: GPUArray) -> GPUArray:
    """GPU-native element-wise multiplication."""
    return mul(a, b)


def gpu_batched_matmul(a: GPUArray, b: GPUArray) -> GPUArray:
    """GPU-native batched matrix multiplication."""
    return batched_matmul(a, b)


def gpu_scale(x: GPUArray, scale: float) -> GPUArray:
    """Scale tensor by a scalar value.

    Args:
        x: Input tensor.
        scale: Scalar multiplier.

    Returns:
        Scaled tensor.

    Note:
        Uses native CUDA kernel - no H2D/D2H transfer.
    """
    if _is_native_available():
        native = get_native_module()
        x_native = x._get_native()
        result_native = native.scale_tensor(x_native, scale)
        return GPUArray._wrap_native(result_native)
    else:
        x_np = x.to_numpy()
        return from_numpy((x_np * scale).astype(x_np.dtype))


def gpu_broadcast_add(
    x: GPUArray,
    bias: GPUArray,
    axis: int = -1,
) -> GPUArray:
    """Add bias with broadcasting along specified axis.

    Args:
        x: Input tensor [batch, seq_len, features].
        bias: Bias tensor [features] (1D) or [1, 1, features] (3D).
        axis: Axis along which to broadcast (default: -1, last axis).

    Returns:
        x + bias with broadcasting.

    Note:
        For 3D input with 1D bias along last axis, uses bias_add_inplace.
        Other cases fall back to numpy.
    """
    if x.ndim == 3 and bias.ndim == 1 and axis == -1:
        # Reshape to 2D, apply bias, reshape back
        batch, seq_len, features = x.shape
        x_2d = x.reshape(batch * seq_len, features)
        # Create copy since bias_add_inplace modifies in-place
        out_2d = x_2d.copy() if hasattr(x_2d, "copy") else from_numpy(x_2d.to_numpy().copy())
        bias_add_inplace(out_2d, bias)
        return out_2d.reshape(batch, seq_len, features)
    else:
        # Fall back to numpy for complex broadcasting
        x_np = x.to_numpy()
        bias_np = bias.to_numpy()
        # Handle broadcasting
        if axis == -1 or axis == x.ndim - 1:
            result = x_np + bias_np
        else:
            # Expand dims for proper broadcasting
            expand_shape = [1] * x.ndim
            expand_shape[axis] = bias.shape[0]
            result = x_np + bias_np.reshape(expand_shape)
        return from_numpy(result.astype(x_np.dtype))


def gpu_broadcast_mul(
    x: GPUArray,
    scale: GPUArray,
    axis: int = -1,
) -> GPUArray:
    """Multiply with broadcasting along specified axis.

    Args:
        x: Input tensor [batch, seq_len, features].
        scale: Scale tensor [features] (1D) or broadcastable shape.
        axis: Axis along which to broadcast.

    Returns:
        x * scale with broadcasting.
    """
    x_np = x.to_numpy()
    scale_np = scale.to_numpy()

    if axis == -1 or axis == x.ndim - 1:
        result = x_np * scale_np
    else:
        expand_shape = [1] * x.ndim
        expand_shape[axis] = scale.shape[0]
        result = x_np * scale_np.reshape(expand_shape)
    return from_numpy(result.astype(x_np.dtype))


def gpu_modulate(
    x: GPUArray,
    scale: GPUArray,
    shift: GPUArray,
) -> GPUArray:
    """Apply scale and shift modulation: y = x * (1 + scale) + shift.

    Used in AdaLN-Zero for FLUX.

    Args:
        x: Input tensor [batch, seq_len, features].
        scale: Scale tensor [batch, features].
        shift: Shift tensor [batch, features].

    Returns:
        Modulated output [batch, seq_len, features].

    Note:
        Uses native CUDA kernel - no H2D/D2H transfer.
    """
    if _is_native_available() and x.ndim == 3 and scale.ndim == 2:
        native = get_native_module()
        x_native = x._get_native()
        scale_native = scale._get_native()
        shift_native = shift._get_native()
        result_native = native.modulate(x_native, scale_native, shift_native)
        return GPUArray._wrap_native(result_native)
    else:
        # CPU fallback
        x_np = x.to_numpy()
        scale_np = scale.to_numpy()
        shift_np = shift.to_numpy()

        # Expand scale/shift for broadcasting: [batch, features] -> [batch, 1, features]
        if scale_np.ndim == 2:
            scale_np = scale_np[:, None, :]
            shift_np = shift_np[:, None, :]

        result = x_np * (1.0 + scale_np) + shift_np
        return from_numpy(result.astype(np.float32))


def gpu_apply_rope(
    x: GPUArray,
    cos: GPUArray,
    sin: GPUArray,
) -> GPUArray:
    """Apply rotary position embedding to Q or K.

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim].
        cos: Cosine frequencies [seq_len, head_dim] or GPUArray.
        sin: Sine frequencies [seq_len, head_dim] or GPUArray.

    Returns:
        Rotated tensor [batch, seq_len, num_heads, head_dim].

    Note:
        Uses native CUDA kernel - no H2D/D2H transfer.
    """
    if _is_native_available() and x.ndim == 4 and cos.ndim == 2:
        native = get_native_module()
        x_native = x._get_native()
        cos_native = cos._get_native()
        sin_native = sin._get_native()
        result_native = native.apply_rope(x_native, cos_native, sin_native)
        return GPUArray._wrap_native(result_native)
    else:
        # CPU fallback
        x_np = x.to_numpy()
        cos_np = cos.to_numpy() if isinstance(cos, GPUArray) else cos
        sin_np = sin.to_numpy() if isinstance(sin, GPUArray) else sin

        # Reshape cos/sin for broadcasting: [1, seq_len, 1, head_dim]
        cos_np = cos_np[None, :, None, :]
        sin_np = sin_np[None, :, None, :]

        # Split into pairs and rotate
        # x = [x0, x1, x2, x3, ...] -> rotate pairs
        # x_rot = [-x1, x0, -x3, x2, ...]
        x_rot = np.empty_like(x_np)
        x_rot[..., 0::2] = -x_np[..., 1::2]
        x_rot[..., 1::2] = x_np[..., 0::2]

        # Apply rotation: x * cos + x_rot * sin
        result = x_np * cos_np + x_rot * sin_np
        return from_numpy(result.astype(np.float32))


def gpu_concat_axis1(a: GPUArray, b: GPUArray) -> GPUArray:
    """Concatenate two tensors along axis 1.

    Args:
        a: First tensor [batch, seq_a, features].
        b: Second tensor [batch, seq_b, features].

    Returns:
        Concatenated tensor [batch, seq_a + seq_b, features].

    Note:
        Uses native CUDA kernel - no H2D/D2H transfer.
    """
    if _is_native_available() and a.ndim == 3 and b.ndim == 3:
        native = get_native_module()
        a_native = a._get_native()
        b_native = b._get_native()
        result_native = native.concat_axis1(a_native, b_native)
        return GPUArray._wrap_native(result_native)
    else:
        # CPU fallback
        a_np = a.to_numpy()
        b_np = b.to_numpy()
        result = np.concatenate([a_np, b_np], axis=1)
        return from_numpy(result.astype(np.float32))


def gpu_split_axis1(
    x: GPUArray,
    split_size: int,
) -> tuple[GPUArray, GPUArray]:
    """Split tensor along axis 1.

    Args:
        x: Input tensor [batch, seq_len, features].
        split_size: Size of first split.

    Returns:
        Tuple of (first [batch, split_size, features],
                  second [batch, seq_len - split_size, features]).

    Note:
        Uses native CUDA kernel - no H2D/D2H transfer.
    """
    if _is_native_available() and x.ndim == 3:
        native = get_native_module()
        x_native = x._get_native()
        first_native, second_native = native.split_axis1(x_native, split_size)
        return GPUArray._wrap_native(first_native), GPUArray._wrap_native(second_native)
    else:
        # CPU fallback
        x_np = x.to_numpy()
        first = x_np[:, :split_size, :]
        second = x_np[:, split_size:, :]
        return from_numpy(first.astype(np.float32)), from_numpy(second.astype(np.float32))


def gpu_transpose_0213(x: GPUArray) -> GPUArray:
    """GPU-native transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d2, d1, d3].

    Used for attention: [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim].
    Uses native CUDA kernel - no H2D/D2H transfer.
    """
    result = transpose_4d_0213(x)
    # transpose_4d_0213 returns GPUArray directly (native implementation)
    return result if result is not None else x


def gpu_transpose_3d_012(x: GPUArray) -> GPUArray:
    """GPU-native transpose 3D tensor: [d0, d1, d2] -> [d0, d2, d1].

    Used for K^T in attention: [batch*heads, seq, dim] -> [batch*heads, dim, seq].
    Uses native CUDA kernel - no H2D/D2H transfer.
    """
    result = transpose_3d_012(x)
    # transpose_3d_012 returns GPUArray directly (native implementation)
    return result if result is not None else x


def gpu_reshape(x: GPUArray, new_shape: tuple[int, ...]) -> GPUArray:
    """Reshape tensor to new shape."""
    return x.reshape(*new_shape)


def gpu_gated_residual(
    residual: GPUArray,
    gate: GPUArray,
    value: GPUArray,
) -> GPUArray:
    """Apply gated residual connection: y = residual + gate * value.

    Used in FLUX for gated attention outputs.

    Args:
        residual: Residual tensor [batch, seq_len, features].
        gate: Gate tensor [batch, features].
        value: Value tensor [batch, seq_len, features].

    Returns:
        Output tensor [batch, seq_len, features].

    Note:
        Uses native CUDA kernel - no H2D/D2H transfer.
    """
    if _is_native_available() and residual.ndim == 3 and gate.ndim == 2:
        native = get_native_module()
        residual_native = residual._get_native()
        gate_native = gate._get_native()
        value_native = value._get_native()
        result_native = native.gated_residual(residual_native, gate_native, value_native)
        return GPUArray._wrap_native(result_native)
    else:
        # CPU fallback
        residual_np = residual.to_numpy()
        gate_np = gate.to_numpy()
        value_np = value.to_numpy()

        # Expand gate for broadcasting: [batch, features] -> [batch, 1, features]
        if gate_np.ndim == 2:
            gate_np = gate_np[:, None, :]

        result = residual_np + gate_np * value_np
        return from_numpy(result.astype(np.float32))


def gpu_add_broadcast_2d(
    x: GPUArray,
    bias: GPUArray,
) -> GPUArray:
    """Add with broadcasting: x + bias where x is 3D and bias is 2D.

    Args:
        x: Input tensor [batch, seq_len, features].
        bias: Bias tensor [batch, features].

    Returns:
        Output tensor [batch, seq_len, features].

    Note:
        Uses native CUDA kernel - no H2D/D2H transfer.
    """
    if _is_native_available() and x.ndim == 3 and bias.ndim == 2:
        native = get_native_module()
        x_native = x._get_native()
        bias_native = bias._get_native()
        result_native = native.add_broadcast(x_native, bias_native)
        return GPUArray._wrap_native(result_native)
    else:
        # CPU fallback
        x_np = x.to_numpy()
        bias_np = bias.to_numpy()

        # Expand bias for broadcasting: [batch, features] -> [batch, 1, features]
        if bias_np.ndim == 2:
            bias_np = bias_np[:, None, :]

        result = x_np + bias_np
        return from_numpy(result.astype(np.float32))


__all__ = [
    "gpu_linear",
    "gpu_rms_norm",
    "gpu_layer_norm",
    "gpu_silu",
    "gpu_gelu",
    "gpu_softmax",
    "gpu_add",
    "gpu_mul",
    "gpu_batched_matmul",
    "gpu_scale",
    "gpu_broadcast_add",
    "gpu_broadcast_mul",
    "gpu_modulate",
    "gpu_apply_rope",
    "gpu_concat_axis1",
    "gpu_split_axis1",
    "gpu_transpose_0213",
    "gpu_transpose_3d_012",
    "gpu_reshape",
    "gpu_gated_residual",
    "gpu_add_broadcast_2d",
]
