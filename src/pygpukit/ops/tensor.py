"""Tensor manipulation operations for GPUArrays.

Corresponds to native/ops/tensor/.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype, _validate_same_dtype

# =============================================================================
# Concatenation Operations
# =============================================================================


def concat_axis0(a: GPUArray, b: GPUArray) -> GPUArray:
    """Concatenate two tensors along axis 0.

    Args:
        a: First tensor of shape [dim0_a, ...].
        b: Second tensor of shape [dim0_b, ...].

    Returns:
        Concatenated tensor of shape [dim0_a + dim0_b, ...].

    Raises:
        ValueError: If shapes don't match along non-concatenation axes.
    """
    _validate_same_dtype(a, b, "concat_axis0")

    if a.ndim != b.ndim:
        raise ValueError(f"concat_axis0: dimension mismatch ({a.ndim}D vs {b.ndim}D)")

    for i in range(1, a.ndim):
        if a.shape[i] != b.shape[i]:
            raise ValueError(
                f"concat_axis0: shape mismatch at axis {i} ({a.shape[i]} vs {b.shape[i]})"
            )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _concat_axis0_native(a, b)
    else:
        return _concat_axis0_cpu(a, b)


def _concat_axis0_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of concat_axis0."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result = np.concatenate([a_np, b_np], axis=0)
    return from_numpy(result)


def _concat_axis0_native(a: GPUArray, b: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of concat_axis0."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    b_native = b._get_native()
    c_native = native.concat_axis0(a_native, b_native)
    return GPUArray._wrap_native(c_native)


# =============================================================================
# Repeat Operations
# =============================================================================


def repeat_interleave_axis1(input: GPUArray, repeats: int) -> GPUArray:
    """Repeat tensor elements along axis 1 (interleaved).

    For GQA: expands [n_heads_kv, seq_len, head_dim] to [n_heads, seq_len, head_dim]
    by repeating each KV head `repeats` times.

    Args:
        input: Input tensor of shape [dim0, dim1, dim2].
        repeats: Number of times to repeat each element along axis 1.

    Returns:
        Tensor of shape [dim0, dim1 * repeats, dim2].
    """
    _validate_float_dtype(input, "repeat_interleave_axis1")

    if input.ndim != 3:
        raise ValueError(
            f"repeat_interleave_axis1 expects 3D input [d0, d1, d2], got {input.ndim}D"
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _repeat_interleave_axis1_native(input, repeats)
    else:
        return _repeat_interleave_axis1_cpu(input, repeats)


def _repeat_interleave_axis1_cpu(input: GPUArray, repeats: int) -> GPUArray:
    """CPU implementation of repeat_interleave_axis1."""
    x = input.to_numpy()
    # np.repeat with axis=1 gives interleaved repeat
    result = np.repeat(x, repeats, axis=1)
    return from_numpy(result)


def _repeat_interleave_axis1_native(input: GPUArray, repeats: int) -> GPUArray:
    """Native C++ CUDA implementation of repeat_interleave_axis1."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    c_native = native.repeat_interleave_axis1(input_native, repeats)
    return GPUArray._wrap_native(c_native)


# =============================================================================
# Transpose Operations
# =============================================================================


def transpose_3d_021(input: GPUArray, *, out: GPUArray | None = None) -> GPUArray | None:
    """Transpose 3D tensor: [d0, d1, d2] -> [d1, d0, d2].

    Swaps axes 0 and 1 while keeping axis 2 in place.
    Useful for converting [seq_len, n_heads, head_dim] to [n_heads, seq_len, head_dim].

    Args:
        input: 3D tensor to transpose.
        out: Optional pre-allocated output buffer for CUDA Graph capture.
             If provided, must have shape [d1, d0, d2] and same dtype as input.

    Returns:
        Transposed tensor with axes 0 and 1 swapped.
        Returns None if out is provided (in-place operation).
    """
    _validate_float_dtype(input, "transpose_3d_021")

    if input.ndim != 3:
        raise ValueError(f"transpose_3d_021 expects 3D input, got {input.ndim}D")

    backend = get_backend()

    # Native transpose_3d_021 supports float32/float16/bfloat16
    if isinstance(backend, NativeBackend) and backend.is_available():
        dtype_str = str(input.dtype)
        if dtype_str in ("float32", "float16", "bfloat16"):
            return _transpose_3d_021_native(input, out=out)
        else:
            if out is not None:
                raise NotImplementedError(
                    "transpose_3d_021: out parameter not supported for CPU fallback"
                )
            return _transpose_3d_021_cpu(input)
    else:
        if out is not None:
            raise NotImplementedError(
                "transpose_3d_021: out parameter not supported for CPU fallback"
            )
        return _transpose_3d_021_cpu(input)


def _transpose_3d_021_cpu(input: GPUArray) -> GPUArray:
    """CPU implementation of transpose_3d_021."""
    x = input.to_numpy()
    result = np.transpose(x, (1, 0, 2)).copy()
    return from_numpy(result)


def _transpose_3d_021_native(input: GPUArray, *, out: GPUArray | None = None) -> GPUArray | None:
    """Native C++ CUDA implementation of transpose_3d_021."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()

    if out is not None:
        out_native = out._get_native()
        native.transpose_3d_021_(input_native, out_native)
        return None
    else:
        c_native = native.transpose_3d_021(input_native)
        return GPUArray._wrap_native(c_native)


def transpose_4d_0213(input: GPUArray, *, out: GPUArray | None = None) -> GPUArray | None:
    """Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d2, d1, d3].

    Swaps axes 1 and 2 while keeping axes 0 and 3 in place.
    Common in attention operations to convert:
    - [batch, seq, heads, dim] -> [batch, heads, seq, dim]

    Args:
        input: 4D tensor to transpose.
        out: Optional pre-allocated output buffer for CUDA Graph capture.
             If provided, must have shape [d0, d2, d1, d3] and same dtype as input.

    Returns:
        Transposed tensor with axes 1 and 2 swapped.
        Returns None if out is provided (in-place operation).
    """
    _validate_float_dtype(input, "transpose_4d_0213")

    if input.ndim != 4:
        raise ValueError(f"transpose_4d_0213 expects 4D input, got {input.ndim}D")

    backend = get_backend()

    # Native transpose_4d_0213 supports float32/float16/bfloat16
    if isinstance(backend, NativeBackend) and backend.is_available():
        dtype_str = str(input.dtype)
        if dtype_str in ("float32", "float16", "bfloat16"):
            return _transpose_4d_0213_native(input, out=out)
        else:
            if out is not None:
                raise NotImplementedError(
                    "transpose_4d_0213: out parameter not supported for CPU fallback"
                )
            return _transpose_4d_0213_cpu(input)
    else:
        if out is not None:
            raise NotImplementedError(
                "transpose_4d_0213: out parameter not supported for CPU fallback"
            )
        return _transpose_4d_0213_cpu(input)


def _transpose_4d_0213_cpu(input: GPUArray) -> GPUArray:
    """CPU fallback for transpose_4d_0213."""
    x = input.to_numpy()
    result = np.transpose(x, (0, 2, 1, 3)).copy()
    return from_numpy(result)


def _transpose_4d_0213_native(input: GPUArray, *, out: GPUArray | None = None) -> GPUArray | None:
    """Native C++ CUDA implementation of transpose_4d_0213."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()

    if out is not None:
        out_native = out._get_native()
        native.transpose_4d_0213_(input_native, out_native)
        return None
    else:
        c_native = native.transpose_4d_0213(input_native)
        return GPUArray._wrap_native(c_native)


def transpose_3d_012(input: GPUArray, *, out: GPUArray | None = None) -> GPUArray | None:
    """Transpose 3D tensor: [d0, d1, d2] -> [d0, d2, d1].

    Swaps last two axes while keeping axis 0 in place.
    Useful for attention operations where K needs to be transposed.

    Args:
        input: 3D tensor to transpose.
        out: Optional pre-allocated output buffer for CUDA Graph capture.
             If provided, must have shape [d0, d2, d1] and same dtype as input.

    Returns:
        Transposed tensor with last two axes swapped.
        Returns None if out is provided (in-place operation).
    """
    _validate_float_dtype(input, "transpose_3d_012")

    if input.ndim != 3:
        raise ValueError(f"transpose_3d_012 expects 3D input, got {input.ndim}D")

    backend = get_backend()

    # Native transpose_3d_012 supports float32/float16/bfloat16
    if isinstance(backend, NativeBackend) and backend.is_available():
        dtype_str = str(input.dtype)
        if dtype_str in ("float32", "float16", "bfloat16"):
            return _transpose_3d_012_native(input, out=out)
        else:
            if out is not None:
                raise NotImplementedError(
                    "transpose_3d_012: out parameter not supported for CPU fallback"
                )
            return _transpose_3d_012_cpu(input)
    else:
        if out is not None:
            raise NotImplementedError(
                "transpose_3d_012: out parameter not supported for CPU fallback"
            )
        return _transpose_3d_012_cpu(input)


def _transpose_3d_012_cpu(input: GPUArray) -> GPUArray:
    """CPU implementation of transpose_3d_012."""
    x = input.to_numpy()
    result = np.transpose(x, (0, 2, 1)).copy()
    return from_numpy(result)


def _transpose_3d_012_native(input: GPUArray, *, out: GPUArray | None = None) -> GPUArray | None:
    """Native C++ CUDA implementation of transpose_3d_012."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()

    if out is not None:
        out_native = out._get_native()
        native.transpose_3d_012_(input_native, out_native)
        return None
    else:
        c_native = native.transpose_3d_012(input_native)
        return GPUArray._wrap_native(c_native)


def transpose_4d_0132(input: GPUArray, *, out: GPUArray | None = None) -> GPUArray | None:
    """Transpose 4D tensor: [d0, d1, d2, d3] -> [d0, d1, d3, d2].

    Swaps last two axes while keeping axes 0 and 1 in place.
    Useful for K^T in attention operations.

    Args:
        input: 4D tensor to transpose.
        out: Optional pre-allocated output buffer for CUDA Graph capture.
             If provided, must have shape [d0, d1, d3, d2] and same dtype as input.

    Returns:
        Transposed tensor with last two axes swapped.
        Returns None if out is provided (in-place operation).
    """
    _validate_float_dtype(input, "transpose_4d_0132")

    if input.ndim != 4:
        raise ValueError(f"transpose_4d_0132 expects 4D input, got {input.ndim}D")

    backend = get_backend()

    # Native transpose_4d_0132 supports float32/float16/bfloat16
    if isinstance(backend, NativeBackend) and backend.is_available():
        dtype_str = str(input.dtype)
        if dtype_str in ("float32", "float16", "bfloat16"):
            return _transpose_4d_0132_native(input, out=out)
        else:
            if out is not None:
                raise NotImplementedError(
                    "transpose_4d_0132: out parameter not supported for CPU fallback"
                )
            return _transpose_4d_0132_cpu(input)
    else:
        if out is not None:
            raise NotImplementedError(
                "transpose_4d_0132: out parameter not supported for CPU fallback"
            )
        return _transpose_4d_0132_cpu(input)


def _transpose_4d_0132_cpu(input: GPUArray) -> GPUArray:
    """CPU fallback for transpose_4d_0132."""
    x = input.to_numpy()
    result = np.transpose(x, (0, 1, 3, 2)).copy()
    return from_numpy(result)


def _transpose_4d_0132_native(input: GPUArray, *, out: GPUArray | None = None) -> GPUArray | None:
    """Native C++ CUDA implementation of transpose_4d_0132."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()

    if out is not None:
        out_native = out._get_native()
        native.transpose_4d_0132_(input_native, out_native)
        return None
    else:
        c_native = native.transpose_4d_0132(input_native)
        return GPUArray._wrap_native(c_native)


# =============================================================================
# Reshape Operations
# =============================================================================


def reshape_copy(
    input: GPUArray,
    new_shape: tuple[int, ...] | None = None,
    *,
    out: GPUArray | None = None,
) -> GPUArray | None:
    """Reshape tensor with copy (ensures contiguous output).

    Args:
        input: Input tensor to reshape.
        new_shape: Target shape (total elements must match).
                   Required if out is not provided.
        out: Optional pre-allocated output buffer for CUDA Graph capture.
             If provided, new_shape is ignored and output shape is determined by out.

    Returns:
        Reshaped tensor with new shape.
        Returns None if out is provided (in-place operation).

    Raises:
        ValueError: If total element count doesn't match.
    """
    _validate_float_dtype(input, "reshape_copy")

    # Determine target shape
    if out is not None:
        target_shape = out.shape
    elif new_shape is not None:
        target_shape = new_shape
    else:
        raise ValueError("reshape_copy: either new_shape or out must be provided")

    # Verify total size
    input_size = 1
    for dim in input.shape:
        input_size *= dim

    output_size = 1
    for dim in target_shape:
        output_size *= dim

    if input_size != output_size:
        raise ValueError(f"reshape_copy: total size mismatch ({input_size} vs {output_size})")

    backend = get_backend()

    # Native reshape_copy supports float32/float16/bfloat16
    if isinstance(backend, NativeBackend) and backend.is_available():
        dtype_str = str(input.dtype)
        if dtype_str in ("float32", "float16", "bfloat16"):
            return _reshape_copy_native(input, target_shape, out=out)
        else:
            if out is not None:
                raise NotImplementedError(
                    "reshape_copy: out parameter not supported for CPU fallback"
                )
            return _reshape_copy_cpu(input, target_shape)
    else:
        if out is not None:
            raise NotImplementedError("reshape_copy: out parameter not supported for CPU fallback")
        return _reshape_copy_cpu(input, target_shape)


def _reshape_copy_cpu(input: GPUArray, new_shape: tuple[int, ...]) -> GPUArray:
    """CPU implementation of reshape_copy."""
    x = input.to_numpy()
    result = x.reshape(new_shape).copy()
    return from_numpy(result)


def _reshape_copy_native(
    input: GPUArray,
    new_shape: tuple[int, ...],
    *,
    out: GPUArray | None = None,
) -> GPUArray | None:
    """Native C++ CUDA implementation of reshape_copy."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()

    if out is not None:
        out_native = out._get_native()
        native.reshape_copy_(input_native, out_native)
        return None
    else:
        c_native = native.reshape_copy(input_native, list(new_shape))
        return GPUArray._wrap_native(c_native)


# =============================================================================
# Dtype Cast Operations
# =============================================================================


def cast_f32_to_bf16(src: GPUArray) -> GPUArray:
    """Cast float32 to bfloat16 on GPU.

    Uses __float2bfloat16_rn for round-to-nearest-even.

    Args:
        src: Source tensor (float32).

    Returns:
        New tensor in bfloat16.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    src_native = src._get_native()
    result_native = native.cast_f32_to_bf16(src_native)
    return GPUArray._wrap_native(result_native)


def cast_f32_to_f16(src: GPUArray) -> GPUArray:
    """Cast float32 to float16 on GPU.

    Args:
        src: Source tensor (float32).

    Returns:
        New tensor in float16.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    src_native = src._get_native()
    result_native = native.cast_f32_to_f16(src_native)
    return GPUArray._wrap_native(result_native)


def cast_bf16_to_f32(src: GPUArray) -> GPUArray:
    """Cast bfloat16 to float32 on GPU.

    Args:
        src: Source tensor (bfloat16).

    Returns:
        New tensor in float32.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    src_native = src._get_native()
    result_native = native.cast_bf16_to_f32(src_native)
    return GPUArray._wrap_native(result_native)


def cast_f16_to_f32(src: GPUArray) -> GPUArray:
    """Cast float16 to float32 on GPU.

    Args:
        src: Source tensor (float16).

    Returns:
        New tensor in float32.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    src_native = src._get_native()
    result_native = native.cast_f16_to_f32(src_native)
    return GPUArray._wrap_native(result_native)
