"""Linear layer operations for GPUArrays.

Corresponds to native/ops/nn/linear/ and native/ops/nn/tensor/.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype


def bias_add_inplace(output: GPUArray, bias: GPUArray) -> None:
    """Add bias to output in-place.

    Computes: output[batch, features] += bias[features]

    Args:
        output: Output array of shape [batch, features] (modified in-place).
        bias: Bias array of shape [features].

    Raises:
        ValueError: If shapes don't match or dtypes don't match.
    """
    _validate_float_dtype(output, "bias_add_inplace")

    if output.ndim != 2:
        raise ValueError(
            f"bias_add_inplace expects 2D output [batch, features], got {output.ndim}D"
        )
    if bias.ndim != 1:
        raise ValueError(f"bias_add_inplace expects 1D bias [features], got {bias.ndim}D")
    if output.dtype != bias.dtype:
        raise ValueError("bias_add_inplace: output and bias must have same dtype")

    features = output.shape[1]
    if bias.shape[0] != features:
        raise ValueError(
            f"bias_add_inplace: bias size {bias.shape[0]} must match features {features}"
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        _bias_add_inplace_native(output, bias)
    else:
        _bias_add_inplace_cpu(output, bias)


def _bias_add_inplace_cpu(output: GPUArray, bias: GPUArray) -> None:
    """CPU implementation of bias_add_inplace."""
    # For CPU backend, we need to get numpy arrays, modify, and update
    output_np = output.to_numpy()
    bias_np = bias.to_numpy()
    output_np += bias_np
    # Note: This creates a new array - for CPU backend, in-place is not truly in-place
    # The native backend does true in-place modification
    output._data = from_numpy(output_np)._data


def _bias_add_inplace_native(output: GPUArray, bias: GPUArray) -> None:
    """Native C++ CUDA implementation of bias_add_inplace (true in-place)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    output_native = output._get_native()
    bias_native = bias._get_native()
    native.bias_add_inplace(output_native, bias_native)


def split_qkv_batch(
    qkv: GPUArray,
    q_out: GPUArray,
    k_out: GPUArray,
    v_out: GPUArray,
    q_dim: int,
    k_dim: int,
    v_dim: int,
) -> None:
    """Split fused QKV projection output into separate Q, K, V tensors.

    This is a zero-allocation operation designed for CUDA Graph compatibility.
    Output buffers must be pre-allocated.

    Args:
        qkv: Fused QKV tensor [seq_len, q_dim + k_dim + v_dim].
        q_out: Pre-allocated Q output buffer [seq_len, q_dim] or [seq_len, n_heads, head_dim].
        k_out: Pre-allocated K output buffer [seq_len, k_dim] or [seq_len, n_kv_heads, head_dim].
        v_out: Pre-allocated V output buffer [seq_len, v_dim] or [seq_len, n_kv_heads, head_dim].
        q_dim: Size of Q projection (num_heads * head_dim).
        k_dim: Size of K projection (num_kv_heads * head_dim).
        v_dim: Size of V projection (num_kv_heads * head_dim).

    Note:
        The output buffers can be 2D [seq_len, dim] or 3D [seq_len, heads, head_dim]
        as long as the total size matches. The kernel writes linearly.
    """
    from pygpukit.core.backend import get_backend, get_native_module

    backend = get_backend()
    if not backend.is_available():
        raise RuntimeError("split_qkv_batch requires GPU backend")

    native = get_native_module()
    native.split_qkv_batch(
        qkv._get_native(),
        q_out._get_native(),
        k_out._get_native(),
        v_out._get_native(),
        q_dim,
        k_dim,
        v_dim,
    )


def slice_rows_range_ptr(
    table: GPUArray,
    out: GPUArray,
    start_pos_buf: GPUArray,
    count: int,
) -> None:
    """Slice consecutive rows from table using GPU-stored start position.

    This is a zero-allocation operation designed for CUDA Graph compatibility.
    The start position is read from a GPU buffer, enabling graph replay with
    different positions without H2D copies.

    Args:
        table: Source table of shape [num_rows, row_dim].
        out: Pre-allocated output buffer of shape [count, row_dim].
        start_pos_buf: GPU buffer containing start position [1] int32.
        count: Number of consecutive rows to copy.

    Example:
        # During CUDA Graph capture
        slice_rows_range_ptr(rope_cos_table, cos_batch, start_pos_buf, batch_size)
        # Copies cos_batch[i, :] = rope_cos_table[start_pos + i, :]
    """
    from pygpukit.core.backend import get_backend, get_native_module

    backend = get_backend()
    if not backend.is_available():
        raise RuntimeError("slice_rows_range_ptr requires GPU backend")

    native = get_native_module()
    native.slice_rows_range_ptr(
        table._get_native(),
        out._get_native(),
        start_pos_buf._get_native(),
        count,
    )


__all__ = [
    "bias_add_inplace",
    "split_qkv_batch",
    "slice_rows_range_ptr",
]
