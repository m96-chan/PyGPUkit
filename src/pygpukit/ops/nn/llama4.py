"""Llama 4 architecture specific operations.

Corresponds to native/ops/nn/llama4/.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype


def l2norm(
    input: GPUArray,
    eps: float = 1e-6,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """L2 Normalization (Llama4TextL2Norm).

    Computes: x * rsqrt(mean(x^2) + eps)

    Unlike RMSNorm, no gamma scaling is applied.
    Used for QK normalization in Llama 4 attention.

    Args:
        input: Input array of any shape. Normalization is applied over the last dimension.
        eps: Small epsilon for numerical stability.
        out: Optional output buffer. If provided, result is written in-place
            (for CUDA Graph capture).

    Returns:
        A new GPUArray containing the normalized output (or out if provided).

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(input, "l2norm")

    if input.ndim < 1:
        raise ValueError(f"l2norm expects at least 1D input, got {input.ndim}D")

    # Validate out array if provided
    if out is not None:
        if out.shape != input.shape:
            raise ValueError(f"out shape {out.shape} does not match input shape {input.shape}")
        if out.dtype != input.dtype:
            raise ValueError(f"out dtype {out.dtype} does not match input dtype {input.dtype}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _l2norm_native(input, eps, out=out)
    else:
        return _l2norm_cpu(input, eps, out=out)


def _l2norm_cpu(
    input: GPUArray,
    eps: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """CPU implementation of l2norm."""
    x = input.to_numpy()

    # L2 norm = x * rsqrt(mean(x^2) + eps)
    mean_sq = np.mean(x**2, axis=-1, keepdims=True)
    result = x / np.sqrt(mean_sq + eps)

    if out is not None:
        out_np = out.to_numpy()
        np.copyto(out_np, result)
        out._data = from_numpy(out_np)._data
        return out
    return from_numpy(result)


def _l2norm_native(
    input: GPUArray,
    eps: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ CUDA implementation of l2norm (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()

    if out is not None:
        out_native = out._get_native()
        native.l2norm_(input_native, out_native, eps)
        return out
    else:
        c_native = native.l2norm(input_native, eps)
        return GPUArray._wrap_native(c_native)


def irope_scale_q(
    Q: GPUArray,
    positions: GPUArray,
    attn_scale: float = 0.1,
    floor_scale: float = 8192.0,
) -> GPUArray:
    """Apply iRoPE temperature scaling to Q tensor.

    Formula: scale = log1p(floor((pos + 1) / floor_scale)) * attn_scale + 1.0

    Args:
        Q: Query tensor of shape [seq_len, num_heads, head_dim].
        positions: Position indices of shape [seq_len].
        attn_scale: Attention scale factor (default 0.1).
        floor_scale: Floor scale for temperature calculation (default 8192.0).

    Returns:
        Q tensor with iRoPE temperature scaling applied.

    Raises:
        ValueError: If Q is not 3D.
    """
    _validate_float_dtype(Q, "irope_scale_q")

    if Q.ndim != 3:
        raise ValueError(f"irope_scale_q expects 3D Q [seq_len, num_heads, head_dim], got {Q.ndim}D")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _irope_scale_q_native(Q, positions, attn_scale, floor_scale)
    else:
        return _irope_scale_q_cpu(Q, positions, attn_scale, floor_scale)


def _irope_scale_q_cpu(
    Q: GPUArray,
    positions: GPUArray,
    attn_scale: float,
    floor_scale: float,
) -> GPUArray:
    """CPU implementation of iRoPE Q scaling."""
    q = Q.to_numpy()
    pos = positions.to_numpy()

    # scale = log1p(floor((pos + 1) / floor_scale)) * attn_scale + 1.0
    scale = np.log1p(np.floor((pos + 1) / floor_scale)) * attn_scale + 1.0
    scale = scale[:, None, None]  # [seq_len, 1, 1]

    result = q * scale
    return from_numpy(result)


def _irope_scale_q_native(
    Q: GPUArray,
    positions: GPUArray,
    attn_scale: float,
    floor_scale: float,
) -> GPUArray:
    """Native C++ CUDA implementation of iRoPE Q scaling."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    Q_native = Q._get_native()
    positions_native = positions._get_native()
    c_native = native.irope_scale_q(Q_native, positions_native, attn_scale, floor_scale)
    return GPUArray._wrap_native(c_native)


def sdpa_irope(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    positions: GPUArray,
    attn_scale: float = 0.1,
    floor_scale: float = 8192.0,
    causal_offset: int = 0,
) -> GPUArray:
    """Scaled dot-product attention with iRoPE temperature scaling.

    Fuses temperature scaling into attention computation.

    Args:
        Q: Query tensor of shape [n_heads, q_len, head_dim].
        K: Key tensor of shape [n_kv_heads, kv_len, head_dim].
        V: Value tensor of shape [n_kv_heads, kv_len, head_dim].
        positions: Position indices of shape [q_len].
        attn_scale: Attention scale factor (default 0.1).
        floor_scale: Floor scale for temperature calculation (default 8192.0).
        causal_offset: Offset for causal mask (default 0).

    Returns:
        Attention output of shape [n_heads, q_len, head_dim].

    Raises:
        ValueError: If Q/K/V are not 3D or have mismatched dtypes.
    """
    _validate_float_dtype(Q, "sdpa_irope")

    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("sdpa_irope expects 3D Q, K, V [heads, seq, head_dim]")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError("sdpa_irope: Q/K/V must have same dtype")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _sdpa_irope_native(Q, K, V, positions, attn_scale, floor_scale, causal_offset)
    else:
        return _sdpa_irope_cpu(Q, K, V, positions, attn_scale, floor_scale, causal_offset)


def _sdpa_irope_cpu(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    positions: GPUArray,
    attn_scale: float,
    floor_scale: float,
    causal_offset: int,
) -> GPUArray:
    """CPU implementation of SDPA with iRoPE."""
    q = Q.to_numpy()
    k = K.to_numpy()
    v = V.to_numpy()
    pos = positions.to_numpy()

    n_heads, q_len, head_dim = q.shape
    n_kv_heads, kv_len, _ = k.shape

    # Compute temperature scale
    scale = np.log1p(np.floor((pos + 1) / floor_scale)) * attn_scale + 1.0

    # GQA expansion
    kv_repeat = n_heads // n_kv_heads

    output = np.zeros_like(q)
    for h in range(n_heads):
        kv_h = h // kv_repeat
        for i in range(q_len):
            # Q @ K^T with temperature scaling
            scores = np.dot(q[h, i], k[kv_h].T) * scale[i] / np.sqrt(head_dim)

            # Causal mask
            for j in range(kv_len):
                if j > i + causal_offset:
                    scores[j] = float("-inf")

            # Softmax
            scores_max: float = np.max(scores)
            scores_exp = np.exp(scores - scores_max)
            scores_softmax = scores_exp / np.sum(scores_exp)

            # V weighted sum
            output[h, i] = np.dot(scores_softmax, v[kv_h])

    return from_numpy(output.astype(q.dtype))


def _sdpa_irope_native(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    positions: GPUArray,
    attn_scale: float,
    floor_scale: float,
    causal_offset: int,
) -> GPUArray:
    """Native C++ CUDA implementation of SDPA with iRoPE."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    Q_native = Q._get_native()
    K_native = K._get_native()
    V_native = V._get_native()
    positions_native = positions._get_native()
    c_native = native.sdpa_irope(
        Q_native, K_native, V_native, positions_native,
        attn_scale, floor_scale, causal_offset
    )
    return GPUArray._wrap_native(c_native)


__all__ = [
    "l2norm",
    "irope_scale_q",
    "sdpa_irope",
]
