"""RoPE (Rotary Position Embedding) operations for GPUArrays.

Corresponds to native/ops/nn/rope/.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype


def rope_inplace(
    q: GPUArray,
    k: GPUArray,
    cos: GPUArray,
    sin: GPUArray,
) -> None:
    """Apply Rotary Position Embedding (RoPE) to Q and K tensors in-place.

    Args:
        q: Query tensor of shape [seq_len, n_heads_q, head_dim] (modified in-place).
        k: Key tensor of shape [seq_len, n_heads_k, head_dim] (modified in-place).
        cos: Precomputed cosine of shape [seq_len, head_dim].
        sin: Precomputed sine of shape [seq_len, head_dim].

    Note:
        This operation modifies q and k in-place.
        Works with GQA (n_heads_k can be different from n_heads_q).
    """
    _validate_float_dtype(q, "rope_inplace")

    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("rope_inplace expects 3D q, k [seq_len, n_heads, head_dim]")
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("rope_inplace expects 2D cos, sin [seq_len, head_dim]")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        _rope_inplace_native(q, k, cos, sin)
    else:
        _rope_inplace_cpu(q, k, cos, sin)


def _rope_inplace_cpu(
    q: GPUArray,
    k: GPUArray,
    cos: GPUArray,
    sin: GPUArray,
) -> None:
    """CPU implementation of rope_inplace."""

    q_np = q.to_numpy()
    k_np = k.to_numpy()
    cos_np = cos.to_numpy()
    sin_np = sin.to_numpy()

    seq_len, n_heads_q, head_dim = q_np.shape
    n_heads_k = k_np.shape[1]
    half_dim = head_dim // 2

    # Apply RoPE to Q
    for s in range(seq_len):
        c = cos_np[s, :half_dim]
        sn = sin_np[s, :half_dim]
        for h in range(n_heads_q):
            q0 = q_np[s, h, :half_dim].copy()
            q1 = q_np[s, h, half_dim:].copy()
            q_np[s, h, :half_dim] = q0 * c - q1 * sn
            q_np[s, h, half_dim:] = q1 * c + q0 * sn

    # Apply RoPE to K
    for s in range(seq_len):
        c = cos_np[s, :half_dim]
        sn = sin_np[s, :half_dim]
        for h in range(n_heads_k):
            k0 = k_np[s, h, :half_dim].copy()
            k1 = k_np[s, h, half_dim:].copy()
            k_np[s, h, :half_dim] = k0 * c - k1 * sn
            k_np[s, h, half_dim:] = k1 * c + k0 * sn

    # Update the GPUArray data in-place
    q._data = from_numpy(q_np)._data
    k._data = from_numpy(k_np)._data


def _rope_inplace_native(
    q: GPUArray,
    k: GPUArray,
    cos: GPUArray,
    sin: GPUArray,
) -> None:
    """Native C++ CUDA implementation of rope_inplace."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    q_native = q._get_native()
    k_native = k._get_native()
    cos_native = cos._get_native()
    sin_native = sin._get_native()
    native.rope_inplace(q_native, k_native, cos_native, sin_native)


def rope_inplace_f32table(
    q: GPUArray,
    k: GPUArray,
    cos: GPUArray,
    sin: GPUArray,
) -> None:
    """Apply RoPE with FP32 cos/sin tables (higher precision for bf16/f16).

    Uses FP32 cos/sin tables for higher precision computation, avoiding
    the need to convert tables to bf16/f16.

    Args:
        q: Query tensor [seq_len, n_heads_q, head_dim] (bf16 or f16, modified in-place).
        k: Key tensor [seq_len, n_heads_k, head_dim] (bf16 or f16, modified in-place).
        cos: Precomputed cosine [seq_len, head_dim] (f32).
        sin: Precomputed sine [seq_len, head_dim] (f32).
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    q_native = q._get_native()
    k_native = k._get_native()
    cos_native = cos._get_native()
    sin_native = sin._get_native()
    native.rope_inplace_f32table(q_native, k_native, cos_native, sin_native)


__all__ = [
    "rope_inplace",
    "rope_inplace_f32table",
]
