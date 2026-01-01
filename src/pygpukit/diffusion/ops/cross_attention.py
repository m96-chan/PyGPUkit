"""Cross-Attention for diffusion models.

Cross-attention enables conditioning on text embeddings.
Unlike causal self-attention, cross-attention is bidirectional
and uses different sequence lengths for Q (image) and K/V (text).
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy


def cross_attention(
    query: GPUArray,
    key: GPUArray,
    value: GPUArray,
    scale: float = 0.0,
    mask: GPUArray | None = None,
) -> GPUArray:
    """Cross-Attention (non-causal).

    Computes attention where query comes from one modality (e.g., image)
    and key/value come from another modality (e.g., text).

    Args:
        query: Query tensor [B, H, N_q, D] (image features)
        key: Key tensor [B, H, N_kv, D] (text features)
        value: Value tensor [B, H, N_kv, D] (text features)
        scale: Attention scale. If <= 0, uses 1/sqrt(D).
        mask: Optional attention mask [B, N_q, N_kv] or [N_q, N_kv].

    Returns:
        Output tensor [B, H, N_q, D].

    Note:
        Unlike sdpa_causal, this is bidirectional (no causal mask).
        N_q can differ from N_kv.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("cross_attention expects 4D inputs [B, H, N, D]")

    B, H, N_q, D = query.shape
    _, _, N_kv, _ = key.shape

    if key.shape != value.shape:
        raise ValueError("key and value must have same shape")
    if key.shape[0] != B or key.shape[1] != H or key.shape[3] != D:
        raise ValueError("key/value batch, heads, or head_dim mismatch with query")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _cross_attention_native(query, key, value, scale, mask)
    else:
        return _cross_attention_cpu(query, key, value, scale, mask)


def _cross_attention_cpu(
    query: GPUArray,
    key: GPUArray,
    value: GPUArray,
    scale: float,
    mask: GPUArray | None,
) -> GPUArray:
    """CPU implementation of cross-attention."""
    q = query.to_numpy()
    k = key.to_numpy()
    v = value.to_numpy()

    _, _, _, D = q.shape

    if scale <= 0:
        scale = 1.0 / np.sqrt(D)

    # Compute attention scores: [B, H, N_q, N_kv]
    # q @ k^T
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

    # Apply mask if provided
    if mask is not None:
        mask_np = mask.to_numpy()
        # Broadcast mask to [B, H, N_q, N_kv]
        if mask_np.ndim == 2:
            mask_np = mask_np[np.newaxis, np.newaxis, :, :]
        elif mask_np.ndim == 3:
            mask_np = mask_np[:, np.newaxis, :, :]
        scores = scores + mask_np

    # Softmax over key dimension
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # Weighted sum of values: [B, H, N_q, D]
    output = np.matmul(weights, v)

    return from_numpy(output.astype(q.dtype))


def _cross_attention_native(
    query: GPUArray,
    key: GPUArray,
    value: GPUArray,
    scale: float,
    mask: GPUArray | None,
) -> GPUArray:
    """Native CUDA implementation of cross-attention."""
    # TODO: Implement native CUDA kernel for cross-attention
    return _cross_attention_cpu(query, key, value, scale, mask)


def self_attention(
    query: GPUArray,
    key: GPUArray,
    value: GPUArray,
    scale: float = 0.0,
) -> GPUArray:
    """Self-Attention (non-causal, bidirectional).

    Same as cross_attention but typically Q, K, V come from the same source.

    Args:
        query: Query tensor [B, H, N, D]
        key: Key tensor [B, H, N, D]
        value: Value tensor [B, H, N, D]
        scale: Attention scale.

    Returns:
        Output tensor [B, H, N, D].
    """
    return cross_attention(query, key, value, scale, mask=None)


__all__ = [
    "cross_attention",
    "self_attention",
]
