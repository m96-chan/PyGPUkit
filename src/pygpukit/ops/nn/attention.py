"""Attention operations for GPUArrays.

Corresponds to native/ops/nn/attention/.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype


def sdpa_causal(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    scale: float = 0.0,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Scaled Dot-Product Attention with causal mask.

    Computes attention with automatic causal masking for autoregressive
    sequence generation. This is the core attention operation used in
    transformer models.

    Algorithm:
        scores = Q @ K^T / scale
        scores = apply_causal_mask(scores)
        weights = softmax(scores)
        output = weights @ V

    Args:
        Q: Query tensor of shape [n_heads, q_len, head_dim].
        K: Key tensor of shape [n_heads, kv_len, head_dim].
        V: Value tensor of shape [n_heads, kv_len, head_dim].
        scale: Scaling factor (typically 1/sqrt(head_dim)).
               If <= 0, computed automatically from head_dim.
        out: Optional output buffer [n_heads, q_len, head_dim].
             If provided, result is written in-place (for CUDA Graph capture).

    Returns:
        Output tensor of shape [n_heads, q_len, head_dim].

    Raises:
        ValueError: If shapes or dtypes don't match.

    Note:
        For KV cache usage during inference, kv_len >= q_len.
        The causal mask ensures query at position i can only attend
        to key positions 0 to (kv_len - q_len + i).
    """
    _validate_float_dtype(Q, "sdpa_causal")

    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("sdpa_causal expects 3D inputs [n_heads, seq_len, head_dim]")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError("sdpa_causal: Q, K, V must have same dtype")

    n_heads, q_len, head_dim = Q.shape

    if K.shape[0] != n_heads or V.shape[0] != n_heads:
        raise ValueError("sdpa_causal: n_heads mismatch")
    if K.shape[2] != head_dim or V.shape[2] != head_dim:
        raise ValueError("sdpa_causal: head_dim mismatch")
    if K.shape[1] != V.shape[1]:
        raise ValueError("sdpa_causal: K and V seq_len mismatch")

    # Validate out array if provided
    if out is not None:
        if out.shape != (n_heads, q_len, head_dim):
            raise ValueError(
                f"out shape {out.shape} does not match expected {(n_heads, q_len, head_dim)}"
            )
        if out.dtype != Q.dtype:
            raise ValueError(f"out dtype {out.dtype} does not match Q dtype {Q.dtype}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _sdpa_causal_native(Q, K, V, scale, out=out)
    else:
        return _sdpa_causal_cpu(Q, K, V, scale, out=out)


def _sdpa_causal_cpu(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    scale: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """CPU implementation of SDPA with causal mask."""
    q = Q.to_numpy()
    k = K.to_numpy()
    v = V.to_numpy()

    n_heads, q_len, head_dim = q.shape
    kv_len = k.shape[1]

    if scale <= 0:
        scale = 1.0 / np.sqrt(head_dim)

    # scores: [n_heads, q_len, kv_len]
    scores = np.matmul(q, k.transpose(0, 2, 1)) * scale

    # Create causal mask
    causal_offset = kv_len - q_len
    for i in range(q_len):
        max_attend = causal_offset + i + 1
        if max_attend < kv_len:
            scores[:, i, max_attend:] = -np.inf

    # Softmax over last dimension
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # output: [n_heads, q_len, head_dim]
    output = np.matmul(weights, v)

    if out is not None:
        out_np = out.to_numpy()
        np.copyto(out_np, output.astype(q.dtype))
        out._data = from_numpy(out_np)._data
        return out
    return from_numpy(output.astype(q.dtype))


def _sdpa_causal_native(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    scale: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ CUDA implementation of SDPA with causal mask."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    q_native = Q._get_native()
    k_native = K._get_native()
    v_native = V._get_native()

    if out is not None:
        out_native = out._get_native()
        native.sdpa_causal_(q_native, k_native, v_native, out_native, scale)
        return out
    else:
        c_native = native.sdpa_causal(q_native, k_native, v_native, scale)
        return GPUArray._wrap_native(c_native)


def sdpa_causal_fixed_cache(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    out: GPUArray,
    context_len: int,
    scale: float = 0.0,
) -> None:
    """SDPA with fixed-length KV cache for CUDA Graph capture.

    This variant is designed for use with pre-allocated KV caches where
    the buffer size (max_seq_len) is larger than the actual context length.

    Args:
        Q: Query tensor of shape [n_heads, q_len, head_dim].
        K: Key cache of shape [n_heads, max_seq_len, head_dim].
        V: Value cache of shape [n_heads, max_seq_len, head_dim].
        out: Pre-allocated output buffer [n_heads, q_len, head_dim].
        context_len: Actual number of valid tokens in KV cache.
        scale: Scaling factor (typically 1/sqrt(head_dim)).
               If <= 0, computed automatically from head_dim.

    Raises:
        ValueError: If shapes or dtypes don't match, or context_len is invalid.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    q_native = Q._get_native()
    k_native = K._get_native()
    v_native = V._get_native()
    out_native = out._get_native()

    native.sdpa_causal_fixed_cache(q_native, k_native, v_native, out_native, context_len, scale)


def sdpa_causal_fixed_cache_ptr(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    out: GPUArray,
    context_len_buf: GPUArray,
    max_kv_len: int,
    scale: float = 0.0,
) -> None:
    """SDPA with pointer-based context_len for CUDA Graph replay.

    This variant reads context_len from a GPU buffer at runtime, enabling
    CUDA Graph replay with dynamic context lengths without re-capture.

    Args:
        Q: Query tensor of shape [n_heads, q_len, head_dim].
        K: Key cache of shape [n_heads, max_seq_len, head_dim].
        V: Value cache of shape [n_heads, max_seq_len, head_dim].
        out: Pre-allocated output buffer [n_heads, q_len, head_dim].
        context_len_buf: GPU int32 buffer containing actual context_len [1].
        max_kv_len: Maximum context length (for shared memory allocation
                    during graph capture). Must be <= K.shape[1].
        scale: Scaling factor (typically 1/sqrt(head_dim)).
               If <= 0, computed automatically from head_dim.

    Note:
        For CUDA Graph: capture with max_kv_len, then update context_len_buf
        before each replay to change the effective context length.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    q_native = Q._get_native()
    k_native = K._get_native()
    v_native = V._get_native()
    out_native = out._get_native()
    ctx_buf_native = context_len_buf._get_native()

    native.sdpa_causal_fixed_cache_ptr(
        q_native, k_native, v_native, out_native, ctx_buf_native, max_kv_len, scale
    )


def fa3_fp8_available() -> bool:
    """Check if FA3 FP8 (FP8 Q@K^T with block-scale MMA) is available.

    Requires SM120+ (Blackwell GeForce RTX 5000 series).

    Returns:
        True if FA3 FP8 is available on the current device.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    return native.fa3_fp8_available()


def get_sm_version() -> int:
    """Get the SM (Streaming Multiprocessor) version of the current device.

    Returns:
        SM version as an integer (e.g., 120 for SM120).
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    return native.get_sm_version()


def sdpa_causal_fp8(
    Q: GPUArray,
    K: GPUArray,
    V: GPUArray,
    out: GPUArray,
    scale: float = 0.0,
) -> None:
    """SDPA with FP8 Q@K^T using block-scale MMA (SM120+).

    This is an optimized Flash Attention 3 variant that uses FP8 block-scale MMA
    for Q@K^T computation, providing ~50% memory bandwidth reduction with only
    ~0.25% expected error vs BF16 FA3.

    The implementation:
    - Quantizes Q and K to FP8 E4M3 internally (with per-32-element scales)
    - Uses `mma.sync.aligned.block_scale.m16n8k32.f32.e4m3.e4m3` for Q@K^T
    - Keeps V as BF16 and uses WMMA for P@V (FP8 V gives ~18% error)
    - Uses TMA for efficient async loading

    Args:
        Q: Query tensor of shape [n_heads, q_len, head_dim], BFloat16.
        K: Key tensor of shape [n_heads, kv_len, head_dim], BFloat16.
        V: Value tensor of shape [n_heads, kv_len, head_dim], BFloat16.
        out: Output buffer of shape [n_heads, q_len, head_dim], BFloat16.
        scale: Scaling factor (typically 1/sqrt(head_dim)).
               If <= 0, computed automatically from head_dim.

    Raises:
        RuntimeError: If FA3 FP8 is not available (requires SM120+).
        ValueError: If shapes or dtypes don't match.

    Note:
        Requires SM120+ (RTX 5090 or newer). Use fa3_fp8_available() to check.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    if not native.fa3_fp8_available():
        raise RuntimeError("FA3 FP8 requires SM120+ (Blackwell GeForce)")

    _validate_float_dtype(Q, "sdpa_causal_fp8")

    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("sdpa_causal_fp8 expects 3D inputs [n_heads, seq_len, head_dim]")
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError("sdpa_causal_fp8: Q, K, V must have same dtype (BFloat16)")
    if out.dtype != Q.dtype:
        raise ValueError("sdpa_causal_fp8: out must have same dtype as Q")

    n_heads, q_len, head_dim = Q.shape

    if K.shape[0] != n_heads or V.shape[0] != n_heads:
        raise ValueError("sdpa_causal_fp8: n_heads mismatch")
    if K.shape[2] != head_dim or V.shape[2] != head_dim:
        raise ValueError("sdpa_causal_fp8: head_dim mismatch")
    if K.shape[1] != V.shape[1]:
        raise ValueError("sdpa_causal_fp8: K and V seq_len mismatch")

    if out.shape != (n_heads, q_len, head_dim):
        raise ValueError(
            f"out shape {out.shape} does not match expected {(n_heads, q_len, head_dim)}"
        )

    q_native = Q._get_native()
    k_native = K._get_native()
    v_native = V._get_native()
    out_native = out._get_native()

    native.sdpa_causal_fp8(q_native, k_native, v_native, out_native, scale)


def test_fp8_mma_direct() -> None:
    """Run direct FP8 MMA test to debug C fragment layout.

    This test runs the FP8 block-scale MMA instruction with known sparse inputs
    and prints the results to verify correct fragment loading and output layout.

    Requires SM120+ (RTX 5090 or newer).
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    native.test_fp8_mma_direct()


__all__ = [
    "sdpa_causal",
    "sdpa_causal_fixed_cache",
    "sdpa_causal_fixed_cache_ptr",
    "fa3_fp8_available",
    "get_sm_version",
    "sdpa_causal_fp8",
    "test_fp8_mma_direct",
]
