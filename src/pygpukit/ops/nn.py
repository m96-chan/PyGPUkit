"""Neural network operations for GPUArrays.

Corresponds to native/ops/nn/.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype

# =============================================================================
# Activation Functions
# =============================================================================


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


# =============================================================================
# Normalization Layers
# =============================================================================


def layernorm(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float = 1e-5,
) -> GPUArray:
    """Layer normalization.

    Computes: (x - mean) / sqrt(var + eps) * gamma + beta

    Args:
        input: Input array of shape [batch, features] or [batch, seq_len, features].
        gamma: Scale parameter of shape [features].
        beta: Bias parameter of shape [features].
        eps: Small epsilon for numerical stability.

    Returns:
        A new GPUArray containing the normalized output.

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(input, "layernorm")

    if input.ndim not in (2, 3):
        raise ValueError(f"layernorm expects 2D or 3D input, got {input.ndim}D")
    if gamma.ndim != 1 or beta.ndim != 1:
        raise ValueError("layernorm expects 1D gamma and beta")
    if input.dtype != gamma.dtype or input.dtype != beta.dtype:
        raise ValueError("layernorm: all inputs must have same dtype")

    features = input.shape[-1]  # Last dimension is features
    if gamma.shape[0] != features or beta.shape[0] != features:
        raise ValueError(
            f"layernorm: gamma/beta size {gamma.shape[0]} must match features {features}"
        )

    # Handle 3D input by reshaping to 2D, processing, and reshaping back
    if input.ndim == 3:
        batch, seq_len, feat = input.shape
        input_2d = input.reshape(batch * seq_len, feat)
        result_2d = _layernorm_dispatch(input_2d, gamma, beta, eps)
        return result_2d.reshape(batch, seq_len, feat)
    else:
        return _layernorm_dispatch(input, gamma, beta, eps)


def _layernorm_dispatch(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float,
) -> GPUArray:
    """Dispatch layernorm to native or CPU implementation."""
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _layernorm_native(input, gamma, beta, eps)
    else:
        return _layernorm_cpu(input, gamma, beta, eps)


def _layernorm_cpu(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float,
) -> GPUArray:
    """CPU implementation of layernorm."""
    x = input.to_numpy()
    g = gamma.to_numpy()
    b = beta.to_numpy()

    # Compute mean and variance along features axis
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)

    # Normalize
    normalized = (x - mean) / np.sqrt(var + eps)

    # Apply affine transform
    result = normalized * g + b
    return from_numpy(result)


def _layernorm_native(
    input: GPUArray,
    gamma: GPUArray,
    beta: GPUArray,
    eps: float,
) -> GPUArray:
    """Native C++ CUDA implementation of layernorm (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    gamma_native = gamma._get_native()
    beta_native = beta._get_native()
    c_native = native.layernorm(input_native, gamma_native, beta_native, eps)
    return GPUArray._wrap_native(c_native)


def rmsnorm(
    input: GPUArray,
    gamma: GPUArray,
    eps: float = 1e-5,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """RMS Normalization (Root Mean Square Normalization).

    Computes: x / sqrt(mean(x^2) + eps) * gamma

    Simpler than LayerNorm (no mean subtraction, no beta).
    Used in Llama and other modern LLMs.

    Args:
        input: Input array of shape [batch, features].
        gamma: Scale parameter of shape [features].
        eps: Small epsilon for numerical stability.
        out: Optional output buffer. If provided, result is written in-place
            (for CUDA Graph capture).

    Returns:
        A new GPUArray containing the normalized output (or out if provided).

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(input, "rmsnorm")

    if input.ndim != 2:
        raise ValueError(f"rmsnorm expects 2D input [batch, features], got {input.ndim}D")
    if gamma.ndim != 1:
        raise ValueError("rmsnorm expects 1D gamma")
    if input.dtype != gamma.dtype:
        raise ValueError("rmsnorm: all inputs must have same dtype")

    features = input.shape[1]
    if gamma.shape[0] != features:
        raise ValueError(f"rmsnorm: gamma size {gamma.shape[0]} must match features {features}")

    # Validate out array if provided
    if out is not None:
        if out.shape != input.shape:
            raise ValueError(f"out shape {out.shape} does not match input shape {input.shape}")
        if out.dtype != input.dtype:
            raise ValueError(f"out dtype {out.dtype} does not match input dtype {input.dtype}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _rmsnorm_native(input, gamma, eps, out=out)
    else:
        return _rmsnorm_cpu(input, gamma, eps, out=out)


def _rmsnorm_cpu(
    input: GPUArray,
    gamma: GPUArray,
    eps: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """CPU implementation of rmsnorm."""
    x = input.to_numpy()
    g = gamma.to_numpy()

    # RMS = sqrt(mean(x^2) + eps)
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + eps)

    # Normalize and scale
    result = (x / rms) * g

    if out is not None:
        out_np = out.to_numpy()
        np.copyto(out_np, result)
        out._data = from_numpy(out_np)._data
        return out
    return from_numpy(result)


def _rmsnorm_native(
    input: GPUArray,
    gamma: GPUArray,
    eps: float,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ CUDA implementation of rmsnorm (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    gamma_native = gamma._get_native()

    if out is not None:
        out_native = out._get_native()
        native.rmsnorm_(input_native, gamma_native, out_native, eps)
        return out
    else:
        c_native = native.rmsnorm(input_native, gamma_native, eps)
        return GPUArray._wrap_native(c_native)


# =============================================================================
# Bias Operations
# =============================================================================


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


# =============================================================================
# Attention Operations
# =============================================================================


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


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================


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


# =============================================================================
# QKV Split Operations
# =============================================================================


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
