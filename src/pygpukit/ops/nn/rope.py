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


def rope_init_ntk_aware(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    scale: float = 1.0,
) -> tuple[GPUArray, GPUArray]:
    """Initialize RoPE with NTK-aware frequency scaling.

    NTK-aware interpolation scales the base frequency instead of positions:
    base' = base * scale^(dim / (dim - 2))

    This preserves high-frequency components better than linear interpolation.

    Args:
        max_seq_len: Maximum sequence length.
        head_dim: Dimension per head.
        base: Base for frequency computation (default 10000).
        scale: Context extension scale factor (e.g., 2.0 for 2x context).

    Returns:
        Tuple of (cos_table, sin_table) each of shape [max_seq_len, head_dim].

    Example:
        >>> cos, sin = rope_init_ntk_aware(8192, 128, scale=2.0)
        >>> rope_inplace(q, k, cos, sin)
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    cos_native, sin_native = native.rope_init_ntk_aware(max_seq_len, head_dim, base, scale)
    return GPUArray._wrap_native(cos_native), GPUArray._wrap_native(sin_native)


def rope_init_yarn(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    scale: float = 1.0,
    original_max_len: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 0.1,
) -> tuple[GPUArray, GPUArray]:
    """Initialize RoPE with YaRN dimension-wise interpolation.

    YaRN (Yet another RoPE extensioN) combines NTK with attention scaling
    and dimension-wise interpolation for state-of-the-art context extension.

    Different frequency bands are handled differently:
    - Low frequency (local attention): no interpolation
    - High frequency: full interpolation
    - Mid frequency: gradual transition

    Args:
        max_seq_len: Maximum sequence length (extended).
        head_dim: Dimension per head.
        base: Base for frequency computation (default 10000).
        scale: Context extension scale factor.
        original_max_len: Original training context length.
        beta_fast: Fast wavelength threshold (default 32).
        beta_slow: Slow wavelength threshold (default 1).
        mscale: Attention scaling factor (default 0.1).

    Returns:
        Tuple of (cos_table, sin_table) each of shape [max_seq_len, head_dim].

    Example:
        >>> cos, sin = rope_init_yarn(32768, 128, scale=4.0, original_max_len=4096)
        >>> rope_inplace(q, k, cos, sin)
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    cos_native, sin_native = native.rope_init_yarn(
        max_seq_len, head_dim, base, scale, original_max_len, beta_fast, beta_slow, mscale
    )
    return GPUArray._wrap_native(cos_native), GPUArray._wrap_native(sin_native)


def rope_init_linear(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    scale: float = 1.0,
) -> tuple[GPUArray, GPUArray]:
    """Initialize RoPE with linear position interpolation.

    Simple baseline: pos' = pos / scale.
    Works but degrades quality at high scales.

    Args:
        max_seq_len: Maximum sequence length.
        head_dim: Dimension per head.
        base: Base for frequency computation (default 10000).
        scale: Context extension scale factor.

    Returns:
        Tuple of (cos_table, sin_table) each of shape [max_seq_len, head_dim].
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    cos_native, sin_native = native.rope_init_linear(max_seq_len, head_dim, base, scale)
    return GPUArray._wrap_native(cos_native), GPUArray._wrap_native(sin_native)


def pope_init_encoding(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> GPUArray:
    """Initialize sinusoidal positional encoding table (PoPE).

    PoPE is an additive positional encoding alternative to RoPE.
    Uses sinusoidal encoding: PE(pos, 2i) = sin(pos / base^(2i/d))
                               PE(pos, 2i+1) = cos(pos / base^(2i/d))

    Args:
        max_seq_len: Maximum sequence length.
        head_dim: Dimension per head.
        base: Base for frequency computation (default 10000).

    Returns:
        Encoding tensor of shape [max_seq_len, head_dim].

    Example:
        >>> encoding = pope_init_encoding(2048, 128)
        >>> pope_inplace(q, k, encoding)
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    encoding_native = native.pope_init_encoding(max_seq_len, head_dim, base)
    return GPUArray._wrap_native(encoding_native)


def pope_inplace(
    q: GPUArray,
    k: GPUArray,
    encoding: GPUArray,
    start_pos: int = 0,
) -> None:
    """Apply additive positional encoding to Q and K in-place.

    PoPE adds positional information by simple addition (vs RoPE's rotation).
    Simpler compute but limited extrapolation compared to RoPE.

    Args:
        q: Query tensor [seq_len, n_heads_q, head_dim] (modified in-place).
        k: Key tensor [seq_len, n_heads_k, head_dim] (modified in-place).
        encoding: Position encoding [max_seq_len, head_dim] (f32).
        start_pos: Starting position for incremental decoding.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    native.pope_inplace(q._get_native(), k._get_native(), encoding._get_native(), start_pos)


def alibi_init_slopes(num_heads: int) -> GPUArray:
    """Initialize ALiBi head-specific slopes.

    ALiBi (Attention with Linear Biases) adds a linear bias to attention
    scores based on query-key distance: scores[i,j] -= slope * |i - j|

    Each head gets a different slope: m_h = 2^(-8 * h / num_heads)

    Args:
        num_heads: Number of attention heads.

    Returns:
        Slopes tensor of shape [num_heads].

    Example:
        >>> slopes = alibi_init_slopes(32)
        >>> bias = alibi_compute_bias(512, 32, slopes)
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    slopes_native = native.alibi_init_slopes(num_heads)
    return GPUArray._wrap_native(slopes_native)


def alibi_compute_bias(
    seq_len: int,
    num_heads: int,
    slopes: GPUArray,
    causal: bool = True,
) -> GPUArray:
    """Compute ALiBi bias matrix for attention.

    Creates a bias tensor to be added to attention scores.
    For causal attention, positions j > i are masked with -inf.

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        slopes: Head-specific slopes [num_heads].
        causal: Whether to apply causal masking (default True).

    Returns:
        Bias tensor of shape [num_heads, seq_len, seq_len].
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    bias_native = native.alibi_compute_bias(seq_len, num_heads, slopes._get_native(), causal)
    return GPUArray._wrap_native(bias_native)


def alibi_add_bias(
    scores: GPUArray,
    slopes: GPUArray,
    start_pos: int = 0,
) -> None:
    """Add ALiBi bias to attention scores in-place.

    Efficiently adds position-dependent bias during incremental decoding.

    Args:
        scores: Attention scores [batch, num_heads, q_len, kv_len] (modified in-place).
        slopes: Head-specific slopes [num_heads].
        start_pos: Starting position for incremental decoding.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    native.alibi_add_bias(scores._get_native(), slopes._get_native(), start_pos)


__all__ = [
    "rope_inplace",
    "rope_inplace_f32table",
    # RoPE extensions
    "rope_init_ntk_aware",
    "rope_init_yarn",
    "rope_init_linear",
    # PoPE
    "pope_init_encoding",
    "pope_inplace",
    # ALiBi
    "alibi_init_slopes",
    "alibi_compute_bias",
    "alibi_add_bias",
]
