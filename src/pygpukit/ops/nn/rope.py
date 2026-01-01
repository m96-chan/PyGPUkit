"""RoPE (Rotary Position Embedding) operations for GPUArrays.

Corresponds to native/ops/nn/rope/.
"""

from __future__ import annotations

import numpy as np

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
    backend = get_backend()

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
    backend.copy_host_to_device(q_np.ravel(), q._device_ptr)
    backend.copy_host_to_device(k_np.ravel(), k._device_ptr)


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
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        cos_native, sin_native = native.rope_init_ntk_aware(max_seq_len, head_dim, base, scale)
        return GPUArray._wrap_native(cos_native), GPUArray._wrap_native(sin_native)
    else:
        return _rope_init_ntk_aware_cpu(max_seq_len, head_dim, base, scale)


def _rope_init_ntk_aware_cpu(
    max_seq_len: int,
    head_dim: int,
    base: float,
    scale: float,
) -> tuple[GPUArray, GPUArray]:
    """CPU implementation of NTK-aware RoPE initialization."""
    # NTK-aware scaling: base' = base * scale^(dim / (dim - 2))
    scaled_base = base * (scale ** (head_dim / (head_dim - 2))) if scale > 1.0 else base

    # Compute inverse frequencies
    half_dim = head_dim // 2
    inv_freq = 1.0 / (scaled_base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))

    # Compute positions
    positions = np.arange(max_seq_len, dtype=np.float32)

    # Compute angles: [max_seq_len, half_dim]
    angles = np.outer(positions, inv_freq)

    # Compute cos and sin, then interleave to get [max_seq_len, head_dim]
    cos_half = np.cos(angles)
    sin_half = np.sin(angles)

    # Interleave: [cos0, cos0, cos1, cos1, ...] for compatibility with RoPE apply
    cos_table = np.zeros((max_seq_len, head_dim), dtype=np.float32)
    sin_table = np.zeros((max_seq_len, head_dim), dtype=np.float32)
    cos_table[:, 0::2] = cos_half
    cos_table[:, 1::2] = cos_half
    sin_table[:, 0::2] = sin_half
    sin_table[:, 1::2] = sin_half

    return from_numpy(cos_table), from_numpy(sin_table)


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
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        cos_native, sin_native = native.rope_init_yarn(
            max_seq_len,
            head_dim,
            base,
            scale,
            original_max_len,
            beta_fast,
            beta_slow,
            mscale,
        )
        return GPUArray._wrap_native(cos_native), GPUArray._wrap_native(sin_native)
    else:
        return _rope_init_yarn_cpu(
            max_seq_len, head_dim, base, scale, original_max_len, beta_fast, beta_slow
        )


def _rope_init_yarn_cpu(
    max_seq_len: int,
    head_dim: int,
    base: float,
    scale: float,
    original_max_len: int,
    beta_fast: float,
    beta_slow: float,
) -> tuple[GPUArray, GPUArray]:
    """CPU implementation of YaRN RoPE initialization."""
    half_dim = head_dim // 2

    # Compute base frequencies
    inv_freq = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))

    # Compute wavelengths for each dimension
    wavelengths = 2 * np.pi / inv_freq

    # Compute interpolation factors (YaRN dimension-wise interpolation)
    low_freq_wavelen = original_max_len / beta_slow
    high_freq_wavelen = original_max_len / beta_fast

    # Interpolation factor: 0 = no interpolation, 1 = full interpolation
    smooth = np.clip(
        (wavelengths - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen), 0, 1
    )

    # Apply interpolation: mix between original and scaled frequencies
    scaled_inv_freq = inv_freq / scale
    interpolated_inv_freq = (1 - smooth) * scaled_inv_freq + smooth * inv_freq

    # Compute positions
    positions = np.arange(max_seq_len, dtype=np.float32)

    # Compute angles
    angles = np.outer(positions, interpolated_inv_freq)

    # Compute cos and sin
    cos_half = np.cos(angles)
    sin_half = np.sin(angles)

    # Interleave
    cos_table = np.zeros((max_seq_len, head_dim), dtype=np.float32)
    sin_table = np.zeros((max_seq_len, head_dim), dtype=np.float32)
    cos_table[:, 0::2] = cos_half
    cos_table[:, 1::2] = cos_half
    sin_table[:, 0::2] = sin_half
    sin_table[:, 1::2] = sin_half

    return from_numpy(cos_table), from_numpy(sin_table)


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
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        cos_native, sin_native = native.rope_init_linear(max_seq_len, head_dim, base, scale)
        return GPUArray._wrap_native(cos_native), GPUArray._wrap_native(sin_native)
    else:
        return _rope_init_linear_cpu(max_seq_len, head_dim, base, scale)


def _rope_init_linear_cpu(
    max_seq_len: int,
    head_dim: int,
    base: float,
    scale: float,
) -> tuple[GPUArray, GPUArray]:
    """CPU implementation of linear position interpolation RoPE."""
    half_dim = head_dim // 2

    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))

    # Compute scaled positions (linear interpolation: pos' = pos / scale)
    positions = np.arange(max_seq_len, dtype=np.float32) / scale

    # Compute angles
    angles = np.outer(positions, inv_freq)

    # Compute cos and sin
    cos_half = np.cos(angles)
    sin_half = np.sin(angles)

    # Interleave
    cos_table = np.zeros((max_seq_len, head_dim), dtype=np.float32)
    sin_table = np.zeros((max_seq_len, head_dim), dtype=np.float32)
    cos_table[:, 0::2] = cos_half
    cos_table[:, 1::2] = cos_half
    sin_table[:, 0::2] = sin_half
    sin_table[:, 1::2] = sin_half

    return from_numpy(cos_table), from_numpy(sin_table)


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
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        encoding_native = native.pope_init_encoding(max_seq_len, head_dim, base)
        return GPUArray._wrap_native(encoding_native)
    else:
        return _pope_init_encoding_cpu(max_seq_len, head_dim, base)


def _pope_init_encoding_cpu(
    max_seq_len: int,
    head_dim: int,
    base: float,
) -> GPUArray:
    """CPU implementation of sinusoidal positional encoding."""
    encoding = np.zeros((max_seq_len, head_dim), dtype=np.float32)

    positions = np.arange(max_seq_len, dtype=np.float32)
    half_dim = head_dim // 2

    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))

    # Compute angles
    angles = np.outer(positions, inv_freq)

    # PE(pos, 2i) = sin, PE(pos, 2i+1) = cos
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)

    return from_numpy(encoding)


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
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        native.pope_inplace(q._get_native(), k._get_native(), encoding._get_native(), start_pos)
    else:
        _pope_inplace_cpu(q, k, encoding, start_pos)


def _pope_inplace_cpu(
    q: GPUArray,
    k: GPUArray,
    encoding: GPUArray,
    start_pos: int,
) -> None:
    """CPU implementation of PoPE in-place application."""
    backend = get_backend()

    q_np = q.to_numpy()
    k_np = k.to_numpy()
    enc_np = encoding.to_numpy()

    seq_len = q_np.shape[0]
    n_heads_q = q_np.shape[1]
    n_heads_k = k_np.shape[1]

    # Add positional encoding to each position
    for s in range(seq_len):
        pos = start_pos + s
        enc_pos = enc_np[pos]

        # Add to all heads
        for h in range(n_heads_q):
            q_np[s, h] = q_np[s, h] + enc_pos

        for h in range(n_heads_k):
            k_np[s, h] = k_np[s, h] + enc_pos

    # Update the GPUArray data in-place
    backend.copy_host_to_device(q_np.ravel(), q._device_ptr)
    backend.copy_host_to_device(k_np.ravel(), k._device_ptr)


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
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        slopes_native = native.alibi_init_slopes(num_heads)
        return GPUArray._wrap_native(slopes_native)
    else:
        return _alibi_init_slopes_cpu(num_heads)


def _alibi_init_slopes_cpu(num_heads: int) -> GPUArray:
    """CPU implementation of ALiBi slopes initialization."""
    # m_h = 2^(-8 * (h+1) / num_heads)
    slopes = np.array([2 ** (-8 * (h + 1) / num_heads) for h in range(num_heads)], dtype=np.float32)
    return from_numpy(slopes)


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
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        bias_native = native.alibi_compute_bias(seq_len, num_heads, slopes._get_native(), causal)
        return GPUArray._wrap_native(bias_native)
    else:
        return _alibi_compute_bias_cpu(seq_len, num_heads, slopes, causal)


def _alibi_compute_bias_cpu(
    seq_len: int,
    num_heads: int,
    slopes: GPUArray,
    causal: bool,
) -> GPUArray:
    """CPU implementation of ALiBi bias computation."""
    slopes_np = slopes.to_numpy()

    # Create bias tensor [num_heads, seq_len, seq_len]
    bias = np.zeros((num_heads, seq_len, seq_len), dtype=np.float32)

    # Compute distance matrix
    for h in range(num_heads):
        slope = slopes_np[h]
        for i in range(seq_len):
            for j in range(seq_len):
                if causal and j > i:
                    # Causal mask: future positions are masked
                    bias[h, i, j] = -1e9
                else:
                    # ALiBi bias: -slope * distance
                    bias[h, i, j] = -slope * (i - j)

    return from_numpy(bias)


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
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        native.alibi_add_bias(scores._get_native(), slopes._get_native(), start_pos)
    else:
        _alibi_add_bias_cpu(scores, slopes, start_pos)


def _alibi_add_bias_cpu(
    scores: GPUArray,
    slopes: GPUArray,
    start_pos: int,
) -> None:
    """CPU implementation of ALiBi in-place bias addition."""
    backend = get_backend()

    scores_np = scores.to_numpy()
    slopes_np = slopes.to_numpy()

    # scores shape: [batch, num_heads, q_len, kv_len]
    batch, num_heads, q_len, kv_len = scores_np.shape

    for b in range(batch):
        for h in range(num_heads):
            slope = slopes_np[h]
            for qi in range(q_len):
                q_pos = start_pos + qi
                for kj in range(kv_len):
                    # Distance from query position to key position
                    distance = q_pos - kj
                    scores_np[b, h, qi, kj] -= slope * distance

    # Update the GPUArray data in-place
    backend.copy_host_to_device(scores_np.ravel(), scores._device_ptr)


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
