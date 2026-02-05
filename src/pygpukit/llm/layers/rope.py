"""Rotary Position Embedding (RoPE) utilities for PyGPUkit LLM.

Provides:
- precompute_freqs_cis: Precompute RoPE cos/sin tables (1D)
- apply_rotary_pos_emb_numpy: Apply RoPE on CPU (numpy)
- precompute_freqs_cis_3d: Precompute Multimodal RoPE cos/sin tables (3D)
- apply_multimodal_rotary_pos_emb_numpy: Apply Multimodal RoPE (3D) on CPU
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def precompute_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute rotary embedding cos/sin tables."""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    cos = np.concatenate([cos, cos], axis=-1)
    sin = np.concatenate([sin, sin], axis=-1)
    return cos, sin


def apply_rotary_pos_emb_numpy(
    q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rotary position embeddings to Q and K (numpy version)."""

    def rotate_half(x: np.ndarray) -> np.ndarray:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return np.concatenate([-x2, x1], axis=-1)

    cos = cos[:, np.newaxis, :]
    sin = sin[:, np.newaxis, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Multimodal RoPE (3D) for Qwen3-TTS
# =============================================================================


def precompute_freqs_cis_3d(
    head_dim: int,
    max_t: int,
    max_h: int,
    max_w: int,
    mrope_section: Sequence[int],
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute Multimodal RoPE cos/sin tables for 3D positions.

    Qwen3-TTS uses Multimodal RoPE to encode temporal (t), height (h), and
    width (w) positions separately. The head_dim is split into sections,
    each receiving position embeddings from a different dimension.

    Args:
        head_dim: Total head dimension.
        max_t: Maximum temporal positions.
        max_h: Maximum height positions.
        max_w: Maximum width positions.
        mrope_section: Section sizes for [t, h, w] dimensions.
            Example: [16, 16, 16] splits head_dim=48 into 3 equal parts.
        theta: Base for frequency computation (default 10000).

    Returns:
        Tuple of (cos_t, sin_t, cos_h, sin_h, cos_w, sin_w).
        Each has shape [max_pos, section_dim] where section_dim is from mrope_section.

    Example:
        >>> cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = precompute_freqs_cis_3d(
        ...     head_dim=48, max_t=1024, max_h=32, max_w=32, mrope_section=[16, 16, 16]
        ... )
    """
    if len(mrope_section) != 3:
        raise ValueError("mrope_section must have exactly 3 elements [t, h, w]")

    section_t, section_h, section_w = mrope_section

    if section_t + section_h + section_w != head_dim // 2:
        raise ValueError(
            f"Sum of mrope_section ({sum(mrope_section)}) must equal head_dim // 2 ({head_dim // 2})"
        )

    def compute_freqs(section_dim: int, max_pos: int) -> tuple[np.ndarray, np.ndarray]:
        freqs = 1.0 / (theta ** (np.arange(0, section_dim, dtype=np.float32) / section_dim))
        positions = np.arange(max_pos, dtype=np.float32)
        angles = np.outer(positions, freqs)
        return np.cos(angles), np.sin(angles)

    cos_t, sin_t = compute_freqs(section_t, max_t)
    cos_h, sin_h = compute_freqs(section_h, max_h)
    cos_w, sin_w = compute_freqs(section_w, max_w)

    return cos_t, sin_t, cos_h, sin_h, cos_w, sin_w


def build_mrope_cos_sin(
    positions_t: np.ndarray,
    positions_h: np.ndarray,
    positions_w: np.ndarray,
    cos_t: np.ndarray,
    sin_t: np.ndarray,
    cos_h: np.ndarray,
    sin_h: np.ndarray,
    cos_w: np.ndarray,
    sin_w: np.ndarray,
    mrope_section: Sequence[int],
    interleaved: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build combined cos/sin tables from 3D position indices.

    Given position indices for each dimension (t, h, w), gathers the
    corresponding cos/sin values and concatenates them.

    Args:
        positions_t: Temporal position indices [seq_len].
        positions_h: Height position indices [seq_len].
        positions_w: Width position indices [seq_len].
        cos_t, sin_t: Precomputed cos/sin for temporal [max_t, section_t].
        cos_h, sin_h: Precomputed cos/sin for height [max_h, section_h].
        cos_w, sin_w: Precomputed cos/sin for width [max_w, section_w].
        mrope_section: Section sizes [t, h, w].
        interleaved: If True, interleave [t0, h0, w0, t1, h1, w1, ...].
                     If False, concatenate [all_t, all_h, all_w].

    Returns:
        Tuple of (cos, sin) each of shape [seq_len, head_dim].
    """
    # Gather cos/sin for each position (seq_len = len(positions_t))
    cos_t_gathered = cos_t[positions_t]  # [seq_len, section_t]
    sin_t_gathered = sin_t[positions_t]
    cos_h_gathered = cos_h[positions_h]  # [seq_len, section_h]
    sin_h_gathered = sin_h[positions_h]
    cos_w_gathered = cos_w[positions_w]  # [seq_len, section_w]
    sin_w_gathered = sin_w[positions_w]

    if interleaved:
        # Interleaved layout: [t0, h0, w0, t1, h1, w1, ...]
        section_t, section_h, section_w = mrope_section
        min_section = min(section_t, section_h, section_w)

        cos_interleaved = []
        sin_interleaved = []
        for i in range(min_section):
            cos_interleaved.append(cos_t_gathered[:, i : i + 1])
            cos_interleaved.append(cos_h_gathered[:, i : i + 1])
            cos_interleaved.append(cos_w_gathered[:, i : i + 1])
            sin_interleaved.append(sin_t_gathered[:, i : i + 1])
            sin_interleaved.append(sin_h_gathered[:, i : i + 1])
            sin_interleaved.append(sin_w_gathered[:, i : i + 1])

        # Handle remaining elements if sections are unequal
        if section_t > min_section:
            cos_interleaved.append(cos_t_gathered[:, min_section:])
            sin_interleaved.append(sin_t_gathered[:, min_section:])
        if section_h > min_section:
            cos_interleaved.append(cos_h_gathered[:, min_section:])
            sin_interleaved.append(sin_h_gathered[:, min_section:])
        if section_w > min_section:
            cos_interleaved.append(cos_w_gathered[:, min_section:])
            sin_interleaved.append(sin_w_gathered[:, min_section:])

        cos_half = np.concatenate(cos_interleaved, axis=-1)
        sin_half = np.concatenate(sin_interleaved, axis=-1)
    else:
        # Sequential layout: [all_t, all_h, all_w]
        cos_half = np.concatenate([cos_t_gathered, cos_h_gathered, cos_w_gathered], axis=-1)
        sin_half = np.concatenate([sin_t_gathered, sin_h_gathered, sin_w_gathered], axis=-1)

    # Duplicate for full head_dim (RoPE applies to pairs)
    cos = np.concatenate([cos_half, cos_half], axis=-1)
    sin = np.concatenate([sin_half, sin_half], axis=-1)

    return cos.astype(np.float32), sin.astype(np.float32)


def apply_multimodal_rotary_pos_emb_numpy(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Multimodal RoPE to Q and K (numpy version).

    Same as standard RoPE application, but with cos/sin built from 3D positions.

    Args:
        q: Query tensor [..., seq_len, n_heads, head_dim] or [seq_len, n_heads, head_dim].
        k: Key tensor [..., seq_len, n_heads, head_dim] or [seq_len, n_heads, head_dim].
        cos: Cosine table [seq_len, head_dim].
        sin: Sine table [seq_len, head_dim].

    Returns:
        Tuple of (q_embed, k_embed) with same shapes as inputs.
    """

    def rotate_half(x: np.ndarray) -> np.ndarray:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return np.concatenate([-x2, x1], axis=-1)

    # Expand cos/sin for broadcasting: [seq_len, 1, head_dim]
    cos = cos[:, np.newaxis, :]
    sin = sin[:, np.newaxis, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def precompute_freqs_cis_text(
    head_dim: int,
    max_seq_len: int,
    mrope_section: Sequence[int],
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE for text-only mode in Multimodal models.

    For text, all 3 dimensions (t, h, w) use the same position index.
    This is equivalent to standard 1D RoPE but with the mrope_section split.

    Args:
        head_dim: Total head dimension.
        max_seq_len: Maximum sequence length.
        mrope_section: Section sizes [t, h, w] (must sum to head_dim // 2).
        theta: Base for frequency computation.

    Returns:
        Tuple of (cos, sin) each of shape [max_seq_len, head_dim].
    """
    # For text, positions are identical across all dimensions
    cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = precompute_freqs_cis_3d(
        head_dim=head_dim,
        max_t=max_seq_len,
        max_h=max_seq_len,
        max_w=max_seq_len,
        mrope_section=mrope_section,
        theta=theta,
    )

    # All positions are the same for text
    positions = np.arange(max_seq_len, dtype=np.int64)

    cos, sin = build_mrope_cos_sin(
        positions_t=positions,
        positions_h=positions,
        positions_w=positions,
        cos_t=cos_t,
        sin_t=sin_t,
        cos_h=cos_h,
        sin_h=sin_h,
        cos_w=cos_w,
        sin_w=sin_w,
        mrope_section=mrope_section,
        interleaved=False,
    )

    return cos, sin


__all__ = [
    # Standard 1D RoPE
    "precompute_freqs_cis",
    "apply_rotary_pos_emb_numpy",
    # Multimodal 3D RoPE (Qwen3-TTS)
    "precompute_freqs_cis_3d",
    "build_mrope_cos_sin",
    "apply_multimodal_rotary_pos_emb_numpy",
    "precompute_freqs_cis_text",
]
