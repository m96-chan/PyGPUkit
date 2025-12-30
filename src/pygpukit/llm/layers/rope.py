"""Rotary Position Embedding (RoPE) utilities for PyGPUkit LLM.

Provides:
- precompute_freqs_cis: Precompute RoPE cos/sin tables
- apply_rotary_pos_emb_numpy: Apply RoPE on CPU (numpy)
"""

from __future__ import annotations

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


__all__ = [
    "precompute_freqs_cis",
    "apply_rotary_pos_emb_numpy",
]
