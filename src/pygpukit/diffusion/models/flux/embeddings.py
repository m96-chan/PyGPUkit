"""Embedding modules for FLUX.

Provides RoPE position embeddings and timestep/text embeddings.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy


def get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D rotary position embedding frequencies.

    Args:
        dim: Embedding dimension (will use dim/2 frequencies).
        pos: Position indices [seq_len].
        theta: Base frequency.

    Returns:
        Tuple of (cos, sin) each [seq_len, dim].
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

    # Outer product: [seq_len] x [dim/2] -> [seq_len, dim/2]
    freqs = np.outer(pos.astype(np.float32), inv_freq)

    # Compute cos and sin
    freqs_cos = np.cos(freqs)  # [seq_len, dim/2]
    freqs_sin = np.sin(freqs)  # [seq_len, dim/2]

    # Repeat interleave to full dimension: [a,b,c] -> [a,a,b,b,c,c]
    freqs_cos = np.repeat(freqs_cos, 2, axis=1)  # [seq_len, dim]
    freqs_sin = np.repeat(freqs_sin, 2, axis=1)  # [seq_len, dim]

    return freqs_cos, freqs_sin


def get_rope_frequencies(
    img_ids: np.ndarray,
    txt_ids: np.ndarray,
    axes_dim: tuple[int, int, int] = (16, 56, 56),
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RoPE frequencies for FLUX.

    FLUX uses 3D position encoding: (text_idx, img_height, img_width).
    The axes_dim specifies the dimension allocated to each axis.

    Args:
        img_ids: Image position IDs [img_seq_len, 3].
        txt_ids: Text position IDs [txt_seq_len, 3].
        axes_dim: Dimensions for each axis (16, 56, 56) = 128 total.
        theta: Base frequency.

    Returns:
        Tuple of (cos, sin) each [txt_seq_len + img_seq_len, sum(axes_dim)].
    """
    # Concatenate text and image IDs
    ids = np.concatenate([txt_ids, img_ids], axis=0)  # [total_seq, 3]

    all_cos = []
    all_sin = []

    for i, dim in enumerate(axes_dim):
        cos_i, sin_i = get_1d_rotary_pos_embed(dim, ids[:, i], theta)
        all_cos.append(cos_i)
        all_sin.append(sin_i)

    # Concatenate along embedding dimension
    freqs_cos = np.concatenate(all_cos, axis=1)  # [seq_len, 128]
    freqs_sin = np.concatenate(all_sin, axis=1)

    return freqs_cos.astype(np.float32), freqs_sin.astype(np.float32)


def apply_rope(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """Apply rotary position embedding to Q or K.

    Args:
        x: Input tensor [B, seq_len, num_heads, head_dim].
        cos: Cosine frequencies [seq_len, head_dim].
        sin: Sine frequencies [seq_len, head_dim].

    Returns:
        Rotated tensor [B, seq_len, num_heads, head_dim].
    """
    # Reshape cos/sin for broadcasting: [1, seq_len, 1, head_dim]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    # Split into pairs and rotate
    # x = [x0, x1, x2, x3, ...] -> rotate pairs
    # x_rot = [-x1, x0, -x3, x2, ...]
    x_rot = np.empty_like(x)
    x_rot[..., 0::2] = -x[..., 1::2]
    x_rot[..., 1::2] = x[..., 0::2]

    # Apply rotation: x * cos + x_rot * sin
    return x * cos + x_rot * sin


def timestep_embedding(
    timestep: np.ndarray,
    dim: int = 256,
    max_period: float = 10000.0,
) -> np.ndarray:
    """Sinusoidal timestep embedding.

    Args:
        timestep: Timestep values [B].
        dim: Embedding dimension.
        max_period: Maximum period for frequencies.

    Returns:
        Timestep embeddings [B, dim].
    """
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
    args = timestep[:, None].astype(np.float32) * freqs[None, :]
    embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)

    if dim % 2 == 1:
        embedding = np.concatenate([embedding, np.zeros_like(embedding[:, :1])], axis=-1)

    return embedding.astype(np.float32)


def combined_timestep_text_embedding(
    timestep: np.ndarray,
    pooled_text: GPUArray,
    time_proj_weight: GPUArray,
    time_proj_bias: GPUArray | None,
    text_proj_weight: GPUArray,
    text_proj_bias: GPUArray | None,
    out_proj_weight: GPUArray,
    out_proj_bias: GPUArray | None,
    embedding_dim: int = 3072,
) -> GPUArray:
    """Combined timestep and pooled text embedding for FLUX.

    Structure:
        timestep -> sinusoidal(256) -> Linear(time_embed_dim) -> SiLU -> Linear(embedding_dim)
        pooled_text -> Linear(embedding_dim)
        combined = timestep_embed + text_embed

    Args:
        timestep: Timestep values [B].
        pooled_text: Pooled text embedding [B, pooled_dim].
        time_proj_weight, time_proj_bias: Time projection.
        text_proj_weight, text_proj_bias: Text projection.
        out_proj_weight, out_proj_bias: Output projection.
        embedding_dim: Output embedding dimension.

    Returns:
        Combined embedding [B, embedding_dim].
    """
    timestep.shape[0]

    # Timestep embedding: sinusoidal -> Linear -> SiLU -> Linear
    t_emb = timestep_embedding(timestep, dim=256)  # [B, 256]

    # First projection
    t_proj_w = time_proj_weight.to_numpy().T.astype(np.float32)
    t_emb = t_emb @ t_proj_w
    if time_proj_bias is not None:
        t_emb = t_emb + time_proj_bias.to_numpy()

    # SiLU activation
    t_emb = t_emb * (1.0 / (1.0 + np.exp(-t_emb)))

    # Output projection
    out_w = out_proj_weight.to_numpy().T.astype(np.float32)
    t_emb = t_emb @ out_w
    if out_proj_bias is not None:
        t_emb = t_emb + out_proj_bias.to_numpy()

    # Text embedding projection
    pooled_np = pooled_text.to_numpy()
    text_w = text_proj_weight.to_numpy().T.astype(np.float32)
    text_emb = pooled_np @ text_w
    if text_proj_bias is not None:
        text_emb = text_emb + text_proj_bias.to_numpy()

    # Combine
    combined = t_emb + text_emb

    return from_numpy(combined.astype(np.float32))


def prepare_image_ids(
    batch_size: int,
    height: int,
    width: int,
    patch_size: int = 1,
) -> np.ndarray:
    """Prepare image position IDs for RoPE.

    Args:
        batch_size: Batch size.
        height: Latent height (after VAE encoding).
        width: Latent width.
        patch_size: Patch size (1 for FLUX).

    Returns:
        Image IDs [batch_size, h*w, 3] with (0, row, col) format.
    """
    h = height // patch_size
    w = width // patch_size

    # Create grid
    rows = np.arange(h)
    cols = np.arange(w)
    row_ids, col_ids = np.meshgrid(rows, cols, indexing="ij")

    # Flatten: [h, w] -> [h*w]
    row_ids = row_ids.flatten()
    col_ids = col_ids.flatten()

    # Stack: [h*w, 3] with (text_idx=0, row, col)
    img_ids = np.stack(
        [
            np.zeros_like(row_ids),  # text dimension (0 for images)
            row_ids,
            col_ids,
        ],
        axis=-1,
    )

    # Expand for batch: [B, h*w, 3]
    img_ids = np.tile(img_ids[None, :, :], (batch_size, 1, 1))

    return img_ids.astype(np.float32)


def prepare_text_ids(
    batch_size: int,
    seq_len: int,
) -> np.ndarray:
    """Prepare text position IDs for RoPE.

    Args:
        batch_size: Batch size.
        seq_len: Text sequence length.

    Returns:
        Text IDs [batch_size, seq_len, 3] with (idx, 0, 0) format.
    """
    # Text uses only the first dimension
    text_ids = np.stack(
        [
            np.arange(seq_len),  # text index
            np.zeros(seq_len),  # row = 0
            np.zeros(seq_len),  # col = 0
        ],
        axis=-1,
    )

    # Expand for batch
    text_ids = np.tile(text_ids[None, :, :], (batch_size, 1, 1))

    return text_ids.astype(np.float32)


__all__ = [
    "get_1d_rotary_pos_embed",
    "get_rope_frequencies",
    "apply_rope",
    "timestep_embedding",
    "combined_timestep_text_embedding",
    "prepare_image_ids",
    "prepare_text_ids",
]
