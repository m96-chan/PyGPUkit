"""Embedding modules for DiT.

Provides patch embedding, timestep embedding, and caption projection.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.ops.matmul.generic import matmul


def sinusoidal_embedding(positions: np.ndarray, dim: int) -> np.ndarray:
    """Sinusoidal positional embedding.

    Args:
        positions: Position indices [N] or values [B].
        dim: Embedding dimension.

    Returns:
        Embeddings [N, dim] or [B, dim].
    """
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim == 0:
        positions = positions.reshape(1)

    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
    emb = positions[:, None] * emb[None, :]

    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)

    if dim % 2 == 1:
        emb = np.pad(emb, ((0, 0), (0, 1)))

    return emb.astype(np.float32)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int | tuple[int, int]) -> np.ndarray:
    """2D sinusoidal position embeddings for a grid of patches.

    Matches diffusers PatchEmbed implementation:
    - Patches are in column-major order (h varies first, then w)
    - Embeddings are concatenated as [height_embed, width_embed]

    Args:
        embed_dim: Embedding dimension.
        grid_size: Grid size (H, W) or single int for square grid.

    Returns:
        Position embeddings [H*W, embed_dim].
    """
    if isinstance(grid_size, int):
        grid_h, grid_w = grid_size, grid_size
    else:
        grid_h, grid_w = grid_size

    # Create position arrays
    grid_h_pos = np.arange(grid_h, dtype=np.float32)
    grid_w_pos = np.arange(grid_w, dtype=np.float32)

    # Create 2D grid in column-major order (h varies first)
    # This matches diffusers: for each column, iterate through rows
    h_grid, w_grid = np.meshgrid(grid_h_pos, grid_w_pos, indexing="ij")
    # Flatten in Fortran order (column-major) to match diffusers patch ordering
    h_flat = h_grid.flatten("F")  # [H*W]
    w_flat = w_grid.flatten("F")  # [H*W]

    # Get embeddings for each dimension
    emb_h = sinusoidal_embedding(h_flat, embed_dim // 2)  # height embedding
    emb_w = sinusoidal_embedding(w_flat, embed_dim // 2)  # width embedding

    # Concatenate: [height_embed, width_embed]
    pos_embed = np.concatenate([emb_h, emb_w], axis=-1)  # [H*W, embed_dim]

    return pos_embed.astype(np.float32)


def patch_embed(
    x: GPUArray,
    proj_weight: GPUArray,
    proj_bias: GPUArray | None,
    patch_size: int = 2,
) -> GPUArray:
    """Patch embedding via convolution-like projection.

    PixArt structure:
        pos_embed.proj.weight: [D, C, patch_size, patch_size]
        pos_embed.proj.bias: [D]

    Args:
        x: Input image [B, C, H, W].
        proj_weight: Projection weight [D, C, patch_size, patch_size].
        proj_bias: Projection bias [D].
        patch_size: Size of each patch.

    Returns:
        Patch embeddings [B, num_patches, D].
    """
    x_np = x.to_numpy()
    w_np = proj_weight.to_numpy()

    B, C, H, W = x_np.shape
    D = w_np.shape[0]

    h_patches = H // patch_size
    w_patches = W // patch_size
    num_patches = h_patches * w_patches

    # Reshape image to patches [B, num_patches, C * patch_size * patch_size]
    x_patches = x_np.reshape(B, C, h_patches, patch_size, w_patches, patch_size)
    x_patches = x_patches.transpose(0, 2, 4, 1, 3, 5)  # [B, h, w, C, p, p]
    x_patches = x_patches.reshape(B, num_patches, C * patch_size * patch_size)

    # Reshape weight to 2D: [D, C * patch_size * patch_size]
    w_2d = w_np.reshape(D, -1).T.astype(np.float32)  # [C*p*p, D]

    # Project patches
    x_2d = x_patches.reshape(B * num_patches, -1).astype(np.float32)
    output = matmul(from_numpy(x_2d), from_numpy(w_2d)).to_numpy()

    if proj_bias is not None:
        output = output + proj_bias.to_numpy()

    return from_numpy(output.reshape(B, num_patches, D).astype(np.float32))


def timestep_embedding(
    timestep: float | np.ndarray,
    dim: int,
    linear1_weight: GPUArray | None = None,
    linear1_bias: GPUArray | None = None,
    linear2_weight: GPUArray | None = None,
    linear2_bias: GPUArray | None = None,
    batch_size: int = 1,
) -> GPUArray:
    """Timestep embedding with optional MLP.

    PixArt structure:
        adaln_single.emb.timestep_embedder.linear_1: [D, 256]
        adaln_single.emb.timestep_embedder.linear_2: [D, D]

    Args:
        timestep: Timestep value(s).
        dim: Embedding dimension.
        linear1_weight, linear1_bias: First MLP layer.
        linear2_weight, linear2_bias: Second MLP layer.
        batch_size: Batch size for scalar timestep.

    Returns:
        Timestep embedding [B, D].
    """
    if isinstance(timestep, (int, float)):
        t = np.array([timestep] * batch_size, dtype=np.float32)
    else:
        t = np.asarray(timestep, dtype=np.float32)

    # Initial sinusoidal embedding
    if linear1_weight is not None:
        # Use 256-dim embedding for MLP input
        t_emb = sinusoidal_embedding(t, 256)
    else:
        t_emb = sinusoidal_embedding(t, dim)

    # Apply MLP if weights available
    if linear1_weight is not None:
        w1 = linear1_weight.to_numpy().T.astype(np.float32)
        t_emb = matmul(from_numpy(t_emb), from_numpy(w1)).to_numpy()
        if linear1_bias is not None:
            t_emb = t_emb + linear1_bias.to_numpy()
        # SiLU activation
        t_emb = t_emb * (1.0 / (1.0 + np.exp(-t_emb)))

        if linear2_weight is not None:
            w2 = linear2_weight.to_numpy().T.astype(np.float32)
            t_emb = matmul(from_numpy(t_emb.astype(np.float32)), from_numpy(w2)).to_numpy()
            if linear2_bias is not None:
                t_emb = t_emb + linear2_bias.to_numpy()

    return from_numpy(t_emb.astype(np.float32))


def caption_projection(
    text_embeds: GPUArray,
    linear1_weight: GPUArray,
    linear1_bias: GPUArray | None,
    linear2_weight: GPUArray,
    linear2_bias: GPUArray | None,
) -> GPUArray:
    """Project text embeddings to model dimension.

    PixArt structure:
        caption_projection.linear_1: [D, text_dim]
        caption_projection.linear_2: [D, D]

    Args:
        text_embeds: Text embeddings [B, seq_len, text_dim].
        linear1_weight, linear1_bias: First projection layer.
        linear2_weight, linear2_bias: Second projection layer.

    Returns:
        Projected embeddings [B, seq_len, D].
    """
    x_np = text_embeds.to_numpy()
    B, N, text_dim = x_np.shape

    # First projection
    w1 = linear1_weight.to_numpy().T.astype(np.float32)
    x_2d: np.ndarray = x_np.reshape(B * N, text_dim).astype(np.float32)
    x_proj = matmul(from_numpy(x_2d), from_numpy(w1)).to_numpy()
    if linear1_bias is not None:
        x_proj = x_proj + linear1_bias.to_numpy()

    # SiLU activation
    x_proj = x_proj * (1.0 / (1.0 + np.exp(-x_proj)))

    # Second projection
    w2 = linear2_weight.to_numpy().T.astype(np.float32)
    D = w2.shape[-1]
    x_out = matmul(from_numpy(x_proj.astype(np.float32)), from_numpy(w2)).to_numpy()
    if linear2_bias is not None:
        x_out = x_out + linear2_bias.to_numpy()

    return from_numpy(x_out.reshape(B, N, D).astype(np.float32))


def unpatchify(
    x: GPUArray,
    H: int,
    W: int,
    out_channels: int,
    patch_size: int,
    proj_weight: GPUArray,
    proj_bias: GPUArray | None,
) -> GPUArray:
    """Convert patch tokens back to image.

    Args:
        x: Patch tokens [B, num_patches, D].
        H, W: Original image height/width.
        out_channels: Number of output channels.
        patch_size: Patch size.
        proj_weight: Output projection [out_dim, D].
        proj_bias: Output bias [out_dim].

    Returns:
        Output image [B, out_channels, H, W].
    """
    x_np = x.to_numpy()
    B, num_patches, D = x_np.shape

    h_patches = H // patch_size
    w_patches = W // patch_size

    # Project to output dimension
    w = proj_weight.to_numpy().T.astype(np.float32)  # [D, out_dim]
    x_2d: np.ndarray = x_np.reshape(B * num_patches, D).astype(np.float32)
    output = matmul(from_numpy(x_2d), from_numpy(w)).to_numpy()

    if proj_bias is not None:
        output = output + proj_bias.to_numpy()

    # Reshape to image
    # proj_out outputs [num_patches, C*p*p] where the order is [p, p, C] (row-major)
    # So reshape to [B, h, w, p, p, C] then transpose to [B, C, h, p, w, p]
    output = output.reshape(B, h_patches, w_patches, patch_size, patch_size, out_channels)
    output = output.transpose(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
    output = output.reshape(B, out_channels, H, W)

    return from_numpy(output.astype(np.float32))


__all__ = [
    "sinusoidal_embedding",
    "patch_embed",
    "timestep_embedding",
    "caption_projection",
    "unpatchify",
]
