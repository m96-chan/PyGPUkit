"""Timestep embedding for diffusion models.

Provides sinusoidal positional embeddings for timesteps,
following the Transformer/DDPM convention.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy


def sinusoidal_timestep_embedding(
    timesteps: GPUArray | np.ndarray,
    embedding_dim: int,
    max_period: float = 10000.0,
    dtype: str = "float32",
) -> GPUArray:
    """Sinusoidal timestep embedding.

    Creates positional embeddings for timesteps using sine and cosine functions
    at different frequencies, following the Transformer convention.

    Args:
        timesteps: Timestep values [B] (can be float or int).
        embedding_dim: Dimension of the embedding.
        max_period: Maximum period for the sinusoidal functions.
        dtype: Output dtype.

    Returns:
        Embeddings of shape [B, embedding_dim].
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _sinusoidal_embedding_native(timesteps, embedding_dim, max_period, dtype)
    else:
        return _sinusoidal_embedding_cpu(timesteps, embedding_dim, max_period, dtype)


def _sinusoidal_embedding_cpu(
    timesteps: GPUArray | np.ndarray,
    embedding_dim: int,
    max_period: float,
    dtype: str,
) -> GPUArray:
    """CPU implementation of sinusoidal timestep embedding."""
    t: np.ndarray
    if isinstance(timesteps, GPUArray):
        t = timesteps.to_numpy().astype(np.float32)
    else:
        t = np.asarray(timesteps, dtype=np.float32)

    if t.ndim == 0:
        t = t.reshape(1)

    batch_size = t.shape[0]
    half_dim = embedding_dim // 2

    # Compute frequencies
    freqs = np.exp(-np.log(max_period) * np.arange(half_dim, dtype=np.float32) / half_dim)

    # Compute arguments: [B, half_dim]
    args = t[:, np.newaxis] * freqs[np.newaxis, :]

    # Compute sin and cos embeddings
    emb_sin = np.sin(args)
    emb_cos = np.cos(args)

    # Interleave sin and cos
    embedding = np.zeros((batch_size, embedding_dim), dtype=np.float32)
    embedding[:, 0::2] = emb_sin
    embedding[:, 1::2] = emb_cos

    # Handle odd embedding_dim
    if embedding_dim % 2 == 1:
        embedding = np.concatenate([embedding, np.zeros((batch_size, 1))], axis=1)

    if dtype == "float16":
        embedding = embedding.astype(np.float16)
    elif dtype == "bfloat16":
        # NumPy doesn't support bfloat16, keep as float32
        pass

    return from_numpy(embedding)


def _sinusoidal_embedding_native(
    timesteps: GPUArray | np.ndarray,
    embedding_dim: int,
    max_period: float,
    dtype: str,
) -> GPUArray:
    """Native CUDA implementation of sinusoidal embedding."""
    # TODO: Implement native CUDA kernel
    return _sinusoidal_embedding_cpu(timesteps, embedding_dim, max_period, dtype)


def timestep_mlp(
    timestep_embedding: GPUArray,
    fc1_weight: GPUArray,
    fc1_bias: GPUArray,
    fc2_weight: GPUArray,
    fc2_bias: GPUArray,
) -> GPUArray:
    """MLP for processing timestep embeddings.

    Common pattern: Linear -> SiLU -> Linear

    Args:
        timestep_embedding: Input embeddings [B, D].
        fc1_weight: First linear weight [hidden_dim, D].
        fc1_bias: First linear bias [hidden_dim].
        fc2_weight: Second linear weight [out_dim, hidden_dim].
        fc2_bias: Second linear bias [out_dim].

    Returns:
        Processed embeddings [B, out_dim].
    """
    from pygpukit.ops.nn import silu

    # Linear 1
    x = timestep_embedding.to_numpy()
    w1 = fc1_weight.to_numpy()
    b1 = fc1_bias.to_numpy()
    h = np.dot(x, w1.T) + b1

    # SiLU
    h_gpu = from_numpy(h.astype(x.dtype))
    h_silu = silu(h_gpu)

    # Linear 2
    h2 = h_silu.to_numpy()
    w2 = fc2_weight.to_numpy()
    b2 = fc2_bias.to_numpy()
    out = np.dot(h2, w2.T) + b2

    return from_numpy(out.astype(x.dtype))


__all__ = [
    "sinusoidal_timestep_embedding",
    "timestep_mlp",
]
