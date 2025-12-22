"""Embedding and KV cache operations for GPUArrays.

Corresponds to native/ops/embedding/.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray

# =============================================================================
# Embedding Lookup Operations
# =============================================================================


def embedding_lookup(embed_matrix: GPUArray, out: GPUArray, token_id: int) -> None:
    """Lookup embedding on GPU without CPU transfer.

    For CUDA Graph: no allocation, no CPU->GPU transfer.

    Args:
        embed_matrix: Embedding matrix [vocab_size, hidden_size].
        out: Pre-allocated output buffer [1, hidden_size].
        token_id: Token index to lookup.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    embed_native = embed_matrix._get_native()
    out_native = out._get_native()
    native.embedding_lookup(embed_native, out_native, token_id)


def embedding_lookup_ptr(embed_matrix: GPUArray, out: GPUArray, token_id_buf: GPUArray) -> None:
    """Lookup embedding reading index from GPU buffer.

    For CUDA Graph replay: index is read from GPU memory, allowing
    graph replay with different indices without recapturing.

    Args:
        embed_matrix: Embedding matrix [vocab_size, hidden_size].
        out: Pre-allocated output buffer [1, hidden_size].
        token_id_buf: GPUArray[1] int32 containing token/position value.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    embed_native = embed_matrix._get_native()
    out_native = out._get_native()
    token_id_buf_native = token_id_buf._get_native()
    native.embedding_lookup_ptr(embed_native, out_native, token_id_buf_native)


def embedding_lookup_batch(
    embed_matrix: GPUArray,
    out: GPUArray,
    token_ids_buf: GPUArray,
    batch_size: int,
) -> None:
    """Batch embedding lookup from GPU token ID array.

    For CUDA Graph batch decode: looks up multiple tokens at once.
    out[i, :] = embed_matrix[token_ids[i], :]

    Args:
        embed_matrix: Embedding matrix [vocab_size, hidden_size]
        out: Output buffer [batch_size, hidden_size] (pre-allocated)
        token_ids_buf: GPU buffer containing token IDs [max_batch_size] int32
        batch_size: Number of tokens to look up (actual batch size)
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    embed_native = embed_matrix._get_native()
    out_native = out._get_native()
    token_ids_buf_native = token_ids_buf._get_native()
    native.embedding_lookup_batch(embed_native, out_native, token_ids_buf_native, batch_size)


# =============================================================================
# KV Cache Operations
# =============================================================================


def kv_cache_update(new_kv: GPUArray, cache: GPUArray, position: int) -> None:
    """Update KV cache at a single position (decode step).

    Used for fixed-length KV cache with CUDA Graph support.
    Copies new K or V values to a specific position in the pre-allocated cache.

    Args:
        new_kv: New K or V tensor of shape [1, num_kv_heads, head_dim].
        cache: Pre-allocated cache tensor of shape [max_seq_len, num_kv_heads, head_dim].
        position: Position index in cache where to write (0-indexed).

    Raises:
        ValueError: If shapes are incompatible.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    new_kv_native = new_kv._get_native()
    cache_native = cache._get_native()
    native.kv_cache_update(new_kv_native, cache_native, position)


def kv_cache_prefill(new_kv: GPUArray, cache: GPUArray, start_pos: int = 0) -> None:
    """Prefill KV cache from sequence (prefill step).

    Used for fixed-length KV cache with CUDA Graph support.
    Copies K or V values from prefill to the pre-allocated cache.

    Args:
        new_kv: K or V tensor from prefill of shape [seq_len, num_kv_heads, head_dim].
        cache: Pre-allocated cache tensor of shape [max_seq_len, num_kv_heads, head_dim].
        start_pos: Starting position in cache (default 0).

    Raises:
        ValueError: If shapes are incompatible.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    new_kv_native = new_kv._get_native()
    cache_native = cache._get_native()
    native.kv_cache_prefill(new_kv_native, cache_native, start_pos)


def kv_cache_update_gqa(new_kv: GPUArray, cache: GPUArray, num_heads: int, position: int) -> None:
    """Update GQA-expanded KV cache at a single position (decode step).

    For CUDA Graph optimization: writes to transposed, GQA-expanded cache.
    Eliminates per-step transpose and GQA expansion overhead.

    Args:
        new_kv: K or V tensor of shape [1, num_kv_heads, head_dim].
        cache: Pre-allocated cache of shape [num_heads, max_seq_len, head_dim].
        num_heads: Total number of attention heads.
        position: Position in cache to update.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    new_kv_native = new_kv._get_native()
    cache_native = cache._get_native()
    native.kv_cache_update_gqa(new_kv_native, cache_native, num_heads, position)


def kv_cache_prefill_gqa(
    new_kv: GPUArray, cache: GPUArray, num_heads: int, start_pos: int = 0
) -> None:
    """Prefill GQA-expanded KV cache from sequence.

    For CUDA Graph optimization: writes to transposed, GQA-expanded cache.
    Eliminates per-step transpose and GQA expansion overhead.

    Args:
        new_kv: K or V tensor of shape [seq_len, num_kv_heads, head_dim].
        cache: Pre-allocated cache of shape [num_heads, max_seq_len, head_dim].
        num_heads: Total number of attention heads.
        start_pos: Starting position in cache (default 0).
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    new_kv_native = new_kv._get_native()
    cache_native = cache._get_native()
    native.kv_cache_prefill_gqa(new_kv_native, cache_native, num_heads, start_pos)


def kv_cache_update_gqa_ptr(
    new_kv: GPUArray, cache: GPUArray, num_heads: int, position_buf: GPUArray
) -> None:
    """Update GQA-expanded KV cache reading position from GPU buffer.

    For CUDA Graph replay: position is read from GPU memory, allowing
    graph replay with different positions without recapturing.

    Args:
        new_kv: K or V tensor of shape [1, num_kv_heads, head_dim].
        cache: Pre-allocated cache of shape [num_heads, max_seq_len, head_dim].
        num_heads: Total number of attention heads.
        position_buf: GPUArray[1] int32 containing position value.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    new_kv_native = new_kv._get_native()
    cache_native = cache._get_native()
    position_buf_native = position_buf._get_native()
    native.kv_cache_update_gqa_ptr(new_kv_native, cache_native, num_heads, position_buf_native)
