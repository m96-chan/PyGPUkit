"""GPU-native attention modules for FLUX.

Provides joint attention (for double blocks) and single attention mechanisms.
All operations stay on GPU to minimize H2D/D2H transfers.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.diffusion.models.flux.ops import (
    gpu_apply_rope,
    gpu_batched_matmul,
    gpu_concat_axis1,
    gpu_linear,
    gpu_rms_norm,
    gpu_scale,
    gpu_softmax,
    gpu_split_axis1,
    gpu_transpose_0213,
    gpu_transpose_3d_012,
)


def rms_norm(
    x: GPUArray,
    weight: GPUArray | None = None,
    eps: float = 1e-6,
) -> GPUArray:
    """RMS normalization (used per-head in FLUX attention).

    Args:
        x: Input tensor [..., dim].
        weight: Optional learnable scale parameter [dim].
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor [..., dim].
    """
    if weight is not None:
        return gpu_rms_norm(x, weight, eps)
    else:
        # RMS norm without weight - fall back to numpy for now
        x_np = x.to_numpy()
        rms = np.sqrt(np.mean(x_np**2, axis=-1, keepdims=True) + eps)
        normed = x_np / rms
        return from_numpy(normed.astype(np.float32))


def layer_norm(x: GPUArray | np.ndarray, eps: float = 1e-6) -> GPUArray | np.ndarray:
    """Layer normalization (returns same type as input).

    Args:
        x: Input tensor [..., dim].
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor [..., dim].

    Note:
        Uses native CUDA kernel for GPUArray input.
    """
    if isinstance(x, GPUArray):
        from pygpukit.diffusion.models.flux.ops import gpu_layer_norm

        return gpu_layer_norm(x, eps)
    else:
        # numpy input
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)


def joint_attention(
    hidden_states: GPUArray,
    encoder_hidden_states: GPUArray,
    q_weight: GPUArray,
    k_weight: GPUArray,
    v_weight: GPUArray,
    q_bias: GPUArray | None,
    k_bias: GPUArray | None,
    v_bias: GPUArray | None,
    add_q_weight: GPUArray,
    add_k_weight: GPUArray,
    add_v_weight: GPUArray,
    add_q_bias: GPUArray | None,
    add_k_bias: GPUArray | None,
    add_v_bias: GPUArray | None,
    out_weight: GPUArray,
    out_bias: GPUArray | None,
    add_out_weight: GPUArray,
    add_out_bias: GPUArray | None,
    norm_q_weight: GPUArray,
    norm_k_weight: GPUArray,
    norm_added_q_weight: GPUArray,
    norm_added_k_weight: GPUArray,
    rope_cos: np.ndarray | GPUArray,
    rope_sin: np.ndarray | GPUArray,
    num_heads: int = 24,
    head_dim: int = 128,
) -> tuple[GPUArray, GPUArray]:
    """GPU-native joint attention for FLUX double blocks.

    Both image and text tokens attend to each other via concatenated K/V.
    Most operations stay on GPU to minimize transfers.

    Args:
        hidden_states: Image hidden states [B, img_len, D].
        encoder_hidden_states: Text hidden states [B, txt_len, D].
        q/k/v_weight: Image Q/K/V projections [D, D].
        add_q/k/v_weight: Text Q/K/V projections [D, D].
        out_weight: Image output projection [D, D].
        add_out_weight: Text output projection [D, D].
        norm_q/k_weight: RMSNorm weights for image Q/K [head_dim].
        norm_added_q/k_weight: RMSNorm weights for text Q/K [head_dim].
        rope_cos, rope_sin: RoPE frequencies [txt_len + img_len, head_dim].
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Tuple of (image_output, text_output).
    """
    B = hidden_states.shape[0]
    img_len = hidden_states.shape[1]
    txt_len = encoder_hidden_states.shape[1]
    D = hidden_states.shape[2]
    total_len = txt_len + img_len

    # Project image Q, K, V using GPU-native linear
    q_img = gpu_linear(hidden_states, q_weight, q_bias)
    k_img = gpu_linear(hidden_states, k_weight, k_bias)
    v_img = gpu_linear(hidden_states, v_weight, v_bias)

    # Project text Q, K, V
    q_txt = gpu_linear(encoder_hidden_states, add_q_weight, add_q_bias)
    k_txt = gpu_linear(encoder_hidden_states, add_k_weight, add_k_bias)
    v_txt = gpu_linear(encoder_hidden_states, add_v_weight, add_v_bias)

    # Reshape to [B, seq_len, num_heads, head_dim]
    q_img = q_img.reshape(B, img_len, num_heads, head_dim)
    k_img = k_img.reshape(B, img_len, num_heads, head_dim)
    v_img = v_img.reshape(B, img_len, num_heads, head_dim)

    q_txt = q_txt.reshape(B, txt_len, num_heads, head_dim)
    k_txt = k_txt.reshape(B, txt_len, num_heads, head_dim)
    v_txt = v_txt.reshape(B, txt_len, num_heads, head_dim)

    # Apply RMS norm per head with learnable weights
    q_img = gpu_rms_norm(q_img, norm_q_weight)
    k_img = gpu_rms_norm(k_img, norm_k_weight)
    q_txt = gpu_rms_norm(q_txt, norm_added_q_weight)
    k_txt = gpu_rms_norm(k_txt, norm_added_k_weight)

    # Concatenate: [text, image] along seq dimension
    q = gpu_concat_axis1(q_txt, q_img)  # [B, total_len, heads, head_dim]
    k = gpu_concat_axis1(k_txt, k_img)
    v = gpu_concat_axis1(v_txt, v_img)

    # Convert rope to GPUArray if numpy
    if isinstance(rope_cos, np.ndarray):
        rope_cos = from_numpy(rope_cos.astype(np.float32))
    if isinstance(rope_sin, np.ndarray):
        rope_sin = from_numpy(rope_sin.astype(np.float32))

    # Apply RoPE to Q and K
    q = gpu_apply_rope(q, rope_cos, rope_sin)
    k = gpu_apply_rope(k, rope_cos, rope_sin)

    # Transpose for attention: [B, seq_len, heads, head_dim] -> [B, heads, seq_len, head_dim]
    q = gpu_transpose_0213(q)
    k = gpu_transpose_0213(k)
    v = gpu_transpose_0213(v)

    # Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
    scale = 1.0 / np.sqrt(head_dim)

    # Reshape for batched matmul: [B*num_heads, seq_len, head_dim]
    q_flat = q.reshape(B * num_heads, total_len, head_dim)
    k_flat = k.reshape(B * num_heads, total_len, head_dim)
    v_flat = v.reshape(B * num_heads, total_len, head_dim)

    # Q @ K^T: [B*heads, seq, dim] @ [B*heads, dim, seq] -> [B*heads, seq, seq]
    # Use GPU-native transpose for K^T (no H2D/D2H transfer)
    k_t = gpu_transpose_3d_012(k_flat)  # [B*heads, seq, dim] -> [B*heads, dim, seq]
    scores = gpu_batched_matmul(q_flat, k_t)
    scores = gpu_scale(scores, scale)

    # Softmax over last axis
    attn_weights = gpu_softmax(scores, axis=-1)

    # Attention @ V: [B*heads, seq, seq] @ [B*heads, seq, dim] -> [B*heads, seq, dim]
    attn_out = gpu_batched_matmul(attn_weights, v_flat)

    # Reshape back: [B, heads, total_len, head_dim] -> [B, total_len, D]
    attn_out = attn_out.reshape(B, num_heads, total_len, head_dim)
    attn_out = gpu_transpose_0213(attn_out)  # [B, total_len, heads, head_dim]
    attn_out = attn_out.reshape(B, total_len, D)

    # Split back to text and image
    txt_out, img_out = gpu_split_axis1(attn_out, txt_len)

    # Output projections
    img_final = gpu_linear(img_out, out_weight, out_bias)
    txt_final = gpu_linear(txt_out, add_out_weight, add_out_bias)

    return img_final, txt_final


def single_attention(
    hidden_states: GPUArray,
    q_weight: GPUArray,
    k_weight: GPUArray,
    v_weight: GPUArray,
    q_bias: GPUArray | None,
    k_bias: GPUArray | None,
    v_bias: GPUArray | None,
    norm_q_weight: GPUArray,
    norm_k_weight: GPUArray,
    rope_cos: np.ndarray | GPUArray,
    rope_sin: np.ndarray | GPUArray,
    num_heads: int = 24,
    head_dim: int = 128,
) -> GPUArray:
    """GPU-native single self-attention for FLUX single blocks.

    Operates on concatenated [text, image] sequence.

    Args:
        hidden_states: Concatenated hidden states [B, total_len, D].
        q/k/v_weight: Q/K/V projections [D, D].
        norm_q/k_weight: RMSNorm weights for Q/K [head_dim].
        rope_cos, rope_sin: RoPE frequencies [total_len, head_dim].
        num_heads: Number of attention heads.
        head_dim: Dimension per head.

    Returns:
        Attention output [B, total_len, D] (no output projection in single blocks).
    """
    B = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    D = hidden_states.shape[2]

    # Project Q, K, V using GPU-native linear
    q = gpu_linear(hidden_states, q_weight, q_bias)
    k = gpu_linear(hidden_states, k_weight, k_bias)
    v = gpu_linear(hidden_states, v_weight, v_bias)

    # Reshape to [B, seq_len, num_heads, head_dim]
    q = q.reshape(B, seq_len, num_heads, head_dim)
    k = k.reshape(B, seq_len, num_heads, head_dim)
    v = v.reshape(B, seq_len, num_heads, head_dim)

    # Apply RMS norm per head with learnable weights
    q = gpu_rms_norm(q, norm_q_weight)
    k = gpu_rms_norm(k, norm_k_weight)

    # Convert rope to GPUArray if numpy
    if isinstance(rope_cos, np.ndarray):
        rope_cos = from_numpy(rope_cos.astype(np.float32))
    if isinstance(rope_sin, np.ndarray):
        rope_sin = from_numpy(rope_sin.astype(np.float32))

    # Apply RoPE
    q = gpu_apply_rope(q, rope_cos, rope_sin)
    k = gpu_apply_rope(k, rope_cos, rope_sin)

    # Transpose for attention: [B, seq_len, heads, head_dim] -> [B, heads, seq_len, head_dim]
    q = gpu_transpose_0213(q)
    k = gpu_transpose_0213(k)
    v = gpu_transpose_0213(v)

    # Compute attention
    scale = 1.0 / np.sqrt(head_dim)

    # Reshape for batched matmul: [B*num_heads, seq_len, head_dim]
    q_flat = q.reshape(B * num_heads, seq_len, head_dim)
    k_flat = k.reshape(B * num_heads, seq_len, head_dim)
    v_flat = v.reshape(B * num_heads, seq_len, head_dim)

    # Q @ K^T - Use GPU-native transpose (no H2D/D2H transfer)
    k_t = gpu_transpose_3d_012(k_flat)  # [B*heads, seq, dim] -> [B*heads, dim, seq]
    scores = gpu_batched_matmul(q_flat, k_t)
    scores = gpu_scale(scores, scale)

    # Softmax
    attn_weights = gpu_softmax(scores, axis=-1)

    # Attention @ V
    attn_out = gpu_batched_matmul(attn_weights, v_flat)

    # Reshape back: [B, seq_len, D]
    attn_out = attn_out.reshape(B, num_heads, seq_len, head_dim)
    attn_out = gpu_transpose_0213(attn_out)  # [B, seq_len, heads, head_dim]
    attn_out = attn_out.reshape(B, seq_len, D)

    return attn_out


__all__ = [
    "rms_norm",
    "layer_norm",
    "joint_attention",
    "single_attention",
]
