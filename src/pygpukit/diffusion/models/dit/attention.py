"""Attention modules for DiT.

Provides Self-Attention and Cross-Attention with GPU matmul.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.ops.matmul.generic import batched_matmul, matmul


def self_attention(
    x: GPUArray,
    q_weight: GPUArray,
    k_weight: GPUArray,
    v_weight: GPUArray,
    out_weight: GPUArray,
    q_bias: GPUArray | None = None,
    k_bias: GPUArray | None = None,
    v_bias: GPUArray | None = None,
    out_bias: GPUArray | None = None,
    num_heads: int = 16,
) -> GPUArray:
    """Self-attention with GPU matmul.

    Args:
        x: Input tensor [B, N, D].
        q_weight, k_weight, v_weight: QKV projection weights [D, D].
        out_weight: Output projection weight [D, D].
        q_bias, k_bias, v_bias, out_bias: Optional biases [D].
        num_heads: Number of attention heads.

    Returns:
        Attention output [B, N, D].
    """
    x_np = x.to_numpy()
    B, N, D = x_np.shape
    head_dim = D // num_heads

    # Project Q, K, V
    x_2d = from_numpy(x_np.reshape(B * N, D).astype(np.float32))

    q_w = q_weight.to_numpy().T.astype(np.float32)
    k_w = k_weight.to_numpy().T.astype(np.float32)
    v_w = v_weight.to_numpy().T.astype(np.float32)

    q = matmul(x_2d, from_numpy(q_w)).to_numpy()
    k = matmul(x_2d, from_numpy(k_w)).to_numpy()
    v = matmul(x_2d, from_numpy(v_w)).to_numpy()

    # Add biases
    if q_bias is not None:
        q = q + q_bias.to_numpy()
    if k_bias is not None:
        k = k + k_bias.to_numpy()
    if v_bias is not None:
        v = v + v_bias.to_numpy()

    # Reshape to [B, num_heads, N, head_dim]
    q = q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Attention scores: [B*H, N, head_dim] @ [B*H, head_dim, N] -> [B*H, N, N]
    scale = 1.0 / np.sqrt(head_dim)
    q_flat = q.reshape(B * num_heads, N, head_dim)
    k_flat = k.reshape(B * num_heads, N, head_dim)
    v_flat = v.reshape(B * num_heads, N, head_dim)

    q_gpu = from_numpy(q_flat.astype(np.float32))
    k_t_gpu = from_numpy(k_flat.transpose(0, 2, 1).astype(np.float32))
    scores = batched_matmul(q_gpu, k_t_gpu).to_numpy() * scale

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

    # Attention output: [B*H, N, N] @ [B*H, N, head_dim] -> [B*H, N, head_dim]
    attn_gpu = from_numpy(attn_weights.astype(np.float32))
    v_gpu = from_numpy(v_flat.astype(np.float32))
    attn_out = batched_matmul(attn_gpu, v_gpu).to_numpy()

    # Reshape back: [B, N, D]
    attn_out = attn_out.reshape(B, num_heads, N, head_dim)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B * N, D)

    # Output projection
    out_w = out_weight.to_numpy().T.astype(np.float32)
    output = matmul(from_numpy(attn_out.astype(np.float32)), from_numpy(out_w)).to_numpy()
    if out_bias is not None:
        output = output + out_bias.to_numpy()

    return from_numpy(output.reshape(B, N, D).astype(np.float32))


def cross_attention(
    x: GPUArray,
    context: GPUArray,
    q_weight: GPUArray,
    k_weight: GPUArray,
    v_weight: GPUArray,
    out_weight: GPUArray,
    q_bias: GPUArray | None = None,
    k_bias: GPUArray | None = None,
    v_bias: GPUArray | None = None,
    out_bias: GPUArray | None = None,
    num_heads: int = 16,
) -> GPUArray:
    """Cross-attention with GPU matmul.

    Args:
        x: Query input [B, N, D].
        context: Key/Value input [B, M, context_dim].
        q_weight: Query projection [D, D].
        k_weight: Key projection [context_dim, D].
        v_weight: Value projection [context_dim, D].
        out_weight: Output projection [D, D].
        num_heads: Number of attention heads.

    Returns:
        Attention output [B, N, D].
    """
    x_np = x.to_numpy()
    ctx_np = context.to_numpy()
    B, N, D = x_np.shape
    _, M, ctx_dim = ctx_np.shape
    head_dim = D // num_heads

    # Project Q from x
    x_2d = from_numpy(x_np.reshape(B * N, D).astype(np.float32))
    q_w = q_weight.to_numpy().T.astype(np.float32)
    q = matmul(x_2d, from_numpy(q_w)).to_numpy()
    if q_bias is not None:
        q = q + q_bias.to_numpy()

    # Project K, V from context
    ctx_2d = from_numpy(ctx_np.reshape(B * M, ctx_dim).astype(np.float32))
    k_w = k_weight.to_numpy().T.astype(np.float32)
    v_w = v_weight.to_numpy().T.astype(np.float32)
    k = matmul(ctx_2d, from_numpy(k_w)).to_numpy()
    v = matmul(ctx_2d, from_numpy(v_w)).to_numpy()
    if k_bias is not None:
        k = k + k_bias.to_numpy()
    if v_bias is not None:
        v = v + v_bias.to_numpy()

    # Reshape to [B, num_heads, seq, head_dim]
    q = q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, M, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, M, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Attention scores: [B*H, N, head_dim] @ [B*H, head_dim, M] -> [B*H, N, M]
    scale = 1.0 / np.sqrt(head_dim)
    q_flat = q.reshape(B * num_heads, N, head_dim)
    k_flat = k.reshape(B * num_heads, M, head_dim)
    v_flat = v.reshape(B * num_heads, M, head_dim)

    q_gpu = from_numpy(q_flat.astype(np.float32))
    k_t_gpu = from_numpy(k_flat.transpose(0, 2, 1).astype(np.float32))
    scores = batched_matmul(q_gpu, k_t_gpu).to_numpy() * scale

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

    # Attention output: [B*H, N, M] @ [B*H, M, head_dim] -> [B*H, N, head_dim]
    attn_gpu = from_numpy(attn_weights.astype(np.float32))
    v_gpu = from_numpy(v_flat.astype(np.float32))
    attn_out = batched_matmul(attn_gpu, v_gpu).to_numpy()

    # Reshape back: [B, N, D]
    attn_out = attn_out.reshape(B, num_heads, N, head_dim)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B * N, D)

    # Output projection
    out_w = out_weight.to_numpy().T.astype(np.float32)
    output = matmul(from_numpy(attn_out), from_numpy(out_w)).to_numpy()
    if out_bias is not None:
        output = output + out_bias.to_numpy()

    return from_numpy(output.reshape(B, N, D).astype(np.float32))


__all__ = ["self_attention", "cross_attention"]
