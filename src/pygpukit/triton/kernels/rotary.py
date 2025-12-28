"""
Rotary Position Embedding (RoPE) Triton kernel.

Not optimized for maximum performance - focus on correctness and iteration speed.
"""

from typing import TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    from ..wrapper import TritonArray


@triton.jit
def _rotary_fwd_kernel(
    X,  # Input tensor pointer [batch, seq, num_heads, head_dim]
    COS,  # Cosine cache pointer [seq, head_dim/2]
    SIN,  # Sine cache pointer [seq, head_dim/2]
    Y,  # Output tensor pointer
    stride_xb,  # Batch stride
    stride_xs,  # Seq stride
    stride_xh,  # Head stride
    stride_xd,  # Dim stride
    stride_yb,
    stride_ys,
    stride_yh,
    stride_yd,
    stride_cos_s,  # Seq stride for cos/sin
    stride_cos_d,  # Dim stride for cos/sin
    seq_len,
    num_heads,
    head_dim,
    BLOCK_DIM: tl.constexpr,
):
    """RoPE forward kernel."""
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # Base pointers
    X += batch_idx * stride_xb + seq_idx * stride_xs + head_idx * stride_xh
    Y += batch_idx * stride_yb + seq_idx * stride_ys + head_idx * stride_yh
    COS += seq_idx * stride_cos_s
    SIN += seq_idx * stride_cos_s

    half_dim = head_dim // 2

    # Process first half and second half together
    for off in range(0, half_dim, BLOCK_DIM):
        cols = off + tl.arange(0, BLOCK_DIM)
        mask = cols < half_dim

        # Load x1 (first half) and x2 (second half)
        x1 = tl.load(X + cols * stride_xd, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X + (cols + half_dim) * stride_xd, mask=mask, other=0.0).to(tl.float32)

        # Load cos and sin
        cos = tl.load(COS + cols * stride_cos_d, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(SIN + cols * stride_cos_d, mask=mask, other=0.0).to(tl.float32)

        # Apply rotation
        # y1 = x1 * cos - x2 * sin
        # y2 = x1 * sin + x2 * cos
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos

        # Store
        tl.store(Y + cols * stride_yd, y1, mask=mask)
        tl.store(Y + (cols + half_dim) * stride_yd, y2, mask=mask)


def rotary(
    x: "TritonArray",
    cos: "TritonArray",
    sin: "TritonArray",
    out: "TritonArray",
) -> None:
    """
    RoPE (Rotary Position Embedding) operation (in-place output).

    Note: Output must be pre-allocated. This follows PyGPUkit's
    "explicit allocation" principle.

    Args:
        x: Input tensor [batch, seq, num_heads, head_dim] (TritonArray)
        cos: Cosine cache [seq, head_dim/2] (TritonArray)
        sin: Sine cache [seq, head_dim/2] (TritonArray)
        out: Output tensor [batch, seq, num_heads, head_dim] (TritonArray, pre-allocated)
    """
    batch, seq_len, num_heads, head_dim = x.shape

    # Choose block size
    half_dim = head_dim // 2
    BLOCK_DIM = triton.next_power_of_2(half_dim)
    BLOCK_DIM = min(BLOCK_DIM, 128)

    # Launch kernel
    grid = (batch, seq_len, num_heads)
    _rotary_fwd_kernel[grid](
        x,
        cos,
        sin,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        cos.stride(0),
        cos.stride(1),
        seq_len,
        num_heads,
        head_dim,
        BLOCK_DIM=BLOCK_DIM,
    )
