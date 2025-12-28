"""
LayerNorm Triton kernel.

Not optimized for maximum performance - focus on correctness and iteration speed.
"""

from typing import TYPE_CHECKING, Optional

import triton
import triton.language as tl

if TYPE_CHECKING:
    from ..wrapper import TritonArray


@triton.jit
def _layernorm_fwd_kernel(
    X,  # Input tensor pointer
    W,  # Weight tensor pointer
    B,  # Bias tensor pointer (can be None)
    Y,  # Output tensor pointer
    stride_x,  # Stride for X rows
    stride_y,  # Stride for Y rows
    N,  # Hidden dimension
    eps,  # Epsilon for numerical stability
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """LayerNorm forward kernel."""
    row = tl.program_id(0)

    # Compute offsets for this row
    X += row * stride_x
    Y += row * stride_y

    # First pass: compute mean
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x

    mean = tl.sum(_sum, axis=0) / N

    # Second pass: compute variance
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        _sum_sq += diff * diff

    var = tl.sum(_sum_sq, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    # Normalize, scale, and shift
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w
        if HAS_BIAS:
            b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
            y += b
        tl.store(Y + cols, y, mask=mask)


def layernorm(
    x: "TritonArray",
    weight: "TritonArray",
    out: "TritonArray",
    bias: Optional["TritonArray"] = None,
    eps: float = 1e-5,
) -> None:
    """
    LayerNorm operation (in-place output).

    Note: Output must be pre-allocated. This follows PyGPUkit's
    "explicit allocation" principle.

    Args:
        x: Input tensor [..., hidden_size] (TritonArray)
        weight: Weight tensor [hidden_size] (TritonArray)
        out: Output tensor [..., hidden_size] (TritonArray, pre-allocated)
        bias: Bias tensor [hidden_size] (optional)
        eps: Epsilon for numerical stability
    """
    # Get dimensions
    shape = x.shape
    N = shape[-1]  # hidden dimension
    M = x.numel // N  # batch dimension (flattened)

    # Compute strides
    stride_x = x.stride(-2) if x.ndim > 1 else N
    stride_out = out.stride(-2) if out.ndim > 1 else N

    # Choose block size
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    # Handle None bias
    has_bias = bias is not None
    bias_ptr = bias if has_bias else weight  # Dummy pointer

    # Launch kernel
    grid = (M,)
    _layernorm_fwd_kernel[grid](
        x,
        weight,
        bias_ptr,
        out,
        stride_x,
        stride_out,
        N,
        eps,
        HAS_BIAS=has_bias,
        BLOCK_SIZE=BLOCK_SIZE,
    )
