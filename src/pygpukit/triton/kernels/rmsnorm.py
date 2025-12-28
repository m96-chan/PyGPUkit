"""
RMSNorm Triton kernel.

Not optimized for maximum performance - focus on correctness and iteration speed.
"""

from typing import TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    from ..wrapper import TritonArray


@triton.jit
def _rmsnorm_fwd_kernel(
    X,  # Input tensor pointer
    W,  # Weight tensor pointer
    Y,  # Output tensor pointer
    stride_x,  # Stride for X rows
    stride_y,  # Stride for Y rows
    N,  # Hidden dimension
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm forward kernel."""
    row = tl.program_id(0)

    # Compute offsets for this row
    X += row * stride_x
    Y += row * stride_y

    # Compute mean of squares
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum_sq += x * x

    # Reduce and compute RMS
    sum_sq = tl.sum(_sum_sq, axis=0)
    rms = tl.rsqrt(sum_sq / N + eps)

    # Normalize and apply weight
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rms * w
        tl.store(Y + cols, y, mask=mask)


def rmsnorm(
    x: "TritonArray",
    weight: "TritonArray",
    out: "TritonArray",
    eps: float = 1e-6,
) -> None:
    """
    RMSNorm operation (in-place output).

    Note: Output must be pre-allocated. This follows PyGPUkit's
    "explicit allocation" principle.

    Args:
        x: Input tensor [..., hidden_size] (TritonArray)
        weight: Weight tensor [hidden_size] (TritonArray)
        out: Output tensor [..., hidden_size] (TritonArray, pre-allocated)
        eps: Epsilon for numerical stability
    """
    # Get dimensions
    shape = x.shape
    N = shape[-1]  # hidden dimension
    M = x.numel // N  # batch dimension (flattened)

    # Compute strides
    stride_x = x.stride(-2) if x.ndim > 1 else N
    stride_out = out.stride(-2) if out.ndim > 1 else N

    # Choose block size (simple heuristic)
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    # Launch kernel
    grid = (M,)
    _rmsnorm_fwd_kernel[grid](
        x,
        weight,
        out,
        stride_x,
        stride_out,
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
