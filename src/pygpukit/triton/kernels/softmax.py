"""
Softmax Triton kernel.

Not optimized for maximum performance - focus on correctness and iteration speed.
"""

from typing import TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    from ..wrapper import TritonArray


@triton.jit
def _softmax_fwd_kernel(
    X,  # Input tensor pointer
    Y,  # Output tensor pointer
    stride_x,  # Stride for X rows
    stride_y,  # Stride for Y rows
    N,  # Row length (last dimension)
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax forward kernel (numerically stable)."""
    row = tl.program_id(0)

    # Compute offsets for this row
    X += row * stride_x
    Y += row * stride_y

    # First pass: find max for numerical stability
    _max = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=-float("inf")).to(tl.float32)
        _max = tl.maximum(_max, x)

    max_val = tl.max(_max, axis=0)

    # Second pass: compute exp(x - max) and sum
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=-float("inf")).to(tl.float32)
        exp_x = tl.exp(x - max_val)
        _sum += tl.where(mask, exp_x, 0.0)

    sum_exp = tl.sum(_sum, axis=0)

    # Third pass: normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=-float("inf")).to(tl.float32)
        exp_x = tl.exp(x - max_val)
        y = exp_x / sum_exp
        tl.store(Y + cols, y, mask=mask)


def softmax(
    x: "TritonArray",
    out: "TritonArray",
) -> None:
    """
    Softmax operation on last dimension (in-place output).

    Note: Output must be pre-allocated. This follows PyGPUkit's
    "explicit allocation" principle.

    Args:
        x: Input tensor [..., N] (TritonArray)
        out: Output tensor [..., N] (TritonArray, pre-allocated)
    """
    # Get dimensions
    shape = x.shape
    N = shape[-1]
    M = x.numel // N

    # Compute strides
    stride_x = x.stride(-2) if x.ndim > 1 else N
    stride_out = out.stride(-2) if out.ndim > 1 else N

    # Choose block size
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    # Launch kernel
    grid = (M,)
    _softmax_fwd_kernel[grid](
        x,
        out,
        stride_x,
        stride_out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
