"""Feed-Forward Network modules for DiT.

Provides GEGLU (Gated Linear Unit with GELU) and standard FFN.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.ops.matmul.generic import matmul


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def geglu_ffn(
    x: GPUArray,
    gate_proj_weight: GPUArray,
    gate_proj_bias: GPUArray | None,
    down_proj_weight: GPUArray,
    down_proj_bias: GPUArray | None,
) -> GPUArray:
    """GEGLU Feed-Forward Network.

    Structure:
        x -> Linear(D, 2*D_ff) -> split -> GELU(gate) * up -> Linear(D_ff, D)

    Args:
        x: Input [B, N, D].
        gate_proj_weight: Combined gate+up projection [2*D_ff, D].
        gate_proj_bias: Bias [2*D_ff].
        down_proj_weight: Down projection [D, D_ff].
        down_proj_bias: Bias [D].

    Returns:
        Output [B, N, D].
    """
    x_np = x.to_numpy()
    B, N, D = x_np.shape

    # Combined gate + up projection
    x_2d = from_numpy(x_np.reshape(B * N, D).astype(np.float32))
    gate_w = gate_proj_weight.to_numpy().T.astype(np.float32)  # [D, 2*D_ff]
    hidden = matmul(x_2d, from_numpy(gate_w)).to_numpy()

    if gate_proj_bias is not None:
        hidden = hidden + gate_proj_bias.to_numpy()

    # Split into gate and up
    d_ff = hidden.shape[-1] // 2
    gate = hidden[:, :d_ff]
    up = hidden[:, d_ff:]

    # GEGLU: GELU(gate) * up
    hidden = gelu(gate) * up

    # Down projection - note: down weight expects d_ff input, but we have d_ff/2
    # Check if dimensions match, otherwise use full hidden
    down_w_np = down_proj_weight.to_numpy()
    expected_in = down_w_np.shape[1]  # [D, D_ff] -> D_ff

    if hidden.shape[-1] != expected_in:
        # PixArt uses GELU (not GEGLU) - don't split
        hidden_full = matmul(x_2d, from_numpy(gate_w)).to_numpy()
        if gate_proj_bias is not None:
            hidden_full = hidden_full + gate_proj_bias.to_numpy()
        hidden = gelu(hidden_full)

    # Down projection
    down_w = down_w_np.T.astype(np.float32)  # [D_ff, D]
    output = matmul(from_numpy(hidden.astype(np.float32)), from_numpy(down_w)).to_numpy()

    if down_proj_bias is not None:
        output = output + down_proj_bias.to_numpy()

    return from_numpy(output.reshape(B, N, D).astype(np.float32))


def standard_ffn(
    x: GPUArray,
    up_weight: GPUArray,
    up_bias: GPUArray | None,
    down_weight: GPUArray,
    down_bias: GPUArray | None,
    activation: str = "gelu",
) -> GPUArray:
    """Standard Feed-Forward Network.

    Structure:
        x -> Linear(D, D_ff) -> Activation -> Linear(D_ff, D)

    Args:
        x: Input [B, N, D].
        up_weight: Up projection [D_ff, D].
        up_bias: Bias [D_ff].
        down_weight: Down projection [D, D_ff].
        down_bias: Bias [D].
        activation: Activation function ("gelu", "silu", "relu").

    Returns:
        Output [B, N, D].
    """
    x_np = x.to_numpy()
    B, N, D = x_np.shape

    # Up projection
    x_2d = from_numpy(x_np.reshape(B * N, D).astype(np.float32))
    up_w = up_weight.to_numpy().T.astype(np.float32)
    hidden = matmul(x_2d, from_numpy(up_w)).to_numpy()

    if up_bias is not None:
        hidden = hidden + up_bias.to_numpy()

    # Activation
    if activation == "gelu":
        hidden = gelu(hidden)
    elif activation == "silu":
        hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))
    elif activation == "relu":
        hidden = np.maximum(hidden, 0)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Down projection
    down_w = down_weight.to_numpy().T.astype(np.float32)
    output = matmul(from_numpy(hidden.astype(np.float32)), from_numpy(down_w)).to_numpy()

    if down_bias is not None:
        output = output + down_bias.to_numpy()

    return from_numpy(output.reshape(B, N, D).astype(np.float32))


__all__ = ["geglu_ffn", "standard_ffn", "gelu"]
