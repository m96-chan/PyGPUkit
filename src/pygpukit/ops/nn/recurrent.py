"""Recurrent (LSTM) operations for GPUArrays.

Corresponds to native/ops/nn/recurrent/.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray


def lstm_forward(
    x: GPUArray,
    W_ih: GPUArray,
    W_hh: GPUArray,
    b_ih: GPUArray,
    b_hh: GPUArray,
    h0: GPUArray | None = None,
    c0: GPUArray | None = None,
    reverse: bool = False,
) -> tuple[GPUArray, GPUArray, GPUArray]:
    """LSTM forward pass (unidirectional).

    Implements the standard LSTM equations:
        i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)
        f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)
        g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)
        o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    Args:
        x: Input sequence [batch, seq_len, input_size].
        W_ih: Input-to-hidden weights [4*hidden_size, input_size].
        W_hh: Hidden-to-hidden weights [4*hidden_size, hidden_size].
        b_ih: Input bias [4*hidden_size].
        b_hh: Hidden bias [4*hidden_size].
        h0: Initial hidden state [batch, hidden_size]. If None, zeros.
        c0: Initial cell state [batch, hidden_size]. If None, zeros.
        reverse: If True, process sequence in reverse order.

    Returns:
        Tuple of (output, h_n, c_n):
            output: Hidden states [batch, seq_len, hidden_size]
            h_n: Final hidden state [batch, hidden_size]
            c_n: Final cell state [batch, hidden_size]
    """
    from pygpukit.core.backend import get_backend, get_native_module

    backend = get_backend()
    if not backend.is_available():
        raise RuntimeError("lstm_forward requires GPU backend")

    native = get_native_module()

    # Create zero-sized arrays for None states
    if h0 is None:
        h0_native = native.GPUArray([0], native.Float32)
    else:
        h0_native = h0._get_native()

    if c0 is None:
        c0_native = native.GPUArray([0], native.Float32)
    else:
        c0_native = c0._get_native()

    output_native, h_n_native, c_n_native = native.lstm_forward(
        x._get_native(),
        W_ih._get_native(),
        W_hh._get_native(),
        b_ih._get_native(),
        b_hh._get_native(),
        h0_native,
        c0_native,
        reverse,
    )

    return (
        GPUArray._wrap_native(output_native),
        GPUArray._wrap_native(h_n_native),
        GPUArray._wrap_native(c_n_native),
    )


def lstm_bidirectional(
    x: GPUArray,
    W_ih_fwd: GPUArray,
    W_hh_fwd: GPUArray,
    b_ih_fwd: GPUArray,
    b_hh_fwd: GPUArray,
    W_ih_bwd: GPUArray,
    W_hh_bwd: GPUArray,
    b_ih_bwd: GPUArray,
    b_hh_bwd: GPUArray,
) -> tuple[GPUArray, GPUArray, GPUArray]:
    """Bidirectional LSTM.

    Runs forward and backward LSTM passes and concatenates the outputs.

    Args:
        x: Input sequence [batch, seq_len, input_size].
        W_ih_fwd, W_hh_fwd, b_ih_fwd, b_hh_fwd: Forward LSTM weights.
        W_ih_bwd, W_hh_bwd, b_ih_bwd, b_hh_bwd: Backward LSTM weights.

    Returns:
        Tuple of (output, h_n, c_n):
            output: Concatenated hidden states [batch, seq_len, 2*hidden_size]
            h_n: Stacked final hidden states [2, batch, hidden_size]
            c_n: Stacked final cell states [2, batch, hidden_size]
    """
    from pygpukit.core.backend import get_backend, get_native_module

    backend = get_backend()
    if not backend.is_available():
        raise RuntimeError("lstm_bidirectional requires GPU backend")

    native = get_native_module()

    output_native, h_n_native, c_n_native = native.lstm_bidirectional(
        x._get_native(),
        W_ih_fwd._get_native(),
        W_hh_fwd._get_native(),
        b_ih_fwd._get_native(),
        b_hh_fwd._get_native(),
        W_ih_bwd._get_native(),
        W_hh_bwd._get_native(),
        b_ih_bwd._get_native(),
        b_hh_bwd._get_native(),
    )

    return (
        GPUArray._wrap_native(output_native),
        GPUArray._wrap_native(h_n_native),
        GPUArray._wrap_native(c_n_native),
    )


__all__ = [
    "lstm_forward",
    "lstm_bidirectional",
]
