"""Neural network operations for GPUArrays.

Corresponds to native/ops/nn/.

Provides:
- Activation functions (gelu, silu, sigmoid, tanh)
- Normalization layers (layernorm, rmsnorm)
- Attention operations (sdpa_causal, sdpa_causal_fixed_cache)
- RoPE (rotary position embedding)
- Linear operations (bias_add_inplace, split_qkv_batch)
- Recurrent operations (lstm_forward, lstm_bidirectional)
"""

from __future__ import annotations

# Activation functions
from pygpukit.ops.nn.activation import (
    gelu,
    sigmoid,
    silu,
    tanh,
)

# Attention operations
from pygpukit.ops.nn.attention import (
    sdpa_causal,
    sdpa_causal_fixed_cache,
    sdpa_causal_fixed_cache_ptr,
)

# Linear operations
from pygpukit.ops.nn.linear import (
    bias_add_inplace,
    slice_rows_range_ptr,
    split_qkv_batch,
)

# Normalization layers
from pygpukit.ops.nn.norm import (
    layernorm,
    rmsnorm,
)

# Recurrent operations
from pygpukit.ops.nn.recurrent import (
    lstm_bidirectional,
    lstm_forward,
)

# RoPE operations
from pygpukit.ops.nn.rope import (
    rope_inplace,
    rope_inplace_f32table,
)

__all__ = [
    # Activation
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    # Normalization
    "layernorm",
    "rmsnorm",
    # Attention
    "sdpa_causal",
    "sdpa_causal_fixed_cache",
    "sdpa_causal_fixed_cache_ptr",
    # RoPE
    "rope_inplace",
    "rope_inplace_f32table",
    # Linear
    "bias_add_inplace",
    "split_qkv_batch",
    "slice_rows_range_ptr",
    # Recurrent
    "lstm_forward",
    "lstm_bidirectional",
]
