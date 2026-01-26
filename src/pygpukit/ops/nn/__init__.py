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
    relu2,
    sigmoid,
    silu,
    tanh,
)

# Attention operations
from pygpukit.ops.nn.attention import (
    fa3_fp8_available,
    get_sm_version,
    sdpa_causal,
    sdpa_causal_fixed_cache,
    sdpa_causal_fixed_cache_ptr,
    sdpa_causal_fp8,
    test_fp8_mma_direct,
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

# Llama4 specific operations
from pygpukit.ops.nn.llama4 import (
    irope_scale_q,
    l2norm,
    sdpa_irope,
)

# Recurrent operations
from pygpukit.ops.nn.recurrent import (
    lstm_bidirectional,
    lstm_forward,
)

# Fused operations
from pygpukit.ops.nn.fused import (
    geglu,
    rmsnorm_residual,
    swiglu,
)

# RoPE operations
from pygpukit.ops.nn.rope import (
    alibi_add_bias,
    alibi_compute_bias,
    # ALiBi
    alibi_init_slopes,
    # PoPE
    pope_init_encoding,
    pope_inplace,
    rope_init_linear,
    # RoPE extensions
    rope_init_ntk_aware,
    rope_init_yarn,
    rope_inplace,
    rope_inplace_f32table,
)

__all__ = [
    # Activation
    "gelu",
    "relu2",
    "silu",
    "sigmoid",
    "tanh",
    # Normalization
    "layernorm",
    "rmsnorm",
    "l2norm",
    # Fused operations
    "rmsnorm_residual",
    "swiglu",
    "geglu",
    # Llama4
    "irope_scale_q",
    "sdpa_irope",
    # Attention
    "sdpa_causal",
    "sdpa_causal_fixed_cache",
    "sdpa_causal_fixed_cache_ptr",
    # FA3 FP8 (SM120+)
    "fa3_fp8_available",
    "get_sm_version",
    "sdpa_causal_fp8",
    "test_fp8_mma_direct",
    # RoPE
    "rope_inplace",
    "rope_inplace_f32table",
    # RoPE extensions
    "rope_init_ntk_aware",
    "rope_init_yarn",
    "rope_init_linear",
    # PoPE
    "pope_init_encoding",
    "pope_inplace",
    # ALiBi
    "alibi_init_slopes",
    "alibi_compute_bias",
    "alibi_add_bias",
    # Linear
    "bias_add_inplace",
    "split_qkv_batch",
    "slice_rows_range_ptr",
    # Recurrent
    "lstm_forward",
    "lstm_bidirectional",
]
