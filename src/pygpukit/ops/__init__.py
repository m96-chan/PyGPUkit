"""Operations module for PyGPUkit.

Submodules:
- elementwise: add, sub, mul, div, add_inplace, mul_inplace, copy_to
- unary: exp, log, relu
- reduction: sum, mean, max, softmax
- matmul: matmul, transpose, linear_bias_gelu
- nn: gelu, silu, layernorm, rmsnorm, bias_add_inplace, sdpa_*, rope_*
- embedding: embedding_lookup*, kv_cache_*
- sampling: sample_*, set_sampling_seed
- tensor: concat_*, repeat_*, transpose_3d_*, reshape_copy, cast_*
"""

from pygpukit.ops.basic import (
    # Elementwise
    add,
    add_inplace,
    # Neural Network
    bias_add_inplace,
    # Tensor
    cast_bf16_to_f32,
    cast_f16_to_f32,
    cast_f32_to_bf16,
    cast_f32_to_f16,
    concat_axis0,
    copy_to,
    div,
    # Embedding & KV Cache
    embedding_lookup,
    embedding_lookup_batch,
    embedding_lookup_ptr,
    # Unary
    exp,
    gelu,
    kv_cache_prefill,
    kv_cache_prefill_gqa,
    kv_cache_update,
    kv_cache_update_gqa,
    kv_cache_update_gqa_ptr,
    layernorm,
    # Matmul
    linear_bias_gelu,
    log,
    matmul,
    # Reduction
    max,
    mean,
    mul,
    mul_inplace,
    relu,
    repeat_interleave_axis1,
    reshape_copy,
    rmsnorm,
    rope_inplace,
    rope_inplace_f32table,
    # Sampling
    sample_greedy,
    sample_multinomial,
    sample_token_gpu,
    sample_topk,
    sample_topk_to_buf_ptr,
    sample_topp,
    sdpa_causal,
    sdpa_causal_fixed_cache,
    sdpa_causal_fixed_cache_ptr,
    set_sampling_seed,
    silu,
    slice_rows_range_ptr,
    softmax,
    split_qkv_batch,
    sub,
    sum,
    transpose,
    transpose_3d_021,
)

__all__ = [
    # Elementwise
    "add",
    "sub",
    "mul",
    "div",
    "add_inplace",
    "mul_inplace",
    "copy_to",
    # Unary
    "exp",
    "log",
    "relu",
    # Reduction
    "sum",
    "mean",
    "max",
    "softmax",
    # Matmul
    "matmul",
    "transpose",
    "linear_bias_gelu",
    # Neural Network
    "gelu",
    "silu",
    "layernorm",
    "rmsnorm",
    "bias_add_inplace",
    "sdpa_causal",
    "sdpa_causal_fixed_cache",
    "sdpa_causal_fixed_cache_ptr",
    "rope_inplace",
    "rope_inplace_f32table",
    "split_qkv_batch",
    "slice_rows_range_ptr",
    # Embedding & KV Cache
    "embedding_lookup",
    "embedding_lookup_ptr",
    "embedding_lookup_batch",
    "kv_cache_update",
    "kv_cache_prefill",
    "kv_cache_update_gqa",
    "kv_cache_prefill_gqa",
    "kv_cache_update_gqa_ptr",
    # Sampling
    "sample_token_gpu",
    "sample_topk_to_buf_ptr",
    "sample_greedy",
    "sample_multinomial",
    "sample_topk",
    "sample_topp",
    "set_sampling_seed",
    # Tensor
    "concat_axis0",
    "repeat_interleave_axis1",
    "transpose_3d_021",
    "reshape_copy",
    "cast_f32_to_bf16",
    "cast_f32_to_f16",
    "cast_bf16_to_f32",
    "cast_f16_to_f32",
]
