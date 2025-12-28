"""Basic operations for GPUArrays.

This module re-exports all operations from submodules for backwards compatibility.
For new code, prefer importing from specific submodules:
- pygpukit.ops.elementwise - add, sub, mul, div, add_inplace, mul_inplace, copy_to
- pygpukit.ops.unary - exp, log, relu
- pygpukit.ops.reduction - sum, mean, max, softmax
- pygpukit.ops.matmul - matmul, transpose, linear_bias_gelu
- pygpukit.ops.nn - gelu, silu, layernorm, rmsnorm, bias_add_inplace, sdpa_*, rope_*
- pygpukit.ops.embedding - embedding_lookup*, kv_cache_*
- pygpukit.ops.sampling - sample_*, set_sampling_seed
- pygpukit.ops.tensor - concat_*, repeat_*, transpose_3d_*, reshape_copy, cast_*
"""

from __future__ import annotations

# Re-export validation helpers
from pygpukit.ops._common import (
    _validate_float_dtype,
    _validate_same_dtype,
    _validate_same_shape,
)

# Re-export elementwise operations
from pygpukit.ops.elementwise import (
    add,
    add_inplace,
    clamp,
    copy_to,
    div,
    mul,
    mul_inplace,
    sub,
    where,
)

# Re-export embedding operations
from pygpukit.ops.embedding import (
    embedding_lookup,
    embedding_lookup_batch,
    embedding_lookup_ptr,
    kv_cache_prefill,
    kv_cache_prefill_gqa,
    kv_cache_update,
    kv_cache_update_gqa,
    kv_cache_update_gqa_ptr,
)

# Re-export matmul operations
from pygpukit.ops.matmul import (
    batched_matmul,
    fp8_available,
    fp8_fp8_get_scale_sizes,
    fp8_fp8_sm120_available,
    fp8_get_sizes,
    fp8_init_lut,
    fp8_sm90_available,
    fp8_sm100_available,
    fp8_sm120_available,
    # GEMV operations
    gemv_bf16,
    gemv_fp8_bf16,
    gemv_fp8_bf16_batched,
    gemv_nvf4_available,
    gemv_nvf4_bf16,
    # Grouped GEMM for MoE
    grouped_gemm_fp8_bf16,
    grouped_gemm_init_lut,
    linear_bias_gelu,
    matmul,
    matmul_fp8,
    matmul_fp8_fp8_blockwise_sm120,
    matmul_fp8_fp8_sm120,
    matmul_fp8_sm90,
    matmul_fp8_sm100,
    matmul_fp8_sm120,
    matmul_nvf4_bf16_sm120,
    nvf4_bf16_sm120_available,
    nvf4_get_sizes,
    quantize_bf16_to_nvf4,
    transpose,
    # W8A16 GEMM
    w8a16_gemm_sm120,
)

# Re-export neural network operations
from pygpukit.ops.nn import (
    bias_add_inplace,
    gelu,
    layernorm,
    rmsnorm,
    rope_inplace,
    rope_inplace_f32table,
    sdpa_causal,
    sdpa_causal_fixed_cache,
    sdpa_causal_fixed_cache_ptr,
    sigmoid,
    silu,
    slice_rows_range_ptr,
    split_qkv_batch,
    tanh,
)

# Re-export reduction operations
from pygpukit.ops.reduction import (
    argmax,
    max,
    mean,
    min,
    softmax,
    sum,
    sum_axis,
)

# Re-export sampling operations
from pygpukit.ops.sampling import (
    sample_greedy,
    sample_multinomial,
    sample_token_gpu,
    sample_topk,
    sample_topk_to_buf_ptr,
    sample_topp,
    set_sampling_seed,
)

# Re-export tensor operations
from pygpukit.ops.tensor import (
    cast_bf16_to_f32,
    cast_f16_to_f32,
    cast_f32_to_bf16,
    cast_f32_to_f16,
    concat_axis0,
    repeat_interleave_axis1,
    reshape_copy,
    transpose_3d_021,
    transpose_4d_0213,
)

# Re-export unary operations
from pygpukit.ops.unary import (
    abs,
    cos,
    exp,
    log,
    neg,
    relu,
    rsqrt,
    sin,
    sqrt,
)

__all__ = [
    # Validation helpers
    "_validate_same_shape",
    "_validate_same_dtype",
    "_validate_float_dtype",
    # Elementwise
    "add",
    "sub",
    "mul",
    "div",
    "add_inplace",
    "mul_inplace",
    "copy_to",
    "clamp",
    "where",
    # Unary
    "abs",
    "cos",
    "exp",
    "log",
    "neg",
    "relu",
    "rsqrt",
    "sin",
    "sqrt",
    # Reduction
    "argmax",
    "max",
    "mean",
    "min",
    "softmax",
    "sum",
    "sum_axis",
    # Matmul
    "matmul",
    "batched_matmul",
    "transpose",
    "linear_bias_gelu",
    "matmul_fp8",
    "matmul_fp8_sm90",
    "matmul_fp8_sm100",
    "matmul_fp8_sm120",
    "matmul_nvf4_bf16_sm120",
    "fp8_available",
    "fp8_fp8_sm120_available",
    "fp8_fp8_get_scale_sizes",
    "fp8_sm90_available",
    "fp8_sm100_available",
    "fp8_sm120_available",
    "matmul_fp8_fp8_blockwise_sm120",
    "matmul_fp8_fp8_sm120",
    "nvf4_bf16_sm120_available",
    # GEMV
    "gemv_bf16",
    "gemv_fp8_bf16",
    "gemv_fp8_bf16_batched",
    "gemv_nvf4_bf16",
    "gemv_nvf4_available",
    # W8A16 GEMM
    "w8a16_gemm_sm120",
    # Grouped GEMM for MoE
    "grouped_gemm_fp8_bf16",
    "grouped_gemm_init_lut",
    "fp8_init_lut",
    "fp8_get_sizes",
    "nvf4_get_sizes",
    "quantize_bf16_to_nvf4",
    # Neural Network
    "gelu",
    "sigmoid",
    "silu",
    "tanh",
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
    "transpose_4d_0213",
    "reshape_copy",
    "cast_f32_to_bf16",
    "cast_f32_to_f16",
    "cast_bf16_to_f32",
    "cast_f16_to_f32",
]
