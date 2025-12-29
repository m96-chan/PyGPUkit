# PyGPUkit Python API Reference

## Core

### GPUArray (pygpukit.core.array)
```
GPUArray
  - shape, dtype, size, ndim, nbytes, itemsize
  - device_ptr, on_gpu, last_access
  - to_numpy(), is_contiguous(), contiguous(), clone()
  - astype(), narrow(), view(), slice_rows()
  - transpose(), T, reshape()
  - __add__, __sub__, __mul__, __truediv__, __matmul__, __getitem__
```

### Factory (pygpukit.core.factory)
```
from_numpy(), zeros(), ones(), empty(), arange()
```

---

## Operations (pygpukit.ops)

### matmul.py - Matrix Operations
```
# Basic
matmul, transpose, batched_matmul, linear_bias_gelu

# GEMV
gemv_bf16, gemv_fp8_bf16, gemv_fp8_bf16_batched
gemv_fp8_bf16_opt, gemv_fp8_bf16_opt_batched
gemv_nvf4_bf16

# FP8 GEMM
matmul_fp8, matmul_fp8_sm90, matmul_fp8_sm100, matmul_fp8_sm120
matmul_fp8_fp8_sm120, matmul_fp8_fp8_blockwise_sm120
fp8_fp8_get_scale_sizes, fp8_get_sizes

# NVF4
matmul_nvf4_bf16_sm120, nvf4_get_sizes, quantize_bf16_to_nvf4

# Grouped GEMM (MoE)
grouped_gemm_fp8_bf16, grouped_gemm_init_lut

# W8A16
w8a16_gemm_sm120

# Availability checks
fp8_available, fp8_sm90_available, fp8_sm100_available, fp8_sm120_available
fp8_fp8_sm120_available, nvf4_bf16_sm120_available, gemv_nvf4_available
```

### nn.py - Neural Network Ops
```
# Activations
gelu, silu, sigmoid, tanh

# Normalization
layernorm, rmsnorm, bias_add_inplace

# Attention
sdpa_causal, sdpa_causal_fixed_cache, sdpa_causal_fixed_cache_ptr
rope_inplace, rope_inplace_f32table
split_qkv_batch, slice_rows_range_ptr
```

### elementwise.py - Element-wise Ops
```
add, sub, mul, div
add_inplace, mul_inplace
copy_to, clamp, where
```

### reduction.py - Reduction Ops
```
sum, mean, max, min, argmax, sum_axis, softmax
```

### sampling.py - Token Sampling
```
sample_token_gpu, sample_topk_to_buf_ptr
sample_greedy, sample_multinomial, sample_topk, sample_topp
set_sampling_seed
```

### embedding.py - Embedding & KV Cache
```
embedding_lookup, embedding_lookup_ptr, embedding_lookup_batch
kv_cache_update, kv_cache_prefill
kv_cache_update_gqa, kv_cache_prefill_gqa, kv_cache_update_gqa_ptr
```

### tensor.py - Tensor Manipulation
```
concat_axis0, repeat_interleave_axis1, reshape_copy
transpose_3d_021, transpose_3d_012
transpose_4d_0213, transpose_4d_0132
cast_f32_to_bf16, cast_f32_to_f16, cast_bf16_to_f32, cast_f16_to_f32
```

---

## LLM (pygpukit.llm)

### loader.py - Model Loading
```
load_model_from_safetensors  # Auto-detect model type
load_gpt2_from_safetensors
load_llama_from_safetensors
load_qwen3_from_safetensors
load_mixtral_from_safetensors
repack_model_weights

# FP8
is_fp8_weight, load_fp8_weight_direct, dequantize_fp8_e4m3_block
FP8QuantConfig
```

### layers.py - Layer Classes
```
LinearBF16, LinearFP8
Norm (RMSNorm/LayerNorm)
Attention, MLP, MoELayer, TransformerBlock
```

### model.py - Model Classes
```
CausalTransformerModel
  - generate(), generate_stream()
  - snapshot_kv_cache(), restore_kv_cache()
  - decode_step_self_speculative_lookahead()
  - decode_step_jacobi_lookahead()
```

---

## Last Updated
2025-12-27
