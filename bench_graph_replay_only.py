#!/usr/bin/env python3
"""Measure pure graph.replay() time vs kernel launches."""

import gc
import time

import numpy as np

model_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"

from pygpukit._pygpukit_native import CudaGraph

from pygpukit.core import default_stream, from_numpy
from pygpukit.llm import detect_model_spec, load_model_from_safetensors, load_safetensors
from pygpukit.llm.model import DecodeBuffers, precompute_freqs_cis
from pygpukit.ops.basic import add_inplace, copy_to, embedding_lookup, kv_cache_prefill_gqa, rmsnorm

MAX_SEQ_LEN = 512

print("=" * 60)
print("Pure Graph Replay Benchmark")
print("=" * 60)

print("\nLoading model...")
st = load_safetensors(model_path)
spec = detect_model_spec(st.tensor_names)
model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)
dtype = str(model.embed_tokens.dtype)
use_qk_norm = model.spec is not None and model.spec.use_qk_norm

print("Initializing buffers...")
for block in model.blocks:
    block.attn.init_fixed_cache(MAX_SEQ_LEN, dtype=dtype)

buffers = DecodeBuffers.allocate(model.config, dtype=dtype, use_qk_norm=use_qk_norm)

if model.config.use_rope:
    cos_np, sin_np = precompute_freqs_cis(
        model.config.head_dim, MAX_SEQ_LEN, model.config.rope_theta
    )
    np_dtype = np.float16 if dtype == "float16" else np.float32
    model._rope_cos_gpu = from_numpy(cos_np.astype(np_dtype))
    model._rope_sin_gpu = from_numpy(sin_np.astype(np_dtype))

# Run prefill to initialize KV cache
print("Running prefill...")
input_ids = [1, 2, 3, 4, 5]  # Dummy tokens
hidden, past_key_values = model(input_ids, use_cache=True)
for i, block in enumerate(model.blocks):
    past_k, past_v = past_key_values[i]
    kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
    kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

token_id = 100
position = 5
context_len = 6


# Define inline decode step
def _inline_decode_step():
    embedding_lookup(model.embed_tokens, buffers.hidden, token_id)
    for block in model.blocks:
        rmsnorm(buffers.hidden, block.attn_norm.weight, block.attn_norm.eps, out=buffers.norm_out)
        copy_to(buffers.hidden, buffers.residual)
        model._attention_forward_zero_alloc(
            block.attn,
            buffers.norm_out,
            position,
            context_len,
            buffers,
            use_position_ptr=False,
        )
        add_inplace(buffers.hidden, buffers.residual)
        copy_to(buffers.hidden, buffers.residual)
        rmsnorm(buffers.hidden, block.mlp_norm.weight, block.mlp_norm.eps, out=buffers.norm_out)
        model._mlp_forward_zero_alloc(block.mlp, buffers.norm_out, buffers)
        add_inplace(buffers.hidden, buffers.residual)
    rmsnorm(buffers.hidden, model.final_norm.weight, model.final_norm.eps, out=buffers.norm_out)
    copy_to(buffers.norm_out, buffers.hidden)


# ============================================================
# Test 1: Direct kernel launches (no graph)
# ============================================================
print("\n--- Test 1: Direct Kernel Launches ---")

# Warmup
for _ in range(3):
    _inline_decode_step()
default_stream().synchronize()

# Measure
times_direct = []
for i in range(10):
    default_stream().synchronize()
    start = time.perf_counter()
    _inline_decode_step()
    default_stream().synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    times_direct.append(elapsed)
    print(f"  {i + 1}: {elapsed:.2f} ms")

mean_direct = np.mean(times_direct)
print(f"  Mean: {mean_direct:.2f} ms")

# ============================================================
# Test 2: Graph capture and replay
# ============================================================
print("\n--- Test 2: CUDA Graph Replay ---")

# Capture graph
print("Capturing graph...")
graph = CudaGraph()
gc.disable()
try:
    graph.begin_capture()
    _inline_decode_step()
    graph.end_capture()
finally:
    gc.enable()
print(f"  Captured {graph.num_nodes} nodes")

# Warmup replay
for _ in range(3):
    graph.replay()
graph.synchronize()

# Measure replay
times_graph = []
for i in range(10):
    graph.synchronize()  # Ensure previous is done
    start = time.perf_counter()
    graph.replay()
    graph.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    times_graph.append(elapsed)
    print(f"  {i + 1}: {elapsed:.2f} ms")

mean_graph = np.mean(times_graph)
print(f"  Mean: {mean_graph:.2f} ms")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY (Transformer blocks only, no get_logits)")
print("=" * 60)
print(f"Direct launches: {mean_direct:.2f} ms")
print(f"Graph replay:    {mean_graph:.2f} ms")
print(f"Speedup:         {mean_direct / mean_graph:.2f}x")
print(f"Saved per step:  {mean_direct - mean_graph:.2f} ms")
print("=" * 60)
