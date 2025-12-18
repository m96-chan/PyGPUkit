#!/usr/bin/env python3
"""Benchmark CUDA Graph for LLM inference.

Compares:
- Standard: Normal generation with allocations
- Fixed (Graph off): Fixed KV cache without graph
- Fixed (Graph on): Fixed KV cache with CUDA Graph capture/replay
"""

import time

model_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
tokenizer_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/tokenizer.json"

print("=" * 70)
print(" CUDA Graph Benchmark - LLM Inference")
print("=" * 70)

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(tokenizer_path)

from pygpukit.llm import (
    ChatMessage,
    detect_model_spec,
    format_chat_messages,
    load_model_from_safetensors,
    load_safetensors,
)

# Benchmark parameters
NUM_RUNS = 3
MAX_NEW_TOKENS = 32
MAX_SEQ_LEN = 512

# Prepare input
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is 2+2?"),
]
prompt = format_chat_messages(messages, model_type="qwen3")
input_ids = tokenizer.encode(prompt).ids
print(f"Prompt tokens: {len(input_ids)}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"Runs per mode: {NUM_RUNS}")

results = {}

# =============================================================================
# Benchmark 1: Standard generation
# =============================================================================
print("\n" + "-" * 70)
print(" Mode 1: Standard (model.generate)")
print("-" * 70)

st = load_safetensors(model_path)
spec = detect_model_spec(st.tensor_names)
model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)

# Warm-up
_ = model.generate(input_ids, max_new_tokens=4, temperature=0.7, top_k=50, top_p=0.9)

times_standard = []
for i in range(NUM_RUNS):
    start = time.perf_counter()
    tokens = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
    elapsed = time.perf_counter() - start
    times_standard.append(elapsed)
    generated = len(tokens) - len(input_ids)
    tok_per_sec = generated / elapsed
    print(f"  Run {i + 1}: {generated} tokens in {elapsed:.3f}s = {tok_per_sec:.2f} tok/s")

avg_standard = sum(times_standard) / len(times_standard)
tok_per_sec_standard = MAX_NEW_TOKENS / avg_standard
results["Standard"] = tok_per_sec_standard
del model

# =============================================================================
# Benchmark 2: Fixed Cache (Graph off)
# =============================================================================
print("\n" + "-" * 70)
print(" Mode 2: Fixed Cache (Graph off)")
print("-" * 70)

model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)

# Warm-up
_ = model.generate_cuda_graph(
    input_ids,
    max_new_tokens=4,
    max_seq_len=MAX_SEQ_LEN,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    use_graph=False,
    gpu_sampling=True,
)

times_fixed = []
for i in range(NUM_RUNS):
    # Reload model to reset KV cache state
    del model
    model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)

    start = time.perf_counter()
    tokens = model.generate_cuda_graph(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        use_graph=False,
        gpu_sampling=True,
    )
    elapsed = time.perf_counter() - start
    times_fixed.append(elapsed)
    generated = len(tokens) - len(input_ids)
    tok_per_sec = generated / elapsed
    print(f"  Run {i + 1}: {generated} tokens in {elapsed:.3f}s = {tok_per_sec:.2f} tok/s")

avg_fixed = sum(times_fixed) / len(times_fixed)
tok_per_sec_fixed = MAX_NEW_TOKENS / avg_fixed
results["Fixed (Graph off)"] = tok_per_sec_fixed
del model

# =============================================================================
# Benchmark 3: Fixed Cache (Graph on)
# =============================================================================
print("\n" + "-" * 70)
print(" Mode 3: Fixed Cache (Graph on)")
print("-" * 70)

model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)

times_graph = []
for i in range(NUM_RUNS):
    # Reload model to reset KV cache and graph state
    del model
    model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)

    start = time.perf_counter()
    tokens = model.generate_cuda_graph(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        use_graph=True,
        gpu_sampling=True,
    )
    elapsed = time.perf_counter() - start
    times_graph.append(elapsed)
    generated = len(tokens) - len(input_ids)
    tok_per_sec = generated / elapsed
    print(f"  Run {i + 1}: {generated} tokens in {elapsed:.3f}s = {tok_per_sec:.2f} tok/s")

avg_graph = sum(times_graph) / len(times_graph)
tok_per_sec_graph = MAX_NEW_TOKENS / avg_graph
results["Fixed (Graph on)"] = tok_per_sec_graph

# =============================================================================
# Results Summary
# =============================================================================
print("\n" + "=" * 70)
print(" Results Summary")
print("=" * 70)
print(f"{'Mode':<25} {'tok/s':>10} {'Speedup':>10}")
print("-" * 45)
for mode, tok_s in results.items():
    speedup = tok_s / tok_per_sec_standard
    print(f"{mode:<25} {tok_s:>10.2f} {speedup:>9.2f}x")

print("\n" + "=" * 70)
print(" Benchmark Complete")
print("=" * 70)
