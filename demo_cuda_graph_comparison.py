#!/usr/bin/env python3
"""Demo: CUDA Graph Position Buffer Feature Comparison.

Compares performance of:
1. Current v0.2.10: Graph OFF (use_graph=False)
2. Current v0.2.10: Graph ON (use_graph=True) with position buffer

Uses official Qwen3-8B model for benchmarking.
"""

import sys
import time

# Model paths (Aratako Qwen3-8B from CLAUDE.md)
model_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
tokenizer_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/tokenizer.json"

print("=" * 70)
print(" CUDA Graph Position Buffer Demo - Qwen3-8B")
print("=" * 70)

try:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

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
    ChatMessage(role="user", content="日本の首都はどこですか？"),
]
prompt = format_chat_messages(messages, model_type="qwen3")
input_ids = tokenizer.encode(prompt).ids
print("\nModel: Qwen3-8B (FP16)")
print(f"Prompt tokens: {len(input_ids)}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"Runs per mode: {NUM_RUNS}")

results = {}

# =============================================================================
# Load model once
# =============================================================================
print("\nLoading model...")
st = load_safetensors(model_path)
spec = detect_model_spec(st.tensor_names)
model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)
print("Model loaded!")

# =============================================================================
# Benchmark 1: Standard generation (baseline, no fixed cache)
# =============================================================================
print("\n" + "-" * 70)
print(" Mode 1: Standard (model.generate) - Baseline")
print("-" * 70)

# Warm-up
_ = model.generate(input_ids, max_new_tokens=4, temperature=0.0)

times_standard = []
for i in range(NUM_RUNS):
    start = time.perf_counter()
    tokens = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,  # Deterministic
    )
    elapsed = time.perf_counter() - start
    times_standard.append(elapsed)
    generated = len(tokens) - len(input_ids)
    tok_per_sec = generated / elapsed
    print(f"  Run {i + 1}: {generated} tokens in {elapsed:.3f}s = {tok_per_sec:.2f} tok/s")

    if i == 0:
        # Decode output for first run
        output_text = tokenizer.decode(tokens[len(input_ids) :])
        print(f"  Output: {output_text[:100]}...")

avg_standard = sum(times_standard) / len(times_standard)
tok_per_sec_standard = MAX_NEW_TOKENS / avg_standard
results["Standard"] = tok_per_sec_standard

# =============================================================================
# Benchmark 2: Fixed Cache (Graph OFF)
# =============================================================================
print("\n" + "-" * 70)
print(" Mode 2: Fixed Cache (use_graph=False)")
print("-" * 70)

# Reload model to reset state
del model
model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)

# Warm-up
_ = model.generate_cuda_graph(
    input_ids,
    max_new_tokens=4,
    max_seq_len=MAX_SEQ_LEN,
    temperature=0.0,
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
        temperature=0.0,
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

# =============================================================================
# Benchmark 3: Fixed Cache (Graph ON) - NEW FEATURE
# =============================================================================
print("\n" + "-" * 70)
print(" Mode 3: Fixed Cache (use_graph=True) - CUDA Graph with Position Buffer")
print("-" * 70)

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
        temperature=0.0,
        use_graph=True,  # <-- CUDA Graph enabled!
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
print(" Results Summary - Qwen3-8B (FP16)")
print("=" * 70)
print(f"{'Mode':<30} {'tok/s':>10} {'Speedup':>10}")
print("-" * 50)
for mode, tok_s in results.items():
    speedup = tok_s / tok_per_sec_standard
    print(f"{mode:<30} {tok_s:>10.2f} {speedup:>9.2f}x")

# Graph vs Fixed improvement
graph_vs_fixed = tok_per_sec_graph / tok_per_sec_fixed
print("\n" + "-" * 50)
print(f"CUDA Graph improvement over Fixed (no graph): {(graph_vs_fixed - 1) * 100:.1f}%")

print("\n" + "=" * 70)
print(" Demo Complete")
print("=" * 70)
