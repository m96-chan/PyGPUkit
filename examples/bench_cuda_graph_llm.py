#!/usr/bin/env python3
"""
Benchmark: generate vs generate_cuda_graph

Compares standard generation with fixed-length KV cache generation.
"""

import time
from pathlib import Path

import numpy as np


def main():
    model_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
    tokenizer_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/tokenizer.json"

    print("=" * 70)
    print(" Qwen3-8B: generate vs generate_cuda_graph")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load model
    print("Loading model...")
    from pygpukit.llm import (
        detect_model_spec,
        load_model_from_safetensors,
        load_safetensors,
        format_chat_messages,
        ChatMessage,
    )

    st = load_safetensors(model_path)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)

    print(f"  Layers: {model.config.num_layers}")
    print(f"  Hidden: {model.config.hidden_size}")
    print(f"  Heads: {model.config.num_heads} (Q), {model.config.num_kv_heads} (KV)")

    # Prepare prompt
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is 2+2?"),
    ]
    prompt = format_chat_messages(messages, model_type="qwen3")
    input_ids = tokenizer.encode(prompt).ids
    print(f"\n  Prompt tokens: {len(input_ids)}")

    max_new_tokens = 64

    # Warmup
    print("\nWarmup...")
    _ = model.generate(input_ids, max_new_tokens=5, use_cache=True)

    # Benchmark: Standard generate
    print("\n" + "-" * 50)
    print("Benchmark: model.generate() [standard]")
    print("-" * 50)

    start = time.perf_counter()
    output_standard = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        use_cache=True,
    )
    elapsed_standard = (time.perf_counter() - start) * 1000

    new_tokens_standard = len(output_standard) - len(input_ids)
    tps_standard = new_tokens_standard / (elapsed_standard / 1000)
    ms_per_token_standard = elapsed_standard / new_tokens_standard

    print(f"  Generated: {new_tokens_standard} tokens")
    print(f"  Time: {elapsed_standard:.0f} ms")
    print(f"  Speed: {tps_standard:.2f} tok/s ({ms_per_token_standard:.0f} ms/tok)")

    text_standard = tokenizer.decode(output_standard[len(input_ids):])
    print(f"  Output: {repr(text_standard[:100])}...")

    # Benchmark: generate_cuda_graph (fixed cache)
    print("\n" + "-" * 50)
    print("Benchmark: model.generate_cuda_graph() [fixed cache]")
    print("-" * 50)

    start = time.perf_counter()
    output_graph = model.generate_cuda_graph(
        input_ids,
        max_new_tokens=max_new_tokens,
        max_seq_len=512,
        temperature=0.7,
    )
    elapsed_graph = (time.perf_counter() - start) * 1000

    new_tokens_graph = len(output_graph) - len(input_ids)
    tps_graph = new_tokens_graph / (elapsed_graph / 1000)
    ms_per_token_graph = elapsed_graph / new_tokens_graph

    print(f"  Generated: {new_tokens_graph} tokens")
    print(f"  Time: {elapsed_graph:.0f} ms")
    print(f"  Speed: {tps_graph:.2f} tok/s ({ms_per_token_graph:.0f} ms/tok)")

    text_graph = tokenizer.decode(output_graph[len(input_ids):])
    print(f"  Output: {repr(text_graph[:100])}...")

    # Summary
    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"\n  Standard:    {tps_standard:.2f} tok/s ({ms_per_token_standard:.0f} ms/tok)")
    print(f"  Fixed Cache: {tps_graph:.2f} tok/s ({ms_per_token_graph:.0f} ms/tok)")

    speedup = tps_graph / tps_standard
    print(f"\n  Speedup: {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    exit(main())
