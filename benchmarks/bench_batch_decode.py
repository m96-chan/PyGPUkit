#!/usr/bin/env python3
"""Benchmark batch decode vs sequential decode performance."""

import numpy as np

model_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
tokenizer_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/tokenizer.json"

from tokenizers import Tokenizer

from pygpukit import CudaEvent, event_elapsed_us
from pygpukit.core import default_stream, from_numpy
from pygpukit.llm import (
    ChatMessage,
    detect_model_spec,
    format_chat_messages,
    load_model_from_safetensors,
    load_safetensors,
)
from pygpukit.llm.model import precompute_freqs_cis, sample_token
from pygpukit.ops.basic import kv_cache_prefill_gqa

MAX_SEQ_LEN = 512
NUM_ITERATIONS = 10


def main():
    print("=" * 70)
    print("BATCH DECODE PERFORMANCE BENCHMARK")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is 2+2?"),
    ]
    prompt = format_chat_messages(messages, model_type="qwen3")
    input_ids = tokenizer.encode(prompt).ids
    prefill_len = len(input_ids)

    print(f"\nLoading model... (prefill_len={prefill_len})")
    st = load_safetensors(model_path)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)
    dtype = str(model.embed_tokens.dtype)

    print("Initializing KV cache...")
    for block in model.blocks:
        block.attn.init_fixed_cache(MAX_SEQ_LEN, dtype=dtype)

    if model.config.use_rope:
        cos_np, sin_np = precompute_freqs_cis(
            model.config.head_dim, MAX_SEQ_LEN, model.config.rope_theta
        )
        np_dtype = np.float16 if dtype == "float16" else np.float32
        model._rope_cos_gpu = from_numpy(cos_np.astype(np_dtype))
        model._rope_sin_gpu = from_numpy(sin_np.astype(np_dtype))

    # Prefill
    print("\nRunning prefill...")
    hidden, past_key_values = model(input_ids, use_cache=True)
    for i, block in enumerate(model.blocks):
        past_k, past_v = past_key_values[i]
        kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
        kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

    logits = model.get_logits(hidden)
    last_logits = logits.to_numpy()[-1]
    first_token = sample_token(last_logits, 0.7, 50, 0.9)

    # Store KV cache after prefill
    kv_cache_backup = []
    for block in model.blocks:
        k_backup = block.attn._k_cache.to_numpy().copy()
        v_backup = block.attn._v_cache.to_numpy().copy()
        kv_cache_backup.append((k_backup, v_backup))

    # Generate some tokens for testing
    print("\nGenerating test tokens...")
    test_tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1
    for _ in range(7):  # Generate 7 more tokens (total 8)
        hidden = model._decode_step_fixed_cache(test_tokens[-1], position, context_len)
        logits = model.get_logits(hidden)
        next_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)
        test_tokens.append(next_token)
        position += 1
        context_len += 1

    print(f"Test tokens: {test_tokens}")

    # Benchmark different batch sizes
    batch_sizes = [1, 2, 4, 8]
    results = {}

    start_event = CudaEvent()
    stop_event = CudaEvent()

    for batch_size in batch_sizes:
        if batch_size > len(test_tokens):
            continue

        print(f"\n--- Batch size: {batch_size} ---")

        # Restore KV cache
        for i, block in enumerate(model.blocks):
            k_backup, v_backup = kv_cache_backup[i]
            block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
            block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

        batch_tokens = test_tokens[:batch_size]
        start_pos = prefill_len
        ctx_len = prefill_len + batch_size

        # Warmup
        for _ in range(3):
            if batch_size == 1:
                model._decode_step_fixed_cache(batch_tokens[0], start_pos, start_pos + 1)
            else:
                model._decode_step_fixed_cache_batch(batch_tokens, start_pos, ctx_len)
        default_stream().synchronize()

        # Benchmark
        times = []
        for _ in range(NUM_ITERATIONS):
            # Restore cache each iteration
            for i, block in enumerate(model.blocks):
                k_backup, v_backup = kv_cache_backup[i]
                block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
                block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

            start_event.record()

            if batch_size == 1:
                model._decode_step_fixed_cache(batch_tokens[0], start_pos, start_pos + 1)
            else:
                model._decode_step_fixed_cache_batch(batch_tokens, start_pos, ctx_len)

            stop_event.record()
            stop_event.synchronize()

            elapsed = event_elapsed_us(start_event, stop_event)
            times.append(elapsed)

        mean_time = np.mean(times)
        time_per_token = mean_time / batch_size
        results[batch_size] = {
            "total_us": mean_time,
            "per_token_us": time_per_token,
        }

        print(f"  Total time: {mean_time:.1f} us")
        print(f"  Per token:  {time_per_token:.1f} us")
        print(f"  Throughput: {1_000_000 / time_per_token:.1f} tok/s (theoretical)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Batch Size':>12} {'Total (us)':>12} {'Per Token (us)':>15} {'Speedup':>10}")
    print("-" * 55)

    baseline = results.get(1, {}).get("per_token_us", 1)
    for batch_size in batch_sizes:
        if batch_size not in results:
            continue
        total = results[batch_size]["total_us"]
        per_tok = results[batch_size]["per_token_us"]
        speedup = baseline / per_tok if per_tok > 0 else 0
        print(f"{batch_size:>12} {total:>12.1f} {per_tok:>15.1f} {speedup:>10.2f}x")


if __name__ == "__main__":
    main()
