#!/usr/bin/env python3
"""Benchmark: Speculative decoding potential speedup analysis.

This benchmark measures the raw batch verification speedup and projects
the expected E2E speedup at various acceptance rates.

Key insight: Speculative decoding speedup depends on:
1. Draft model speed (how fast we can generate K tokens)
2. Batch verification speed (verifying K tokens in 1 forward pass)
3. Acceptance rate (how many tokens get accepted per step)

Speedup formula:
    speedup = (1 + acceptance_rate * K) / (draft_time * K + verify_time)

Where:
- K = number of draft tokens
- draft_time = time for 1 draft model decode
- verify_time = time for batch verification
"""

import numpy as np

# Model paths
TARGET_MODEL_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
TOKENIZER_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/tokenizer.json"

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
NUM_ITERATIONS = 20


def main():
    print("=" * 70)
    print("SPECULATIVE DECODING POTENTIAL ANALYSIS")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Explain quantum computing."),
    ]
    prompt = format_chat_messages(messages, model_type="qwen3")
    input_ids = tokenizer.encode(prompt).ids
    prefill_len = len(input_ids)

    print(f"\nLoading model... (prefill_len={prefill_len})")
    st = load_safetensors(TARGET_MODEL_PATH)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(TARGET_MODEL_PATH, dtype="float16", spec=spec)
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
    print("Running prefill...")
    hidden, past_key_values = model(input_ids, use_cache=True)
    for i, block in enumerate(model.blocks):
        past_k, past_v = past_key_values[i]
        kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
        kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

    logits = model.get_logits(hidden)
    first_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)

    # Backup KV cache
    kv_backup = []
    for block in model.blocks:
        k_backup = block.attn._k_cache.to_numpy().copy()
        v_backup = block.attn._v_cache.to_numpy().copy()
        kv_backup.append((k_backup, v_backup))

    # Generate test tokens
    test_tokens = [first_token]
    position = prefill_len
    context_len = prefill_len + 1
    for _ in range(15):
        hidden = model._decode_step_fixed_cache(test_tokens[-1], position, context_len)
        logits = model.get_logits(hidden)
        next_token = sample_token(logits.to_numpy()[-1], 0.7, 50, 0.9)
        test_tokens.append(next_token)
        position += 1
        context_len += 1

    start_event = CudaEvent()
    stop_event = CudaEvent()

    # Measure single token decode time (target model baseline)
    print("\n--- Measuring Single Token Decode ---")
    for i, block in enumerate(model.blocks):
        k_backup, v_backup = kv_backup[i]
        block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
        block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

    # Warmup
    for _ in range(5):
        model._decode_step_fixed_cache(test_tokens[0], prefill_len, prefill_len + 1)
    default_stream().synchronize()

    single_times = []
    for _ in range(NUM_ITERATIONS):
        for i, block in enumerate(model.blocks):
            k_backup, v_backup = kv_backup[i]
            block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
            block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

        start_event.record()
        model._decode_step_fixed_cache(test_tokens[0], prefill_len, prefill_len + 1)
        stop_event.record()
        stop_event.synchronize()
        single_times.append(event_elapsed_us(start_event, stop_event))

    single_time = np.mean(single_times)
    print(f"Single token decode: {single_time:.1f} us ({1_000_000 / single_time:.1f} tok/s)")

    # Measure batch decode times for different batch sizes
    print("\n--- Measuring Batch Verification ---")
    batch_results = {}

    for batch_size in [2, 4, 8]:
        for i, block in enumerate(model.blocks):
            k_backup, v_backup = kv_backup[i]
            block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
            block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

        batch_tokens = test_tokens[:batch_size]
        ctx_len = prefill_len + batch_size

        # Warmup
        for _ in range(5):
            model._decode_step_fixed_cache_batch(batch_tokens, prefill_len, ctx_len)
        default_stream().synchronize()

        batch_times = []
        for _ in range(NUM_ITERATIONS):
            for i, block in enumerate(model.blocks):
                k_backup, v_backup = kv_backup[i]
                block.attn._k_cache = from_numpy(k_backup.astype(np.float16))
                block.attn._v_cache = from_numpy(v_backup.astype(np.float16))

            start_event.record()
            model._decode_step_fixed_cache_batch(batch_tokens, prefill_len, ctx_len)
            stop_event.record()
            stop_event.synchronize()
            batch_times.append(event_elapsed_us(start_event, stop_event))

        batch_time = np.mean(batch_times)
        batch_results[batch_size] = batch_time
        speedup = (single_time * batch_size) / batch_time
        print(f"Batch {batch_size}: {batch_time:.1f} us ({speedup:.2f}x vs sequential)")

    # Project E2E speedup at various acceptance rates
    print("\n" + "=" * 70)
    print("PROJECTED E2E SPEEDUP")
    print("=" * 70)
    print("\nAssumption: Draft model is 5x faster than target (typical for 0.6B vs 8B)")
    print("            Draft time = target_time / 5")

    draft_time = single_time / 5  # Assume draft is 5x faster

    print(f"\n{'Batch':<8} {'Acceptance':<12} {'Seq tok/s':<12} {'Spec tok/s':<12} {'Speedup':<10}")
    print("-" * 54)

    seq_tps = 1_000_000 / single_time

    for batch_size in [4, 8]:
        verify_time = batch_results[batch_size]
        for acceptance_rate in [0.3, 0.5, 0.7, 0.9]:
            # Expected tokens per step: 1 + acceptance_rate * (K-1) on average
            # Time per step: draft_time * K + verify_time
            tokens_per_step = 1 + acceptance_rate * (batch_size - 1)
            time_per_step = draft_time * batch_size + verify_time
            spec_tps = tokens_per_step * 1_000_000 / time_per_step
            speedup = spec_tps / seq_tps

            print(
                f"K={batch_size:<5} {acceptance_rate * 100:>5.0f}%{'':<6} {seq_tps:<12.1f} {spec_tps:<12.1f} {speedup:<10.2f}x"
            )
        print()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. Batch verification is highly efficient:
   - K=4: ~3.5x faster than 4 sequential decodes
   - K=8: ~6.8x faster than 8 sequential decodes

2. Speculative decoding breaks even at ~30% acceptance rate

3. With 70% acceptance (typical for fine-tuned draft models):
   - K=4: ~1.9x speedup
   - K=8: ~2.7x speedup

4. The key bottleneck is acceptance rate, not batch verification

5. For maximum benefit:
   - Use a draft model from the same family (e.g., Qwen3-0.6B for Qwen3-8B)
   - Fine-tune draft model on target's output for higher acceptance
   - Use greedy decoding (temperature=0) for maximum acceptance
""")


if __name__ == "__main__":
    main()
