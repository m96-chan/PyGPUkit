"""Benchmark decode strategies vs legacy model methods.

Compares:
1. Legacy: model._decode_step_fixed_cache() (M=1 non-graph)
2. Strategy: DecodeM1.step() (M=1 non-graph)
3. Legacy Graph: model.init_decode_graph() + _decode_step_graph_replay()
4. Strategy Graph: DecodeM1.init_graph() + step_graph()
"""

import time
import warnings

import numpy as np

# Suppress deprecation warnings for legacy benchmarks
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
MAX_SEQ_LEN = 512
WARMUP_TOKENS = 10
BENCH_TOKENS = 50


def init_kv_caches(model, max_seq_len: int, dtype: str):
    """Initialize KV caches for all layers."""
    for block in model.blocks:
        block.attn.init_fixed_cache(max_seq_len, dtype=dtype)


def prefill_model(model, input_ids, prefill_buffers):
    """Run prefill and copy KV to fixed caches."""
    from pygpukit.ops.basic import kv_cache_prefill_gqa

    hidden, past_key_values = model._prefill_with_buffers(
        input_ids, prefill_buffers, use_cache=True
    )

    for i, block in enumerate(model.blocks):
        past_k, past_v = past_key_values[i]
        kv_cache_prefill_gqa(past_k, block.attn._k_cache, block.attn.num_heads, start_pos=0)
        kv_cache_prefill_gqa(past_v, block.attn._v_cache, block.attn.num_heads, start_pos=0)

    return hidden


def main():
    print("=" * 60)
    print("Strategy Pattern Benchmark")
    print("=" * 60)
    print("Model: Qwen2.5-7B-Instruct")
    print(f"Max seq len: {MAX_SEQ_LEN}")
    print(f"Warmup: {WARMUP_TOKENS} tokens, Bench: {BENCH_TOKENS} tokens")
    print()

    # Load model
    print("Loading model...")
    t0 = time.perf_counter()

    from pygpukit.core import default_stream
    from pygpukit.core.factory import from_numpy
    from pygpukit.llm import load_model_from_safetensors
    from pygpukit.llm.buffers import DecodeBuffers, PrefillBuffers
    from pygpukit.llm.layers import precompute_freqs_cis

    model = load_model_from_safetensors(
        f"{MODEL_PATH}/model.safetensors.index.json",
        dtype="bfloat16",
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  Layers: {len(model.blocks)}, Hidden: {model.config.hidden_size}")

    # Get dtype and other params
    dtype = str(model.embed_tokens.dtype)
    use_qk_norm = model.spec is not None and model.spec.use_qk_norm
    lm_head = model._lm_head if model._lm_head is not None else model.embed_tokens
    vocab_size = lm_head.shape[0]

    # Dummy prompt tokens
    prompt_tokens = list(range(10))
    prefill_len = len(prompt_tokens)

    # Initialize KV cache
    print("\nInitializing KV cache...")
    init_kv_caches(model, MAX_SEQ_LEN, dtype)

    # Pre-compute RoPE tables
    if model.config.use_rope:
        cos_np, sin_np = precompute_freqs_cis(
            model.config.head_dim, MAX_SEQ_LEN, model.config.rope_theta
        )
        np_dtype = np.float16 if dtype == "float16" else np.float32
        model._rope_cos_gpu = from_numpy(cos_np.astype(np_dtype))
        model._rope_sin_gpu = from_numpy(sin_np.astype(np_dtype))

    # Allocate prefill buffers
    prefill_buffers = PrefillBuffers.allocate(
        model.config, max_seq_len=prefill_len, dtype=dtype, use_qk_norm=use_qk_norm
    )

    # Allocate decode buffers (used by strategy)
    decode_buffers = DecodeBuffers.allocate(
        model.config, dtype=dtype, use_qk_norm=use_qk_norm, vocab_size=vocab_size
    )

    # Prefill
    print("Prefilling...")
    prefill_model(model, prompt_tokens, prefill_buffers)

    # =========================================================================
    # Benchmark 1: Legacy M=1 (non-graph)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Benchmark 1: Legacy M=1 (model._decode_step_fixed_cache)")
    print("=" * 60)

    # Re-init caches and prefill
    init_kv_caches(model, MAX_SEQ_LEN, dtype)
    prefill_model(model, prompt_tokens, prefill_buffers)

    # Warmup
    position = prefill_len
    context_len = prefill_len + 1
    token = 1000

    for _ in range(WARMUP_TOKENS):
        model._decode_step_fixed_cache(token, position, context_len)
        position += 1
        context_len += 1

    # Benchmark
    default_stream().synchronize()

    t_start = time.perf_counter()
    for i in range(BENCH_TOKENS):
        model._decode_step_fixed_cache(token + i, position, context_len)
        position += 1
        context_len += 1
    default_stream().synchronize()
    t_legacy_m1 = time.perf_counter() - t_start

    tps_legacy_m1 = BENCH_TOKENS / t_legacy_m1
    print(f"  Time: {t_legacy_m1:.3f}s")
    print(f"  Throughput: {tps_legacy_m1:.1f} tok/s")

    # =========================================================================
    # Benchmark 2: Strategy M=1 (non-graph)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Benchmark 2: Strategy M=1 (DecodeM1.step)")
    print("=" * 60)

    from pygpukit.llm import DecodeM1

    m1 = DecodeM1()
    m1.bind(model)

    # Re-init caches and prefill
    init_kv_caches(model, MAX_SEQ_LEN, dtype)
    prefill_model(model, prompt_tokens, prefill_buffers)

    # Warmup
    position = prefill_len
    context_len = prefill_len + 1

    for _ in range(WARMUP_TOKENS):
        m1.step(token, position, context_len, decode_buffers)
        position += 1
        context_len += 1

    # Benchmark
    default_stream().synchronize()

    t_start = time.perf_counter()
    for i in range(BENCH_TOKENS):
        m1.step(token + i, position, context_len, decode_buffers)
        position += 1
        context_len += 1
    default_stream().synchronize()
    t_strategy_m1 = time.perf_counter() - t_start

    tps_strategy_m1 = BENCH_TOKENS / t_strategy_m1
    print(f"  Time: {t_strategy_m1:.3f}s")
    print(f"  Throughput: {tps_strategy_m1:.1f} tok/s")

    # =========================================================================
    # Benchmark 3: Legacy CUDA Graph
    # =========================================================================
    print("\n" + "=" * 60)
    print("Benchmark 3: Legacy CUDA Graph (model.init_decode_graph)")
    print("=" * 60)

    t_legacy_graph = None
    tps_legacy_graph = None

    try:
        # Re-init caches and prefill
        init_kv_caches(model, MAX_SEQ_LEN, dtype)
        prefill_model(model, prompt_tokens, prefill_buffers)

        # Initialize legacy graph
        model.init_decode_graph(MAX_SEQ_LEN)

        # Warmup
        position = prefill_len
        context_len = prefill_len + 1

        for _ in range(WARMUP_TOKENS):
            model._decode_step_graph_replay(token, position, context_len)
            position += 1
            context_len += 1

        # Benchmark
        default_stream().synchronize()

        t_start = time.perf_counter()
        for i in range(BENCH_TOKENS):
            model._decode_step_graph_replay(token + i, position, context_len)
            position += 1
            context_len += 1
        default_stream().synchronize()
        t_legacy_graph = time.perf_counter() - t_start

        tps_legacy_graph = BENCH_TOKENS / t_legacy_graph
        print(f"  Time: {t_legacy_graph:.3f}s")
        print(f"  Throughput: {tps_legacy_graph:.1f} tok/s")
    except RuntimeError as e:
        print(f"  SKIPPED: {e}")

    # =========================================================================
    # Benchmark 4: Strategy CUDA Graph
    # =========================================================================
    print("\n" + "=" * 60)
    print("Benchmark 4: Strategy CUDA Graph (DecodeM1.init_graph)")
    print("=" * 60)

    t_strategy_graph = None
    tps_strategy_graph = None

    try:
        # Re-init caches and prefill
        init_kv_caches(model, MAX_SEQ_LEN, dtype)
        prefill_model(model, prompt_tokens, prefill_buffers)

        # Create new strategy and init graph
        m1_graph = DecodeM1()
        m1_graph.bind(model)
        m1_graph.init_graph(MAX_SEQ_LEN)

        # Warmup
        position = prefill_len
        context_len = prefill_len + 1

        for _ in range(WARMUP_TOKENS):
            m1_graph.step_graph(token, position, context_len)
            position += 1
            context_len += 1

        # Benchmark
        default_stream().synchronize()

        t_start = time.perf_counter()
        for i in range(BENCH_TOKENS):
            m1_graph.step_graph(token + i, position, context_len)
            position += 1
            context_len += 1
        default_stream().synchronize()
        t_strategy_graph = time.perf_counter() - t_start

        tps_strategy_graph = BENCH_TOKENS / t_strategy_graph
        print(f"  Time: {t_strategy_graph:.3f}s")
        print(f"  Throughput: {tps_strategy_graph:.1f} tok/s")
    except RuntimeError as e:
        print(f"  SKIPPED: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Method':<40} {'Time (s)':<12} {'tok/s':<10}")
    print("-" * 60)
    print(f"{'Legacy M=1 (non-graph)':<40} {t_legacy_m1:<12.3f} {tps_legacy_m1:<10.1f}")
    print(f"{'Strategy M=1 (non-graph)':<40} {t_strategy_m1:<12.3f} {tps_strategy_m1:<10.1f}")
    if t_legacy_graph is not None:
        print(f"{'Legacy CUDA Graph':<40} {t_legacy_graph:<12.3f} {tps_legacy_graph:<10.1f}")
    else:
        print(f"{'Legacy CUDA Graph':<40} {'SKIPPED':<12} {'N/A':<10}")
    if t_strategy_graph is not None:
        print(f"{'Strategy CUDA Graph':<40} {t_strategy_graph:<12.3f} {tps_strategy_graph:<10.1f}")
    else:
        print(f"{'Strategy CUDA Graph':<40} {'SKIPPED':<12} {'N/A':<10}")
    print()

    # Calculate overhead
    overhead_m1 = (t_strategy_m1 - t_legacy_m1) / t_legacy_m1 * 100
    print(f"Strategy overhead (M=1): {overhead_m1:+.1f}%")
    if t_legacy_graph is not None and t_strategy_graph is not None:
        overhead_graph = (t_strategy_graph - t_legacy_graph) / t_legacy_graph * 100
        print(f"Strategy overhead (Graph): {overhead_graph:+.1f}%")
    else:
        print("Strategy overhead (Graph): N/A (CUDA Graph tests skipped)")


if __name__ == "__main__":
    main()
