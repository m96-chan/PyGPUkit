"""Benchmark all decode strategies.

Compares:
1. DecodeM1 - Single token decode (baseline)
2. DecodeBatch - Batch decode
3. DecodeSpeculative - Self-speculative (early layers as draft)
4. DecodeJacobi - Parallel iterative decode
"""

import time
import warnings

import numpy as np

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_PATH = "C:/Users/y_har/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
MAX_SEQ_LEN = 512
WARMUP_TOKENS = 5
BENCH_TOKENS = 30


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
    print("=" * 70)
    print("All Decode Strategies Benchmark")
    print("=" * 70)
    print("Model: Qwen2.5-7B-Instruct (bfloat16)")
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

    # Allocate buffers
    prefill_buffers = PrefillBuffers.allocate(
        model.config, max_seq_len=prefill_len, dtype=dtype, use_qk_norm=use_qk_norm
    )
    decode_buffers = DecodeBuffers.allocate(
        model.config, dtype=dtype, use_qk_norm=use_qk_norm, vocab_size=vocab_size
    )

    # Prefill
    print("Prefilling...")
    prefill_model(model, prompt_tokens, prefill_buffers)

    results = {}

    # =========================================================================
    # Benchmark 1: DecodeM1 (baseline)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Benchmark 1: DecodeM1 (single token decode - baseline)")
    print("=" * 70)

    from pygpukit.llm import DecodeM1

    m1 = DecodeM1()
    m1.bind(model)

    init_kv_caches(model, MAX_SEQ_LEN, dtype)
    prefill_model(model, prompt_tokens, prefill_buffers)

    position = prefill_len
    context_len = prefill_len + 1
    token = 1000

    # Warmup
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
    t_m1 = time.perf_counter() - t_start

    tps_m1 = BENCH_TOKENS / t_m1
    results["DecodeM1"] = {"time": t_m1, "tps": tps_m1, "tokens": BENCH_TOKENS}
    print(f"  Time: {t_m1:.3f}s")
    print(f"  Throughput: {tps_m1:.1f} tok/s")

    # =========================================================================
    # Benchmark 2: DecodeBatch
    # =========================================================================
    print("\n" + "=" * 70)
    print("Benchmark 2: DecodeBatch (batch=8 tokens at once)")
    print("=" * 70)

    from pygpukit.llm import DecodeBatch

    try:
        batch_size = 8
        batch = DecodeBatch(batch_size=batch_size)
        batch.bind(model)

        # Allocate batch buffers
        batch_buffers = DecodeBuffers.allocate(
            model.config,
            dtype=dtype,
            use_qk_norm=use_qk_norm,
            vocab_size=vocab_size,
            max_batch_size=batch_size,
        )

        init_kv_caches(model, MAX_SEQ_LEN, dtype)
        prefill_model(model, prompt_tokens, prefill_buffers)

        position = prefill_len
        context_len = prefill_len + batch_size

        # Calculate how many batch steps
        batch_steps = BENCH_TOKENS // batch_size

        # Warmup
        for _ in range(2):
            token_ids = list(range(1000, 1000 + batch_size))
            batch.step_batch(token_ids, position, context_len, batch_buffers)
            position += batch_size
            context_len += batch_size

        # Reset for benchmark
        init_kv_caches(model, MAX_SEQ_LEN, dtype)
        prefill_model(model, prompt_tokens, prefill_buffers)
        position = prefill_len
        context_len = prefill_len + batch_size

        # Benchmark
        default_stream().synchronize()
        t_start = time.perf_counter()
        total_tokens = 0
        for step in range(batch_steps):
            token_ids = list(range(1000 + step * batch_size, 1000 + (step + 1) * batch_size))
            batch.step_batch(token_ids, position, context_len, batch_buffers)
            position += batch_size
            context_len += batch_size
            total_tokens += batch_size
        default_stream().synchronize()
        t_batch = time.perf_counter() - t_start

        tps_batch = total_tokens / t_batch
        results["DecodeBatch"] = {"time": t_batch, "tps": tps_batch, "tokens": total_tokens}
        print(f"  Batch size: {batch_size}")
        print(f"  Tokens processed: {total_tokens}")
        print(f"  Time: {t_batch:.3f}s")
        print(f"  Throughput: {tps_batch:.1f} tok/s")
    except Exception as e:
        print(f"  SKIPPED: {e}")
        results["DecodeBatch"] = None

    # =========================================================================
    # Benchmark 3: DecodeSpeculative (self-speculative)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Benchmark 3: DecodeSpeculative (self-speculative, draft_layers=8)")
    print("=" * 70)

    from pygpukit.llm import DecodeSpeculative

    try:
        spec = DecodeSpeculative(max_draft_tokens=4, draft_layers=8)
        spec.bind(model)

        init_kv_caches(model, MAX_SEQ_LEN, dtype)
        prefill_model(model, prompt_tokens, prefill_buffers)

        position = prefill_len
        context_len = prefill_len + 1
        token = 1000

        # Warmup
        for _ in range(2):
            accepted, new_pos, stats = spec.step_speculative(token, position, context_len)
            token = accepted[-1] if accepted else token + 1
            position = new_pos
            context_len = new_pos + 1

        # Reset for benchmark
        init_kv_caches(model, MAX_SEQ_LEN, dtype)
        prefill_model(model, prompt_tokens, prefill_buffers)
        position = prefill_len
        context_len = prefill_len + 1
        token = 1000

        # Benchmark
        default_stream().synchronize()
        t_start = time.perf_counter()
        total_tokens = 0
        total_accepted = 0
        total_drafted = 0
        iterations = 0

        while total_tokens < BENCH_TOKENS:
            accepted, new_pos, stats = spec.step_speculative(token, position, context_len)
            total_tokens += len(accepted)
            total_accepted += stats.get("accepted_count", len(accepted))
            total_drafted += stats.get("draft_count", 4)
            token = accepted[-1] if accepted else token + 1
            position = new_pos
            context_len = new_pos + 1
            iterations += 1

        default_stream().synchronize()
        t_spec = time.perf_counter() - t_start

        tps_spec = total_tokens / t_spec
        accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0
        results["DecodeSpeculative"] = {
            "time": t_spec,
            "tps": tps_spec,
            "tokens": total_tokens,
            "accept_rate": accept_rate,
            "iterations": iterations,
        }
        print(f"  Tokens generated: {total_tokens}")
        print(f"  Iterations: {iterations} (avg {total_tokens / iterations:.1f} tok/iter)")
        print(f"  Accept rate: {accept_rate:.1%}")
        print(f"  Time: {t_spec:.3f}s")
        print(f"  Throughput: {tps_spec:.1f} tok/s")
    except Exception as e:
        print(f"  SKIPPED: {e}")
        results["DecodeSpeculative"] = None

    # =========================================================================
    # Benchmark 4: DecodeJacobi
    # =========================================================================
    print("\n" + "=" * 70)
    print("Benchmark 4: DecodeJacobi (parallel iterative, n_tokens=4)")
    print("=" * 70)

    from pygpukit.llm import DecodeJacobi

    try:
        jacobi = DecodeJacobi(n_tokens=4, max_iter=3, init_strategy="repeat")
        jacobi.bind(model)

        init_kv_caches(model, MAX_SEQ_LEN, dtype)
        prefill_model(model, prompt_tokens, prefill_buffers)

        position = prefill_len
        context_len = prefill_len + 1
        token = 1000

        # Warmup
        for _ in range(2):
            accepted, new_pos, stats = jacobi.step_jacobi(token, position, context_len)
            token = accepted[-1] if accepted else token + 1
            position = new_pos
            context_len = new_pos + 1

        # Reset for benchmark
        init_kv_caches(model, MAX_SEQ_LEN, dtype)
        prefill_model(model, prompt_tokens, prefill_buffers)
        position = prefill_len
        context_len = prefill_len + 1
        token = 1000

        # Benchmark
        default_stream().synchronize()
        t_start = time.perf_counter()
        total_tokens = 0
        total_converged = 0
        iterations = 0

        while total_tokens < BENCH_TOKENS:
            accepted, new_pos, stats = jacobi.step_jacobi(token, position, context_len)
            total_tokens += len(accepted)
            if stats.get("converged", False):
                total_converged += 1
            token = accepted[-1] if accepted else token + 1
            position = new_pos
            context_len = new_pos + 1
            iterations += 1

        default_stream().synchronize()
        t_jacobi = time.perf_counter() - t_start

        tps_jacobi = total_tokens / t_jacobi
        converge_rate = total_converged / iterations if iterations > 0 else 0
        results["DecodeJacobi"] = {
            "time": t_jacobi,
            "tps": tps_jacobi,
            "tokens": total_tokens,
            "converge_rate": converge_rate,
            "iterations": iterations,
        }
        print(f"  Tokens generated: {total_tokens}")
        print(f"  Iterations: {iterations} (avg {total_tokens / iterations:.1f} tok/iter)")
        print(f"  Convergence rate: {converge_rate:.1%}")
        print(f"  Time: {t_jacobi:.3f}s")
        print(f"  Throughput: {tps_jacobi:.1f} tok/s")
    except Exception as e:
        print(f"  SKIPPED: {e}")
        results["DecodeJacobi"] = None

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Tokens':<10} {'Time (s)':<12} {'tok/s':<10} {'Speedup':<10}")
    print("-" * 70)

    baseline_tps = results["DecodeM1"]["tps"]

    for name, data in results.items():
        if data is None:
            print(f"{name:<25} {'SKIPPED':<10}")
        else:
            speedup = data["tps"] / baseline_tps
            print(
                f"{name:<25} {data['tokens']:<10} {data['time']:<12.3f} {data['tps']:<10.1f} {speedup:<10.2f}x"
            )

    print()
    print("Notes:")
    print("- DecodeM1: Single token per step (baseline)")
    print("- DecodeBatch: Process multiple tokens in parallel")
    print("- DecodeSpeculative: Self-speculative using early layers as draft")
    print("- DecodeJacobi: Parallel iterative refinement without draft model")


if __name__ == "__main__":
    main()
