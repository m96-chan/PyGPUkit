#!/usr/bin/env python3
"""
Benchmark: Standard vs Fixed Cache KV Cache Strategies

Compares:
1. Standard: Dynamic KV cache (grows with sequence)
2. Fixed Cache: Fixed-length KV cache (pre-allocated, GQA-expanded)

The Fixed Cache strategy is the foundation for CUDA Graph optimization,
which requires deterministic memory layouts and zero allocations during decode.
"""

import argparse
import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    tokens: int
    time_ms: float
    tps: float
    ms_per_token: float
    output_text: str


def run_benchmark(
    model,
    tokenizer,
    input_ids: list[int],
    max_new_tokens: int,
    num_runs: int = 3,
) -> tuple[list[BenchmarkResult], list[BenchmarkResult]]:
    """Run benchmark comparing standard vs fixed cache."""
    standard_results = []
    fixed_results = []

    for run in range(num_runs):
        # Standard generate
        start = time.perf_counter()
        output_standard = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            use_cache=True,
        )
        elapsed_standard = (time.perf_counter() - start) * 1000

        new_tokens = len(output_standard) - len(input_ids)
        tps = new_tokens / (elapsed_standard / 1000)
        ms_per_tok = elapsed_standard / new_tokens
        text = tokenizer.decode(output_standard[len(input_ids) :])

        standard_results.append(
            BenchmarkResult(
                name="Standard",
                tokens=new_tokens,
                time_ms=elapsed_standard,
                tps=tps,
                ms_per_token=ms_per_tok,
                output_text=text,
            )
        )

        # Fixed cache generate
        start = time.perf_counter()
        output_fixed = model.generate_cuda_graph(
            input_ids,
            max_new_tokens=max_new_tokens,
            max_seq_len=512,
            temperature=0.7,
        )
        elapsed_fixed = (time.perf_counter() - start) * 1000

        new_tokens = len(output_fixed) - len(input_ids)
        tps = new_tokens / (elapsed_fixed / 1000)
        ms_per_tok = elapsed_fixed / new_tokens
        text = tokenizer.decode(output_fixed[len(input_ids) :])

        fixed_results.append(
            BenchmarkResult(
                name="Fixed Cache",
                tokens=new_tokens,
                time_ms=elapsed_fixed,
                tps=tps,
                ms_per_token=ms_per_tok,
                output_text=text,
            )
        )

    return standard_results, fixed_results


def print_results(
    standard: list[BenchmarkResult],
    fixed: list[BenchmarkResult],
    show_output: bool = False,
):
    """Print benchmark results with statistics."""
    print("\n" + "=" * 70)
    print(" Benchmark Results")
    print("=" * 70)

    # Standard results
    avg_tps_std = sum(r.tps for r in standard) / len(standard)
    avg_ms_std = sum(r.ms_per_token for r in standard) / len(standard)
    print("\n  Standard (dynamic KV cache):")
    print(f"    Average: {avg_tps_std:.2f} tok/s ({avg_ms_std:.0f} ms/tok)")
    for i, r in enumerate(standard):
        print(f"    Run {i + 1}: {r.tps:.2f} tok/s ({r.time_ms:.0f} ms, {r.tokens} tokens)")
    if show_output:
        print(f"    Output: {standard[-1].output_text[:80]}...")

    # Fixed cache results
    avg_tps_fix = sum(r.tps for r in fixed) / len(fixed)
    avg_ms_fix = sum(r.ms_per_token for r in fixed) / len(fixed)
    print("\n  Fixed Cache (pre-allocated, GQA-expanded):")
    print(f"    Average: {avg_tps_fix:.2f} tok/s ({avg_ms_fix:.0f} ms/tok)")
    for i, r in enumerate(fixed):
        print(f"    Run {i + 1}: {r.tps:.2f} tok/s ({r.time_ms:.0f} ms, {r.tokens} tokens)")
    if show_output:
        print(f"    Output: {fixed[-1].output_text[:80]}...")

    # Summary
    speedup = avg_tps_fix / avg_tps_std
    print("\n" + "-" * 70)
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Fixed Cache is {(speedup - 1) * 100:.1f}% faster than Standard")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache strategies")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--output", action="store_true", help="Show generated output text")
    args = parser.parse_args()

    model_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/model.safetensors.index.json"
    tokenizer_path = "C:/Users/y_har/.cache/huggingface/hub/models--Aratako--Qwen3-8B-ERP-v0.1/snapshots/8311aa4482f02c2de93872e4979887def1841faf/tokenizer.json"

    print("=" * 70)
    print(" PyGPUkit LLM Benchmark: Standard vs Fixed Cache")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load model
    print("Loading model...")
    from pygpukit.llm import (
        ChatMessage,
        detect_model_spec,
        format_chat_messages,
        load_model_from_safetensors,
        load_safetensors,
    )

    st = load_safetensors(model_path)
    spec = detect_model_spec(st.tensor_names)
    model = load_model_from_safetensors(model_path, dtype="float16", spec=spec)

    print("  Model: Qwen3-8B")
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
    print(f"  Max new tokens: {args.tokens}")
    print(f"  Benchmark runs: {args.runs}")

    # Warmup
    print("\nWarmup...")
    _ = model.generate(input_ids, max_new_tokens=5, use_cache=True)

    # Run benchmark
    print(f"\nRunning {args.runs} benchmark iterations...")
    standard_results, fixed_results = run_benchmark(
        model, tokenizer, input_ids, args.tokens, args.runs
    )

    # Print results
    print_results(standard_results, fixed_results, show_output=args.output)

    return 0


if __name__ == "__main__":
    exit(main())
