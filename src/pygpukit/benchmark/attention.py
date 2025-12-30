"""Attention benchmarks."""

from __future__ import annotations

import numpy as np

from .base import Benchmark
from .results import BenchmarkResult


class SDPABenchmark(Benchmark):
    """Scaled Dot-Product Attention benchmark."""

    category = "attention"

    def __init__(
        self,
        seq_lens: list[int] | None = None,
        num_heads: int = 32,
        head_dim: int = 128,
        warmup: int = 10,
        iterations: int = 50,
    ):
        super().__init__(warmup=warmup, iterations=iterations)
        self.seq_lens = seq_lens or [512, 1024, 2048, 4096]
        self.num_heads = num_heads
        self.head_dim = head_dim

    def run(self) -> list[BenchmarkResult]:
        """Run SDPA benchmarks."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        results = []

        for seq_len in self.seq_lens:
            try:
                result = self._benchmark_sdpa(native, seq_len)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  SDPA seq_len={seq_len}: ERROR - {e}")

        return results

    def _benchmark_sdpa(
        self,
        native: object,
        seq_len: int,
    ) -> BenchmarkResult | None:
        """Benchmark SDPA for a given sequence length."""
        import pygpukit as gk

        name = f"sdpa_seq{seq_len}"
        params = {
            "seq_len": seq_len,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
        }

        # Attention FLOPs: 4 * seq_len^2 * head_dim * num_heads
        # (Q@K^T and attn@V, each 2*seq*seq*dim)
        flops = 4.0 * seq_len * seq_len * self.head_dim * self.num_heads

        # Create Q, K, V
        Q = gk.empty((self.num_heads, seq_len, self.head_dim), dtype="bfloat16")
        K = gk.empty((self.num_heads, seq_len, self.head_dim), dtype="bfloat16")
        V = gk.empty((self.num_heads, seq_len, self.head_dim), dtype="bfloat16")
        O = gk.empty((self.num_heads, seq_len, self.head_dim), dtype="bfloat16")

        # Check if native SDPA available
        if not hasattr(native, "sdpa_causal_bf16"):
            return None

        def run_fn() -> None:
            native.sdpa_causal_bf16(
                Q._get_native(),
                K._get_native(),
                V._get_native(),
                O._get_native(),
            )

        return self._measure(name, run_fn, params, flops=flops)


class GQABenchmark(Benchmark):
    """Grouped Query Attention benchmark."""

    category = "attention"

    def __init__(
        self,
        seq_lens: list[int] | None = None,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        warmup: int = 10,
        iterations: int = 50,
    ):
        super().__init__(warmup=warmup, iterations=iterations)
        self.seq_lens = seq_lens or [512, 1024, 2048]
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def run(self) -> list[BenchmarkResult]:
        """Run GQA benchmarks."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        results = []

        for seq_len in self.seq_lens:
            try:
                result = self._benchmark_gqa(native, seq_len)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  GQA seq_len={seq_len}: ERROR - {e}")

        return results

    def _benchmark_gqa(
        self,
        native: object,
        seq_len: int,
    ) -> BenchmarkResult | None:
        """Benchmark GQA."""
        import pygpukit as gk

        name = f"gqa_seq{seq_len}"
        params = {
            "seq_len": seq_len,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }

        # GQA FLOPs (KV heads broadcasted)
        flops = 4.0 * seq_len * seq_len * self.head_dim * self.num_heads

        Q = gk.empty((self.num_heads, seq_len, self.head_dim), dtype="bfloat16")
        K = gk.empty((self.num_kv_heads, seq_len, self.head_dim), dtype="bfloat16")
        V = gk.empty((self.num_kv_heads, seq_len, self.head_dim), dtype="bfloat16")
        O = gk.empty((self.num_heads, seq_len, self.head_dim), dtype="bfloat16")

        if not hasattr(native, "sdpa_causal_gqa_bf16"):
            return None

        def run_fn() -> None:
            native.sdpa_causal_gqa_bf16(
                Q._get_native(),
                K._get_native(),
                V._get_native(),
                O._get_native(),
                self.num_heads // self.num_kv_heads,
            )

        return self._measure(name, run_fn, params, flops=flops)
