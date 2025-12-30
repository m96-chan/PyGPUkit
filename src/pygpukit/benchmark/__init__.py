"""PyGPUkit Benchmark Suite.

Usage:
    from pygpukit.benchmark import BenchmarkSuite

    suite = BenchmarkSuite()
    suite.add_gemm()
    suite.add_gemv()
    report = suite.run()
    report.save("baseline.json")

    # Compare with baseline
    comparison = suite.compare("baseline.json")
    if comparison.has_regression():
        raise RuntimeError("Performance regression detected!")
"""

from __future__ import annotations

from pathlib import Path

from .attention import GQABenchmark, SDPABenchmark
from .base import Benchmark, get_gpu_info, measure_kernel
from .gemm import FP8GEMMBenchmark, GEMMBenchmark
from .gemv import GEMVBenchmark, W8A8GEMVBenchmark
from .results import (
    BenchmarkReport,
    BenchmarkResult,
    ComparisonResult,
    GPUInfo,
    Regression,
    compare_reports,
)

__all__ = [
    "BenchmarkSuite",
    "BenchmarkReport",
    "BenchmarkResult",
    "ComparisonResult",
    "GPUInfo",
    "Regression",
    "Benchmark",
    "GEMMBenchmark",
    "FP8GEMMBenchmark",
    "GEMVBenchmark",
    "W8A8GEMVBenchmark",
    "SDPABenchmark",
    "GQABenchmark",
    "get_gpu_info",
    "measure_kernel",
    "compare_reports",
]


class BenchmarkSuite:
    """Unified benchmark suite for PyGPUkit.

    Example:
        suite = BenchmarkSuite()
        suite.add_gemm(sizes=[(4096, 4096, 4096)])
        suite.add_gemv()
        report = suite.run()
        report.save("results.json")
    """

    def __init__(self, warmup: int = 10, iterations: int = 50, quick: bool = False):
        """Initialize benchmark suite.

        Args:
            warmup: Number of warmup iterations
            iterations: Number of timed iterations
            quick: If True, use reduced warmup/iterations
        """
        if quick:
            warmup = 5
            iterations = 20
        self.warmup = warmup
        self.iterations = iterations
        self.benchmarks: list[Benchmark] = []

    def add_gemm(
        self,
        sizes: list[tuple[int, int, int]] | None = None,
        dtypes: list[str] | None = None,
    ) -> BenchmarkSuite:
        """Add GEMM benchmark.

        Args:
            sizes: List of (M, K, N) tuples
            dtypes: List of dtypes to benchmark (fp32, tf32, bf16, fp16)
        """
        self.benchmarks.append(
            GEMMBenchmark(
                sizes=sizes,
                dtypes=dtypes,
                warmup=self.warmup,
                iterations=self.iterations,
            )
        )
        return self

    def add_fp8_gemm(
        self,
        sizes: list[tuple[int, int, int]] | None = None,
    ) -> BenchmarkSuite:
        """Add FP8 GEMM benchmark (SM120+)."""
        self.benchmarks.append(
            FP8GEMMBenchmark(
                sizes=sizes,
                warmup=self.warmup,
                iterations=self.iterations,
            )
        )
        return self

    def add_gemv(
        self,
        configs: list[tuple[int, int, str]] | None = None,
        dtypes: list[str] | None = None,
    ) -> BenchmarkSuite:
        """Add GEMV benchmark.

        Args:
            configs: List of (K, N, label) tuples
            dtypes: List of dtypes (bf16, fp8, nvf4, int4)
        """
        self.benchmarks.append(
            GEMVBenchmark(
                configs=configs,
                dtypes=dtypes,
                warmup=self.warmup,
                iterations=self.iterations,
            )
        )
        return self

    def add_w8a8_gemv(
        self,
        configs: list[tuple[int, int, str]] | None = None,
    ) -> BenchmarkSuite:
        """Add W8A8 GEMV benchmark."""
        self.benchmarks.append(
            W8A8GEMVBenchmark(
                configs=configs,
                warmup=self.warmup,
                iterations=self.iterations,
            )
        )
        return self

    def add_attention(
        self,
        seq_lens: list[int] | None = None,
        num_heads: int = 32,
        head_dim: int = 128,
    ) -> BenchmarkSuite:
        """Add SDPA benchmark."""
        self.benchmarks.append(
            SDPABenchmark(
                seq_lens=seq_lens,
                num_heads=num_heads,
                head_dim=head_dim,
                warmup=self.warmup,
                iterations=self.iterations,
            )
        )
        return self

    def add_gqa(
        self,
        seq_lens: list[int] | None = None,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
    ) -> BenchmarkSuite:
        """Add GQA benchmark."""
        self.benchmarks.append(
            GQABenchmark(
                seq_lens=seq_lens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                warmup=self.warmup,
                iterations=self.iterations,
            )
        )
        return self

    def add_all(self) -> BenchmarkSuite:
        """Add all available benchmarks with default settings."""
        self.add_gemm()
        self.add_gemv()
        self.add_attention()
        return self

    def run(self, verbose: bool = True) -> BenchmarkReport:
        """Run all benchmarks.

        Args:
            verbose: If True, print progress

        Returns:
            BenchmarkReport with all results
        """
        gpu_info = get_gpu_info()
        report = BenchmarkReport(gpu=gpu_info)

        if verbose:
            print("=" * 60)
            print("PyGPUkit Benchmark Suite")
            print("=" * 60)
            print(f"GPU: {gpu_info.name}")
            print(f"SM: {gpu_info.sm_major}.{gpu_info.sm_minor}")
            print(f"Memory: {gpu_info.memory_gb:.1f} GB")
            print()

        for benchmark in self.benchmarks:
            if verbose:
                print(f"Running {benchmark.__class__.__name__}...")

            results = benchmark.run()
            for result in results:
                report.add(result)
                if verbose:
                    tflops_str = f"{result.tflops:.1f} TFLOPS" if result.tflops else ""
                    print(f"  {result.name}: {result.median_us:.1f} us {tflops_str}")

            if verbose:
                print()

        return report

    def compare(
        self,
        baseline_path: str | Path,
        threshold: float = 0.05,
        verbose: bool = True,
    ) -> ComparisonResult:
        """Run benchmarks and compare with baseline.

        Args:
            baseline_path: Path to baseline JSON file
            threshold: Regression threshold (0.05 = 5%)
            verbose: If True, print comparison summary

        Returns:
            ComparisonResult
        """
        current = self.run(verbose=verbose)
        baseline = BenchmarkReport.load(baseline_path)
        comparison = compare_reports(current, baseline, threshold=threshold)

        if verbose:
            print(comparison.summary())

        return comparison


def run_quick() -> BenchmarkReport:
    """Run quick benchmark suite."""
    suite = BenchmarkSuite(quick=True)
    suite.add_gemm(sizes=[(4096, 4096, 4096)], dtypes=["bf16"])
    suite.add_gemv(dtypes=["bf16"])
    return suite.run()


def run_full() -> BenchmarkReport:
    """Run full benchmark suite."""
    suite = BenchmarkSuite()
    suite.add_all()
    return suite.run()
