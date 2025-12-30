"""GEMM (General Matrix Multiply) benchmarks."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .base import Benchmark
from .results import BenchmarkResult


class GEMMBenchmark(Benchmark):
    """GEMM benchmark for various dtypes and sizes."""

    category = "gemm"

    def __init__(
        self,
        sizes: list[tuple[int, int, int]] | None = None,
        dtypes: list[str] | None = None,
        warmup: int = 10,
        iterations: int = 50,
    ):
        super().__init__(warmup=warmup, iterations=iterations)
        self.sizes = sizes or [
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]
        self.dtypes = dtypes or ["fp32", "tf32", "bf16"]

    def run(self) -> list[BenchmarkResult]:
        """Run GEMM benchmarks."""
        results: list[BenchmarkResult] = []

        for dtype in self.dtypes:
            for M, K, N in self.sizes:
                try:
                    result = self._benchmark_gemm(dtype, M, K, N)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"  GEMM {dtype} {M}x{K}x{N}: ERROR - {e}")

        return results

    def _benchmark_gemm(
        self,
        dtype: str,
        M: int,
        K: int,
        N: int,
    ) -> BenchmarkResult | None:
        """Benchmark single GEMM configuration."""
        import os

        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        name = f"gemm_{dtype}_{M}x{K}x{N}"
        params = {"dtype": dtype, "M": M, "K": K, "N": N}
        flops = 2.0 * M * K * N

        if dtype == "fp32":
            os.environ.pop("PYGPUKIT_ALLOW_TF32", None)
            A = np.random.randn(M, K).astype(np.float32) * 0.1
            B = np.random.randn(K, N).astype(np.float32) * 0.1
            A_gpu = native.from_numpy(A)
            B_gpu = native.from_numpy(B)

            def run_fn() -> None:
                native.matmul(A_gpu, B_gpu)

            def check_fn() -> tuple[bool, float]:
                C_gpu = native.matmul(A_gpu, B_gpu)
                C = C_gpu.to_numpy()
                C_ref = A @ B
                err = float(np.max(np.abs(C - C_ref)) / (np.max(np.abs(C_ref)) + 1e-8))
                return err < 1e-3, err

        elif dtype == "tf32":
            os.environ["PYGPUKIT_ALLOW_TF32"] = "1"
            os.environ["PYGPUKIT_TF32_V2"] = "1"
            A = np.random.randn(M, K).astype(np.float32) * 0.1
            B = np.random.randn(K, N).astype(np.float32) * 0.1
            A_gpu = native.from_numpy(A)
            B_gpu = native.from_numpy(B)

            def run_fn() -> None:
                native.matmul(A_gpu, B_gpu)

            def check_fn() -> tuple[bool, float]:
                C_gpu = native.matmul(A_gpu, B_gpu)
                C = C_gpu.to_numpy()
                C_ref = A @ B
                err = float(np.max(np.abs(C - C_ref)) / (np.max(np.abs(C_ref)) + 1e-8))
                return err < 0.01, err

        elif dtype == "bf16":
            import pygpukit as gk

            A = np.random.randn(M, K).astype(np.float32) * 0.1
            B = np.random.randn(K, N).astype(np.float32) * 0.1
            A_gpu = gk.from_numpy(A).astype(gk.bfloat16)._get_native()
            B_gpu = gk.from_numpy(B).astype(gk.bfloat16)._get_native()

            def run_fn() -> None:
                native.matmul(A_gpu, B_gpu)

            def check_fn() -> tuple[bool, float]:
                import pygpukit as gk

                C_gpu = native.matmul(A_gpu, B_gpu)
                C = gk.GPUArray._wrap_native(C_gpu).astype(gk.float32).to_numpy()
                C_ref = A @ B
                err = float(np.max(np.abs(C - C_ref)) / (np.max(np.abs(C_ref)) + 1e-8))
                return err < 0.05, err

        elif dtype == "fp16":
            A = np.random.randn(M, K).astype(np.float16) * 0.1
            B = np.random.randn(K, N).astype(np.float16) * 0.1
            A_gpu = native.from_numpy(A)
            B_gpu = native.from_numpy(B)

            def run_fn() -> None:
                native.matmul(A_gpu, B_gpu)

            def check_fn() -> tuple[bool, float]:
                C_gpu = native.matmul(A_gpu, B_gpu)
                C = C_gpu.to_numpy().astype(np.float32)
                C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
                err = float(
                    np.max(np.abs(C - C_ref.astype(np.float32)))
                    / (np.max(np.abs(C_ref.astype(np.float32))) + 1e-8)
                )
                return err < 0.05, err

        else:
            return None

        return self._measure(name, run_fn, params, flops=flops, check_fn=check_fn)


class FP8GEMMBenchmark(Benchmark):
    """FP8 GEMM benchmark (SM120+)."""

    category = "gemm"

    def __init__(
        self,
        sizes: list[tuple[int, int, int]] | None = None,
        warmup: int = 10,
        iterations: int = 50,
    ):
        super().__init__(warmup=warmup, iterations=iterations)
        self.sizes = sizes or [
            (1024, 4096, 14336),
            (2048, 4096, 14336),
            (4096, 4096, 14336),
        ]

    def run(self) -> list[BenchmarkResult]:
        """Run FP8 GEMM benchmarks."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        results: list[BenchmarkResult] = []

        # Check SM120 availability
        props = native.get_device_properties(0)
        sm = props.compute_capability_major * 10 + props.compute_capability_minor
        if sm < 120:
            print(f"  FP8 GEMM: Requires SM120+ (current: SM{sm})")
            return results

        for M, K, N in self.sizes:
            try:
                result = self._benchmark_fp8_gemm(native, M, K, N)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  FP8 GEMM {M}x{K}x{N}: ERROR - {e}")

        return results

    def _benchmark_fp8_gemm(
        self,
        native: object,
        M: int,
        K: int,
        N: int,
    ) -> BenchmarkResult | None:
        """Benchmark FP8 GEMM."""
        from pygpukit.core import from_numpy

        name = f"gemm_fp8_{M}x{K}x{N}"
        params = {"dtype": "fp8", "M": M, "K": K, "N": N}
        flops = 2.0 * M * K * N

        A_fp8 = from_numpy(np.random.randint(0, 256, (M, K), dtype=np.uint8))
        B_fp8 = from_numpy(np.random.randint(0, 256, (K, N), dtype=np.uint8))
        C_fp8 = from_numpy(np.zeros((M, N), dtype=np.uint8))

        # Try v5 (cached) kernel
        func = getattr(native, "gemm_fp8_fp8_sm120_v5", None)
        if func is None:
            return None

        def run_fn() -> None:
            func(A_fp8._get_native(), B_fp8._get_native(), C_fp8._get_native())

        return self._measure(name, run_fn, params, flops=flops)
