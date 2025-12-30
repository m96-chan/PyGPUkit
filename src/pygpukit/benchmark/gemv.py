"""GEMV (General Matrix-Vector) benchmarks."""

from __future__ import annotations

import numpy as np

from .base import Benchmark
from .results import BenchmarkResult


# LLM-relevant GEMV configurations
LLM_CONFIGS = [
    # (K, N, label)
    (4096, 4096, "7B_hidden"),
    (4096, 14336, "7B_mlp_up"),
    (14336, 4096, "7B_mlp_down"),
    (8192, 8192, "72B_hidden"),
    (8192, 29568, "72B_mlp_up"),
    (29568, 8192, "72B_mlp_down"),
]


class GEMVBenchmark(Benchmark):
    """GEMV benchmark for LLM decode (M=1)."""

    category = "gemv"

    def __init__(
        self,
        configs: list[tuple[int, int, str]] | None = None,
        dtypes: list[str] | None = None,
        warmup: int = 10,
        iterations: int = 50,
    ):
        super().__init__(warmup=warmup, iterations=iterations)
        self.configs = configs or LLM_CONFIGS
        self.dtypes = dtypes or ["bf16", "fp8", "nvf4"]

    def run(self) -> list[BenchmarkResult]:
        """Run GEMV benchmarks."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        results: list[BenchmarkResult] = []

        for K, N, label in self.configs:
            for dtype in self.dtypes:
                try:
                    result = self._benchmark_gemv(native, K, N, label, dtype)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"  GEMV {dtype} {label}: ERROR - {e}")

        return results

    def _benchmark_gemv(
        self,
        native: object,
        K: int,
        N: int,
        label: str,
        dtype: str,
    ) -> BenchmarkResult | None:
        """Benchmark single GEMV configuration."""
        import pygpukit as gk
        from pygpukit.core import from_numpy

        name = f"gemv_{dtype}_{label}"
        params = {"dtype": dtype, "K": K, "N": N, "label": label}
        flops = 2.0 * K * N  # M=1

        if dtype == "bf16":
            from pygpukit.ops.matmul import gemv_bf16

            A = gk.empty((K,), dtype="bfloat16")
            B = gk.empty((K, N), dtype="bfloat16")
            C = gk.empty((N,), dtype="bfloat16")

            def run_fn() -> None:
                gemv_bf16(A, B, out=C)

        elif dtype == "fp8":
            from pygpukit.ops.matmul import fp8_init_lut, gemv_fp8_bf16

            fp8_init_lut()
            A = gk.empty((K,), dtype="bfloat16")
            B_fp8 = from_numpy(np.zeros((N, K), dtype=np.uint8))
            n_blocks = (N + 127) // 128
            k_blocks = (K + 127) // 128
            B_scale = from_numpy(np.ones((n_blocks, k_blocks), dtype=np.float16).view(np.uint16))
            C = gk.empty((N,), dtype="bfloat16")

            def run_fn() -> None:
                gemv_fp8_bf16(A, B_fp8, B_scale, out=C)

        elif dtype == "nvf4":
            from pygpukit.ops.matmul import gemv_nvf4_available, gemv_nvf4_bf16

            if not gemv_nvf4_available():
                return None

            A = gk.empty((K,), dtype="bfloat16")
            B_nvf4 = from_numpy(np.zeros((K // 2, N), dtype=np.uint8))
            k_scale_blocks = (K + 31) // 32
            B_scale = from_numpy(np.ones((k_scale_blocks, N), dtype=np.uint8))
            C = gk.empty((N,), dtype="bfloat16")

            def run_fn() -> None:
                gemv_nvf4_bf16(A, B_nvf4, B_scale, out=C)

        elif dtype == "int4":
            if not hasattr(native, "int4_gemv_available") or not native.int4_gemv_available():
                return None

            def pack_int4(values: np.ndarray) -> np.ndarray:
                flat = values.reshape(-1)
                low = flat[0::2].astype(np.int32) & 0x0F
                high = flat[1::2].astype(np.int32) & 0x0F
                packed = (high << 4) | low
                new_shape = list(values.shape)
                new_shape[-1] //= 2
                return packed.astype(np.uint8).reshape(new_shape)

            A_raw = np.random.randint(-8, 8, K, dtype=np.int8)
            B_raw = np.random.randint(-8, 8, (N, K), dtype=np.int8)
            A_int4 = from_numpy(pack_int4(A_raw.reshape(1, -1)).reshape(-1))
            B_int4 = from_numpy(pack_int4(B_raw))
            C_int4 = from_numpy(np.zeros(N, dtype=np.int32))

            def run_fn() -> None:
                native.int4_gemv_int32_sm120(
                    A_int4._get_native(), B_int4._get_native(), C_int4._get_native()
                )

        else:
            return None

        return self._measure(name, run_fn, params, flops=flops)


class W8A8GEMVBenchmark(Benchmark):
    """W8A8 (FP8 weights, FP8 activations) GEMV benchmark."""

    category = "gemv"

    def __init__(
        self,
        configs: list[tuple[int, int, str]] | None = None,
        warmup: int = 10,
        iterations: int = 50,
    ):
        super().__init__(warmup=warmup, iterations=iterations)
        self.configs = configs or LLM_CONFIGS[:3]  # Smaller set

    def run(self) -> list[BenchmarkResult]:
        """Run W8A8 GEMV benchmarks."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        results: list[BenchmarkResult] = []

        # Check availability
        if not hasattr(native, "gemv_fp8_fp8_available") or not native.gemv_fp8_fp8_available():
            print("  W8A8 GEMV: Not available")
            return results

        for K, N, label in self.configs:
            try:
                result = self._benchmark_w8a8(native, K, N, label)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  W8A8 GEMV {label}: ERROR - {e}")

        return results

    def _benchmark_w8a8(
        self,
        native: object,
        K: int,
        N: int,
        label: str,
    ) -> BenchmarkResult | None:
        """Benchmark W8A8 GEMV."""
        from pygpukit.core import from_numpy, zeros

        name = f"gemv_w8a8_{label}"
        params = {"dtype": "w8a8", "K": K, "N": N, "label": label}
        flops = 2.0 * K * N

        block_size = 128
        n_scales_k = (K + block_size - 1) // block_size
        n_scales_n = (N + block_size - 1) // block_size

        A_fp8 = from_numpy(np.random.randint(0, 256, K, dtype=np.uint8))
        B_fp8 = from_numpy(np.random.randint(0, 256, (N, K), dtype=np.uint8))
        scale_A = from_numpy(np.ones(n_scales_k, dtype=np.float32))
        scale_B = from_numpy(np.ones(n_scales_n * n_scales_k, dtype=np.float32))
        C = zeros((N,), dtype="bfloat16")

        def run_fn() -> None:
            native.gemv_fp8_fp8_bf16_sm120(
                A_fp8._get_native(),
                B_fp8._get_native(),
                scale_A._get_native(),
                scale_B._get_native(),
                C._get_native(),
            )

        return self._measure(name, run_fn, params, flops=flops)
