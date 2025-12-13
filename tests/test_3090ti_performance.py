"""
TDD Performance Tests for RTX 3090 Ti Optimization (v0.2.2)

RTX 3090 Ti Specs:
- 10752 CUDA cores (84 SMs * 128 cores/SM)
- Boost clock: ~1.86 GHz
- Theoretical FP32: ~40 TFLOPS
- Memory: 24GB GDDR6X, 1008 GB/s bandwidth
- Shared memory: 100KB per SM (configurable)
- L2 cache: 6MB

Performance Targets:
- Target: 35.6 TFLOPS (90% of theoretical)
- Minimum: 22 TFLOPS (must beat PyTorch baseline)
"""
import os
import time

import numpy as np
import pytest

# Setup CUDA DLL path (if CUDA is installed)
cuda_path = os.environ.get(
    "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
)
cuda_bin = os.path.join(cuda_path, "bin")
if os.path.isdir(cuda_bin):
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_bin)

# Skip if native module not available
try:
    import _pygpukit_native as native
except ImportError:
    # Try via pygpukit package
    try:
        from pygpukit import _pygpukit_native as native
    except ImportError:
        pytest.skip("Native module not available", allow_module_level=True)

# RTX 3090 Ti performance constants
RTX_3090TI_THEORETICAL_TFLOPS = 40.0
TARGET_EFFICIENCY = 0.90  # 90%
MINIMUM_EFFICIENCY = 0.55  # 55% (must beat PyTorch)
TARGET_TFLOPS = RTX_3090TI_THEORETICAL_TFLOPS * TARGET_EFFICIENCY  # 35.6
MINIMUM_TFLOPS = RTX_3090TI_THEORETICAL_TFLOPS * MINIMUM_EFFICIENCY  # 22


def compute_tflops(m: int, n: int, k: int, time_sec: float) -> float:
    """Compute TFLOPS for matrix multiplication."""
    flops = 2 * m * n * k  # multiply-add = 2 ops
    return flops / time_sec / 1e12


def benchmark_matmul(m: int, n: int, k: int, warmup: int = 3, iterations: int = 10):
    """Benchmark matmul and return median time and TFLOPS."""
    A_np = np.random.randn(m, k).astype(np.float32)
    B_np = np.random.randn(k, n).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        A_gpu = native.from_numpy(A_np)
        B_gpu = native.from_numpy(B_np)
        _ = native.matmul(A_gpu, B_gpu)

    # Benchmark
    times = []
    for _ in range(iterations):
        A_gpu = native.from_numpy(A_np)
        B_gpu = native.from_numpy(B_np)
        start = time.perf_counter()
        _ = native.matmul(A_gpu, B_gpu)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = np.median(times)
    tflops = compute_tflops(m, n, k, median_time)
    return median_time, tflops


@pytest.fixture(scope="module")
def check_3090ti():
    """Check if running on RTX 3090 Ti or compatible high-end GPU."""
    if not native.is_cuda_available():
        pytest.skip("CUDA not available")

    props = native.get_device_properties(0)
    # Allow RTX 3090, 3090 Ti, 4090, or any high-end Ampere/Ada GPU
    high_end_gpus = ["3090", "4090", "4080", "A100", "H100"]
    is_high_end = any(gpu in props.name for gpu in high_end_gpus)
    if not is_high_end:
        pytest.skip(f"Not a high-end GPU: {props.name}")

    print(f"\nRunning on: {props.name}")
    return props


class TestMinimumPerformance:
    """Tests for minimum performance requirements (22 TFLOPS)."""

    def test_4096x4096_minimum_tflops(self, check_3090ti):
        """4096x4096 matmul must achieve at least 22 TFLOPS."""
        _, tflops = benchmark_matmul(4096, 4096, 4096)
        print(f"\n4096x4096: {tflops:.1f} TFLOPS (minimum: {MINIMUM_TFLOPS})")
        assert tflops >= MINIMUM_TFLOPS, (
            f"4096x4096 matmul achieved only {tflops:.1f} TFLOPS, "
            f"minimum required: {MINIMUM_TFLOPS} TFLOPS"
        )

    def test_8192x8192_minimum_tflops(self, check_3090ti):
        """8192x8192 matmul must achieve at least 22 TFLOPS."""
        _, tflops = benchmark_matmul(8192, 8192, 8192, warmup=2, iterations=5)
        print(f"\n8192x8192: {tflops:.1f} TFLOPS (minimum: {MINIMUM_TFLOPS})")
        assert tflops >= MINIMUM_TFLOPS, (
            f"8192x8192 matmul achieved only {tflops:.1f} TFLOPS, "
            f"minimum required: {MINIMUM_TFLOPS} TFLOPS"
        )

    def test_2048x2048_reasonable_tflops(self, check_3090ti):
        """2048x2048 matmul should achieve at least 15 TFLOPS."""
        min_2k = 15.0  # Lower threshold for smaller matrix
        _, tflops = benchmark_matmul(2048, 2048, 2048)
        print(f"\n2048x2048: {tflops:.1f} TFLOPS (minimum: {min_2k})")
        assert tflops >= min_2k, (
            f"2048x2048 matmul achieved only {tflops:.1f} TFLOPS, "
            f"minimum required: {min_2k} TFLOPS"
        )


class TestTargetPerformance:
    """Tests for target performance (35.6 TFLOPS, 90% efficiency)."""

    def test_4096x4096_target_tflops(self, check_3090ti):
        """4096x4096 matmul should achieve 35.6 TFLOPS target."""
        _, tflops = benchmark_matmul(4096, 4096, 4096)
        print(f"\n4096x4096: {tflops:.1f} TFLOPS (target: {TARGET_TFLOPS})")
        assert tflops >= TARGET_TFLOPS, (
            f"4096x4096 matmul achieved only {tflops:.1f} TFLOPS, "
            f"target: {TARGET_TFLOPS} TFLOPS"
        )

    def test_8192x8192_target_tflops(self, check_3090ti):
        """8192x8192 matmul should achieve 35.6 TFLOPS target."""
        _, tflops = benchmark_matmul(8192, 8192, 8192, warmup=2, iterations=5)
        print(f"\n8192x8192: {tflops:.1f} TFLOPS (target: {TARGET_TFLOPS})")
        assert tflops >= TARGET_TFLOPS, (
            f"8192x8192 matmul achieved only {tflops:.1f} TFLOPS, "
            f"target: {TARGET_TFLOPS} TFLOPS"
        )

    def test_large_matrix_sustained_performance(self, check_3090ti):
        """Large matrices should sustain high performance."""
        sizes = [(4096, 4096, 4096), (6144, 6144, 6144), (8192, 8192, 8192)]
        results = []

        for m, n, k in sizes:
            iters = 5 if m >= 6144 else 10
            _, tflops = benchmark_matmul(m, n, k, warmup=2, iterations=iters)
            results.append((m, tflops))
            print(f"\n{m}x{n}x{k}: {tflops:.1f} TFLOPS")

        # All should be above minimum
        for size, tflops in results:
            assert tflops >= MINIMUM_TFLOPS, (
                f"{size}x{size} only achieved {tflops:.1f} TFLOPS"
            )


class TestNonSquareMatrices:
    """Tests for non-square matrix performance."""

    def test_tall_skinny_matrix(self, check_3090ti):
        """Tall-skinny matrices (common in ML) should be efficient."""
        # Typical inference shape: batch x hidden x output
        _, tflops = benchmark_matmul(8192, 4096, 1024)
        print(f"\n8192x4096x1024 (tall-skinny): {tflops:.1f} TFLOPS")
        # Lower threshold for non-square
        assert tflops >= 15.0, f"Tall-skinny only achieved {tflops:.1f} TFLOPS"

    def test_wide_matrix(self, check_3090ti):
        """Wide matrices should be efficient."""
        _, tflops = benchmark_matmul(1024, 8192, 4096)
        print(f"\n1024x8192x4096 (wide): {tflops:.1f} TFLOPS")
        assert tflops >= 15.0, f"Wide matrix only achieved {tflops:.1f} TFLOPS"

    def test_transformer_attention_shapes(self, check_3090ti):
        """Transformer attention-like shapes should be efficient."""
        # QK^T: (batch*heads, seq, head_dim) x (batch*heads, head_dim, seq)
        # Typical: 32*16=512 batch, 2048 seq, 64 head_dim
        _, tflops = benchmark_matmul(512, 2048, 64)
        print(f"\n512x2048x64 (attention QK): {tflops:.1f} TFLOPS")
        # Small K dimension limits performance
        assert tflops >= 5.0


class TestCorrectness:
    """Verify correctness is maintained with optimizations."""

    def test_matmul_correctness_small(self, check_3090ti):
        """Small matmul should be numerically correct."""
        A = np.random.randn(256, 256).astype(np.float32)
        B = np.random.randn(256, 256).astype(np.float32)

        A_gpu = native.from_numpy(A)
        B_gpu = native.from_numpy(B)
        C_gpu = native.matmul(A_gpu, B_gpu)
        C_result = C_gpu.to_numpy()

        C_expected = A @ B
        rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))
        assert rel_error < 1e-5, f"Relative error too high: {rel_error}"

    def test_matmul_correctness_large(self, check_3090ti):
        """Large matmul should be numerically correct."""
        A = np.random.randn(4096, 4096).astype(np.float32)
        B = np.random.randn(4096, 4096).astype(np.float32)

        A_gpu = native.from_numpy(A)
        B_gpu = native.from_numpy(B)
        C_gpu = native.matmul(A_gpu, B_gpu)
        C_result = C_gpu.to_numpy()

        C_expected = A @ B
        rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))
        assert rel_error < 1e-4, f"Relative error too high: {rel_error}"

    def test_matmul_correctness_non_square(self, check_3090ti):
        """Non-square matmul should be numerically correct."""
        A = np.random.randn(2048, 1024).astype(np.float32)
        B = np.random.randn(1024, 4096).astype(np.float32)

        A_gpu = native.from_numpy(A)
        B_gpu = native.from_numpy(B)
        C_gpu = native.matmul(A_gpu, B_gpu)
        C_result = C_gpu.to_numpy()

        C_expected = A @ B
        rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))
        assert rel_error < 1e-4, f"Relative error too high: {rel_error}"


class TestEfficiencyMetrics:
    """Tests for efficiency metrics and roofline analysis."""

    def test_compute_bound_efficiency(self, check_3090ti):
        """Large square matrices should be compute-bound with high efficiency."""
        _, tflops = benchmark_matmul(8192, 8192, 8192, warmup=2, iterations=5)
        efficiency = tflops / RTX_3090TI_THEORETICAL_TFLOPS
        print(f"\n8192x8192 efficiency: {efficiency*100:.1f}%")
        assert efficiency >= 0.55, f"Efficiency only {efficiency*100:.1f}%"

    def test_memory_bandwidth_utilization(self, check_3090ti):
        """Measure effective memory bandwidth for small K."""
        # Small K = memory-bound
        m, n, k = 8192, 8192, 64
        time_sec, _ = benchmark_matmul(m, n, k)

        # Data transfer: A (m*k) + B (k*n) + C (m*n) in float32
        bytes_transferred = (m * k + k * n + m * n) * 4
        bandwidth_gbps = bytes_transferred / time_sec / 1e9

        print(f"\n{m}x{n}x{k} bandwidth: {bandwidth_gbps:.1f} GB/s")
        # RTX 3090 Ti has 1008 GB/s peak bandwidth
        assert bandwidth_gbps >= 400, f"Bandwidth only {bandwidth_gbps:.1f} GB/s"


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
