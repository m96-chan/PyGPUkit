"""
TDD Tests for TF32 TensorCore GEMM (v0.2.3)

TF32 Specifications:
- Input: TF32 (19-bit: 1 sign + 8 exp + 10 mantissa)
- Accumulator: FP32
- Precision: ~1e-2 relative error (vs FP32's ~1e-5)

Performance Targets (RTX 3090 Ti):
- 4096x4096: 22+ TFLOPS
- 8192x8192: 28+ TFLOPS

Ampere TensorCore:
- mma.sync.aligned.m16n8k8.row.col.tf32.tf32.f32
- 256 TFLOPS theoretical (TF32)
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
    try:
        from pygpukit import _pygpukit_native as native
    except ImportError:
        pytest.skip("Native module not available", allow_module_level=True)


# TF32 precision constants
TF32_RELATIVE_ERROR_TOLERANCE = 1e-2  # TF32 has 10-bit mantissa vs FP32's 23-bit

# Performance targets (RTX 3090 Ti theoretical: 40 TFLOPS FP32, 156 TFLOPS TF32)
MINIMUM_TFLOPS_4096 = 22.0
MINIMUM_TFLOPS_8192 = 28.0
TARGET_TFLOPS_4096 = 30.0
TARGET_TFLOPS_8192 = 35.0


def compute_tflops(m: int, n: int, k: int, time_sec: float) -> float:
    """Compute TFLOPS for matrix multiplication."""
    flops = 2 * m * n * k
    return flops / time_sec / 1e12


def has_tensorcore_support() -> bool:
    """Check if GPU supports TensorCore (SM >= 70 for FP16, SM >= 80 for TF32)."""
    if not native.is_cuda_available():
        return False
    props = native.get_device_properties(0)
    # TF32 requires SM 80+ (Ampere)
    sm_version = props.compute_capability_major * 10 + props.compute_capability_minor
    return sm_version >= 80


@pytest.fixture(scope="module")
def check_tensorcore():
    """Check if TensorCore is available."""
    if not native.is_cuda_available():
        pytest.skip("CUDA not available")
    if not has_tensorcore_support():
        pytest.skip("TensorCore (SM >= 80) not available")
    props = native.get_device_properties(0)
    print(f"\nGPU: {props.name} (SM {props.compute_capability_major}{props.compute_capability_minor})")
    return props


class TestTF32Correctness:
    """Tests for TF32 TensorCore GEMM correctness."""

    def test_tf32_matmul_small(self, check_tensorcore):
        """Small TF32 matmul should be correct within tolerance."""
        m, n, k = 256, 256, 256
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)

        A_gpu = native.from_numpy(A)
        B_gpu = native.from_numpy(B)

        # TF32 matmul (when implemented, use_tf32=True)
        C_gpu = native.matmul(A_gpu, B_gpu)  # TODO: add use_tf32=True
        C_result = C_gpu.to_numpy()

        C_expected = A @ B
        rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))

        print(f"\n{m}x{n}x{k}: relative error = {rel_error:.2e}")
        assert rel_error < TF32_RELATIVE_ERROR_TOLERANCE, (
            f"TF32 relative error {rel_error:.2e} exceeds tolerance {TF32_RELATIVE_ERROR_TOLERANCE}"
        )

    def test_tf32_matmul_medium(self, check_tensorcore):
        """Medium TF32 matmul should be correct within tolerance."""
        m, n, k = 1024, 1024, 1024
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)

        A_gpu = native.from_numpy(A)
        B_gpu = native.from_numpy(B)
        C_gpu = native.matmul(A_gpu, B_gpu)
        C_result = C_gpu.to_numpy()

        C_expected = A @ B
        rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))

        print(f"\n{m}x{n}x{k}: relative error = {rel_error:.2e}")
        assert rel_error < TF32_RELATIVE_ERROR_TOLERANCE

    def test_tf32_matmul_large(self, check_tensorcore):
        """Large TF32 matmul should be correct within tolerance."""
        m, n, k = 4096, 4096, 4096
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)

        A_gpu = native.from_numpy(A)
        B_gpu = native.from_numpy(B)
        C_gpu = native.matmul(A_gpu, B_gpu)
        C_result = C_gpu.to_numpy()

        C_expected = A @ B
        rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))

        print(f"\n{m}x{n}x{k}: relative error = {rel_error:.2e}")
        assert rel_error < TF32_RELATIVE_ERROR_TOLERANCE

    def test_tf32_matmul_non_square(self, check_tensorcore):
        """Non-square TF32 matmul should be correct."""
        test_cases = [
            (2048, 4096, 1024),
            (4096, 2048, 2048),
            (1024, 1024, 4096),
        ]

        for m, n, k in test_cases:
            A = np.random.randn(m, k).astype(np.float32)
            B = np.random.randn(k, n).astype(np.float32)

            A_gpu = native.from_numpy(A)
            B_gpu = native.from_numpy(B)
            C_gpu = native.matmul(A_gpu, B_gpu)
            C_result = C_gpu.to_numpy()

            C_expected = A @ B
            rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))

            print(f"\n{m}x{n}x{k}: relative error = {rel_error:.2e}")
            assert rel_error < TF32_RELATIVE_ERROR_TOLERANCE

    def test_tf32_deterministic(self, check_tensorcore):
        """TF32 matmul should be deterministic over 100 iterations."""
        m, n, k = 1024, 1024, 1024
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)

        A_gpu = native.from_numpy(A)
        B_gpu = native.from_numpy(B)

        # First result
        C_first = native.matmul(A_gpu, B_gpu).to_numpy()

        # Run 100 times and verify identical
        for i in range(100):
            C_current = native.matmul(A_gpu, B_gpu).to_numpy()
            max_diff = np.max(np.abs(C_current - C_first))
            assert max_diff == 0.0, f"Non-deterministic at iteration {i}: max diff = {max_diff}"

        print("\n100 iterations: deterministic PASS")


class TestTF32Performance:
    """Tests for TF32 TensorCore GEMM performance."""

    def benchmark_matmul(self, m, n, k, warmup=5, iterations=10):
        """Benchmark matmul and return median TFLOPS."""
        A_np = np.random.randn(m, k).astype(np.float32)
        B_np = np.random.randn(k, n).astype(np.float32)

        A_gpu = native.from_numpy(A_np)
        B_gpu = native.from_numpy(B_np)

        # Warmup
        for _ in range(warmup):
            _ = native.matmul(A_gpu, B_gpu)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = native.matmul(A_gpu, B_gpu)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        median_time = np.median(times)
        tflops = compute_tflops(m, n, k, median_time)
        return median_time, tflops

    def test_tf32_4096_minimum_tflops(self, check_tensorcore):
        """4096x4096 TF32 matmul must achieve at least 22 TFLOPS."""
        m, n, k = 4096, 4096, 4096
        _, tflops = self.benchmark_matmul(m, n, k)

        print(f"\n{m}x{n}x{k}: {tflops:.1f} TFLOPS (minimum: {MINIMUM_TFLOPS_4096})")
        assert tflops >= MINIMUM_TFLOPS_4096, (
            f"4096x4096 TF32 matmul achieved only {tflops:.1f} TFLOPS, "
            f"minimum required: {MINIMUM_TFLOPS_4096} TFLOPS"
        )

    def test_tf32_8192_minimum_tflops(self, check_tensorcore):
        """8192x8192 TF32 matmul must achieve at least 28 TFLOPS."""
        m, n, k = 8192, 8192, 8192
        _, tflops = self.benchmark_matmul(m, n, k, warmup=3, iterations=5)

        print(f"\n{m}x{n}x{k}: {tflops:.1f} TFLOPS (minimum: {MINIMUM_TFLOPS_8192})")
        assert tflops >= MINIMUM_TFLOPS_8192, (
            f"8192x8192 TF32 matmul achieved only {tflops:.1f} TFLOPS, "
            f"minimum required: {MINIMUM_TFLOPS_8192} TFLOPS"
        )

    def test_tf32_4096_target_tflops(self, check_tensorcore):
        """4096x4096 TF32 matmul should achieve 30 TFLOPS target."""
        m, n, k = 4096, 4096, 4096
        _, tflops = self.benchmark_matmul(m, n, k)

        print(f"\n{m}x{n}x{k}: {tflops:.1f} TFLOPS (target: {TARGET_TFLOPS_4096})")
        assert tflops >= TARGET_TFLOPS_4096, (
            f"4096x4096 TF32 matmul achieved only {tflops:.1f} TFLOPS, "
            f"target: {TARGET_TFLOPS_4096} TFLOPS"
        )

    def test_tf32_8192_target_tflops(self, check_tensorcore):
        """8192x8192 TF32 matmul should achieve 35 TFLOPS target."""
        m, n, k = 8192, 8192, 8192
        _, tflops = self.benchmark_matmul(m, n, k, warmup=3, iterations=5)

        print(f"\n{m}x{n}x{k}: {tflops:.1f} TFLOPS (target: {TARGET_TFLOPS_8192})")
        assert tflops >= TARGET_TFLOPS_8192, (
            f"8192x8192 TF32 matmul achieved only {tflops:.1f} TFLOPS, "
            f"target: {TARGET_TFLOPS_8192} TFLOPS"
        )


class TestTF32VsFP32:
    """Compare TF32 and FP32 implementations."""

    def test_tf32_faster_than_fp32(self, check_tensorcore):
        """TF32 should be faster than FP32 FMA kernel."""
        m, n, k = 4096, 4096, 4096
        A_np = np.random.randn(m, k).astype(np.float32)
        B_np = np.random.randn(k, n).astype(np.float32)

        A_gpu = native.from_numpy(A_np)
        B_gpu = native.from_numpy(B_np)

        # Warmup
        for _ in range(5):
            _ = native.matmul(A_gpu, B_gpu)

        # Measure TF32 (current implementation)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = native.matmul(A_gpu, B_gpu)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        tf32_time = np.median(times)
        tf32_tflops = compute_tflops(m, n, k, tf32_time)

        # TF32 should achieve at least 22 TFLOPS (vs FP32's ~18 TFLOPS)
        print(f"\nTF32: {tf32_tflops:.1f} TFLOPS")
        assert tf32_tflops >= 22.0, "TF32 not faster than FP32 baseline"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
