"""Benchmark Ampere-optimized GEMM kernel."""
import os
import time

import numpy as np

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

# Import native module
try:
    import _pygpukit_native as native
except ImportError:
    from pygpukit import _pygpukit_native as native

props = native.get_device_properties(0)
print(f"GPU: {props.name}")
print()


def verify_correctness(m, n, k):
    """Verify kernel correctness."""
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)

    A_gpu = native.from_numpy(A)
    B_gpu = native.from_numpy(B)
    C_gpu = native.matmul(A_gpu, B_gpu)
    C_result = C_gpu.to_numpy()

    C_expected = A @ B
    rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))
    return rel_error


def benchmark_matmul(m, n, k, warmup=3, iterations=10):
    """Benchmark matmul and return median time and TFLOPS."""
    A_np = np.random.randn(m, k).astype(np.float32)
    B_np = np.random.randn(k, n).astype(np.float32)

    # Pre-allocate GPU arrays
    A_gpu = native.from_numpy(A_np)
    B_gpu = native.from_numpy(B_np)

    # Warmup
    for _ in range(warmup):
        _ = native.matmul(A_gpu, B_gpu)

    # Benchmark (reuse same input arrays)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = native.matmul(A_gpu, B_gpu)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = np.median(times)
    min_time = np.min(times)
    flops = 2 * m * n * k
    tflops_median = flops / median_time / 1e12
    tflops_max = flops / min_time / 1e12
    return median_time, tflops_median, tflops_max


# First verify correctness
print("=== Correctness Verification ===")
for size in [256, 512, 1024, 2048, 4096]:
    error = verify_correctness(size, size, size)
    status = "PASS" if error < 1e-4 else "FAIL"
    print(f"{size}x{size}: relative error = {error:.2e} [{status}]")

print()

# Benchmark different sizes
sizes = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

print("=== Ampere-Optimized GEMM Benchmark ===")
print()
for m, n, k in sizes:
    iters = 5 if m >= 8192 else 10
    time_ms, tflops_med, tflops_max = benchmark_matmul(m, n, k, warmup=5, iterations=iters)
    status = "PASS" if tflops_med >= 22.0 else "FAIL"
    print(f"{m}x{n}x{k}: {tflops_med:.1f} TFLOPS (max: {tflops_max:.1f}) - {time_ms*1000:.2f} ms [{status}]")

print()
print("Target: 22-32 TFLOPS (62-90% efficiency on RTX 3090 Ti)")
print("Minimum: 22 TFLOPS to beat PyTorch baseline")
