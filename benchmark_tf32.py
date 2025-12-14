"""Benchmark TF32 TensorCore GEMM kernel."""
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
print(f"SM: {props.compute_capability_major}.{props.compute_capability_minor}")
print()


def verify_correctness(m, n, k, tolerance=1e-2):
    """Verify kernel correctness with TF32 tolerance."""
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)

    A_gpu = native.from_numpy(A)
    B_gpu = native.from_numpy(B)
    C_gpu = native.matmul(A_gpu, B_gpu)
    C_result = C_gpu.to_numpy()

    C_expected = A @ B
    rel_error = np.max(np.abs(C_result - C_expected)) / np.max(np.abs(C_expected))
    return rel_error


def benchmark_matmul(m, n, k, warmup=5, iterations=10):
    """Benchmark matmul and return median time and TFLOPS."""
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
    min_time = np.min(times)
    flops = 2 * m * n * k
    tflops_median = flops / median_time / 1e12
    tflops_max = flops / min_time / 1e12
    return median_time, tflops_median, tflops_max


# Correctness verification
print("=== Correctness Verification (TF32 tolerance: 1e-2) ===")
for size in [256, 512, 1024, 2048, 4096]:
    error = verify_correctness(size, size, size)
    status = "PASS" if error < 1e-2 else "FAIL"
    print(f"{size}x{size}: relative error = {error:.2e} [{status}]")

print()

# Performance benchmark
sizes = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

print("=== TF32 TensorCore GEMM Benchmark ===")
print()

# Performance targets
TARGETS = {
    2048: 15.0,
    4096: 22.0,
    8192: 28.0,
}

for m, n, k in sizes:
    iters = 5 if m >= 8192 else 10
    time_ms, tflops_med, tflops_max = benchmark_matmul(m, n, k, warmup=5, iterations=iters)
    target = TARGETS.get(m, 20.0)
    status = "PASS" if tflops_med >= target else "FAIL"
    print(f"{m}x{n}x{k}: {tflops_med:.1f} TFLOPS (max: {tflops_max:.1f}) - {time_ms*1000:.2f} ms [{status}]")

print()
print("=== Performance Targets ===")
print("4096x4096: 22 TFLOPS minimum, 30 TFLOPS target")
print("8192x8192: 28 TFLOPS minimum, 35 TFLOPS target")
print()
print("RTX 3090 Ti theoretical: 40 TFLOPS (FP32), 156 TFLOPS (TF32)")
