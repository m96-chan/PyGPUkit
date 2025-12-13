"""Benchmark PyTorch cuBLAS for comparison with PyGPUkit."""
import time
import numpy as np

try:
    import torch
except ImportError:
    print("PyTorch not installed")
    exit(0)

# Check PyTorch CUDA
print("PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Benchmark
sizes = [2048, 4096, 8192]
for size in sizes:
    A = torch.randn(size, size, device="cuda", dtype=torch.float32)
    B = torch.randn(size, size, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(3):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    iterations = 10 if size < 8192 else 5
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = np.median(times)
    tflops = 2 * size**3 / median_time / 1e12
    print(f"{size}x{size}: {tflops:.1f} TFLOPS ({median_time*1000:.2f} ms)")
