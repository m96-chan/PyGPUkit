"""Benchmark: Tiled vs Naive Matmul Performance"""

import os
import sys
import time

# Add CUDA DLLs to PATH
cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
cuda_bin = os.path.join(cuda_path, "bin")
if cuda_bin not in os.environ["PATH"]:
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(cuda_bin)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pygpukit"))

import numpy as np  # noqa: E402

print("=" * 70)
print("Tiled Matmul Benchmark - PyGPUkit v0.2")
print("=" * 70)

try:
    import _pygpukit_native as native  # noqa: E402

    print(f"\nCUDA available: {native.is_cuda_available()}")

    if native.is_cuda_available():
        props = native.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Memory: {props.total_memory / 1024**3:.1f} GB")

        print("\n" + "-" * 70)
        print("Matrix Size | Kernel    | Time (ms) | GFLOPS  | Speedup")
        print("-" * 70)

        sizes = [512, 1024, 2048, 3072, 4096]

        for size in sizes:
            M, N, K = size, size, size

            # Create test matrices
            A_np = np.random.randn(M, K).astype(np.float32)
            B_np = np.random.randn(K, N).astype(np.float32)

            # Warmup
            A_gpu = native.from_numpy(A_np)
            B_gpu = native.from_numpy(B_np)
            _ = native.matmul(A_gpu, B_gpu)

            # Benchmark GPU
            iterations = 5 if size >= 2048 else 10
            times = []
            for _ in range(iterations):
                A_gpu = native.from_numpy(A_np)
                B_gpu = native.from_numpy(B_np)
                start = time.perf_counter()
                C_gpu = native.matmul(A_gpu, B_gpu)
                gpu_time = time.perf_counter() - start
                times.append(gpu_time)

            avg_time = np.median(times)
            gflops = 2 * M * N * K / avg_time / 1e9

            # Check which kernel is used (threshold is 2048)
            kernel = "Tiled" if size >= 2048 else "L2-opt"

            # CPU reference
            start = time.perf_counter()
            C_cpu = np.matmul(A_np, B_np)
            cpu_time = time.perf_counter() - start

            speedup = cpu_time / avg_time

            # Verify correctness
            C_result = C_gpu.to_numpy()
            rel_error = np.max(np.abs(C_result - C_cpu)) / np.max(np.abs(C_cpu))

            print(f"{size:>5}x{size:<5} | {kernel:<9} | {avg_time*1000:>8.2f} | {gflops:>7.1f} | {speedup:>5.1f}x")

            if rel_error > 1e-3:
                print(f"  WARNING: High relative error: {rel_error:.2e}")

        print("-" * 70)
        print("\nTiled kernel should show improved performance for sizes >= 2048")
        print("=" * 70)

except ImportError as e:
    print(f"Error: {e}")
    print("Native module not available")
