"""PyGPUkit v0.1 Optimized Demo - Zero-copy operations"""

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

# Add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "pygpukit"))

import numpy as np  # noqa: E402

print("=" * 70)
print("PyGPUkit v0.1 Optimized Demo - Zero-copy GPU Operations")
print("=" * 70)

try:
    import pygpukit as pgk
    from pygpukit.core.backend import NativeBackend, get_backend, has_native_module

    backend = get_backend()
    is_native = isinstance(backend, NativeBackend) and backend.is_available()

    print(f"\nBackend: {'Native C++ CUDA' if is_native else 'CPU Simulation'}")
    print(f"Native module available: {has_native_module()}")

    if is_native:
        import _pygpukit_native as native

        props = native.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Memory: {props.total_memory / 1024**3:.1f} GB")

    print("\n" + "-" * 70)
    print("Performance Comparison: Optimized (zero-copy) vs Previous (with copies)")
    print("-" * 70)

    # Test 1: Chained operations (where zero-copy shines)
    print("\n1. Chained Operations: (a + b) * c + d")
    n = 1_000_000

    # Create data
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    c_np = np.random.randn(n).astype(np.float32)
    d_np = np.random.randn(n).astype(np.float32)

    # GPU with PyGPUkit (optimized - data stays on GPU between ops)
    start = time.perf_counter()
    a = pgk.from_numpy(a_np)  # H→D once
    b = pgk.from_numpy(b_np)  # H→D once
    c = pgk.from_numpy(c_np)  # H→D once
    d = pgk.from_numpy(d_np)  # H→D once
    result = pgk.add(pgk.mul(pgk.add(a, b), c), d)  # All on GPU
    result_np = result.to_numpy()  # D→H once
    gpu_time = time.perf_counter() - start

    # CPU (NumPy)
    start = time.perf_counter()
    expected = (a_np + b_np) * c_np + d_np
    cpu_time = time.perf_counter() - start

    max_diff = np.max(np.abs(result_np - expected))
    print(f"   Elements: {n:,}")
    print(f"   GPU time: {gpu_time*1000:.3f} ms")
    print(f"   CPU time: {cpu_time*1000:.3f} ms")
    print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
    print(f"   Max diff: {max_diff:.2e}")

    # Test 2: Matrix multiplication chain
    print("\n2. Matrix Multiplication Chain: A @ B @ C")
    M = 512

    A_np = np.random.randn(M, M).astype(np.float32)
    B_np = np.random.randn(M, M).astype(np.float32)
    C_np = np.random.randn(M, M).astype(np.float32)

    # GPU
    start = time.perf_counter()
    A = pgk.from_numpy(A_np)
    B = pgk.from_numpy(B_np)
    C = pgk.from_numpy(C_np)
    result = pgk.matmul(pgk.matmul(A, B), C)
    result_np = result.to_numpy()
    gpu_time = time.perf_counter() - start

    # CPU
    start = time.perf_counter()
    expected = A_np @ B_np @ C_np
    cpu_time = time.perf_counter() - start

    rel_error = np.max(np.abs(result_np - expected)) / np.max(np.abs(expected))
    print(f"   Size: {M}x{M}")
    print(f"   GPU time: {gpu_time*1000:.3f} ms")
    print(f"   CPU time: {cpu_time*1000:.3f} ms")
    print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
    print(f"   Rel error: {rel_error:.2e}")

    # Test 3: Large single operation (where data transfer dominates)
    print("\n3. Large Matrix Multiplication: 2048x2048")
    M = 2048

    A_np = np.random.randn(M, M).astype(np.float32)
    B_np = np.random.randn(M, M).astype(np.float32)

    # Warmup
    A = pgk.from_numpy(A_np)
    B = pgk.from_numpy(B_np)
    _ = pgk.matmul(A, B).to_numpy()

    # GPU (timed)
    start = time.perf_counter()
    A = pgk.from_numpy(A_np)
    B = pgk.from_numpy(B_np)
    result = pgk.matmul(A, B)
    result_np = result.to_numpy()
    gpu_total = time.perf_counter() - start

    # GPU compute only (data already on GPU)
    A = pgk.from_numpy(A_np)
    B = pgk.from_numpy(B_np)
    start = time.perf_counter()
    result = pgk.matmul(A, B)
    gpu_compute = time.perf_counter() - start

    # CPU
    start = time.perf_counter()
    expected = A_np @ B_np
    cpu_time = time.perf_counter() - start

    gflops = 2 * M * M * M / gpu_compute / 1e9
    transfer_overhead = (gpu_total - gpu_compute) / gpu_total * 100

    print(f"   GPU total: {gpu_total*1000:.3f} ms (with H<->D transfer)")
    print(f"   GPU compute: {gpu_compute*1000:.3f} ms (data on GPU)")
    print(f"   CPU time: {cpu_time*1000:.3f} ms")
    print(f"   Transfer overhead: {transfer_overhead:.1f}%")
    print(f"   GPU GFLOPS: {gflops:.1f}")
    print(f"   Speedup (compute only): {cpu_time/gpu_compute:.2f}x")

    # Test 4: Many small operations
    print("\n4. Many Small Operations: 100x add of 10K elements")
    n = 10_000
    iterations = 100

    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)

    # GPU
    start = time.perf_counter()
    a = pgk.from_numpy(a_np)
    b = pgk.from_numpy(b_np)
    result = a
    for _ in range(iterations):
        result = pgk.add(result, b)
    result_np = result.to_numpy()
    gpu_time = time.perf_counter() - start

    # CPU
    start = time.perf_counter()
    result_cpu = a_np.copy()
    for _ in range(iterations):
        result_cpu = result_cpu + b_np
    cpu_time = time.perf_counter() - start

    print(f"   GPU time: {gpu_time*1000:.3f} ms")
    print(f"   CPU time: {cpu_time*1000:.3f} ms")
    print(f"   Speedup: {cpu_time/gpu_time:.2f}x")

    print("\n" + "=" * 70)
    print("Summary: Zero-copy operations significantly reduce overhead for")
    print("chained GPU operations where data stays on device between ops.")
    print("=" * 70)

except ImportError as e:
    print(f"Error: {e}")
    print("Running in CPU-only mode")
