"""TF32 v2 Kernel Benchmark"""
import os
import numpy as np
import time

# Enable v2 kernel
os.environ["PYGPUKIT_TF32_V2"] = "1"

def benchmark():
    import pygpukit as gk

    if not gk.is_cuda_available():
        print("CUDA not available")
        return

    info = gk.get_device_info()
    print(f"Device: {info.name}")
    print(f"Using TF32 v2 kernel: PYGPUKIT_TF32_V2={os.environ.get('PYGPUKIT_TF32_V2', '0')}")

    sizes = [2048, 4096, 8192]

    print("\n" + "=" * 50)
    print("Performance Benchmark (TF32 v2)")
    print("=" * 50)

    for N in sizes:
        M, K = N, N

        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)

        a = gk.from_numpy(a_np)
        b = gk.from_numpy(b_np)

        # Warmup
        for _ in range(5):
            c = gk.matmul(a, b, use_tf32=True)

        # Benchmark
        num_iters = 20
        start = time.perf_counter()
        for _ in range(num_iters):
            c = gk.matmul(a, b, use_tf32=True)
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / num_iters) * 1000
        flops = 2.0 * M * N * K
        tflops = (flops / (avg_time_ms / 1000)) / 1e12

        print(f"{N}x{N}x{N}: {avg_time_ms:.2f} ms, {tflops:.2f} TFLOPS")

    # Correctness check
    print("\n" + "=" * 50)
    print("Correctness Check")
    print("=" * 50)

    all_pass = True
    for N in [256, 512, 1024, 2048]:
        a_np = np.random.randn(N, N).astype(np.float32)
        b_np = np.random.randn(N, N).astype(np.float32)

        a = gk.from_numpy(a_np)
        b = gk.from_numpy(b_np)

        c = gk.matmul(a, b, use_tf32=True)
        c_np = c.to_numpy()

        expected = a_np @ b_np

        abs_error = np.abs(c_np - expected)
        scale = np.maximum(np.abs(expected), np.abs(c_np))
        scale = np.maximum(scale, 1.0)
        rel_error = abs_error / scale
        max_rel_error = np.max(rel_error)
        mean_rel_error = np.mean(rel_error)
        p99_rel_error = np.percentile(rel_error, 99)

        # TF32 has 10 mantissa bits, allow up to 2% error for large matmuls
        status = "PASS" if p99_rel_error < 2e-2 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {N}x{N}: max={max_rel_error:.6f}, mean={mean_rel_error:.6f}, p99={p99_rel_error:.6f} [{status}]")

    print("\n" + "=" * 50)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 50)

if __name__ == "__main__":
    print("=" * 60)
    print("TF32 v2 Kernel Benchmark")
    print("=" * 60)
    benchmark()
