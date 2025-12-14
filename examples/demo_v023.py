"""
PyGPUkit v0.2.3 Demo - TF32 TensorCore GEMM

This demo showcases v0.2.3 features:
1. TF32 TensorCore matmul with use_tf32=True
2. DeviceCapabilities API for TensorCore detection
3. Performance comparison: FP32 vs TF32
4. Correctness validation

Requirements:
- NVIDIA Ampere+ GPU (RTX 30XX, A100, etc.)
- NVIDIA GPU drivers installed
"""

import os
import sys
import time

# Add CUDA DLLs to PATH (Windows)
cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
cuda_bin = os.path.join(cuda_path, "bin")
if cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(cuda_bin)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str):
    print(f"\n--- {title} ---")


def main():
    print_header("PyGPUkit v0.2.3 Demo - TF32 TensorCore GEMM")

    # Import pygpukit
    try:
        import pygpukit as gp
    except ImportError as e:
        print(f"Failed to import pygpukit: {e}")
        return 1

    print(f"PyGPUkit version: {gp.__version__}")

    # =========================================================================
    # 1. Device Capabilities
    # =========================================================================
    print_section("1. Device Capabilities")

    if not gp.is_cuda_available():
        print("CUDA not available - running in CPU simulation mode")
        print("TF32 TensorCore features require an NVIDIA Ampere+ GPU")
        return 0

    # Get device info
    info = gp.get_device_info()
    print(f"GPU: {info.name}")
    print(f"Compute Capability: {info.compute_capability}")
    print(f"Total Memory: {info.total_memory / 1e9:.1f} GB")

    # Get device capabilities (v0.2.3 feature)
    caps = gp.get_device_capabilities()
    print(f"\nDevice Capabilities:")
    print(f"  SM Version: {caps.sm_version}")
    print(f"  TensorCore (TF32): {caps.tensorcore}")
    print(f"  TensorCore (FP16): {caps.tensorcore_fp16}")
    print(f"  TensorCore (BF16): {caps.tensorcore_bf16}")
    print(f"  Async Copy (cp.async): {caps.async_copy}")

    if not caps.tensorcore:
        print("\nWarning: TF32 TensorCore not available (requires SM >= 80)")
        print("FP32 fallback will be used for use_tf32=True")

    # =========================================================================
    # 2. Basic TF32 Matmul
    # =========================================================================
    print_section("2. Basic TF32 Matmul")

    np.random.seed(42)
    M, N, K = 1024, 1024, 1024

    a_np = np.random.rand(M, K).astype(np.float32)
    b_np = np.random.rand(K, N).astype(np.float32)

    a = gp.from_numpy(a_np)
    b = gp.from_numpy(b_np)

    print(f"Matrix size: {M}x{K} @ {K}x{N}")

    # FP32 matmul (default)
    c_fp32 = gp.matmul(a, b, use_tf32=False)
    print(f"FP32 matmul: shape={c_fp32.shape}, dtype={c_fp32.dtype}")

    # TF32 matmul (v0.2.3 feature)
    c_tf32 = gp.matmul(a, b, use_tf32=True)
    print(f"TF32 matmul: shape={c_tf32.shape}, dtype={c_tf32.dtype}")

    # =========================================================================
    # 3. Correctness Validation
    # =========================================================================
    print_section("3. Correctness Validation")

    expected = np.matmul(a_np, b_np)
    result_fp32 = c_fp32.to_numpy()
    result_tf32 = c_tf32.to_numpy()

    # FP32 error
    fp32_abs_err = np.max(np.abs(result_fp32 - expected))
    fp32_rel_err = np.max(np.abs(result_fp32 - expected) / (np.abs(expected) + 1e-8))
    print(f"\nFP32 Error:")
    print(f"  Max absolute error: {fp32_abs_err:.6e}")
    print(f"  Max relative error: {fp32_rel_err:.6e} ({fp32_rel_err*100:.4f}%)")

    # TF32 error (expected to be higher due to reduced precision)
    tf32_abs_err = np.max(np.abs(result_tf32 - expected))
    tf32_rel_err = np.max(np.abs(result_tf32 - expected) / (np.abs(expected) + 1e-8))
    print(f"\nTF32 Error:")
    print(f"  Max absolute error: {tf32_abs_err:.6e}")
    print(f"  Max relative error: {tf32_rel_err:.6e} ({tf32_rel_err*100:.4f}%)")

    # TF32 typically has ~0.1% error per op, accumulating to ~1-5% for large K
    if tf32_rel_err < 0.1:  # 10% threshold
        print("\n  Status: PASS (within TF32 tolerance)")
    else:
        print("\n  Status: WARNING (higher than expected error)")

    # =========================================================================
    # 4. Performance Benchmark
    # =========================================================================
    print_section("4. Performance Benchmark")

    sizes = [(2048, 2048, 2048), (4096, 4096, 4096)]

    # Add 8192 only if enough memory
    if info.total_memory >= 8 * 1024**3:
        sizes.append((8192, 8192, 8192))

    warmup_iters = 3
    bench_iters = 10

    print(f"\nWarmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {bench_iters}")
    print()

    results = []

    for M, N, K in sizes:
        print(f"Matrix size: {M}x{K} @ {K}x{N}")

        a_np = np.random.rand(M, K).astype(np.float32)
        b_np = np.random.rand(K, N).astype(np.float32)
        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        flops = 2.0 * M * N * K

        # Benchmark FP32
        for _ in range(warmup_iters):
            _ = gp.matmul(a, b, use_tf32=False)

        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = gp.matmul(a, b, use_tf32=False)
        fp32_time = (time.perf_counter() - start) / bench_iters
        fp32_tflops = flops / fp32_time / 1e12

        # Benchmark TF32
        for _ in range(warmup_iters):
            _ = gp.matmul(a, b, use_tf32=True)

        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = gp.matmul(a, b, use_tf32=True)
        tf32_time = (time.perf_counter() - start) / bench_iters
        tf32_tflops = flops / tf32_time / 1e12

        speedup = tf32_tflops / fp32_tflops if fp32_tflops > 0 else 0

        print(f"  FP32: {fp32_tflops:6.2f} TFLOPS ({fp32_time*1000:.2f} ms)")
        print(f"  TF32: {tf32_tflops:6.2f} TFLOPS ({tf32_time*1000:.2f} ms)")
        print(f"  Speedup: {speedup:.2f}x")
        print()

        results.append({
            "size": f"{M}x{N}x{K}",
            "fp32": fp32_tflops,
            "tf32": tf32_tflops,
            "speedup": speedup
        })

    # =========================================================================
    # 5. Summary
    # =========================================================================
    print_section("5. Summary")

    print("\nPerformance Results:")
    print("-" * 50)
    print(f"{'Size':<16} {'FP32':>10} {'TF32':>10} {'Speedup':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['size']:<16} {r['fp32']:>8.2f}T {r['tf32']:>8.2f}T {r['speedup']:>9.2f}x")
    print("-" * 50)

    print("\nv0.2.3 Features Demonstrated:")
    print("  [x] gp.matmul(a, b, use_tf32=True)  - TF32 TensorCore matmul")
    print("  [x] gp.get_device_capabilities()   - TensorCore detection")
    print("  [x] DeviceCapabilities.tensorcore  - SM >= 80 check")
    print("  [x] DeviceCapabilities.sm_version  - Compute capability")

    print("\nFor more details, see: docs/tf32_tensorcore_design.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
