"""PyGPUkit v0.1 GPU Demo - RTX 3090 Ti"""

import os
import sys
import time

# Add CUDA DLLs to PATH
cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
cuda_bin = os.path.join(cuda_path, "bin")
if cuda_bin not in os.environ["PATH"]:
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]

# Add DLL directory for Python 3.8+
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(cuda_bin)

# Add native module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "pygpukit"))

import numpy as np  # noqa: E402

print("=" * 60)
print("PyGPUkit v0.1 GPU Demo - RTX 3090 Ti")
print("=" * 60)

# Try to import native module directly
try:
    import _pygpukit_native as native

    print("\n[OK] Native module loaded!")
    print(f"     CUDA available: {native.is_cuda_available()}")

    if native.is_cuda_available():
        print(f"     Device count: {native.get_device_count()}")
        props = native.get_device_properties(0)
        print(f"     Device name: {props.name}")
        print(f"     Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(
            f"     Compute capability: {props.compute_capability_major}.{props.compute_capability_minor}"
        )
        print(f"     SM count: {props.multiprocessor_count}")

        # NVRTC version
        nvrtc_ver = native.get_nvrtc_version()
        print(f"     NVRTC version: {nvrtc_ver[0]}.{nvrtc_ver[1]}")

        print("\n" + "-" * 60)
        print("GPU Operations Demo")
        print("-" * 60)

        # Test 1: Element-wise Add
        print("\n1. Element-wise Addition (1M elements)")
        n_elements = 1_000_000
        a_np = np.random.randn(n_elements).astype(np.float32)
        b_np = np.random.randn(n_elements).astype(np.float32)

        # GPU
        start = time.perf_counter()
        a_gpu = native.from_numpy(a_np)
        b_gpu = native.from_numpy(b_np)
        c_gpu = native.add(a_gpu, b_gpu)
        c_result = c_gpu.to_numpy()
        gpu_time = time.perf_counter() - start

        # CPU
        start = time.perf_counter()
        c_cpu = a_np + b_np
        cpu_time = time.perf_counter() - start

        # Verify
        max_diff = np.max(np.abs(c_result - c_cpu))
        print(f"   GPU time: {gpu_time*1000:.3f} ms")
        print(f"   CPU time: {cpu_time*1000:.3f} ms")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"   Max diff: {max_diff:.2e} (should be ~0)")

        # Test 2: Element-wise Multiply
        print("\n2. Element-wise Multiplication (1M elements)")
        start = time.perf_counter()
        a_gpu = native.from_numpy(a_np)
        b_gpu = native.from_numpy(b_np)
        c_gpu = native.mul(a_gpu, b_gpu)
        c_result = c_gpu.to_numpy()
        gpu_time = time.perf_counter() - start

        start = time.perf_counter()
        c_cpu = a_np * b_np
        cpu_time = time.perf_counter() - start

        max_diff = np.max(np.abs(c_result - c_cpu))
        print(f"   GPU time: {gpu_time*1000:.3f} ms")
        print(f"   CPU time: {cpu_time*1000:.3f} ms")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"   Max diff: {max_diff:.2e}")

        # Test 3: Matrix Multiplication
        print("\n3. Matrix Multiplication (1024x1024)")
        M, N, K = 1024, 1024, 1024
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)

        # GPU
        start = time.perf_counter()
        A_gpu = native.from_numpy(A_np)
        B_gpu = native.from_numpy(B_np)
        C_gpu = native.matmul(A_gpu, B_gpu)
        C_result = C_gpu.to_numpy()
        gpu_time = time.perf_counter() - start

        # CPU (NumPy uses optimized BLAS)
        start = time.perf_counter()
        C_cpu = np.matmul(A_np, B_np)
        cpu_time = time.perf_counter() - start

        max_diff = np.max(np.abs(C_result - C_cpu))
        rel_error = max_diff / np.max(np.abs(C_cpu))
        print(f"   GPU time: {gpu_time*1000:.3f} ms")
        print(f"   CPU time: {cpu_time*1000:.3f} ms")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"   Max diff: {max_diff:.2e}")
        print(f"   Rel error: {rel_error:.2e}")

        # Test 4: JIT Kernel
        print("\n4. JIT Compilation (custom CUDA kernel)")
        kernel_src = """
        extern "C" __global__
        void scale_add(float* x, float scale, float offset, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                x[idx] = x[idx] * scale + offset;
            }
        }
        """

        start = time.perf_counter()
        kernel = native.JITKernel(kernel_src, "scale_add")
        compile_time = time.perf_counter() - start

        print(f"   Compilation time: {compile_time*1000:.3f} ms")
        print(f"   Kernel compiled: {kernel.is_compiled}")
        print(f"   PTX length: {len(kernel.ptx)} bytes")

        # Test 5: Large Matrix Multiplication
        print("\n5. Large Matrix Multiplication (2048x2048)")
        M, N, K = 2048, 2048, 2048
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)

        # Warm up
        A_gpu = native.from_numpy(A_np)
        B_gpu = native.from_numpy(B_np)
        _ = native.matmul(A_gpu, B_gpu)

        # Timed run
        start = time.perf_counter()
        A_gpu = native.from_numpy(A_np)
        B_gpu = native.from_numpy(B_np)
        C_gpu = native.matmul(A_gpu, B_gpu)
        C_result = C_gpu.to_numpy()
        gpu_time = time.perf_counter() - start

        start = time.perf_counter()
        C_cpu = np.matmul(A_np, B_np)
        cpu_time = time.perf_counter() - start

        gflops = 2 * M * N * K / gpu_time / 1e9
        print(f"   GPU time: {gpu_time*1000:.3f} ms")
        print(f"   CPU time: {cpu_time*1000:.3f} ms")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"   GPU GFLOPS: {gflops:.1f}")

        print("\n" + "=" * 60)
        print("All GPU tests completed successfully!")
        print("=" * 60)

    else:
        print("CUDA not available!")

except ImportError as e:
    print(f"\n[ERROR] Failed to load native module: {e}")
    print("\nFalling back to CPU simulation mode...")

    # Import PyGPUkit with CPU backend
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    import pygpukit as pgk

    print(f"\nPyGPUkit version: {pgk.__version__}")
    print("Running in CPU simulation mode")

    # Test basic operations
    a = pgk.from_numpy(np.array([1, 2, 3], dtype=np.float32))
    b = pgk.from_numpy(np.array([4, 5, 6], dtype=np.float32))
    c = pgk.add(a, b)
    print(f"\nAdd result: {c.to_numpy()}")
