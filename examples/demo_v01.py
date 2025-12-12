#!/usr/bin/env python
"""PyGPUkit v0.1 Demo Script."""

import time

import numpy as np

import pygpukit as gp

print("=" * 60)
print("PyGPUkit v0.1 Demo")
print("=" * 60)

# Check if CUDA is available
cuda_available = gp.is_cuda_available()
print(f"\nCUDA Available: {cuda_available}")

# Get device info
info = gp.get_device_info()
print(f"Device: {info.name}")
print(f"Total Memory: {info.total_memory / (1024**3):.2f} GB")
if info.compute_capability:
    print(f"Compute Capability: {info.compute_capability[0]}.{info.compute_capability[1]}")
print(f"Multiprocessors: {info.multiprocessor_count}")

print("\n" + "-" * 60)
print("1. GPUArray Creation")
print("-" * 60)

# Create arrays
x = gp.zeros((1024, 1024), dtype="float32")
print(f"zeros: {x}")
print(f"  shape={x.shape}, size={x.size}, nbytes={x.nbytes}")

y = gp.ones((1024, 1024), dtype="float32")
print(f"ones: {y}")

np_arr = np.random.rand(5, 5).astype(np.float32)
z = gp.from_numpy(np_arr)
print(f"from_numpy: {z}")

print("\n" + "-" * 60)
print("2. Data Transfer (CPU <-> GPU)")
print("-" * 60)

original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(f"Original NumPy array:\n{original}")

gpu_arr = gp.from_numpy(original)
print(f"Transferred to GPU: {gpu_arr}")

back_to_cpu = gpu_arr.to_numpy()
print(f"Back to CPU:\n{back_to_cpu}")
print(f"Data preserved: {np.allclose(original, back_to_cpu)}")

print("\n" + "-" * 60)
print("3. Basic Operations")
print("-" * 60)

a = gp.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float32))
b = gp.from_numpy(np.array([10, 20, 30, 40, 50], dtype=np.float32))

# Add
c = gp.add(a, b)
print(f"add([1,2,3,4,5], [10,20,30,40,50]) = {c.to_numpy()}")

# Mul
d = gp.mul(a, b)
print(f"mul([1,2,3,4,5], [10,20,30,40,50]) = {d.to_numpy()}")

# Matmul
m1 = gp.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
m2 = gp.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32))
m3 = gp.matmul(m1, m2)
print("matmul([[1,2],[3,4]], [[5,6],[7,8]]) =")
print(m3.to_numpy())

print("\n" + "-" * 60)
print("4. JIT Kernel Compilation")
print("-" * 60)

src = """
extern "C" __global__
void scale(float* x, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= factor;
}
"""
kernel = gp.jit(src, func="scale")
print(f"Compiled kernel: {kernel}")
print(f"  name: {kernel.name}")
print(f"  block_size: {kernel.block_size}")
print(f"  is_compiled: {kernel.is_compiled}")

print("\n" + "-" * 60)
print("5. Stream Management")
print("-" * 60)

stream_high = gp.StreamManager().create_stream(priority="high")
stream_low = gp.StreamManager().create_stream(priority="low")
print(f"High priority stream: {stream_high}")
print(f"Low priority stream: {stream_low}")

default = gp.default_stream()
print(f"Default stream: {default}")

print("\n" + "-" * 60)
print("6. Large Matrix Operations")
print("-" * 60)

# Large matrix multiplication
size = 512
A = gp.from_numpy(np.random.rand(size, size).astype(np.float32))
B = gp.from_numpy(np.random.rand(size, size).astype(np.float32))

start = time.perf_counter()
C = gp.matmul(A, B)
result = C.to_numpy()  # Force sync
elapsed = time.perf_counter() - start

print(f"Matrix multiplication {size}x{size}:")
print(f"  Time: {elapsed*1000:.2f} ms")
print(f"  Result shape: {C.shape}")
print(f"  Result sample [0,0]: {result[0,0]:.4f}")

# Verify correctness
A_np = A.to_numpy()
B_np = B.to_numpy()
expected = np.matmul(A_np, B_np)
max_diff = np.max(np.abs(result - expected))
print(f"  Max difference from NumPy: {max_diff:.2e}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
