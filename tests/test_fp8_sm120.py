"""Test FP8 GEMM with compute-sanitizer."""

import numpy as np

from pygpukit.core.factory import from_numpy
from pygpukit.ops import fp8_sm120_available, matmul_fp8_sm120

print(f"FP8 SM120 available: {fp8_sm120_available()}")

if fp8_sm120_available():
    # Use exact tile size (single tile) to eliminate edge cases
    M, N, K = 128, 128, 128
    print(f"Testing with exact tile size: M={M}, N={N}, K={K}")

    A = np.random.randn(M, K).astype(np.float32) * 0.1  # Small values for FP8
    B = np.random.randn(K, N).astype(np.float32) * 0.1

    A_gpu = from_numpy(A)
    B_gpu = from_numpy(B)

    print("Running FP8 GEMM...")
    try:
        C_gpu = matmul_fp8_sm120(A_gpu, B_gpu)
        print("FP8 GEMM succeeded!")
        C = C_gpu.to_numpy()
        print(f"Output shape: {C.shape}, dtype: {C.dtype}")

        # Verify against numpy
        C_ref = A @ B
        rel_error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
        print(f"Relative error vs NumPy: {rel_error:.6e}")
    except Exception as e:
        print(f"FP8 GEMM failed: {e}")
else:
    print("FP8 SM120 not available")
