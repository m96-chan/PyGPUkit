#!/usr/bin/env python3
"""Simple debug test for w8a16_gemm."""

import numpy as np
import pytest

import pygpukit as gk
from pygpukit.core import from_numpy
from pygpukit.core.backend import get_native_module

# Check if native module is available
try:
    _native = get_native_module()
    HAS_NATIVE = _native is not None
except Exception:
    HAS_NATIVE = False

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="Native module not available")

from pygpukit.ops.matmul import (
    fp8_init_lut,
    gemv_fp8_bf16_batched,
    w8a16_gemm_sm120,
)


def f32_to_bf16_numpy(f32: np.ndarray) -> np.ndarray:
    """Convert float32 to bfloat16 (stored as uint16)."""
    uint32_view = f32.view(np.uint32)
    bf16_data = ((uint32_view + 0x7FFF + ((uint32_view >> 16) & 1)) >> 16).astype(np.uint16)
    return bf16_data


def bf16_to_f32_numpy(bf16: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (stored as uint16) to float32."""
    uint32_view = bf16.astype(np.uint32) << 16
    return uint32_view.view(np.float32)


def fp8_e4m3_to_float_numpy(fp8: np.ndarray) -> np.ndarray:
    """Convert FP8 E4M3 to float32."""
    sign = (fp8 >> 7) & 1
    exp = (fp8 >> 3) & 0xF
    mant = fp8 & 0x7

    result = np.zeros_like(fp8, dtype=np.float32)

    # Normal values
    normal = exp > 0
    result[normal] = (
        ((-1.0) ** sign[normal])
        * (2.0 ** (exp[normal].astype(np.float32) - 7))
        * (1.0 + mant[normal].astype(np.float32) / 8.0)
    )

    # Subnormal values
    subnormal = (exp == 0) & (mant > 0)
    result[subnormal] = (
        ((-1.0) ** sign[subnormal]) * (2.0**-6) * (mant[subnormal].astype(np.float32) / 8.0)
    )

    return result


def test_simple():
    """Simple test with known values."""
    native = get_native_module()
    fp8_init_lut()

    # Minimal test: M=1, K=128, N=128 (single block)
    M, K, N = 1, 128, 128

    print(f"\n{'=' * 60}")
    print(f"Simple test: M={M}, K={K}, N={N}")
    print(f"{'=' * 60}")

    # Create A: all 1.0 (in BF16)
    A_f32 = np.ones((M, K), dtype=np.float32)
    A_bf16_np = f32_to_bf16_numpy(A_f32)
    A_bf16 = from_numpy(A_bf16_np)
    A_bf16._dtype = gk.core.dtypes.bfloat16

    # Create B: FP8 values = 0x38 (which is 1.0 in FP8 E4M3)
    # exp=7, mant=0 -> 2^(7-7) * 1.0 = 1.0
    B_fp8_kn = np.full((K, N), 0x38, dtype=np.uint8)

    # Scale = 1.0
    scale_k = (K + 127) // 128
    scale_n = (N + 127) // 128
    scale_f32 = np.ones((scale_k, scale_n), dtype=np.float32)
    scale_bf16_np = f32_to_bf16_numpy(scale_f32)

    # GPU arrays for w8a16_gemm
    B_kn_gpu = from_numpy(B_fp8_kn)
    scale_kn_gpu = from_numpy(scale_bf16_np)
    scale_kn_gpu._dtype = gk.core.dtypes.bfloat16

    # GPU arrays for batched_gemv (needs B[N, K])
    B_nk = B_fp8_kn.T.copy()
    B_nk_gpu = from_numpy(B_nk)
    scale_nk = scale_f32.T.copy()
    scale_nk_bf16_np = f32_to_bf16_numpy(scale_nk)
    scale_nk_gpu = from_numpy(scale_nk_bf16_np)
    scale_nk_gpu._dtype = gk.core.dtypes.bfloat16

    # Run w8a16_gemm
    C_gemm = gk.empty((M, N), dtype="bfloat16")
    C_gemm = w8a16_gemm_sm120(A_bf16, B_kn_gpu, scale_kn_gpu, out=C_gemm)
    native.device_synchronize()

    # Run batched_gemv
    C_gemv = gk.empty((M, N), dtype="bfloat16")
    C_gemv = gemv_fp8_bf16_batched(A_bf16, B_nk_gpu, scale_nk_gpu, out=C_gemv)
    native.device_synchronize()

    # Get results
    C_gemm_bf16 = C_gemm.to_numpy()
    C_gemv_bf16 = C_gemv.to_numpy()
    C_gemm_f32 = bf16_to_f32_numpy(C_gemm_bf16)
    C_gemv_f32 = bf16_to_f32_numpy(C_gemv_bf16)

    # Expected: A (all 1s) @ B (all 1s) = K = 128
    expected = K * 1.0  # = 128.0

    print(f"Expected output (A=1, B=1): {expected}")
    print(f"w8a16_gemm output: {C_gemm_f32[0, :8]}")
    print(f"batched_gemv output: {C_gemv_f32[0, :8]}")
    print()

    # Verify FP8 dequantization
    B_dequant = fp8_e4m3_to_float_numpy(B_fp8_kn)
    print(f"FP8 0x38 dequant: {B_dequant[0, 0]} (expected 1.0)")


def test_identity_matrix():
    """Test with identity-like pattern."""
    native = get_native_module()
    fp8_init_lut()

    M, K, N = 128, 128, 128

    print(f"\n{'=' * 60}")
    print(f"Identity test: M={M}, K={K}, N={N}")
    print(f"{'=' * 60}")

    # A = identity (128x128)
    A_f32 = np.eye(M, K, dtype=np.float32)
    A_bf16_np = f32_to_bf16_numpy(A_f32)
    A_bf16 = from_numpy(A_bf16_np)
    A_bf16._dtype = gk.core.dtypes.bfloat16

    # B = simple pattern: each row k has value (k % 8) * 0.125
    # FP8 for 0.125 = exp=4, mant=0 -> 0x20
    B_f32 = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        B_f32[k, :] = (k % 8) * 0.125

    # Convert to FP8 manually
    # 0.0 -> 0x00
    # 0.125 -> 0x20 (exp=4, mant=0, 2^(4-7) = 0.125)
    # 0.25 -> 0x28 (exp=5, mant=0, 2^(5-7) = 0.25)
    # 0.375 -> 0x2C (exp=5, mant=4, 0.25 * 1.5 = 0.375)
    # 0.5 -> 0x30 (exp=6, mant=0, 2^(6-7) = 0.5)
    # 0.625 -> 0x32 (exp=6, mant=2, 0.5 * 1.25 = 0.625)
    # 0.75 -> 0x34 (exp=6, mant=4, 0.5 * 1.5 = 0.75)
    # 0.875 -> 0x36 (exp=6, mant=6, 0.5 * 1.75 = 0.875)
    fp8_lut = [0x00, 0x20, 0x28, 0x2C, 0x30, 0x32, 0x34, 0x36]
    B_fp8_kn = np.zeros((K, N), dtype=np.uint8)
    for k in range(K):
        B_fp8_kn[k, :] = fp8_lut[k % 8]

    # Verify FP8 conversion
    B_dequant = fp8_e4m3_to_float_numpy(B_fp8_kn)
    print(f"B_dequant[0,0] = {B_dequant[0, 0]} (expected 0.0)")
    print(f"B_dequant[1,0] = {B_dequant[1, 0]} (expected 0.125)")
    print(f"B_dequant[7,0] = {B_dequant[7, 0]} (expected 0.875)")

    # Scale = 1.0
    scale_f32 = np.ones((1, 1), dtype=np.float32)
    scale_bf16_np = f32_to_bf16_numpy(scale_f32)

    # GPU arrays
    B_kn_gpu = from_numpy(B_fp8_kn)
    scale_kn_gpu = from_numpy(scale_bf16_np)
    scale_kn_gpu._dtype = gk.core.dtypes.bfloat16

    B_nk_gpu = from_numpy(B_fp8_kn.T.copy())
    scale_nk_gpu = from_numpy(scale_bf16_np)
    scale_nk_gpu._dtype = gk.core.dtypes.bfloat16

    # Run kernels
    C_gemm = gk.empty((M, N), dtype="bfloat16")
    C_gemm = w8a16_gemm_sm120(A_bf16, B_kn_gpu, scale_kn_gpu, out=C_gemm)
    native.device_synchronize()

    C_gemv = gk.empty((M, N), dtype="bfloat16")
    C_gemv = gemv_fp8_bf16_batched(A_bf16, B_nk_gpu, scale_nk_gpu, out=C_gemv)
    native.device_synchronize()

    # Get results
    C_gemm_f32 = bf16_to_f32_numpy(C_gemm.to_numpy())
    C_gemv_f32 = bf16_to_f32_numpy(C_gemv.to_numpy())

    # Expected: C = A @ B where A is identity, so C = B
    # C[k, n] = B[k, n] = (k % 8) * 0.125

    print("\nExpected C[0,:8] = B[0,:8] = 0.0 (row 0)")
    print(f"w8a16_gemm C[0,:8]: {C_gemm_f32[0, :8]}")
    print(f"batched_gemv C[0,:8]: {C_gemv_f32[0, :8]}")

    print("\nExpected C[1,:8] = B[1,:8] = 0.125 (row 1)")
    print(f"w8a16_gemm C[1,:8]: {C_gemm_f32[1, :8]}")
    print(f"batched_gemv C[1,:8]: {C_gemv_f32[1, :8]}")

    print("\nExpected C[7,:8] = B[7,:8] = 0.875 (row 7)")
    print(f"w8a16_gemm C[7,:8]: {C_gemm_f32[7, :8]}")
    print(f"batched_gemv C[7,:8]: {C_gemv_f32[7, :8]}")


def test_m32():
    """Test with M=32 to check if issue is M-dependent."""
    native = get_native_module()
    fp8_init_lut()

    M, K, N = 32, 128, 128

    print(f"\n{'=' * 60}")
    print(f"M=32 test: M={M}, K={K}, N={N}")
    print(f"{'=' * 60}")

    # A = all 1.0
    A_f32 = np.ones((M, K), dtype=np.float32)
    A_bf16_np = f32_to_bf16_numpy(A_f32)
    A_bf16 = from_numpy(A_bf16_np)
    A_bf16._dtype = gk.core.dtypes.bfloat16

    # B = all 1.0 (FP8 0x38)
    B_fp8_kn = np.full((K, N), 0x38, dtype=np.uint8)

    # Scale = 1.0
    scale_f32 = np.ones((1, 1), dtype=np.float32)
    scale_bf16_np = f32_to_bf16_numpy(scale_f32)

    # GPU arrays
    B_kn_gpu = from_numpy(B_fp8_kn)
    scale_kn_gpu = from_numpy(scale_bf16_np)
    scale_kn_gpu._dtype = gk.core.dtypes.bfloat16

    B_nk_gpu = from_numpy(B_fp8_kn.T.copy())
    scale_nk_gpu = from_numpy(scale_bf16_np)
    scale_nk_gpu._dtype = gk.core.dtypes.bfloat16

    # Run kernels
    C_gemm = gk.empty((M, N), dtype="bfloat16")
    C_gemm = w8a16_gemm_sm120(A_bf16, B_kn_gpu, scale_kn_gpu, out=C_gemm)
    native.device_synchronize()

    C_gemv = gk.empty((M, N), dtype="bfloat16")
    C_gemv = gemv_fp8_bf16_batched(A_bf16, B_nk_gpu, scale_nk_gpu, out=C_gemv)
    native.device_synchronize()

    # Get results
    C_gemm_f32 = bf16_to_f32_numpy(C_gemm.to_numpy())
    C_gemv_f32 = bf16_to_f32_numpy(C_gemv.to_numpy())

    expected = K * 1.0  # = 128.0
    print(f"Expected output (A=1, B=1): {expected}")
    print(f"w8a16_gemm row 0: {C_gemm_f32[0, :4]} (expecting {expected})")
    print(f"w8a16_gemm row 16: {C_gemm_f32[16, :4]} (expecting {expected})")
    print(f"w8a16_gemm row 31: {C_gemm_f32[31, :4]} (expecting {expected})")
    print(f"batched_gemv row 0: {C_gemv_f32[0, :4]}")
    print(f"batched_gemv row 16: {C_gemv_f32[16, :4]}")


def test_k_accumulation():
    """Test K accumulation with simpler B values."""
    native = get_native_module()
    fp8_init_lut()

    M, K, N = 1, 32, 128  # Single K tile

    print(f"\n{'=' * 60}")
    print(f"Single K tile test: M={M}, K={K}, N={N}")
    print(f"{'=' * 60}")

    # A = all 1.0
    A_f32 = np.ones((M, K), dtype=np.float32)
    A_bf16_np = f32_to_bf16_numpy(A_f32)
    A_bf16 = from_numpy(A_bf16_np)
    A_bf16._dtype = gk.core.dtypes.bfloat16

    # B = all 1.0 (FP8 0x38)
    B_fp8_kn = np.full((K, N), 0x38, dtype=np.uint8)

    # Scale = 1.0
    scale_f32 = np.ones((1, 1), dtype=np.float32)
    scale_bf16_np = f32_to_bf16_numpy(scale_f32)

    # GPU arrays
    B_kn_gpu = from_numpy(B_fp8_kn)
    scale_kn_gpu = from_numpy(scale_bf16_np)
    scale_kn_gpu._dtype = gk.core.dtypes.bfloat16

    B_nk_gpu = from_numpy(B_fp8_kn.T.copy())
    scale_nk_gpu = from_numpy(scale_bf16_np)
    scale_nk_gpu._dtype = gk.core.dtypes.bfloat16

    # Run kernels
    C_gemm = gk.empty((M, N), dtype="bfloat16")
    C_gemm = w8a16_gemm_sm120(A_bf16, B_kn_gpu, scale_kn_gpu, out=C_gemm)
    native.device_synchronize()

    C_gemv = gk.empty((M, N), dtype="bfloat16")
    C_gemv = gemv_fp8_bf16_batched(A_bf16, B_nk_gpu, scale_nk_gpu, out=C_gemv)
    native.device_synchronize()

    # Get results
    C_gemm_f32 = bf16_to_f32_numpy(C_gemm.to_numpy())
    C_gemv_f32 = bf16_to_f32_numpy(C_gemv.to_numpy())

    expected = K * 1.0  # = 32.0
    print(f"Expected output (K={K}): {expected}")
    print(f"w8a16_gemm: {C_gemm_f32[0, :8]}")
    print(f"batched_gemv: {C_gemv_f32[0, :8]}")


def test_single_mma():
    """Test with exactly one MMA operation (K=16)."""
    native = get_native_module()
    fp8_init_lut()

    M, K, N = 1, 16, 128  # Exactly one MMA_K

    print(f"\n{'=' * 60}")
    print(f"Single MMA test: M={M}, K={K}, N={N} (MMA_K=16)")
    print(f"{'=' * 60}")

    # A = all 1.0
    A_f32 = np.ones((M, K), dtype=np.float32)
    A_bf16_np = f32_to_bf16_numpy(A_f32)
    A_bf16 = from_numpy(A_bf16_np)
    A_bf16._dtype = gk.core.dtypes.bfloat16

    # B = all 1.0 (FP8 0x38)
    B_fp8_kn = np.full((K, N), 0x38, dtype=np.uint8)

    # Scale = 1.0
    scale_f32 = np.ones((1, 1), dtype=np.float32)
    scale_bf16_np = f32_to_bf16_numpy(scale_f32)

    # GPU arrays
    B_kn_gpu = from_numpy(B_fp8_kn)
    scale_kn_gpu = from_numpy(scale_bf16_np)
    scale_kn_gpu._dtype = gk.core.dtypes.bfloat16

    B_nk_gpu = from_numpy(B_fp8_kn.T.copy())
    scale_nk_gpu = from_numpy(scale_bf16_np)
    scale_nk_gpu._dtype = gk.core.dtypes.bfloat16

    # Run kernels
    C_gemm = gk.empty((M, N), dtype="bfloat16")
    C_gemm = w8a16_gemm_sm120(A_bf16, B_kn_gpu, scale_kn_gpu, out=C_gemm)
    native.device_synchronize()

    C_gemv = gk.empty((M, N), dtype="bfloat16")
    C_gemv = gemv_fp8_bf16_batched(A_bf16, B_nk_gpu, scale_nk_gpu, out=C_gemv)
    native.device_synchronize()

    # Get results
    C_gemm_f32 = bf16_to_f32_numpy(C_gemm.to_numpy())
    C_gemv_f32 = bf16_to_f32_numpy(C_gemv.to_numpy())

    expected = K * 1.0  # = 16.0
    print(f"Expected output (K={K}): {expected}")
    print(f"w8a16_gemm: {C_gemm_f32[0, :8]}")
    print(f"batched_gemv: {C_gemv_f32[0, :8]}")


def test_m16():
    """Test with M=16 (exactly one MMA_M tile per warp)."""
    native = get_native_module()
    fp8_init_lut()

    M, K, N = 16, 128, 128

    print(f"\n{'=' * 60}")
    print(f"M=16 test: M={M}, K={K}, N={N}")
    print(f"{'=' * 60}")

    # A = all 1.0
    A_f32 = np.ones((M, K), dtype=np.float32)
    A_bf16_np = f32_to_bf16_numpy(A_f32)
    A_bf16 = from_numpy(A_bf16_np)
    A_bf16._dtype = gk.core.dtypes.bfloat16

    # B = all 1.0 (FP8 0x38)
    B_fp8_kn = np.full((K, N), 0x38, dtype=np.uint8)

    # Scale = 1.0
    scale_f32 = np.ones((1, 1), dtype=np.float32)
    scale_bf16_np = f32_to_bf16_numpy(scale_f32)

    # GPU arrays
    B_kn_gpu = from_numpy(B_fp8_kn)
    scale_kn_gpu = from_numpy(scale_bf16_np)
    scale_kn_gpu._dtype = gk.core.dtypes.bfloat16

    B_nk_gpu = from_numpy(B_fp8_kn.T.copy())
    scale_nk_gpu = from_numpy(scale_bf16_np)
    scale_nk_gpu._dtype = gk.core.dtypes.bfloat16

    # Run kernels
    C_gemm = gk.empty((M, N), dtype="bfloat16")
    C_gemm = w8a16_gemm_sm120(A_bf16, B_kn_gpu, scale_kn_gpu, out=C_gemm)
    native.device_synchronize()

    C_gemv = gk.empty((M, N), dtype="bfloat16")
    C_gemv = gemv_fp8_bf16_batched(A_bf16, B_nk_gpu, scale_nk_gpu, out=C_gemv)
    native.device_synchronize()

    # Get results
    C_gemm_f32 = bf16_to_f32_numpy(C_gemm.to_numpy())
    C_gemv_f32 = bf16_to_f32_numpy(C_gemv.to_numpy())

    expected = K * 1.0  # = 128.0
    print(f"Expected output: {expected}")
    print(f"w8a16_gemm row 0: {C_gemm_f32[0, :4]}")
    print(f"w8a16_gemm row 15: {C_gemm_f32[15, :4]}")
    print(f"batched_gemv row 0: {C_gemv_f32[0, :4]}")
    print(f"batched_gemv row 15: {C_gemv_f32[15, :4]}")


if __name__ == "__main__":
    test_simple()
    test_identity_matrix()
    test_m32()
    test_k_accumulation()
    test_single_mma()
    test_m16()
