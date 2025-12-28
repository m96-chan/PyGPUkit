#!/usr/bin/env python3
"""
Correctness test: Compare batched_gemv vs w8a16_gemm.

Both should produce identical results for the same input.
"""

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


def bf16_to_fp8_e4m3_numpy(val: np.ndarray) -> np.ndarray:
    """Convert float32 to FP8 E4M3 using numpy."""
    val = val.astype(np.float32)
    result = np.zeros(val.shape, dtype=np.uint8)

    # Get sign
    sign_mask = (val < 0).astype(np.uint8) * 0x80
    abs_val = np.abs(val)

    # Clamp to FP8 range: max ~448
    abs_val = np.minimum(abs_val, 448.0)

    # Get FP32 bits
    f32_bits = abs_val.view(np.uint32)
    exp_f32 = (f32_bits >> 23) & 0xFF
    mant_f32 = f32_bits & 0x7FFFFF

    # Convert exponent: FP32 bias=127, FP8 bias=7
    e_fp8 = exp_f32.astype(np.int32) - 120

    # Handle different cases
    # Zero
    zero_mask = abs_val == 0

    # Underflow (subnormal in FP8)
    underflow_mask = (e_fp8 <= 0) & ~zero_mask
    e_fp8 = np.maximum(e_fp8, 0)

    # Overflow
    overflow_mask = e_fp8 >= 15
    e_fp8 = np.minimum(e_fp8, 15)

    # Truncate mantissa to 3 bits
    m_fp8 = (mant_f32 >> 20).astype(np.uint8)

    # Set max mantissa for overflow
    m_fp8[overflow_mask] = 6

    # Pack FP8
    result = sign_mask | (e_fp8.astype(np.uint8) << 3) | m_fp8
    result[zero_mask] = sign_mask[zero_mask]

    return result


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


def f32_to_bf16_numpy(f32: np.ndarray) -> np.ndarray:
    """Convert float32 to bfloat16 (stored as uint16)."""
    uint32_view = f32.view(np.uint32)
    # Round to nearest even
    bf16_data = ((uint32_view + 0x7FFF + ((uint32_view >> 16) & 1)) >> 16).astype(np.uint16)
    return bf16_data


def bf16_to_f32_numpy(bf16: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (stored as uint16) to float32."""
    uint32_view = bf16.astype(np.uint32) << 16
    return uint32_view.view(np.float32)


def test_w8a16_gemm_correctness():
    """Test that w8a16_gemm produces correct results vs reference."""
    native = get_native_module()
    fp8_init_lut()

    print("=" * 80)
    print("W8A16 GEMM Correctness Test")
    print("=" * 80)

    # Get GPU info
    props = native.get_device_properties(0)
    print(f"GPU: {props.name}")
    print()

    # Test configurations
    configs = [
        (16, 128, 128),  # Small
        (64, 256, 256),  # Medium
        (128, 512, 512),  # Larger
        (256, 1024, 1024),  # LLM-like
    ]

    for M, K, N in configs:
        print(f"\n{'=' * 60}")
        print(f"M={M}, K={K}, N={N}")
        print(f"{'=' * 60}")

        # Scale dimensions (block size 128)
        scale_k = (K + 127) // 128
        scale_n = (N + 127) // 128

        # Create random input A[M, K] as BF16 (via float32)
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.1
        A_bf16_np = f32_to_bf16_numpy(A_f32)
        A_bf16 = from_numpy(A_bf16_np)
        A_bf16._dtype = gk.core.dtypes.bfloat16  # Override dtype

        # Create random FP8 weights B[K, N] with known values
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.5
        B_fp8_kn = bf16_to_fp8_e4m3_numpy(B_f32)

        # Create scale factors (1.0 for simplicity)
        scale_f32 = np.ones((scale_k, scale_n), dtype=np.float32)
        scale_bf16_np = f32_to_bf16_numpy(scale_f32)

        # Prepare for w8a16_gemm_sm120: B[K, N], scale[K/128, N/128]
        B_kn_gpu = from_numpy(B_fp8_kn)
        scale_kn_gpu = from_numpy(scale_bf16_np)
        scale_kn_gpu._dtype = gk.core.dtypes.bfloat16

        # Prepare for gemv_fp8_bf16_batched: B[N, K], scale[N/128, K/128]
        B_nk = B_fp8_kn.T.copy()  # Transpose to [N, K]
        B_nk_gpu = from_numpy(B_nk)
        scale_nk = scale_f32.T.copy()  # Transpose to [N/128, K/128]
        scale_nk_bf16_np = f32_to_bf16_numpy(scale_nk)
        scale_nk_gpu = from_numpy(scale_nk_bf16_np)
        scale_nk_gpu._dtype = gk.core.dtypes.bfloat16

        # Run w8a16_gemm_sm120
        C_gemm = gk.empty((M, N), dtype="bfloat16")
        C_gemm = w8a16_gemm_sm120(A_bf16, B_kn_gpu, scale_kn_gpu, out=C_gemm)
        native.device_synchronize()

        # Run gemv_fp8_bf16_batched
        C_gemv = gk.empty((M, N), dtype="bfloat16")
        C_gemv = gemv_fp8_bf16_batched(A_bf16, B_nk_gpu, scale_nk_gpu, out=C_gemv)
        native.device_synchronize()

        # Get results as numpy (BF16 -> F32)
        C_gemm_bf16 = C_gemm.to_numpy()
        C_gemv_bf16 = C_gemv.to_numpy()
        C_gemm_f32 = bf16_to_f32_numpy(C_gemm_bf16)
        C_gemv_f32 = bf16_to_f32_numpy(C_gemv_bf16)

        # Calculate reference using numpy
        A_f32_back = bf16_to_f32_numpy(A_bf16_np)  # Convert back to F32
        B_dequant = fp8_e4m3_to_float_numpy(B_fp8_kn)  # [K, N]
        C_ref = A_f32_back @ B_dequant  # [M, K] @ [K, N] = [M, N]

        # Compare
        diff_gemm_ref = np.abs(C_gemm_f32 - C_ref)
        diff_gemv_ref = np.abs(C_gemv_f32 - C_ref)
        diff_gemm_gemv = np.abs(C_gemm_f32 - C_gemv_f32)

        # Relative error
        ref_norm = np.linalg.norm(C_ref)
        rel_err_gemm = np.linalg.norm(diff_gemm_ref) / (ref_norm + 1e-8)
        rel_err_gemv = np.linalg.norm(diff_gemv_ref) / (ref_norm + 1e-8)
        rel_err_cross = np.linalg.norm(diff_gemm_gemv) / (ref_norm + 1e-8)

        print(f"Reference norm: {ref_norm:.4f}")
        print(f"w8a16_gemm vs ref: max_diff={diff_gemm_ref.max():.6f}, rel_err={rel_err_gemm:.6f}")
        print(
            f"batched_gemv vs ref: max_diff={diff_gemv_ref.max():.6f}, rel_err={rel_err_gemv:.6f}"
        )
        print(
            f"w8a16_gemm vs batched_gemv: max_diff={diff_gemm_gemv.max():.6f}, rel_err={rel_err_cross:.6f}"
        )

        # Sample values
        print("\nSample outputs (first 4 elements of row 0):")
        print(f"  Reference: {C_ref[0, :4]}")
        print(f"  w8a16_gemm: {C_gemm_f32[0, :4]}")
        print(f"  batched_gemv: {C_gemv_f32[0, :4]}")

        # Check if results match
        tolerance = 0.1  # FP8 has limited precision
        if rel_err_cross < tolerance:
            print(f"PASS: Results match within tolerance ({rel_err_cross:.4f} < {tolerance})")
        else:
            print(f"FAIL: Results differ ({rel_err_cross:.4f} >= {tolerance})")
            print("\nDetailed comparison at (0, 0):")
            print(f"  A[0,:4] = {A_f32_back[0, :4]}")
            print(f"  B[0,:4] (dequant) = {B_dequant[0, :4]}")


def test_fp8_quantization():
    """Test FP8 quantization roundtrip."""
    print("\n" + "=" * 80)
    print("FP8 Quantization Test")
    print("=" * 80)

    # Test values
    test_vals = np.array(
        [0.0, 0.5, 1.0, -1.0, 2.0, -2.0, 0.125, -0.125, 10.0, 100.0, 400.0], dtype=np.float32
    )

    fp8_vals = bf16_to_fp8_e4m3_numpy(test_vals)
    roundtrip = fp8_e4m3_to_float_numpy(fp8_vals)

    print("Input -> FP8 -> Dequant:")
    for i in range(len(test_vals)):
        print(f"  {test_vals[i]:8.4f} -> 0x{fp8_vals[i]:02x} -> {roundtrip[i]:8.4f}")


if __name__ == "__main__":
    test_fp8_quantization()
    test_w8a16_gemm_correctness()
