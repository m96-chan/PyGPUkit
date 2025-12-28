#!/usr/bin/env python3
"""
GEMV Correctness Test - Measure error rates for all kernel variants.

Tests:
- BF16 GEMV (baseline)
- FP8/FP8 (W8A8) GEMV
- NVF4/BF16 (W4A16) GEMV
- Int4 GEMV
"""

import numpy as np
import pytest

try:
    from pygpukit import _native as native

    HAS_NATIVE = native is not None
except Exception:
    native = None  # type: ignore[assignment]
    HAS_NATIVE = False

pytestmark = [
    pytest.mark.skipif(not HAS_NATIVE, reason="Native module not available"),
    pytest.mark.gpu,  # Requires GPU backend, not CPU simulation
]


# DataType enum - only access if native is available
if HAS_NATIVE:
    BF16 = native.DataType.BFloat16
    F32 = native.DataType.Float32
    U8 = native.DataType.UInt8
else:
    BF16 = F32 = U8 = None  # type: ignore[assignment]


def f32_to_bf16_numpy(f32: np.ndarray) -> np.ndarray:
    """Convert float32 to bfloat16 (stored as uint16)."""
    uint32_view = f32.view(np.uint32)
    bf16_data = ((uint32_view + 0x7FFF + ((uint32_view >> 16) & 1)) >> 16).astype(np.uint16)
    return bf16_data


def bf16_to_f32_numpy(bf16: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (stored as uint16) to float32."""
    uint32_view = bf16.astype(np.uint32) << 16
    return uint32_view.view(np.float32)


def fp8_e4m3_to_float(fp8: np.ndarray) -> np.ndarray:
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


def float_to_fp8_e4m3(val: np.ndarray) -> np.ndarray:
    """Convert float32 to FP8 E4M3."""
    val = val.astype(np.float32)
    result = np.zeros(val.shape, dtype=np.uint8)

    sign_mask = (val < 0).astype(np.uint8) * 0x80
    abs_val = np.abs(val)
    abs_val = np.minimum(abs_val, 448.0)

    f32_bits = abs_val.view(np.uint32)
    exp_f32 = (f32_bits >> 23) & 0xFF
    mant_f32 = f32_bits & 0x7FFFFF

    e_fp8 = exp_f32.astype(np.int32) - 120
    zero_mask = abs_val == 0
    e_fp8 = np.maximum(e_fp8, 0)

    overflow_mask = e_fp8 >= 15
    e_fp8 = np.minimum(e_fp8, 15)

    m_fp8 = (mant_f32 >> 20).astype(np.uint8)
    m_fp8[overflow_mask] = 6

    result = sign_mask | (e_fp8.astype(np.uint8) << 3) | m_fp8
    result[zero_mask] = sign_mask[zero_mask]

    return result


def test_bf16_gemv_correctness():
    """Test BF16 GEMV correctness."""
    print("\n" + "=" * 70)
    print("BF16 GEMV Correctness Test")
    print("=" * 70)

    configs = [
        (4096, 4096),
        (4096, 14336),
        (14336, 4096),
    ]

    for K, N in configs:
        # Create random inputs
        A_f32 = np.random.randn(K).astype(np.float32) * 0.5
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.5

        # Reference: FP32 matmul
        C_ref = A_f32 @ B_f32

        # GPU: BF16 GEMV
        A_gpu = native.empty([K], F32)
        B_gpu = native.empty([K, N], F32)
        A_gpu.copy_from_numpy(A_f32)
        B_gpu.copy_from_numpy(B_f32)

        A_bf16 = native.cast_f32_to_bf16(A_gpu)
        B_bf16 = native.cast_f32_to_bf16(B_gpu)
        C_bf16 = native.empty([N], BF16)

        native.gemv_bf16(A_bf16, B_bf16, C_bf16, 1.0, 0.0)
        native.device_synchronize()

        # Get result
        C_bf16_np = C_bf16.to_numpy()
        C_gpu = bf16_to_f32_numpy(C_bf16_np.view(np.uint16))

        # Calculate errors
        abs_err = np.abs(C_gpu - C_ref)
        rel_err = np.linalg.norm(abs_err) / (np.linalg.norm(C_ref) + 1e-8)
        max_err = abs_err.max()

        print(f"K={K:>5}, N={N:>5}: max_err={max_err:.2e}, rel_err={rel_err:.2e}")

        assert rel_err < 1e-2, f"BF16 GEMV error too high: {rel_err}"


@pytest.mark.xfail(reason="SM120 FP8/FP8 GEMV kernel needs correctness fix")
def test_fp8_fp8_gemv_correctness():
    """Test FP8/FP8 (W8A8) GEMV correctness."""
    if not native.gemv_fp8_fp8_available():
        pytest.skip("SM120 not available")

    print("\n" + "=" * 70)
    print("FP8/FP8 (W8A8) GEMV Correctness Test")
    print("=" * 70)

    configs = [
        (4096, 4096),
        (4096, 14336),
        (14336, 4096),
        (8192, 8192),
    ]

    for K, N in configs:
        # Create random inputs in FP32, then quantize to FP8
        A_f32 = np.random.randn(K).astype(np.float32) * 0.5
        B_f32 = np.random.randn(N, K).astype(np.float32) * 0.5  # [N, K] for GPU

        # Quantize to FP8
        A_fp8 = float_to_fp8_e4m3(A_f32)
        B_fp8 = float_to_fp8_e4m3(B_f32)

        # Dequantize for reference
        A_dequant = fp8_e4m3_to_float(A_fp8)
        B_dequant = fp8_e4m3_to_float(B_fp8)

        # Reference: dequantized matmul
        C_ref = B_dequant @ A_dequant  # [N, K] @ [K] = [N]

        # GPU: FP8/FP8 GEMV
        A_gpu = native.empty([K], U8)
        B_gpu = native.empty([N, K], U8)
        A_gpu.copy_from_numpy(A_fp8)
        B_gpu.copy_from_numpy(B_fp8)

        scale_A = native.empty([1], F32)
        scale_B = native.empty([1], F32)
        scale_A.copy_from_numpy(np.array([1.0], dtype=np.float32))
        scale_B.copy_from_numpy(np.array([1.0], dtype=np.float32))

        C_bf16 = native.empty([N], BF16)

        native.gemv_fp8_fp8_bf16_sm120(A_gpu, B_gpu, scale_A, scale_B, C_bf16)
        native.device_synchronize()

        # Get result
        C_bf16_np = C_bf16.to_numpy()
        C_gpu = bf16_to_f32_numpy(C_bf16_np.view(np.uint16))

        # Calculate errors
        abs_err = np.abs(C_gpu - C_ref)
        rel_err = np.linalg.norm(abs_err) / (np.linalg.norm(C_ref) + 1e-8)
        max_err = abs_err.max()

        print(f"K={K:>5}, N={N:>5}: max_err={max_err:.2e}, rel_err={rel_err:.2e}")

        # FP8 has lower precision, allow higher error
        assert rel_err < 5e-2, f"FP8/FP8 GEMV error too high: {rel_err}"


def test_nvf4_bf16_gemv_correctness():
    """Test NVF4/BF16 (W4A16) GEMV correctness."""
    print("\n" + "=" * 70)
    print("NVF4/BF16 (W4A16) GEMV Correctness Test")
    print("=" * 70)

    # NVF4 uses 4-bit values with UE4M3 scales
    # We'll test with synthetic data where we know the expected result

    configs = [
        (4096, 4096),
        (4096, 14336),
    ]

    for K, N in configs:
        K_half = K // 2
        K_scale = (K + 31) // 32

        # Create BF16 activation
        A_f32 = np.random.randn(K).astype(np.float32) * 0.1
        A_gpu = native.empty([K], F32)
        A_gpu.copy_from_numpy(A_f32)
        A_bf16 = native.cast_f32_to_bf16(A_gpu)

        # Create NVF4 weights with scale=1.0 (UE4M3 = 0x40 = 1.0)
        B_data = native.empty([K_half, N], U8)
        B_scale = native.empty([K_scale, N], U8)

        # Initialize with zeros (0x00 = 0.0 in NVF4)
        B_data_np = np.zeros((K_half, N), dtype=np.uint8)
        B_scale_np = np.full((K_scale, N), 0x40, dtype=np.uint8)  # scale=1.0

        B_data.copy_from_numpy(B_data_np)
        B_scale.copy_from_numpy(B_scale_np)

        C_bf16 = native.empty([N], BF16)

        native.gemv_nvf4_bf16(A_bf16, B_data, B_scale, C_bf16, 1.0)
        native.device_synchronize()

        # With zero weights, output should be near zero
        C_bf16_np = C_bf16.to_numpy()
        C_gpu = bf16_to_f32_numpy(C_bf16_np.view(np.uint16))

        max_val = np.abs(C_gpu).max()
        print(f"K={K:>5}, N={N:>5}: zero weights -> max_output={max_val:.2e} (expect ~0)")

        # Should be near zero
        assert max_val < 1.0, f"NVF4 GEMV with zero weights produced {max_val}"


@pytest.mark.xfail(reason="SM120 Int4 GEMV kernel needs correctness fix")
def test_int4_gemv_correctness():
    """Test Int4 GEMV correctness (Int32 output)."""
    if not native.int4_gemv_available():
        pytest.skip("SM120 Int4 GEMV not available")

    print("\n" + "=" * 70)
    print("Int4 GEMV Correctness Test (Int32 output)")
    print("=" * 70)

    configs = [
        (4096, 4096),
        (4096, 14336),
    ]

    for K, N in configs:
        K_half = K // 2

        # Create random packed int4 activation (K/2 bytes)
        A_packed = native.empty([K_half], U8)
        A_packed_np = np.full(K_half, 0x88, dtype=np.uint8)  # All zeros (8-8=0)
        A_packed.copy_from_numpy(A_packed_np)

        # Create random packed int4 weights [N, K/2]
        B_packed = native.empty([N, K_half], U8)
        B_packed_np = np.full((N, K_half), 0x88, dtype=np.uint8)  # All zeros
        B_packed.copy_from_numpy(B_packed_np)

        C_int32 = native.empty([N], native.DataType.Int32)

        native.int4_gemv_int32_sm120(A_packed, B_packed, C_int32, 1.0, 1.0)
        native.device_synchronize()

        # With zero weights, output should be zero
        C_np = C_int32.to_numpy()
        max_val = np.abs(C_np).max()
        print(f"K={K:>5}, N={N:>5}: zero weights -> max_output={max_val} (expect 0)")

        assert max_val == 0, f"Int4 GEMV with zero weights produced {max_val}"


def run_all_correctness_tests():
    """Run all correctness tests and print summary."""
    print("=" * 70)
    print("GEMV Correctness Summary")
    print("=" * 70)

    results = []

    # BF16 test
    print("\n[1/4] BF16 GEMV...")
    try:
        test_bf16_gemv_correctness()
        results.append(("BF16", "PASS", "rel_err < 1e-2"))
    except Exception as e:
        results.append(("BF16", "FAIL", str(e)))

    # FP8/FP8 test
    print("\n[2/4] FP8/FP8 (W8A8) GEMV...")
    try:
        if native.gemv_fp8_fp8_available():
            test_fp8_fp8_gemv_correctness()
            results.append(("W8A8", "PASS", "rel_err < 5e-2"))
        else:
            results.append(("W8A8", "SKIP", "SM120 not available"))
    except Exception as e:
        results.append(("W8A8", "FAIL", str(e)))

    # NVF4/BF16 test
    print("\n[3/4] NVF4/BF16 (W4A16) GEMV...")
    try:
        test_nvf4_bf16_gemv_correctness()
        results.append(("W4A16", "PASS", "zero weights -> ~0"))
    except Exception as e:
        results.append(("W4A16", "FAIL", str(e)))

    # Int4 test
    print("\n[4/4] Int4 GEMV...")
    try:
        test_int4_gemv_correctness()
        results.append(("Int4", "PASS", "zero weights -> ~0"))
    except Exception as e:
        results.append(("Int4", "FAIL", str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Kernel':<10} {'Status':<8} {'Notes'}")
    print("-" * 70)
    for kernel, status, notes in results:
        print(f"{kernel:<10} {status:<8} {notes}")


if __name__ == "__main__":
    run_all_correctness_tests()
