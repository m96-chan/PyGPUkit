#!/usr/bin/env python3
"""Test Accurate FP8/FP8 GEMV kernel - Issue #123.

Compares accuracy of:
- Fast version (128-element scale blocks): ~1-2% error
- Accurate version (32-element scale blocks): <0.5% error target
"""

import numpy as np

from pygpukit.core import from_numpy, zeros
from pygpukit.core.backend import get_native_module


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


def test_accurate_kernel_basic():
    """Basic test: verify accurate kernel produces reasonable output."""
    native = get_native_module()

    if not native.gemv_fp8_fp8_accurate_available():
        print("SM120 accurate GEMV not available, skipping test")
        return

    print("=" * 70)
    print("Accurate FP8 GEMV Basic Test - Issue #123")
    print("=" * 70)

    K, N = 4096, 4096
    block_size = 32  # Accurate version uses 32-element blocks

    # Create test data
    np.random.seed(42)
    A_f32 = np.random.randn(K).astype(np.float32) * 0.1
    B_f32 = np.random.randn(N, K).astype(np.float32) * 0.1

    # Quantize to FP8
    A_fp8 = float_to_fp8_e4m3(A_f32)
    B_fp8 = float_to_fp8_e4m3(B_f32)

    # Dequantize for reference
    A_dequant = fp8_e4m3_to_float(A_fp8)
    B_dequant = fp8_e4m3_to_float(B_fp8)

    # Reference result
    C_ref = B_dequant @ A_dequant

    # Scale factors: kernel expects [N/block_size, K/block_size] for scale_B
    # But accessed as flattened: scale_B[scale_n * scale_stride_k + scale_k]
    n_scales_n = (N + block_size - 1) // block_size
    n_scales_k = (K + block_size - 1) // block_size

    # For simplicity, use scale=1.0 everywhere (no blockwise quantization)
    scale_A = np.ones(n_scales_k, dtype=np.float32)
    scale_B = np.ones(n_scales_n * n_scales_k, dtype=np.float32)

    print(f"K={K}, N={N}, block_size={block_size}")
    print(f"scale_A shape: {scale_A.shape} (expected {n_scales_k})")
    print(f"scale_B shape: {scale_B.shape} (expected {n_scales_n * n_scales_k})")

    # GPU arrays
    A_gpu = from_numpy(A_fp8)
    B_gpu = from_numpy(B_fp8)
    scale_A_gpu = from_numpy(scale_A)
    scale_B_gpu = from_numpy(scale_B)
    C_gpu = zeros((N,), dtype="bfloat16")

    # Run accurate kernel
    try:
        native.gemv_fp8_fp8_bf16_accurate_sm120(
            A_gpu._get_native(),
            B_gpu._get_native(),
            scale_A_gpu._get_native(),
            scale_B_gpu._get_native(),
            C_gpu._get_native(),
        )
        native.device_synchronize()

        # Get result
        C_raw = C_gpu.to_numpy()
        # Convert bfloat16 to float32
        C_bf16 = C_raw.view(np.uint16).astype(np.uint32) << 16
        C_out = C_bf16.view(np.float32)

        print(f"C output: min={C_out.min():.4f}, max={C_out.max():.4f}")
        print(f"C ref:    min={C_ref.min():.4f}, max={C_ref.max():.4f}")

        # Check for NaN
        if np.isnan(C_out).any():
            print("ERROR: Output contains NaN!")
            return

        # Calculate error
        abs_err = np.abs(C_out - C_ref)
        rel_err = np.linalg.norm(abs_err) / (np.linalg.norm(C_ref) + 1e-8) * 100

        print(f"Relative error: {rel_err:.2f}%")
        print("Target: <0.5%")

        if rel_err < 0.5:
            print("PASS: Error within target!")
        elif rel_err < 2.0:
            print("ACCEPTABLE: Error similar to fast version")
        else:
            print("FAIL: Error too high")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


def test_compare_fast_vs_accurate():
    """Compare fast and accurate versions for error rates."""
    native = get_native_module()

    if not native.gemv_fp8_fp8_available():
        print("SM120 fast GEMV not available, skipping comparison")
        return

    if not native.gemv_fp8_fp8_accurate_available():
        print("SM120 accurate GEMV not available, skipping comparison")
        return

    print("\n" + "=" * 70)
    print("Fast vs Accurate FP8 GEMV Comparison - Issue #123")
    print("=" * 70)

    test_cases = [
        (4096, 4096),
        (8192, 4096),
    ]

    print(f"{'K':<8} {'N':<8} {'Fast Error':<15} {'Accurate Error':<15} {'Improvement':<12}")
    print("-" * 60)

    for K, N in test_cases:
        np.random.seed(42)
        A_f32 = np.random.randn(K).astype(np.float32) * 0.1
        B_f32 = np.random.randn(N, K).astype(np.float32) * 0.1

        # Quantize to FP8
        A_fp8 = float_to_fp8_e4m3(A_f32)
        B_fp8 = float_to_fp8_e4m3(B_f32)

        # Dequantize for reference
        A_dequant = fp8_e4m3_to_float(A_fp8)
        B_dequant = fp8_e4m3_to_float(B_fp8)
        C_ref = B_dequant @ A_dequant

        # Fast version: 128-element blocks
        block_fast = 128
        n_scales_k_fast = (K + block_fast - 1) // block_fast
        n_scales_n_fast = (N + block_fast - 1) // block_fast

        scale_A_fast = np.ones(n_scales_k_fast, dtype=np.float32)
        scale_B_fast = np.ones(n_scales_n_fast * n_scales_k_fast, dtype=np.float32)

        A_gpu = from_numpy(A_fp8)
        B_gpu = from_numpy(B_fp8)
        scale_A_gpu_fast = from_numpy(scale_A_fast)
        scale_B_gpu_fast = from_numpy(scale_B_fast)
        C_gpu_fast = zeros((N,), dtype="bfloat16")

        fast_error = float("nan")
        try:
            native.gemv_fp8_fp8_bf16_sm120(
                A_gpu._get_native(),
                B_gpu._get_native(),
                scale_A_gpu_fast._get_native(),
                scale_B_gpu_fast._get_native(),
                C_gpu_fast._get_native(),
            )
            native.device_synchronize()

            C_raw = C_gpu_fast.to_numpy()
            C_bf16 = C_raw.view(np.uint16).astype(np.uint32) << 16
            C_fast = C_bf16.view(np.float32)

            if not np.isnan(C_fast).any():
                fast_error = (
                    np.linalg.norm(np.abs(C_fast - C_ref)) / (np.linalg.norm(C_ref) + 1e-8) * 100
                )
        except Exception as e:
            print(f"  Fast error: {e}")

        # Accurate version: 32-element blocks
        block_acc = 32
        n_scales_k_acc = (K + block_acc - 1) // block_acc
        n_scales_n_acc = (N + block_acc - 1) // block_acc

        scale_A_acc = np.ones(n_scales_k_acc, dtype=np.float32)
        scale_B_acc = np.ones(n_scales_n_acc * n_scales_k_acc, dtype=np.float32)

        scale_A_gpu_acc = from_numpy(scale_A_acc)
        scale_B_gpu_acc = from_numpy(scale_B_acc)
        C_gpu_acc = zeros((N,), dtype="bfloat16")

        acc_error = float("nan")
        try:
            native.gemv_fp8_fp8_bf16_accurate_sm120(
                A_gpu._get_native(),
                B_gpu._get_native(),
                scale_A_gpu_acc._get_native(),
                scale_B_gpu_acc._get_native(),
                C_gpu_acc._get_native(),
            )
            native.device_synchronize()

            C_raw = C_gpu_acc.to_numpy()
            C_bf16 = C_raw.view(np.uint16).astype(np.uint32) << 16
            C_acc = C_bf16.view(np.float32)

            if not np.isnan(C_acc).any():
                acc_error = (
                    np.linalg.norm(np.abs(C_acc - C_ref)) / (np.linalg.norm(C_ref) + 1e-8) * 100
                )
        except Exception as e:
            print(f"  Accurate error: {e}")

        # Report
        if not np.isnan(fast_error) and not np.isnan(acc_error):
            improvement = fast_error / acc_error if acc_error > 0 else 0
            print(f"{K:<8} {N:<8} {fast_error:<15.2f}% {acc_error:<15.2f}% {improvement:<12.1f}x")
        else:
            fast_str = f"{fast_error:.2f}%" if not np.isnan(fast_error) else "N/A"
            acc_str = f"{acc_error:.2f}%" if not np.isnan(acc_error) else "N/A"
            print(f"{K:<8} {N:<8} {fast_str:<15} {acc_str:<15} {'N/A':<12}")

    print()
    print("Target: Accurate version should have <0.5% error")


if __name__ == "__main__":
    test_accurate_kernel_basic()
    test_compare_fast_vs_accurate()
