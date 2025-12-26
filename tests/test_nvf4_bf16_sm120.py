"""Test NVF4-BF16 GEMM for SM120 (Blackwell GeForce)."""

import numpy as np

from pygpukit.core.factory import from_numpy
from pygpukit.ops import matmul_nvf4_bf16_sm120, nvf4_bf16_sm120_available


def bf16_to_f32(bf16_uint16: np.ndarray) -> np.ndarray:
    """Convert BFloat16 (stored as uint16) to float32.

    BFloat16 is the top 16 bits of float32, so we just left-shift by 16.
    """
    # Ensure input is uint16
    bf16_uint16 = bf16_uint16.astype(np.uint16)

    # Shift to get float32 bits
    f32_bits = bf16_uint16.astype(np.uint32) << 16

    # View as float32
    return f32_bits.view(np.float32)


def f32_to_bf16(f32: np.ndarray) -> np.ndarray:
    """Convert float32 to BFloat16 (stored as uint16).

    Just take the top 16 bits of the float32 representation.
    """
    f32 = f32.astype(np.float32)
    f32_bits = f32.view(np.uint32)
    bf16_bits = (f32_bits >> 16).astype(np.uint16)
    return bf16_bits


def test_nvf4_bf16_gemm():
    """Test NVF4-BF16 GEMM correctness."""
    print(f"NVF4-BF16 SM120 available: {nvf4_bf16_sm120_available()}")

    if not nvf4_bf16_sm120_available():
        print("NVF4-BF16 SM120 not available, skipping test")
        return

    # Test with simple values first: all 2.0
    # Expected result: 2.0 * 2.0 * K = 512 for K=128
    M, N, K = 128, 128, 128
    print(f"Testing with dimensions: M={M}, N={N}, K={K}")

    # Create input data in float32, then convert to BF16 (uint16)
    A_f32 = np.full((M, K), 2.0, dtype=np.float32)
    B_f32 = np.full((K, N), 2.0, dtype=np.float32)

    # Convert to BFloat16 representation (uint16)
    A_bf16 = f32_to_bf16(A_f32)
    B_bf16 = f32_to_bf16(B_f32)

    print(f"A[0,0] as uint16: {A_bf16[0, 0]} (0x{A_bf16[0, 0]:04X})")
    print(f"B[0,0] as uint16: {B_bf16[0, 0]} (0x{B_bf16[0, 0]:04X})")

    # Upload to GPU
    A_gpu = from_numpy(A_bf16)
    B_gpu = from_numpy(B_bf16)

    print(f"A_gpu dtype: {A_gpu.dtype}")
    print(f"B_gpu dtype: {B_gpu.dtype}")

    print("Running NVF4-BF16 GEMM...")
    try:
        C_gpu = matmul_nvf4_bf16_sm120(A_gpu, B_gpu)
        print("NVF4-BF16 GEMM succeeded!")

        # Get result as uint16 (raw BFloat16 storage)
        C_uint16 = C_gpu.to_numpy()
        print(f"C[0,0] as uint16: {C_uint16[0, 0]} (0x{C_uint16[0, 0]:04X})")

        # Convert to float32 for verification
        C_f32 = bf16_to_f32(C_uint16)
        print(f"C[0,0] as float32: {C_f32[0, 0]}")
        print(f"Output shape: {C_f32.shape}, dtype: {C_f32.dtype}")

        # Expected: 2.0 * 2.0 * 128 = 512.0
        expected = 512.0
        actual = C_f32[0, 0]
        print(f"Expected: {expected}, Actual: {actual}")

        if abs(actual - expected) < 1.0:  # Allow small tolerance for quantization
            print("PASS: NVF4-BF16 GEMM produces correct result!")
        else:
            print(f"FAIL: Expected {expected}, got {actual}")

        # Test with NVF4-appropriate random values
        # NVF4 values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} and negatives
        print("\n--- Testing with NVF4-appropriate random values ---")
        nvf4_values = np.array(
            [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        )  # Positive values only for simpler test
        A_rand = np.random.choice(nvf4_values, size=(M, K)).astype(np.float32)
        B_rand = np.random.choice(nvf4_values, size=(K, N)).astype(np.float32)

        A_rand_bf16 = f32_to_bf16(A_rand)
        B_rand_bf16 = f32_to_bf16(B_rand)

        A_rand_gpu = from_numpy(A_rand_bf16)
        B_rand_gpu = from_numpy(B_rand_bf16)

        C_rand_gpu = matmul_nvf4_bf16_sm120(A_rand_gpu, B_rand_gpu)
        C_rand_uint16 = C_rand_gpu.to_numpy()
        C_rand_f32 = bf16_to_f32(C_rand_uint16)

        # Reference: use BF16 precision for comparison
        A_rand_ref = bf16_to_f32(A_rand_bf16)
        B_rand_ref = bf16_to_f32(B_rand_bf16)
        C_ref = A_rand_ref @ B_rand_ref

        # Compare
        abs_error = np.abs(C_rand_f32 - C_ref).mean()
        ref_scale = np.abs(C_ref).mean()
        rel_error = abs_error / ref_scale if ref_scale > 0 else abs_error
        print(f"Mean absolute error: {abs_error:.6e}")
        print(f"Reference mean absolute: {ref_scale:.6e}")
        print(f"Relative error: {rel_error:.2%}")

        # With exact NVF4 values as input, quantization should be exact
        if rel_error < 0.05:  # Allow 5% for BF16 accumulation errors
            print("PASS: NVF4-BF16 GEMM with random values!")
        else:
            print(f"FAIL: Large relative error {rel_error:.2%}")

    except Exception as e:
        print(f"NVF4-BF16 GEMM failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_nvf4_bf16_gemm()
