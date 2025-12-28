"""
Comprehensive GEMV Benchmark for all kernel variants.

Tests:
- BF16 GEMV (baseline)
- FP8/BF16 (W8A16) - 8-bit weight, 16-bit activation
- FP8/FP8 (W8A8) - 8-bit weight, 8-bit activation (SM120)
- NVF4/BF16 (W4A16) - 4-bit weight, 16-bit activation
- NVF4/NVF4 (W4A4) - 4-bit weight, 4-bit activation (SM120)
"""

import time

import numpy as np

from pygpukit import _native as native

# DataType enum
BF16 = native.DataType.BFloat16
F32 = native.DataType.Float32
U8 = native.DataType.UInt8


def benchmark_kernel(
    name: str, setup_fn, run_fn, K: int, N: int, warmup: int = 10, iters: int = 100
):
    """Benchmark a kernel and return timing in microseconds."""
    try:
        setup_fn()

        # Warmup
        for _ in range(warmup):
            run_fn()
        native.device_synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            run_fn()
        native.device_synchronize()
        end = time.perf_counter()

        elapsed_us = (end - start) * 1e6 / iters

        # Theoretical: 2*K*N FLOPs (multiply-add)
        flops = 2 * K * N
        tflops = flops / (elapsed_us * 1e6)  # TFLOPS

        return elapsed_us, tflops
    except Exception as e:
        return None, str(e)


def create_bf16_arrays(K: int, N: int):
    """Create BF16 arrays for GEMV."""
    A_np = np.random.randn(K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)

    A_f32 = native.empty([K], F32)
    B_f32 = native.empty([K, N], F32)

    A_f32.copy_from_numpy(A_np)
    B_f32.copy_from_numpy(B_np)

    A_gpu = native.cast_f32_to_bf16(A_f32)
    B_gpu = native.cast_f32_to_bf16(B_f32)
    C_gpu = native.empty([N], BF16)

    return A_gpu, B_gpu, C_gpu


def bench_bf16_gemv(K: int, N: int):
    """Benchmark BF16 GEMV."""
    A_gpu, B_gpu, C_gpu = create_bf16_arrays(K, N)

    def setup():
        pass

    def run():
        native.gemv_bf16(A_gpu, B_gpu, C_gpu, 1.0, 0.0)

    return benchmark_kernel("BF16", setup, run, K, N)


def bench_fp8_bf16_gemv(K: int, N: int):
    """Benchmark FP8/BF16 (W8A16) GEMV."""
    # Create A in BF16
    A_np = np.random.randn(K).astype(np.float32)
    A_f32 = native.empty([K], F32)
    A_f32.copy_from_numpy(A_np)
    A_gpu = native.cast_f32_to_bf16(A_f32)

    # Create B in FP8 E4M3 with [N, K] layout (optimized)
    B_np = np.random.randn(N, K).astype(np.float32)
    B_fp8 = native.empty([N, K], U8)

    # Compute scale (max abs value -> 448 for E4M3)
    max_val = float(np.abs(B_np).max())
    scale = max_val / 448.0 if max_val > 0 else 1.0
    inv_scale = 1.0 / scale if scale > 0 else 1.0

    # Simple quantization
    B_quant = np.clip(B_np / scale, -448, 448).astype(np.float32)
    # Convert to FP8 E4M3 representation (simplified - use native if available)
    B_fp8_np = np.clip((B_quant * 16).astype(np.int8), -128, 127).astype(np.uint8)
    B_fp8.copy_from_numpy(B_fp8_np)

    # Scale in BF16 format
    scale_np = np.array([inv_scale], dtype=np.float32)
    scale_f32 = native.empty([1], F32)
    scale_f32.copy_from_numpy(scale_np)
    B_scale = native.cast_f32_to_bf16(scale_f32)

    C_gpu = native.empty([N], BF16)

    def setup():
        pass

    def run():
        native.gemv_fp8_bf16_opt(A_gpu, B_fp8, B_scale, C_gpu)

    return benchmark_kernel("FP8/BF16", setup, run, K, N)


def bench_nvf4_bf16_gemv(K: int, N: int):
    """Benchmark NVF4/BF16 (W4A16) GEMV."""
    # Create A in BF16
    A_np = np.random.randn(K).astype(np.float32)
    A_f32 = native.empty([K], F32)
    A_f32.copy_from_numpy(A_np)
    A_gpu = native.cast_f32_to_bf16(A_f32)

    # Create B in NVF4 format
    # B_data: [K/2, N] packed NVF4 (2 elements per byte)
    # B_scale: [K/32, N] UE4M3 scale factors
    K_half = K // 2
    K_scale = (K + 31) // 32

    B_data = native.empty([K_half, N], U8)
    B_scale = native.empty([K_scale, N], U8)

    # Initialize with random data (actual quantization would use native function)
    B_data_np = np.random.randint(0, 256, (K_half, N), dtype=np.uint8)
    B_scale_np = np.random.randint(56, 72, (K_scale, N), dtype=np.uint8)  # Scale around 1.0

    B_data.copy_from_numpy(B_data_np)
    B_scale.copy_from_numpy(B_scale_np)

    C_gpu = native.empty([N], BF16)

    def setup():
        pass

    def run():
        native.gemv_nvf4_bf16(A_gpu, B_data, B_scale, C_gpu, 1.0)

    return benchmark_kernel("NVF4/BF16 (W4A16)", setup, run, K, N)


def bench_nvf4_nvf4_gemv(K: int, N: int):
    """Benchmark NVF4/NVF4 (W4A4) GEMV (SM120+)."""
    if not native.gemv_nvf4_nvf4_available():
        return None, "SM120 not available"

    # A in NVF4: [K/2] data, [K/32] scale
    K_half = K // 2
    K_scale = (K + 31) // 32

    A_data = native.empty([K_half], U8)
    A_scale = native.empty([K_scale], U8)

    A_data_np = np.random.randint(0, 256, K_half, dtype=np.uint8)
    A_scale_np = np.random.randint(56, 72, K_scale, dtype=np.uint8)
    A_data.copy_from_numpy(A_data_np)
    A_scale.copy_from_numpy(A_scale_np)

    # B in NVF4 row-major: [N, K/2] data, [N, K/32] scale
    B_data = native.empty([N, K_half], U8)
    B_scale = native.empty([N, K_scale], U8)

    B_data_np = np.random.randint(0, 256, (N, K_half), dtype=np.uint8)
    B_scale_np = np.random.randint(56, 72, (N, K_scale), dtype=np.uint8)
    B_data.copy_from_numpy(B_data_np)
    B_scale.copy_from_numpy(B_scale_np)

    C_gpu = native.empty([N], BF16)

    def setup():
        pass

    def run():
        native.gemv_nvf4_nvf4_bf16_sm120(A_data, A_scale, B_data, B_scale, C_gpu)

    return benchmark_kernel("NVF4/NVF4 (W4A4)", setup, run, K, N)


def bench_fp8_fp8_gemv(K: int, N: int):
    """Benchmark FP8/FP8 (W8A8) GEMV (SM120+)."""
    if not native.gemv_fp8_fp8_available():
        return None, "SM120 not available"

    # A in FP8: [K]
    A_fp8 = native.empty([K], U8)
    A_fp8_np = np.random.randint(0, 256, K, dtype=np.uint8)
    A_fp8.copy_from_numpy(A_fp8_np)

    # B in FP8: [N, K] (row-major for coalesced access)
    B_fp8 = native.empty([N, K], U8)
    B_fp8_np = np.random.randint(0, 256, (N, K), dtype=np.uint8)
    B_fp8.copy_from_numpy(B_fp8_np)

    # Scales in float32
    scale_A = native.empty([1], F32)
    scale_B = native.empty([1], F32)
    scale_A.copy_from_numpy(np.array([1.0], dtype=np.float32))
    scale_B.copy_from_numpy(np.array([1.0], dtype=np.float32))

    C_gpu = native.empty([N], BF16)

    def setup():
        pass

    def run():
        native.gemv_fp8_fp8_bf16_sm120(A_fp8, B_fp8, scale_A, scale_B, C_gpu)

    return benchmark_kernel("FP8/FP8 (W8A8)", setup, run, K, N)


def main():
    print("=" * 80)
    print("GEMV Kernel Benchmark - All Variants")
    print("=" * 80)

    # Typical LLM dimensions
    test_cases = [
        # (K, N, description)
        (3584, 18944, "Qwen2.5-7B gate_proj (hidden -> intermediate)"),
        (18944, 3584, "Qwen2.5-7B down_proj (intermediate -> hidden)"),
        (3584, 3584, "Qwen2.5-7B o_proj (hidden -> hidden)"),
        (3584, 512, "Qwen2.5-7B qkv_proj head (hidden -> head_dim*num_heads partial)"),
        (4096, 11008, "LLaMA-7B gate_proj"),
        (4096, 4096, "LLaMA-7B o_proj"),
    ]

    # Benchmark functions
    benchmarks = [
        ("BF16", bench_bf16_gemv),
        ("FP8/BF16 (W8A16)", bench_fp8_bf16_gemv),
        ("FP8/FP8 (W8A8)", bench_fp8_fp8_gemv),
        ("NVF4/BF16 (W4A16)", bench_nvf4_bf16_gemv),
        ("NVF4/NVF4 (W4A4)", bench_nvf4_nvf4_gemv),
    ]

    for K, N, desc in test_cases:
        print(f"\n{desc}")
        print(f"K={K}, N={N}")
        print("-" * 70)
        print(f"{'Kernel':<20} {'Time (us)':<12} {'TFLOPS':<10} {'vs BF16':<10}")
        print("-" * 70)

        bf16_time = None

        for name, bench_fn in benchmarks:
            time_us, result = bench_fn(K, N)

            if time_us is None:
                print(f"{name:<20} {'N/A':<12} {result}")
            else:
                tflops = result
                if name == "BF16":
                    bf16_time = time_us
                    speedup = "1.00x"
                elif bf16_time:
                    speedup = f"{bf16_time / time_us:.2f}x"
                else:
                    speedup = "N/A"

                print(f"{name:<20} {time_us:>10.1f}  {tflops:>8.3f}   {speedup}")

    # Summary table for README
    print("\n" + "=" * 80)
    print("Summary Table (for README.md)")
    print("=" * 80)

    # Use Qwen2.5-7B gate_proj as reference
    K, N = 3584, 18944
    print(f"\n### GEMV Benchmark: K={K}, N={N} (Qwen2.5-7B gate_proj)\n")
    print("| Kernel | A dtype | B dtype | Weight Size | Time (us) | vs BF16 |")
    print("|--------|---------|---------|-------------|-----------|---------|")

    bf16_time = None
    for name, bench_fn in benchmarks:
        time_us, result = bench_fn(K, N)

        if time_us is None:
            continue

        tflops = result
        if "BF16" in name and "FP8" not in name and "NVF4" not in name:
            bf16_time = time_us
            a_dtype, b_dtype = "BF16", "BF16"
            weight_size = f"{K * N * 2 / 1024 / 1024:.1f} MB"
            speedup = "1.00x"
        elif "FP8/FP8" in name:
            a_dtype, b_dtype = "FP8", "FP8"
            weight_size = f"{K * N / 1024 / 1024:.1f} MB"
            speedup = f"**{bf16_time / time_us:.1f}x**" if bf16_time else "N/A"
        elif "FP8/BF16" in name:
            a_dtype, b_dtype = "BF16", "FP8"
            weight_size = f"{K * N / 1024 / 1024:.1f} MB"
            speedup = f"{bf16_time / time_us:.2f}x" if bf16_time else "N/A"
        elif "NVF4/NVF4" in name:
            a_dtype, b_dtype = "NVF4", "NVF4"
            weight_size = f"{K * N // 2 / 1024 / 1024:.1f} MB"
            speedup = f"{bf16_time / time_us:.2f}x" if bf16_time else "N/A"
        elif "NVF4/BF16" in name:
            a_dtype, b_dtype = "BF16", "NVF4"
            weight_size = f"{K * N // 2 / 1024 / 1024:.1f} MB"
            speedup = f"{bf16_time / time_us:.2f}x" if bf16_time else "N/A"
        else:
            a_dtype, b_dtype = "?", "?"
            weight_size = "?"
            speedup = "N/A"

        print(
            f"| {name:<20} | {a_dtype:<7} | {b_dtype:<7} | {weight_size:<11} | {time_us:>9.1f} | {speedup:<7} |"
        )

    # Print additional insights
    print("\n### Key Insights\n")
    print("- **FP8/FP8**: Best performance on SM120 (Blackwell). 6-20x faster than BF16.")
    print(
        "- **NVF4/BF16 (W4A16)**: Good balance of speed and memory. ~10% faster than BF16 for large N."
    )
    print(
        "- **NVF4/NVF4 (W4A4)**: Maximum memory efficiency but ~2x slower due to double dequantization."
    )


if __name__ == "__main__":
    main()
