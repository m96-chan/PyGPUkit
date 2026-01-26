"""Benchmark fused NN kernels vs separate operations."""
import time
import numpy as np
import pygpukit as pk
from pygpukit.core.backend import get_native_module


def _sync():
    """Sync GPU."""
    native = get_native_module()
    native.device_synchronize()


def benchmark(name, fn, warmup=10, iterations=100):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        fn()
    _sync()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync()
    end = time.perf_counter()

    total_ms = (end - start) * 1000
    per_iter_us = total_ms * 1000 / iterations
    return per_iter_us


def benchmark_swiglu():
    """Benchmark fused SwiGLU vs separate silu + mul."""
    print("=" * 70)
    print("Benchmark: SwiGLU (Fused vs Separate)")
    print("=" * 70)

    configs = [
        (1, 4096, 14336),    # Qwen-7B single token
        (32, 4096, 14336),   # Qwen-7B batch
        (1, 3584, 18944),    # Qwen-32B single token
        (8, 3584, 18944),    # Qwen-32B batch
    ]

    print(f"{'Batch':>6} {'Features':>10} {'Fused (us)':>12} {'Separate (us)':>14} {'Speedup':>8}")
    print("-" * 70)

    for batch, features, hidden_features in configs:
        # Test with intermediate_size (hidden_features)
        shape = (batch, hidden_features)

        np.random.seed(42)
        gate = pk.from_numpy(np.random.randn(*shape).astype(np.float32)).astype(pk.bfloat16)
        up = pk.from_numpy(np.random.randn(*shape).astype(np.float32)).astype(pk.bfloat16)
        out = pk.zeros(shape, dtype=pk.bfloat16)

        # Fused kernel
        def fused_op():
            pk.ops.nn.swiglu(gate, up, out=out)

        # Separate kernels
        def separate_op():
            silu_gate = pk.ops.nn.silu(gate)
            _ = silu_gate * up

        fused_us = benchmark("fused", fused_op)
        separate_us = benchmark("separate", separate_op)
        speedup = separate_us / fused_us

        print(f"{batch:>6} {hidden_features:>10} {fused_us:>12.2f} {separate_us:>14.2f} {speedup:>7.2f}x")

    print()


def benchmark_rmsnorm_residual():
    """Benchmark fused RMSNorm+Residual vs separate add + rmsnorm."""
    print("=" * 70)
    print("Benchmark: RMSNorm + Residual (Fused vs Separate)")
    print("=" * 70)

    configs = [
        (1, 4096),      # Qwen-7B single token
        (32, 4096),     # Qwen-7B batch
        (1, 3584),      # Qwen-32B single token
        (8, 3584),      # Qwen-32B batch
        (128, 4096),    # Large batch
    ]

    print(f"{'Batch':>6} {'Features':>10} {'Fused (us)':>12} {'Separate (us)':>14} {'Speedup':>8}")
    print("-" * 70)

    for batch, features in configs:
        np.random.seed(42)
        x = pk.from_numpy(np.random.randn(batch, features).astype(np.float32)).astype(pk.bfloat16)
        residual = pk.from_numpy(np.random.randn(batch, features).astype(np.float32)).astype(pk.bfloat16)
        gamma = pk.from_numpy(np.random.randn(features).astype(np.float32) * 0.1 + 1.0).astype(pk.bfloat16)
        out = pk.zeros((batch, features), dtype=pk.bfloat16)

        # Fused kernel
        def fused_op():
            pk.ops.nn.rmsnorm_residual(x, residual, gamma, out=out)

        # Separate kernels
        def separate_op():
            z = x + residual
            _ = pk.ops.nn.rmsnorm(z, gamma)

        fused_us = benchmark("fused", fused_op)
        separate_us = benchmark("separate", separate_op)
        speedup = separate_us / fused_us

        print(f"{batch:>6} {features:>10} {fused_us:>12.2f} {separate_us:>14.2f} {speedup:>7.2f}x")

    print()


def benchmark_geglu():
    """Benchmark fused GeGLU vs separate gelu + mul."""
    print("=" * 70)
    print("Benchmark: GeGLU (Fused vs Separate)")
    print("=" * 70)

    configs = [
        (1, 14336),     # Single token, large intermediate
        (32, 14336),    # Batch
        (1, 18944),     # Larger model
        (8, 18944),     # Batch
    ]

    print(f"{'Batch':>6} {'Features':>10} {'Fused (us)':>12} {'Separate (us)':>14} {'Speedup':>8}")
    print("-" * 70)

    for batch, features in configs:
        np.random.seed(42)
        gate = pk.from_numpy(np.random.randn(batch, features).astype(np.float32)).astype(pk.bfloat16)
        up = pk.from_numpy(np.random.randn(batch, features).astype(np.float32)).astype(pk.bfloat16)
        out = pk.zeros((batch, features), dtype=pk.bfloat16)

        # Fused kernel
        def fused_op():
            pk.ops.nn.geglu(gate, up, out=out)

        # Separate kernels
        def separate_op():
            gelu_gate = pk.ops.nn.gelu(gate)
            _ = gelu_gate * up

        fused_us = benchmark("fused", fused_op)
        separate_us = benchmark("separate", separate_op)
        speedup = separate_us / fused_us

        print(f"{batch:>6} {features:>10} {fused_us:>12.2f} {separate_us:>14.2f} {speedup:>7.2f}x")

    print()


if __name__ == "__main__":
    print("Fused Kernel Performance Benchmarks")
    print("=" * 70)
    print()

    benchmark_swiglu()
    benchmark_rmsnorm_residual()
    benchmark_geglu()

    print("=" * 70)
    print("Notes:")
    print("- Fused kernels reduce kernel launch overhead and memory bandwidth")
    print("- Expected speedup: 1.5-2x for memory-bound operations")
    print("- Speedup varies with batch size (larger batch = more parallelism)")
