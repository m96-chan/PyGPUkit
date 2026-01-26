"""Test fused NN kernels for correctness."""
import numpy as np
import pygpukit as pk


def bf16_to_float(arr):
    """Convert BF16 (stored as uint16) to float32."""
    if arr.dtype == np.uint16:
        return (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.astype(np.float32)


def to_float32(gpu_arr):
    """Convert GPUArray to numpy float32."""
    np_arr = gpu_arr.to_numpy()
    return bf16_to_float(np_arr)


def test_rmsnorm_residual():
    """Test fused RMSNorm + Residual against separate operations."""
    print("=" * 60)
    print("Test: Fused RMSNorm + Residual")
    print("=" * 60)

    batch_size, features = 32, 4096
    eps = 1e-5

    # Create test data
    np.random.seed(42)
    x_np = np.random.randn(batch_size, features).astype(np.float32) * 0.5
    residual_np = np.random.randn(batch_size, features).astype(np.float32) * 0.5
    gamma_np = np.random.randn(features).astype(np.float32) * 0.1 + 1.0

    # Expected: rmsnorm(x + residual) * gamma
    z_np = x_np + residual_np
    rms = np.sqrt((z_np ** 2).mean(axis=-1, keepdims=True) + eps)
    expected = (z_np / rms) * gamma_np

    # GPU computation with fused kernel
    x_gpu = pk.from_numpy(x_np).astype(pk.bfloat16)
    residual_gpu = pk.from_numpy(residual_np).astype(pk.bfloat16)
    gamma_gpu = pk.from_numpy(gamma_np).astype(pk.bfloat16)

    result_gpu = pk.ops.nn.rmsnorm_residual(x_gpu, residual_gpu, gamma_gpu, eps)
    result_np = to_float32(result_gpu)

    # Compare
    diff = np.abs(result_np - expected)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Relative error (avoiding div by zero)
    mask = np.abs(expected) > 0.01
    rel_err = diff[mask] / np.abs(expected[mask]) if mask.sum() > 0 else np.array([0])

    print(f"Expected range: [{expected.min():.4f}, {expected.max():.4f}]")
    print(f"Result range:   [{result_np.min():.4f}, {result_np.max():.4f}]")
    print(f"Max abs diff:   {max_diff:.6f}")
    print(f"Mean abs diff:  {mean_diff:.6f}")
    print(f"Mean rel error: {rel_err.mean() * 100:.2f}%")
    print(f"Max rel error:  {rel_err.max() * 100:.2f}%")

    passed = max_diff < 0.05  # BF16 tolerance
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_swiglu():
    """Test fused SwiGLU against separate silu * up."""
    print("=" * 60)
    print("Test: Fused SwiGLU")
    print("=" * 60)

    batch_size, features = 32, 4096

    # Create test data
    np.random.seed(42)
    gate_np = np.random.randn(batch_size, features).astype(np.float32) * 0.5
    up_np = np.random.randn(batch_size, features).astype(np.float32) * 0.5

    # Expected: silu(gate) * up
    # silu(x) = x / (1 + exp(-x))
    silu_gate = gate_np / (1 + np.exp(-gate_np))
    expected = silu_gate * up_np

    # GPU computation with fused kernel
    gate_gpu = pk.from_numpy(gate_np).astype(pk.bfloat16)
    up_gpu = pk.from_numpy(up_np).astype(pk.bfloat16)

    result_gpu = pk.ops.nn.swiglu(gate_gpu, up_gpu)
    result_np = to_float32(result_gpu)

    # Compare
    diff = np.abs(result_np - expected)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Relative error
    mask = np.abs(expected) > 0.01
    rel_err = diff[mask] / np.abs(expected[mask]) if mask.sum() > 0 else np.array([0])

    print(f"Expected range: [{expected.min():.4f}, {expected.max():.4f}]")
    print(f"Result range:   [{result_np.min():.4f}, {result_np.max():.4f}]")
    print(f"Max abs diff:   {max_diff:.6f}")
    print(f"Mean abs diff:  {mean_diff:.6f}")
    print(f"Mean rel error: {rel_err.mean() * 100:.2f}%")
    print(f"Max rel error:  {rel_err.max() * 100:.2f}%")

    passed = max_diff < 0.05  # BF16 tolerance
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_geglu():
    """Test fused GeGLU against separate gelu * up."""
    print("=" * 60)
    print("Test: Fused GeGLU")
    print("=" * 60)

    batch_size, features = 32, 4096

    # Create test data
    np.random.seed(42)
    gate_np = np.random.randn(batch_size, features).astype(np.float32) * 0.5
    up_np = np.random.randn(batch_size, features).astype(np.float32) * 0.5

    # Expected: gelu(gate) * up
    # gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c1 = 0.7978845608  # sqrt(2/pi)
    c2 = 0.044715
    gelu_gate = gate_np * 0.5 * (1 + np.tanh(c1 * (gate_np + c2 * gate_np ** 3)))
    expected = gelu_gate * up_np

    # GPU computation with fused kernel
    gate_gpu = pk.from_numpy(gate_np).astype(pk.bfloat16)
    up_gpu = pk.from_numpy(up_np).astype(pk.bfloat16)

    result_gpu = pk.ops.nn.geglu(gate_gpu, up_gpu)
    result_np = to_float32(result_gpu)

    # Compare
    diff = np.abs(result_np - expected)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Relative error
    mask = np.abs(expected) > 0.01
    rel_err = diff[mask] / np.abs(expected[mask]) if mask.sum() > 0 else np.array([0])

    print(f"Expected range: [{expected.min():.4f}, {expected.max():.4f}]")
    print(f"Result range:   [{result_np.min():.4f}, {result_np.max():.4f}]")
    print(f"Max abs diff:   {max_diff:.6f}")
    print(f"Mean abs diff:  {mean_diff:.6f}")
    print(f"Mean rel error: {rel_err.mean() * 100:.2f}%")
    print(f"Max rel error:  {rel_err.max() * 100:.2f}%")

    passed = max_diff < 0.05  # BF16 tolerance
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_swiglu_vs_separate():
    """Test fused SwiGLU against separate GPU silu + mul."""
    print("=" * 60)
    print("Test: Fused SwiGLU vs Separate GPU ops")
    print("=" * 60)

    batch_size, features = 32, 4096

    np.random.seed(42)
    gate_np = np.random.randn(batch_size, features).astype(np.float32) * 0.5
    up_np = np.random.randn(batch_size, features).astype(np.float32) * 0.5

    gate_gpu = pk.from_numpy(gate_np).astype(pk.bfloat16)
    up_gpu = pk.from_numpy(up_np).astype(pk.bfloat16)

    # Fused kernel
    fused_result = pk.ops.nn.swiglu(gate_gpu, up_gpu)

    # Separate kernels
    silu_gate = pk.ops.nn.silu(gate_gpu)
    separate_result = silu_gate * up_gpu

    fused_np = to_float32(fused_result)
    separate_np = to_float32(separate_result)

    diff = np.abs(fused_np - separate_np)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Fused range:    [{fused_np.min():.4f}, {fused_np.max():.4f}]")
    print(f"Separate range: [{separate_np.min():.4f}, {separate_np.max():.4f}]")
    print(f"Max diff:       {max_diff:.6f}")
    print(f"Mean diff:      {mean_diff:.6f}")

    # Should match almost exactly since same hardware
    passed = max_diff < 1e-5
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


if __name__ == "__main__":
    print("Fused Kernel Correctness Tests")
    print("=" * 60)
    print()

    results = []
    results.append(("RMSNorm + Residual", test_rmsnorm_residual()))
    results.append(("SwiGLU", test_swiglu()))
    results.append(("GeGLU", test_geglu()))
    results.append(("SwiGLU vs Separate", test_swiglu_vs_separate()))

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
