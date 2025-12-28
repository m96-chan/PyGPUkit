"""Test all Triton kernels with PyGPUkit."""

import numpy as np
import pytest

# Check if native module and Triton are available
try:
    import pygpukit._pygpukit_native as native

    from pygpukit.triton import from_gpuarray, kernels, triton_available

    HAS_NATIVE = native is not None
    HAS_TRITON = triton_available()
except ImportError:
    native = None  # type: ignore[assignment]
    HAS_NATIVE = False
    HAS_TRITON = False

pytestmark = [
    pytest.mark.skipif(not HAS_NATIVE, reason="Native module not available"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton not available"),
    pytest.mark.gpu,
]


def rmsnorm_numpy(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Reference RMSNorm implementation in NumPy."""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def layernorm_numpy(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None, eps: float = 1e-5
) -> np.ndarray:
    """Reference LayerNorm implementation in NumPy."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    result = normalized * weight
    if bias is not None:
        result += bias
    return result


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    """Reference Softmax implementation in NumPy."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def test_rmsnorm():
    """Test RMSNorm kernel."""
    print("\n=== RMSNorm Test ===")
    batch, seq, hidden = 2, 4, 128

    x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
    w_np = np.random.randn(hidden).astype(np.float32)
    expected = rmsnorm_numpy(x_np, w_np)

    x = native.from_numpy(x_np)
    w = native.from_numpy(w_np)
    out = native.empty([batch, seq, hidden], native.Float32)

    tx, tw, tout = from_gpuarray(x), from_gpuarray(w), from_gpuarray(out)
    kernels.rmsnorm(tx, tw, tout, eps=1e-6)
    native.device_synchronize()

    out_np = out.to_numpy()
    max_diff = np.max(np.abs(out_np - expected))

    passed = np.allclose(out_np, expected, rtol=1e-4, atol=1e-4)
    print(f"Max diff: {max_diff:.6e} - {'PASS' if passed else 'FAIL'}")
    return passed


def test_layernorm():
    """Test LayerNorm kernel."""
    print("\n=== LayerNorm Test ===")
    batch, seq, hidden = 2, 4, 128

    x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
    w_np = np.random.randn(hidden).astype(np.float32)
    b_np = np.random.randn(hidden).astype(np.float32)
    expected = layernorm_numpy(x_np, w_np, b_np)

    x = native.from_numpy(x_np)
    w = native.from_numpy(w_np)
    b = native.from_numpy(b_np)
    y = native.empty([batch, seq, hidden], native.Float32)

    tx, tw, tb, tout = from_gpuarray(x), from_gpuarray(w), from_gpuarray(b), from_gpuarray(y)
    kernels.layernorm(tx, tw, tout, bias=tb, eps=1e-5)
    native.device_synchronize()

    y_np = y.to_numpy()
    max_diff = np.max(np.abs(y_np - expected))

    passed = np.allclose(y_np, expected, rtol=1e-4, atol=1e-4)
    print(f"Max diff: {max_diff:.6e} - {'PASS' if passed else 'FAIL'}")
    return passed


def test_softmax():
    """Test Softmax kernel."""
    print("\n=== Softmax Test ===")
    batch, seq = 4, 128

    x_np = np.random.randn(batch, seq).astype(np.float32)
    expected = softmax_numpy(x_np)

    x = native.from_numpy(x_np)
    y = native.empty([batch, seq], native.Float32)

    tx, tout = from_gpuarray(x), from_gpuarray(y)
    kernels.softmax(tx, tout)
    native.device_synchronize()

    y_np = y.to_numpy()
    max_diff = np.max(np.abs(y_np - expected))

    passed = np.allclose(y_np, expected, rtol=1e-4, atol=1e-4)
    print(f"Max diff: {max_diff:.6e} - {'PASS' if passed else 'FAIL'}")
    return passed


def test_rotary():
    """Test Rotary (RoPE) kernel."""
    print("\n=== Rotary (RoPE) Test ===")
    batch, seq, num_heads, head_dim = 1, 4, 4, 64

    x_np = np.random.randn(batch, seq, num_heads, head_dim).astype(np.float32)
    half_dim = head_dim // 2

    # Create cos/sin tables
    positions = np.arange(seq).reshape(-1, 1)
    dims = np.arange(half_dim).reshape(1, -1)
    theta = 10000.0 ** (-2.0 * dims / head_dim)
    angles = positions * theta
    cos_np = np.cos(angles).astype(np.float32)
    sin_np = np.sin(angles).astype(np.float32)

    # Reference implementation
    x1 = x_np[..., :half_dim]
    x2 = x_np[..., half_dim:]
    cos_expanded = cos_np[np.newaxis, :, np.newaxis, :]
    sin_expanded = sin_np[np.newaxis, :, np.newaxis, :]
    y1 = x1 * cos_expanded - x2 * sin_expanded
    y2 = x1 * sin_expanded + x2 * cos_expanded
    expected = np.concatenate([y1, y2], axis=-1)

    x = native.from_numpy(x_np)
    cos = native.from_numpy(cos_np)
    sin = native.from_numpy(sin_np)
    y = native.empty([batch, seq, num_heads, head_dim], native.Float32)

    tx, tcos, tsin, tout = (
        from_gpuarray(x),
        from_gpuarray(cos),
        from_gpuarray(sin),
        from_gpuarray(y),
    )
    kernels.rotary(tx, tcos, tsin, tout)
    native.device_synchronize()

    y_np = y.to_numpy()
    max_diff = np.max(np.abs(y_np - expected))

    passed = np.allclose(y_np, expected, rtol=1e-4, atol=1e-4)
    print(f"Max diff: {max_diff:.6e} - {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("=" * 50)
    print("PyGPUkit Triton Kernel Tests")
    print("(No PyTorch CUDA required!)")
    print("=" * 50)

    results = []
    results.append(("RMSNorm", test_rmsnorm()))
    results.append(("LayerNorm", test_layernorm()))
    results.append(("Softmax", test_softmax()))
    results.append(("Rotary", test_rotary()))

    print("\n" + "=" * 50)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
