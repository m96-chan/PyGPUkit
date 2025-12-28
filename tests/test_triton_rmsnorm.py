"""Test Triton RMSNorm kernel with PyGPUkit."""

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


def test_rmsnorm():
    """Test RMSNorm kernel."""
    batch, seq, hidden = 2, 4, 128

    # Create test data
    x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
    w_np = np.random.randn(hidden).astype(np.float32)

    # Expected result
    expected = rmsnorm_numpy(x_np, w_np)

    # Create PyGPUkit arrays
    x = native.from_numpy(x_np)
    w = native.from_numpy(w_np)
    y = native.empty([batch, seq, hidden], native.Float32)

    # Wrap for Triton
    tx = from_gpuarray(x)
    tw = from_gpuarray(w)
    tout = from_gpuarray(y)

    print(f"Input shape: {tx.shape}")
    print(f"Weight shape: {tw.shape}")
    print(f"Output shape: {tout.shape}")

    # Run kernel
    kernels.rmsnorm(tx, tw, tout, eps=1e-6)
    native.device_synchronize()

    # Check result
    y_np = y.to_numpy()

    # Compare
    max_diff = np.max(np.abs(y_np - expected))
    mean_diff = np.mean(np.abs(y_np - expected))

    print(f"\nMax diff: {max_diff:.6e}")
    print(f"Mean diff: {mean_diff:.6e}")

    if np.allclose(y_np, expected, rtol=1e-4, atol=1e-4):
        print("Result: PASS")
    else:
        print("Result: FAIL")
        print(f"Expected[:2,:2,:4]:\n{expected[:2, :2, :4]}")
        print(f"Got[:2,:2,:4]:\n{y_np[:2, :2, :4]}")


if __name__ == "__main__":
    test_rmsnorm()
