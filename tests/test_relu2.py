"""Tests for ReLU squared (relu2) activation function."""

import numpy as np
import pytest

from pygpukit import from_numpy
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.dtypes import DataType
from pygpukit.ops.nn import relu2


def is_gpu_available():
    """Check if GPU backend is available."""
    backend = get_backend()
    return isinstance(backend, NativeBackend) and backend.is_available()


def relu2_reference(x: np.ndarray) -> np.ndarray:
    """Reference implementation of ReLU squared."""
    relu_val = np.maximum(0, x)
    return relu_val * relu_val


class TestRelu2:
    """Test ReLU squared activation."""

    def test_relu2_basic_f32(self):
        """Test basic ReLU squared with float32."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        expected = relu2_reference(x)

        x_gpu = from_numpy(x)
        y_gpu = relu2(x_gpu)
        y = y_gpu.to_numpy()

        np.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_relu2_negative_values(self):
        """Test that negative values become 0."""
        x = np.array([-5.0, -3.0, -1.0, -0.5], dtype=np.float32)
        x_gpu = from_numpy(x)
        y_gpu = relu2(x_gpu)
        y = y_gpu.to_numpy()

        np.testing.assert_allclose(y, np.zeros_like(x), rtol=1e-5)

    def test_relu2_positive_values(self):
        """Test that positive values are squared."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        expected = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)

        x_gpu = from_numpy(x)
        y_gpu = relu2(x_gpu)
        y = y_gpu.to_numpy()

        np.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_relu2_2d_array(self):
        """Test ReLU squared with 2D array."""
        x = np.random.randn(32, 64).astype(np.float32)
        expected = relu2_reference(x)

        x_gpu = from_numpy(x)
        y_gpu = relu2(x_gpu)
        y = y_gpu.to_numpy()

        np.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_relu2_3d_array(self):
        """Test ReLU squared with 3D array (batch, seq, hidden)."""
        x = np.random.randn(4, 128, 256).astype(np.float32)
        expected = relu2_reference(x)

        x_gpu = from_numpy(x)
        y_gpu = relu2(x_gpu)
        y = y_gpu.to_numpy()

        np.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_relu2_bf16(self):
        """Test ReLU squared with bfloat16."""
        if not is_gpu_available():
            pytest.skip("BF16 requires GPU")

        x = np.random.randn(64, 128).astype(np.float32)
        expected = relu2_reference(x)

        # Convert to bf16 on GPU
        x_gpu = from_numpy(x).astype(DataType.from_string("bfloat16"))
        y_gpu = relu2(x_gpu)
        y = y_gpu.astype(DataType.from_string("float32")).to_numpy()

        # BF16 has lower precision
        np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_relu2_f16(self):
        """Test ReLU squared with float16."""
        if not is_gpu_available():
            pytest.skip("F16 requires GPU")

        x = np.random.randn(64, 128).astype(np.float32)
        expected = relu2_reference(x)

        # Convert to f16 on GPU
        x_gpu = from_numpy(x).astype(DataType.from_string("float16"))
        y_gpu = relu2(x_gpu)
        y = y_gpu.astype(DataType.from_string("float32")).to_numpy()

        # F16 has lower precision
        np.testing.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)

    def test_relu2_with_output_buffer(self):
        """Test ReLU squared with pre-allocated output buffer."""
        x = np.random.randn(32, 64).astype(np.float32)
        expected = relu2_reference(x)

        x_gpu = from_numpy(x)
        out_gpu = from_numpy(np.zeros_like(x))

        result = relu2(x_gpu, out=out_gpu)
        y = out_gpu.to_numpy()

        # Verify output buffer contains correct values
        np.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_relu2_preserves_shape(self):
        """Test that ReLU squared preserves input shape."""
        shapes = [(10,), (10, 20), (5, 10, 15), (2, 3, 4, 5)]
        for shape in shapes:
            x = np.random.randn(*shape).astype(np.float32)
            x_gpu = from_numpy(x)
            y_gpu = relu2(x_gpu)
            assert y_gpu.shape == shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
