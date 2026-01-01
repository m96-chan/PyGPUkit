"""NumPy validation tests for FLUX GPU kernels.

Tests for Issue #187: FLUX.1 performance optimization kernels.
"""

import numpy as np
import pytest

from pygpukit.core import GPUArray
from pygpukit.core.factory import from_numpy


def _to_numpy(arr: GPUArray) -> np.ndarray:
    """Convert GPUArray to numpy."""
    return arr.to_numpy()


class TestFluxKernels:
    """Test FLUX-specific GPU kernels against NumPy reference."""

    def test_layer_norm_simple(self) -> None:
        """Test layer_norm_simple kernel."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        if not hasattr(native, "layer_norm_simple"):
            pytest.skip("layer_norm_simple not available")

        B, N, D = 2, 4, 8
        x_np = np.random.randn(B, N, D).astype(np.float32)

        # NumPy reference
        mean = x_np.mean(axis=-1, keepdims=True)
        var = x_np.var(axis=-1, keepdims=True)
        expected = (x_np - mean) / np.sqrt(var + 1e-5)

        # GPU implementation
        x_gpu = from_numpy(x_np)
        result = native.layer_norm_simple(x_gpu._get_native())
        result_np = GPUArray._wrap_native(result).to_numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)

    def test_modulate(self) -> None:
        """Test modulate kernel."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        if not hasattr(native, "modulate"):
            pytest.skip("modulate not available")

        B, N, D = 2, 4, 8
        x_np = np.random.randn(B, N, D).astype(np.float32)
        scale_np = np.random.randn(B, D).astype(np.float32)
        shift_np = np.random.randn(B, D).astype(np.float32)

        # NumPy reference: y = x * (1 + scale[:, None, :]) + shift[:, None, :]
        expected = x_np * (1 + scale_np[:, np.newaxis, :]) + shift_np[:, np.newaxis, :]

        # GPU implementation
        x_gpu = from_numpy(x_np)
        scale_gpu = from_numpy(scale_np)
        shift_gpu = from_numpy(shift_np)
        result = native.modulate(
            x_gpu._get_native(), scale_gpu._get_native(), shift_gpu._get_native()
        )
        result_np = GPUArray._wrap_native(result).to_numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)

    def test_gated_residual(self) -> None:
        """Test gated_residual kernel."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        if not hasattr(native, "gated_residual"):
            pytest.skip("gated_residual not available")

        B, N, D = 2, 4, 8
        residual_np = np.random.randn(B, N, D).astype(np.float32)
        gate_np = np.random.randn(B, D).astype(np.float32)
        value_np = np.random.randn(B, N, D).astype(np.float32)

        # NumPy reference: y = residual + gate[:, None, :] * value
        expected = residual_np + gate_np[:, np.newaxis, :] * value_np

        # GPU implementation
        residual_gpu = from_numpy(residual_np)
        gate_gpu = from_numpy(gate_np)
        value_gpu = from_numpy(value_np)
        result = native.gated_residual(
            residual_gpu._get_native(), gate_gpu._get_native(), value_gpu._get_native()
        )
        result_np = GPUArray._wrap_native(result).to_numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)

    def test_scale_tensor(self) -> None:
        """Test scale_tensor kernel."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        if not hasattr(native, "scale_tensor"):
            pytest.skip("scale_tensor not available")

        B, N, D = 2, 4, 8
        x_np = np.random.randn(B, N, D).astype(np.float32)
        scale = 2.5

        # NumPy reference
        expected = x_np * scale

        # GPU implementation
        x_gpu = from_numpy(x_np)
        result = native.scale_tensor(x_gpu._get_native(), scale)
        result_np = GPUArray._wrap_native(result).to_numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)

    def test_concat_axis1(self) -> None:
        """Test concat_axis1 kernel."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        if not hasattr(native, "concat_axis1"):
            pytest.skip("concat_axis1 not available")

        B, N1, N2, D = 2, 4, 3, 8
        a_np = np.random.randn(B, N1, D).astype(np.float32)
        b_np = np.random.randn(B, N2, D).astype(np.float32)

        # NumPy reference
        expected = np.concatenate([a_np, b_np], axis=1)

        # GPU implementation
        a_gpu = from_numpy(a_np)
        b_gpu = from_numpy(b_np)
        result = native.concat_axis1(a_gpu._get_native(), b_gpu._get_native())
        result_np = GPUArray._wrap_native(result).to_numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)

    def test_split_axis1(self) -> None:
        """Test split_axis1 kernel."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        if not hasattr(native, "split_axis1"):
            pytest.skip("split_axis1 not available")

        B, N, D = 2, 7, 8
        split_size = 4
        x_np = np.random.randn(B, N, D).astype(np.float32)

        # NumPy reference
        expected_first = x_np[:, :split_size, :]
        expected_second = x_np[:, split_size:, :]

        # GPU implementation
        x_gpu = from_numpy(x_np)
        result = native.split_axis1(x_gpu._get_native(), split_size)
        first_np = GPUArray._wrap_native(result[0]).to_numpy()
        second_np = GPUArray._wrap_native(result[1]).to_numpy()

        np.testing.assert_allclose(first_np, expected_first, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(second_np, expected_second, rtol=1e-4, atol=1e-5)

    def test_add_broadcast(self) -> None:
        """Test add_broadcast kernel."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        if not hasattr(native, "add_broadcast"):
            pytest.skip("add_broadcast not available")

        B, N, D = 2, 4, 8
        x_np = np.random.randn(B, N, D).astype(np.float32)
        bias_np = np.random.randn(B, D).astype(np.float32)

        # NumPy reference: x + bias[:, None, :]
        expected = x_np + bias_np[:, np.newaxis, :]

        # GPU implementation
        x_gpu = from_numpy(x_np)
        bias_gpu = from_numpy(bias_np)
        result = native.add_broadcast(x_gpu._get_native(), bias_gpu._get_native())
        result_np = GPUArray._wrap_native(result).to_numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)

    def test_layer_norm_modulate(self) -> None:
        """Test layer_norm_modulate kernel (fused)."""
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        if not hasattr(native, "layer_norm_modulate"):
            pytest.skip("layer_norm_modulate not available")

        B, N, D = 2, 4, 8
        x_np = np.random.randn(B, N, D).astype(np.float32)
        scale_np = np.random.randn(B, D).astype(np.float32)
        shift_np = np.random.randn(B, D).astype(np.float32)

        # NumPy reference: LayerNorm(x) * (1 + scale) + shift
        mean = x_np.mean(axis=-1, keepdims=True)
        var = x_np.var(axis=-1, keepdims=True)
        normalized = (x_np - mean) / np.sqrt(var + 1e-5)
        expected = (
            normalized * (1 + scale_np[:, np.newaxis, :]) + shift_np[:, np.newaxis, :]
        )

        # GPU implementation
        x_gpu = from_numpy(x_np)
        scale_gpu = from_numpy(scale_np)
        shift_gpu = from_numpy(shift_np)
        result = native.layer_norm_modulate(
            x_gpu._get_native(), scale_gpu._get_native(), shift_gpu._get_native()
        )
        result_np = GPUArray._wrap_native(result).to_numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-4, atol=1e-5)


class TestBatchedMatmul:
    """Test batched matmul with cuBLAS."""

    def test_batched_matmul_3d(self) -> None:
        """Test 3D batched matmul."""
        from pygpukit.ops.matmul import batched_matmul

        batch, M, K, N = 4, 32, 64, 48
        a_np = np.random.randn(batch, M, K).astype(np.float32)
        b_np = np.random.randn(batch, K, N).astype(np.float32)

        # NumPy reference
        expected = np.matmul(a_np, b_np)

        # GPU implementation
        a_gpu = from_numpy(a_np)
        b_gpu = from_numpy(b_np)
        result = batched_matmul(a_gpu, b_gpu)
        result_np = result.to_numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-3, atol=1e-4)

    def test_batched_matmul_4d(self) -> None:
        """Test 4D batched matmul."""
        from pygpukit.ops.matmul import batched_matmul

        batch1, batch2, M, K, N = 2, 8, 16, 32, 24
        a_np = np.random.randn(batch1, batch2, M, K).astype(np.float32)
        b_np = np.random.randn(batch1, batch2, K, N).astype(np.float32)

        # NumPy reference
        expected = np.matmul(a_np, b_np)

        # GPU implementation
        a_gpu = from_numpy(a_np)
        b_gpu = from_numpy(b_np)
        result = batched_matmul(a_gpu, b_gpu)
        result_np = result.to_numpy()

        np.testing.assert_allclose(result_np, expected, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
