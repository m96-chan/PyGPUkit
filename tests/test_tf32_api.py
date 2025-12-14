"""Tests for TF32 API integration (TDD).

These tests verify the use_tf32 parameter for matmul and
the Rust-side DeviceCapabilities.tensorcore support.
"""

import numpy as np
import pytest

import pygpukit as gp


class TestMatmulTF32API:
    """Tests for matmul use_tf32 parameter."""

    def test_matmul_use_tf32_false_default(self):
        """Test that use_tf32=False is the default behavior."""
        np.random.seed(42)
        a_np = np.random.rand(64, 64).astype(np.float32)
        b_np = np.random.rand(64, 64).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        # Default behavior (no use_tf32 arg) should use FP32
        c = gp.matmul(a, b)

        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        # FP32 should be very accurate
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_matmul_use_tf32_explicit_false(self):
        """Test matmul with explicit use_tf32=False."""
        np.random.seed(42)
        a_np = np.random.rand(64, 64).astype(np.float32)
        b_np = np.random.rand(64, 64).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b, use_tf32=False)

        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_matmul_use_tf32_true_correctness(self):
        """Test matmul with use_tf32=True produces correct results within TF32 tolerance."""
        np.random.seed(42)
        # Large enough to trigger TF32 kernel
        a_np = np.random.rand(1024, 1024).astype(np.float32)
        b_np = np.random.rand(1024, 1024).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b, use_tf32=True)

        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)

        # TF32 has lower precision (~0.1% relative error per op)
        # For 1024 accumulations, expect ~1-5% relative error
        rel_error = np.abs(result - expected) / (np.abs(expected) + 1e-8)
        max_rel_error = np.max(rel_error)
        assert max_rel_error < 0.1, f"TF32 relative error too high: {max_rel_error}"

    def test_matmul_use_tf32_small_matrix_fallback(self):
        """Test that small matrices with use_tf32=True still work (may fallback to FP32)."""
        np.random.seed(42)
        a_np = np.random.rand(16, 16).astype(np.float32)
        b_np = np.random.rand(16, 16).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        # Small matrix - implementation may use FP32 fallback
        c = gp.matmul(a, b, use_tf32=True)

        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        # Should still be reasonably accurate
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_matmul_use_tf32_float64_raises(self):
        """Test that use_tf32=True with float64 raises an error."""
        a_np = np.random.rand(64, 64).astype(np.float64)
        b_np = np.random.rand(64, 64).astype(np.float64)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        # TF32 only works with float32 - should raise RuntimeError
        with pytest.raises(RuntimeError, match="float32"):
            gp.matmul(a, b, use_tf32=True)

    def test_matmul_use_tf32_rectangular(self):
        """Test TF32 matmul with rectangular matrices."""
        np.random.seed(42)
        a_np = np.random.rand(512, 1024).astype(np.float32)
        b_np = np.random.rand(1024, 768).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b, use_tf32=True)

        assert c.shape == (512, 768)
        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)

        rel_error = np.abs(result - expected) / (np.abs(expected) + 1e-8)
        max_rel_error = np.max(rel_error)
        assert max_rel_error < 0.1, f"TF32 relative error too high: {max_rel_error}"


class TestDeviceCapabilities:
    """Tests for DeviceCapabilities from Rust."""

    def test_device_capabilities_exists(self):
        """Test that DeviceCapabilities class is available."""
        assert hasattr(gp, 'DeviceCapabilities') or hasattr(gp, 'get_device_capabilities')

    def test_device_capabilities_tensorcore_field(self):
        """Test that DeviceCapabilities has tensorcore field."""
        # Get capabilities for current device
        caps = gp.get_device_capabilities()

        assert hasattr(caps, 'tensorcore')
        assert isinstance(caps.tensorcore, bool)

    def test_device_capabilities_sm_version(self):
        """Test that DeviceCapabilities has SM version info."""
        caps = gp.get_device_capabilities()

        assert hasattr(caps, 'sm_version') or hasattr(caps, 'compute_capability')

    def test_tensorcore_requires_sm80(self):
        """Test that tensorcore is True only for SM >= 80."""
        caps = gp.get_device_capabilities()

        sm_version = getattr(caps, 'sm_version', None) or getattr(caps, 'compute_capability', 0)
        if sm_version >= 80:
            # Ampere or newer should have tensor cores
            assert caps.tensorcore is True
        else:
            # Older GPUs don't have TF32 tensor cores
            assert caps.tensorcore is False


class TestKernelTypeRust:
    """Tests for Rust kernel type enum."""

    def test_kernel_type_exists(self):
        """Test that KernelType enum is available from Rust."""
        # This should be exposed via pygpukit._pygpukit_rust
        try:
            from pygpukit._pygpukit_rust import KernelType
            assert hasattr(KernelType, 'Tf32Mma') or hasattr(KernelType, 'TF32_MMA')
        except ImportError:
            # Rust module may not be built yet - skip
            pytest.skip("Rust module not available")

    def test_kernel_type_fp32_exists(self):
        """Test that FP32 kernel type exists."""
        try:
            from pygpukit._pygpukit_rust import KernelType
            assert hasattr(KernelType, 'Fp32Fma') or hasattr(KernelType, 'FP32_FMA')
        except ImportError:
            pytest.skip("Rust module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
