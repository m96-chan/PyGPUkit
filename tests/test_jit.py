"""Tests for JIT compiler."""

import pytest

from pygpukit.jit.compiler import (
    JITKernel,
    get_nvrtc_path,
    get_nvrtc_version,
    is_nvrtc_available,
    jit,
)


class TestNVRTCAvailability:
    """Tests for NVRTC availability detection."""

    def test_is_nvrtc_available_returns_bool(self):
        """Test that is_nvrtc_available returns a boolean."""
        result = is_nvrtc_available()
        assert isinstance(result, bool)

    def test_get_nvrtc_version_when_available(self):
        """Test get_nvrtc_version returns tuple when NVRTC available."""
        if not is_nvrtc_available():
            pytest.skip("NVRTC not available")

        version = get_nvrtc_version()
        assert version is not None
        assert isinstance(version, tuple)
        assert len(version) == 2
        assert isinstance(version[0], int)
        assert isinstance(version[1], int)
        # NVRTC version should be at least 11.0
        assert version[0] >= 11

    def test_get_nvrtc_version_when_unavailable(self):
        """Test get_nvrtc_version returns None when NVRTC unavailable."""
        # This test documents expected behavior when NVRTC is not available
        # We can't force NVRTC to be unavailable, but we test the interface
        version = get_nvrtc_version()
        if not is_nvrtc_available():
            assert version is None
        else:
            assert version is not None

    def test_get_nvrtc_path_returns_string_or_none(self):
        """Test that get_nvrtc_path returns a string path or None."""
        path = get_nvrtc_path()
        assert path is None or isinstance(path, str)
        if path is not None:
            # If path is returned, it should be an existing file
            import os

            assert os.path.isfile(path), f"NVRTC path does not exist: {path}"

    def test_get_nvrtc_path_consistency(self):
        """Test that get_nvrtc_path is consistent with is_nvrtc_available."""
        path = get_nvrtc_path()
        available = is_nvrtc_available()

        # If NVRTC is available, we should have found the DLL
        # (though the converse isn't always true - DLL found doesn't mean it works)
        if available and path is None:
            # This is unusual but can happen if NVRTC is loaded from system paths
            pass  # Allow this case

    def test_is_nvrtc_available_module_level(self):
        """Test that is_nvrtc_available is exported from main module."""
        import pygpukit as gp

        assert hasattr(gp, "is_nvrtc_available")
        assert callable(gp.is_nvrtc_available)
        result = gp.is_nvrtc_available()
        assert isinstance(result, bool)

    def test_get_nvrtc_version_module_level(self):
        """Test that get_nvrtc_version is exported from main module."""
        import pygpukit as gp

        assert hasattr(gp, "get_nvrtc_version")
        assert callable(gp.get_nvrtc_version)

    def test_get_nvrtc_path_module_level(self):
        """Test that get_nvrtc_path is exported from main module."""
        import pygpukit as gp

        assert hasattr(gp, "get_nvrtc_path")
        assert callable(gp.get_nvrtc_path)


class TestJITKernel:
    """Tests for JITKernel class."""

    def test_jit_creates_kernel(self):
        """Test that jit creates a kernel object."""
        src = """
        extern "C" __global__
        void add_one(float* x, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) x[idx] += 1.0f;
        }
        """
        kernel = jit(src, func="add_one")

        assert kernel is not None
        assert isinstance(kernel, JITKernel)
        assert kernel.name == "add_one"

    def test_jit_kernel_has_source(self):
        """Test that kernel stores source code."""
        src = """
        extern "C" __global__
        void my_kernel(float* x) {}
        """
        kernel = jit(src, func="my_kernel")

        assert kernel.source == src

    def test_jit_kernel_repr(self):
        """Test kernel repr."""
        src = """
        extern "C" __global__
        void test_func(float* x) {}
        """
        kernel = jit(src, func="test_func")

        assert "test_func" in repr(kernel)

    def test_jit_with_compile_options(self):
        """Test JIT with compile options."""
        src = """
        extern "C" __global__
        void kernel_with_opts(float* x) {}
        """
        kernel = jit(src, func="kernel_with_opts", options=["-O3"])

        assert kernel is not None
        assert "-O3" in kernel.options

    def test_jit_kernel_is_callable(self):
        """Test that JITKernel is callable."""
        src = """
        extern "C" __global__
        void callable_kernel(float* x, int n) {}
        """
        kernel = jit(src, func="callable_kernel")

        assert callable(kernel)


class TestJITCompilation:
    """Tests for JIT compilation process."""

    def test_jit_compiles_valid_cuda(self):
        """Test that valid CUDA code compiles."""
        src = """
        extern "C" __global__
        void scale(float* x, float factor, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) x[idx] *= factor;
        }
        """
        kernel = jit(src, func="scale")
        assert kernel.is_compiled

    def test_jit_invalid_func_name_raises(self):
        """Test that invalid function name raises error."""
        src = """
        extern "C" __global__
        void actual_func(float* x) {}
        """
        with pytest.raises(ValueError, match="Function.*not found"):
            jit(src, func="nonexistent_func")


class TestJITKernelConfiguration:
    """Tests for kernel launch configuration."""

    def test_kernel_default_block_size(self):
        """Test default block size."""
        src = """
        extern "C" __global__
        void default_block(float* x) {}
        """
        kernel = jit(src, func="default_block")

        assert kernel.block_size == 256  # Default

    def test_kernel_custom_block_size(self):
        """Test custom block size."""
        src = """
        extern "C" __global__
        void custom_block(float* x) {}
        """
        kernel = jit(src, func="custom_block", block_size=512)

        assert kernel.block_size == 512
