"""Tests for JIT compiler."""

import pytest

from pygpukit.jit.compiler import JITKernel, jit


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
