"""Basic operations for GPUArrays."""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import CUDABackend, get_backend
from pygpukit.core.factory import from_numpy


def _validate_same_shape(a: GPUArray, b: GPUArray, op_name: str) -> None:
    """Validate that two arrays have the same shape."""
    if a.shape != b.shape:
        raise ValueError(
            f"{op_name} requires arrays of same shape, got {a.shape} and {b.shape}"
        )


def _validate_same_dtype(a: GPUArray, b: GPUArray, op_name: str) -> None:
    """Validate that two arrays have the same dtype."""
    if a.dtype != b.dtype:
        raise ValueError(
            f"{op_name} requires arrays of same dtype, got {a.dtype} and {b.dtype}"
        )


def add(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise addition of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        A new GPUArray containing the element-wise sum.

    Raises:
        ValueError: If shapes don't match.
    """
    _validate_same_shape(a, b, "add")
    _validate_same_dtype(a, b, "add")

    backend = get_backend()

    if isinstance(backend, CUDABackend) and backend.is_available():
        # Use CUDA kernel for real GPU
        return _add_cuda(a, b)
    else:
        # CPU simulation
        return _add_cpu(a, b)


def _add_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of add."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = a_np + b_np
    return from_numpy(result_np)


def _add_cuda(a: GPUArray, b: GPUArray) -> GPUArray:
    """CUDA implementation of add."""
    from pygpukit.core.factory import empty
    from pygpukit.jit.compiler import jit

    kernel_src = '''
    extern "C" __global__
    void add_kernel(const float* a, const float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    '''

    kernel = jit(kernel_src, func="add_kernel")
    c = empty(a.shape, dtype=a.dtype)

    kernel(a.device_ptr, b.device_ptr, c.device_ptr, a.size)
    get_backend().synchronize()

    return c


def mul(a: GPUArray, b: GPUArray) -> GPUArray:
    """Element-wise multiplication of two arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        A new GPUArray containing the element-wise product.

    Raises:
        ValueError: If shapes don't match.
    """
    _validate_same_shape(a, b, "mul")
    _validate_same_dtype(a, b, "mul")

    backend = get_backend()

    if isinstance(backend, CUDABackend) and backend.is_available():
        return _mul_cuda(a, b)
    else:
        return _mul_cpu(a, b)


def _mul_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of mul."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = a_np * b_np
    return from_numpy(result_np)


def _mul_cuda(a: GPUArray, b: GPUArray) -> GPUArray:
    """CUDA implementation of mul."""
    from pygpukit.core.factory import empty
    from pygpukit.jit.compiler import jit

    kernel_src = '''
    extern "C" __global__
    void mul_kernel(const float* a, const float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] * b[idx];
        }
    }
    '''

    kernel = jit(kernel_src, func="mul_kernel")
    c = empty(a.shape, dtype=a.dtype)

    kernel(a.device_ptr, b.device_ptr, c.device_ptr, a.size)
    get_backend().synchronize()

    return c


def matmul(a: GPUArray, b: GPUArray) -> GPUArray:
    """Matrix multiplication of two 2D arrays.

    Args:
        a: First input array (M x K).
        b: Second input array (K x N).

    Returns:
        A new GPUArray containing the matrix product (M x N).

    Raises:
        ValueError: If arrays are not 2D or dimensions don't match.
    """
    if a.ndim != 2:
        raise ValueError(f"matmul requires 2D arrays, got {a.ndim}D for first argument")
    if b.ndim != 2:
        raise ValueError(f"matmul requires 2D arrays, got {b.ndim}D for second argument")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul dimension mismatch: {a.shape} @ {b.shape} "
            f"(inner dimensions {a.shape[1]} and {b.shape[0]} must match)"
        )

    _validate_same_dtype(a, b, "matmul")

    backend = get_backend()

    if isinstance(backend, CUDABackend) and backend.is_available():
        return _matmul_cuda(a, b)
    else:
        return _matmul_cpu(a, b)


def _matmul_cpu(a: GPUArray, b: GPUArray) -> GPUArray:
    """CPU implementation of matmul."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = np.matmul(a_np, b_np)
    return from_numpy(result_np)


def _matmul_cuda(a: GPUArray, b: GPUArray) -> GPUArray:
    """CUDA implementation of matmul."""
    from pygpukit.core.factory import empty
    from pygpukit.jit.compiler import jit

    M, K = a.shape
    _, N = b.shape

    kernel_src = '''
    extern "C" __global__
    void matmul_kernel(const float* A, const float* B, float* C,
                       int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    '''

    kernel = jit(kernel_src, func="matmul_kernel")
    c = empty((M, N), dtype=a.dtype)

    # Launch with 2D grid
    block_size = 16
    grid_x = (N + block_size - 1) // block_size
    grid_y = (M + block_size - 1) // block_size

    kernel(
        a.device_ptr,
        b.device_ptr,
        c.device_ptr,
        M,
        N,
        K,
        grid_size=(grid_x, grid_y),
    )
    get_backend().synchronize()

    return c
