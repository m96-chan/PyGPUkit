"""Matrix multiplication operations for GPUArrays.

Corresponds to native/ops/matmul/.
"""

from __future__ import annotations

import warnings

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy
from pygpukit.ops._common import _validate_float_dtype, _validate_same_dtype


def matmul(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
    use_tf32: bool | None = None,
) -> GPUArray:
    """Matrix multiplication of two 2D arrays.

    Args:
        a: First input array (M x K).
        b: Second input array (K x N).
        out: Optional output array (M x N). If provided, result is written to this
            array instead of allocating a new one. This enables CUDA Graph capture
            since no memory allocation occurs during the operation.
        use_tf32: Whether to use TF32 TensorCore acceleration (Ampere+ only).
            - None (default): Use PYGPUKIT_ALLOW_TF32 environment variable
            - True: Force TF32 mode (requires SM >= 80 and float32)
            - False: Force FP32 mode

    Returns:
        The result GPUArray (M x N). If out is provided, returns out.

    Raises:
        ValueError: If arrays are not 2D or dimensions don't match.
        RuntimeError: If use_tf32=True but GPU doesn't support it or dtype is not float32.

    Example:
        # Allocate new output
        y = pk.matmul(x, W)

        # Write to existing buffer (for CUDA Graph capture)
        pk.matmul(x, W, out=y)
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

    # Validate out array if provided
    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(f"out shape {out.shape} does not match expected {expected_shape}")
        if out.dtype != a.dtype:
            raise ValueError(f"out dtype {out.dtype} does not match input dtype {a.dtype}")

    # Check TF32 dtype requirement early (before backend dispatch)
    if use_tf32 is True:
        from pygpukit.core.dtypes import float32

        if a.dtype != float32:
            raise RuntimeError("TF32 matmul requires float32 dtype")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_native(a, b, out=out, use_tf32=use_tf32)
    else:
        return _matmul_cpu(a, b, out=out)


def _matmul_cpu(a: GPUArray, b: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """CPU implementation of matmul."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    if out is not None:
        out_np = out.to_numpy()
        np.matmul(a_np, b_np, out=out_np)
        # Copy back to GPU - this is inefficient but CPU backend is for fallback only
        out._data = from_numpy(out_np)._data
        return out
    else:
        result_np = np.matmul(a_np, b_np)
        return from_numpy(result_np)


def _matmul_native(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
    use_tf32: bool | None = None,
) -> GPUArray:
    """Native C++ CUDA implementation of matmul (zero-copy).

    Args:
        a: First input array.
        b: Second input array.
        out: Optional output array. If provided, result is written in-place.
        use_tf32: Whether to use TF32 TensorCore acceleration.
            None means use environment variable PYGPUKIT_ALLOW_TF32.
    """

    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays (zero-copy if already native)
    a_native = a._get_native()
    b_native = b._get_native()

    if out is not None:
        # In-place operation - write to existing buffer
        out_native = out._get_native()
        if use_tf32 is not None:
            native.matmul_tf32_(a_native, b_native, out_native, use_tf32)
        else:
            native.matmul_(a_native, b_native, out_native)
        return out
    else:
        # Allocate new output
        if use_tf32 is not None:
            c_native = native.matmul_tf32(a_native, b_native, use_tf32)
        else:
            c_native = native.matmul(a_native, b_native)
        return GPUArray._wrap_native(c_native)


def transpose(a: GPUArray) -> GPUArray:
    """Matrix transpose.

    Args:
        a: Input array of shape [rows, cols].

    Returns:
        A new GPUArray of shape [cols, rows] containing a.T.

    Raises:
        ValueError: If input is not 2D or dtype is not a float type.
    """
    _validate_float_dtype(a, "transpose")

    if a.ndim != 2:
        raise ValueError(f"transpose expects 2D input [rows, cols], got {a.ndim}D")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _transpose_native(a)
    else:
        return _transpose_cpu(a)


def _transpose_cpu(a: GPUArray) -> GPUArray:
    """CPU implementation of transpose."""
    a_np = a.to_numpy()
    return from_numpy(a_np.T.copy())


def _transpose_native(a: GPUArray) -> GPUArray:
    """Native C++ CUDA implementation of transpose (zero-copy)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    a_native = a._get_native()
    c_native = native.transpose(a_native)
    return GPUArray._wrap_native(c_native)


def linear_bias_gelu(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray,
) -> GPUArray:
    """Fused linear + bias + GELU operation.

    Computes: output = gelu(input @ weight^T + bias)

    When dimensions are multiples of 16, this uses CUTLASS TensorCore
    epilogue fusion for efficiency. Otherwise, falls back to separate
    matmul + bias_add + gelu operations.

    Args:
        input: Input array of shape [batch, in_features].
        weight: Weight array of shape [out_features, in_features].
        bias: Bias array of shape [out_features].

    Returns:
        A new GPUArray of shape [batch, out_features].

    Raises:
        ValueError: If shapes or dtypes don't match.

    Note:
        Best performance when dimensions are multiples of 16 (uses TensorCore).
        Non-aligned dimensions use native fallback path.
    """
    _validate_float_dtype(input, "linear_bias_gelu")

    if input.ndim != 2:
        raise ValueError(
            f"linear_bias_gelu expects 2D input [batch, in_features], got {input.ndim}D"
        )
    if weight.ndim != 2:
        raise ValueError(
            f"linear_bias_gelu expects 2D weight [out_features, in_features], got {weight.ndim}D"
        )
    if bias.ndim != 1:
        raise ValueError(f"linear_bias_gelu expects 1D bias [out_features], got {bias.ndim}D")

    if input.dtype != weight.dtype or input.dtype != bias.dtype:
        raise ValueError("linear_bias_gelu: all inputs must have same dtype")

    in_features = input.shape[1]
    out_features = weight.shape[0]

    if weight.shape[1] != in_features:
        raise ValueError(
            f"linear_bias_gelu: weight.shape[1]={weight.shape[1]} must match "
            f"input.shape[1]={in_features}"
        )
    if bias.shape[0] != out_features:
        raise ValueError(
            f"linear_bias_gelu: bias.shape[0]={bias.shape[0]} must match "
            f"weight.shape[0]={out_features}"
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _linear_bias_gelu_native(input, weight, bias)
    else:
        return _linear_bias_gelu_cpu(input, weight, bias)


def _linear_bias_gelu_cpu(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray,
) -> GPUArray:
    """CPU implementation of linear_bias_gelu."""
    x = input.to_numpy()
    w = weight.to_numpy()
    b = bias.to_numpy()

    # Linear: y = x @ w.T + b
    y = x @ w.T + b

    # GELU approximation (same as GPU kernel)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    result = y * 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (y + 0.044715 * y**3)))

    return from_numpy(result.astype(x.dtype))


def _linear_bias_gelu_native(
    input: GPUArray,
    weight: GPUArray,
    bias: GPUArray,
) -> GPUArray:
    """Native C++ CUDA implementation of linear_bias_gelu (CUTLASS fused kernel)."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()
    input_native = input._get_native()
    weight_native = weight._get_native()
    bias_native = bias._get_native()
    c_native = native.linear_bias_gelu(input_native, weight_native, bias_native)
    return GPUArray._wrap_native(c_native)


def batched_matmul(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Batched matrix multiplication for 3D and 4D tensors.

    Supports:
    - 3D: [batch, M, K] @ [batch, K, N] -> [batch, M, N]
    - 4D: [batch1, batch2, M, K] @ [batch1, batch2, K, N] -> [batch1, batch2, M, N]

    Args:
        a: First input array (3D or 4D).
        b: Second input array (3D or 4D).
        out: Optional output array. If provided, result is written in-place.

    Returns:
        The result GPUArray with shape [..., M, N].

    Raises:
        ValueError: If arrays are not 3D/4D or dimensions don't match.
    """
    if a.ndim not in (3, 4):
        raise ValueError(f"batched_matmul requires 3D or 4D arrays, got {a.ndim}D")
    if b.ndim not in (3, 4):
        raise ValueError(f"batched_matmul requires 3D or 4D arrays, got {b.ndim}D")
    if a.ndim != b.ndim:
        raise ValueError(f"batched_matmul requires same ndim, got {a.ndim}D and {b.ndim}D")

    _validate_same_dtype(a, b, "batched_matmul")

    # Extract dimensions
    if a.ndim == 3:
        batch = a.shape[0]
        M, K = a.shape[1], a.shape[2]
        K2, N = b.shape[1], b.shape[2]
        if b.shape[0] != batch:
            raise ValueError(f"Batch dimension mismatch: {a.shape[0]} vs {b.shape[0]}")
        if K != K2:
            raise ValueError(f"Inner dimension mismatch: {K} vs {K2}")
        out_shape = (batch, M, N)
        batch_count = batch
    else:  # 4D
        batch1, batch2 = a.shape[0], a.shape[1]
        M, K = a.shape[2], a.shape[3]
        K2, N = b.shape[2], b.shape[3]
        if b.shape[0] != batch1 or b.shape[1] != batch2:
            raise ValueError(
                f"Batch dimensions mismatch: ({batch1}, {batch2}) vs ({b.shape[0]}, {b.shape[1]})"
            )
        if K != K2:
            raise ValueError(f"Inner dimension mismatch: {K} vs {K2}")
        out_shape = (batch1, batch2, M, N)
        batch_count = batch1 * batch2

    # Validate output
    if out is not None:
        if out.shape != out_shape:
            raise ValueError(f"out shape {out.shape} does not match expected {out_shape}")
        if out.dtype != a.dtype:
            raise ValueError(f"out dtype {out.dtype} does not match input dtype {a.dtype}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _batched_matmul_native(a, b, M, N, K, batch_count, out_shape, out=out)
    else:
        return _batched_matmul_cpu(a, b, out=out)


def _batched_matmul_cpu(a: GPUArray, b: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
    """CPU implementation of batched_matmul."""
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    result_np = np.matmul(a_np, b_np)
    result = from_numpy(result_np)

    if out is not None:
        # Copy result to output buffer
        from ..ops.elementwise import copy_to

        copy_to(result, out)
        return out
    else:
        return result


def _batched_matmul_loop(
    a: GPUArray, b: GPUArray, out_shape: tuple[int, ...], *, out: GPUArray | None = None
) -> GPUArray:
    """GPU batched matmul using loop over individual matmuls.

    This is a fallback for when CUTLASS strided batched GEMM is not available
    (e.g., SM 120). Uses native matmul kernel for each batch element.
    """
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Reshape to 3D for easier iteration: [batch, M, K] @ [batch, K, N]
    if a.ndim == 4:
        batch1, batch2 = a.shape[0], a.shape[1]
        M, K = a.shape[2], a.shape[3]
        N = b.shape[3]
        total_batch = batch1 * batch2

        a_3d = a.reshape(total_batch, M, K)
        b_3d = b.reshape(total_batch, K, N)
    else:
        total_batch = a.shape[0]
        M, K = a.shape[1], a.shape[2]
        N = b.shape[2]

        a_3d = a
        b_3d = b

    # Allocate output
    if out is None:
        out_native = native.empty(list(out_shape), native.DataType.Float32)
        out = GPUArray._wrap_native(out_native)

    # Perform batched matmul via loop
    for i in range(total_batch):
        # Extract slice (creates view/copy depending on implementation)
        a_i = a_3d.to_numpy()[i]
        b_i = b_3d.to_numpy()[i]

        a_gpu = from_numpy(a_i)
        b_gpu = from_numpy(b_i)

        # Compute matmul for this batch element
        c_gpu = matmul(a_gpu, b_gpu)

        # Copy result to output
        out_np = out.to_numpy()
        if a.ndim == 4:
            i1, i2 = i // batch2, i % batch2
            out_np[i1, i2] = c_gpu.to_numpy()
        else:
            out_np[i] = c_gpu.to_numpy()
        out = from_numpy(out_np)

    return out


def _batched_matmul_native(
    a: GPUArray,
    b: GPUArray,
    M: int,
    N: int,
    K: int,
    batch_count: int,
    out_shape: tuple[int, ...],
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native cuBLASLt strided batched GEMM implementation."""
    from pygpukit.core.backend import get_native_module
    from pygpukit.core.dtypes import float32

    native = get_native_module()

    # Currently only FP32 supported via cuBLASLt strided batched
    if a.dtype != float32:
        warnings.warn(
            f"batched_matmul: GPU kernel requires float32, got {a.dtype}. Using CPU fallback (slow)",
            RuntimeWarning,
            stacklevel=3,
        )
        return _batched_matmul_cpu(a, b, out=out)

    # Compute strides for strided batched GEMM
    strideA = M * K
    strideB = K * N
    strideC = M * N

    # Get native arrays
    a_native = a._get_native()
    b_native = b._get_native()

    # Allocate output if needed (using native allocation)
    if out is None:
        out_native = native.empty(list(out_shape), native.DataType.Float32)
        out = GPUArray._wrap_native(out_native)
    else:
        out_native = out._get_native()

    # Call strided batched GEMM with CPU fallback for unsupported architectures
    try:
        native.gemm_strided_batched_fp32(
            a_native,
            b_native,
            out_native,
            M,
            N,
            K,
            batch_count,
            strideA,
            strideB,
            strideC,
        )
    except RuntimeError:
        # CUTLASS not available/failed (e.g., SM 120) - fall back to CPU
        warnings.warn(
            "batched_matmul: CUTLASS kernel failed, using CPU fallback (slow)",
            RuntimeWarning,
            stacklevel=3,
        )
        return _batched_matmul_cpu(a, b, out=out)

    return out
