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
        ValueError: If input is not 2D.
    """
    if a.ndim != 2:
        raise ValueError(f"transpose expects 2D input [rows, cols], got {a.ndim}D")

    from pygpukit.core.dtypes import uint8

    backend = get_backend()

    # For uint8 (FP8 weights), use CPU fallback since native transpose
    # doesn't support integer types
    if a.dtype == uint8:
        return _transpose_cpu(a)

    _validate_float_dtype(a, "transpose")

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


def fp8_available() -> bool:
    """Check if FP8 GEMM is available (any backend).

    Returns:
        True if FP8 GEMM is available (requires SM90+ GPU).
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.fp8_available()
    else:
        return False


def fp8_sm90_available() -> bool:
    """Check if FP8 GEMM is available on SM90 (Hopper).

    Returns:
        True if FP8 GEMM is available (requires SM90+ and CUTLASS SM90 support).
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.fp8_sm90_available()
    else:
        return False


def fp8_sm100_available() -> bool:
    """Check if FP8 GEMM is available on SM100 (Blackwell datacenter).

    This may work on SM120 (Blackwell GeForce) as a fallback since both
    are Blackwell architecture.

    Returns:
        True if FP8 GEMM is available (requires SM100+ and CUTLASS SM100 support).
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.fp8_sm100_available()
    else:
        return False


def fp8_sm120_available() -> bool:
    """Check if FP8 GEMM is available on SM120 (Blackwell GeForce).

    Note: Currently disabled due to CUTLASS bug #2902.

    Returns:
        True if FP8 GEMM is available (requires SM120+ and CUTLASS SM120 support).
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.fp8_sm120_available()
    else:
        return False


def fp8_fp8_sm120_available() -> bool:
    """Check if Pure FP8 I/O GEMM is available on SM120 (Blackwell GeForce).

    This is for FP8 models where weights and activations are already in FP8 format.

    Returns:
        True if Pure FP8 GEMM is available (requires SM120+ and CUTLASS SM120 support).
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.fp8_fp8_sm120_available()
    else:
        return False


def matmul_fp8_fp8_sm120(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Pure FP8 I/O matrix multiplication for SM120 (Blackwell GeForce).

    This function takes FP8 E4M3 inputs directly (no conversion from FP32),
    performs the GEMM using CUTLASS FP8 kernels, and returns FP8 E4M3 output.

    This is optimized for FP8 models (Llama 3.1 FP8, etc.) where weights
    and activations are already quantized to FP8.

    Args:
        a: First input array (M x K), FP8 E4M3 stored as uint8.
        b: Second input array (K x N), FP8 E4M3 stored as uint8.
           Should be in ColumnMajor format (pre-transposed).
        out: Optional output array (M x N), uint8. If provided, result is
            written to this array instead of allocating a new one.

    Returns:
        The result GPUArray (M x N), FP8 E4M3 stored as uint8.

    Raises:
        ValueError: If arrays are not 2D, dtypes are not uint8, or dimensions don't match.
        RuntimeError: If FP8 SM120 is not available.

    Example:
        >>> import pygpukit as gk
        >>> # Assuming A and B are already FP8 quantized (stored as uint8)
        >>> A = gk.from_numpy(fp8_a_data)  # [M, K] uint8
        >>> B = gk.from_numpy(fp8_b_data)  # [K, N] uint8 (ColumnMajor)
        >>> C = gk.ops.matmul_fp8_fp8_sm120(A, B)  # [M, N] uint8
    """
    from pygpukit.core.dtypes import uint8

    if a.ndim != 2:
        raise ValueError(
            f"matmul_fp8_fp8_sm120 requires 2D arrays, got {a.ndim}D for first argument"
        )
    if b.ndim != 2:
        raise ValueError(
            f"matmul_fp8_fp8_sm120 requires 2D arrays, got {b.ndim}D for second argument"
        )

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul_fp8_fp8_sm120 dimension mismatch: {a.shape} @ {b.shape} "
            f"(inner dimensions {a.shape[1]} and {b.shape[0]} must match)"
        )

    if a.dtype != uint8 or b.dtype != uint8:
        raise ValueError("matmul_fp8_fp8_sm120 requires uint8 inputs (FP8 E4M3)")

    if not fp8_fp8_sm120_available():
        raise RuntimeError("Pure FP8 SM120 GEMM is not available. Requires SM120+ GPU.")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_fp8_fp8_sm120_native(a, b, out=out)
    else:
        raise RuntimeError("Pure FP8 SM120 GEMM requires native backend")


def _matmul_fp8_fp8_sm120_native(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ implementation of Pure FP8 I/O GEMM for SM120."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays
    a_native = a._get_native()
    b_native = b._get_native()

    # Allocate output if needed
    if out is None:
        M, K = a.shape
        N = b.shape[1]
        out_native = native.empty([M, N], native.DataType.UInt8)
        out = GPUArray._wrap_native(out_native)
    else:
        out_native = out._get_native()

    # Call Pure FP8 GEMM
    native.gemm_fp8_fp8_sm120(a_native, b_native, out_native)

    return out


def fp8_fp8_get_scale_sizes(M: int, N: int, K: int) -> tuple[int, int]:
    """Get scale factor sizes for FP8 blockwise GEMM.

    Returns the required sizes for scale_A and scale_B arrays for the
    given problem dimensions. These sizes depend on the internal tile
    configuration of the CUTLASS kernel.

    Args:
        M: Number of rows in A and output.
        N: Number of columns in B and output.
        K: Inner dimension (columns of A, rows of B).

    Returns:
        Tuple of (scale_A_size, scale_B_size) as integers.

    Example:
        >>> sfa_size, sfb_size = fp8_fp8_get_scale_sizes(256, 256, 256)
        >>> scale_A = pk.from_numpy(np.ones(sfa_size, dtype=np.float32))
        >>> scale_B = pk.from_numpy(np.ones(sfb_size, dtype=np.float32))
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.fp8_fp8_get_scale_sizes(M, N, K)
    else:
        return (0, 0)


def matmul_fp8_fp8_blockwise_sm120(
    a: GPUArray,
    b: GPUArray,
    scale_a: GPUArray,
    scale_b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Blockwise scaled FP8 I/O matrix multiplication for SM120.

    This function takes FP8 E4M3 inputs with per-block scale factors,
    performs the GEMM using CUTLASS FP8 kernels, and returns FP8 E4M3 output.

    The scale factors are applied per block during the GEMM computation,
    enabling better precision for FP8 models with varied value ranges.

    Args:
        a: First input array (M x K), FP8 E4M3 stored as uint8.
        b: Second input array (K x N), FP8 E4M3 stored as uint8.
           Should be in ColumnMajor format (pre-transposed).
        scale_a: Scale factors for A, float32. Size from fp8_fp8_get_scale_sizes().
        scale_b: Scale factors for B, float32. Size from fp8_fp8_get_scale_sizes().
        out: Optional output array (M x N), uint8. If provided, result is
            written to this array instead of allocating a new one.

    Returns:
        The result GPUArray (M x N), FP8 E4M3 stored as uint8.

    Raises:
        ValueError: If arrays are not 2D, dtypes are wrong, or dimensions don't match.
        RuntimeError: If FP8 SM120 is not available.

    Example:
        >>> import pygpukit as gk
        >>> from pygpukit.ops import fp8_fp8_get_scale_sizes, matmul_fp8_fp8_blockwise_sm120
        >>> M, N, K = 256, 256, 256
        >>> sfa_size, sfb_size = fp8_fp8_get_scale_sizes(M, N, K)
        >>> scale_A = gk.from_numpy(np.ones(sfa_size, dtype=np.float32))
        >>> scale_B = gk.from_numpy(np.ones(sfb_size, dtype=np.float32))
        >>> C = matmul_fp8_fp8_blockwise_sm120(A_fp8, B_fp8, scale_A, scale_B)
    """
    from pygpukit.core.dtypes import float32, uint8

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8_fp8_blockwise_sm120 requires 2D arrays, got {a.ndim}D for A")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8_fp8_blockwise_sm120 requires 2D arrays, got {b.ndim}D for B")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul_fp8_fp8_blockwise_sm120 dimension mismatch: {a.shape} @ {b.shape} "
            f"(inner dimensions {a.shape[1]} and {b.shape[0]} must match)"
        )

    if a.dtype != uint8 or b.dtype != uint8:
        raise ValueError("matmul_fp8_fp8_blockwise_sm120 requires uint8 inputs (FP8)")

    if scale_a.dtype != float32 or scale_b.dtype != float32:
        raise ValueError("matmul_fp8_fp8_blockwise_sm120 requires float32 scale factors")

    if not fp8_fp8_sm120_available():
        raise RuntimeError("FP8 blockwise SM120 GEMM is not available. Requires SM120+.")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_fp8_fp8_blockwise_sm120_native(a, b, scale_a, scale_b, out=out)
    else:
        raise RuntimeError("FP8 blockwise SM120 GEMM requires native backend")


def _matmul_fp8_fp8_blockwise_sm120_native(
    a: GPUArray,
    b: GPUArray,
    scale_a: GPUArray,
    scale_b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ implementation of blockwise FP8 I/O GEMM for SM120."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays
    a_native = a._get_native()
    b_native = b._get_native()
    scale_a_native = scale_a._get_native()
    scale_b_native = scale_b._get_native()

    # Allocate output if needed
    if out is None:
        M, K = a.shape
        N = b.shape[1]
        out_native = native.empty([M, N], native.DataType.UInt8)
        out = GPUArray._wrap_native(out_native)
    else:
        out_native = out._get_native()

    # Call blockwise FP8 GEMM
    native.gemm_fp8_fp8_blockwise_sm120(
        a_native, b_native, out_native, scale_a_native, scale_b_native
    )

    return out


def matmul_fp8_sm100(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 matrix multiplication for SM100 (Blackwell datacenter).

    This function takes FP32 inputs, internally quantizes them to FP8,
    performs the GEMM using CUTLASS FP8 kernels with BF16 accumulation,
    and returns the result as FP32.

    This may work on SM120 (Blackwell GeForce) as a fallback since both
    are Blackwell architecture.

    Args:
        a: First input array (M x K), FP32.
        b: Second input array (K x N), FP32.
        out: Optional output array (M x N), FP32. If provided, result is
            written to this array instead of allocating a new one.

    Returns:
        The result GPUArray (M x N), FP32.

    Raises:
        ValueError: If arrays are not 2D, not FP32, or dimensions don't match.
        RuntimeError: If FP8 SM100 GEMM is not available or kernel fails.

    Example:
        >>> import pygpukit as gk
        >>> A = gk.from_numpy(np.random.randn(1024, 1024).astype(np.float32) * 0.1)
        >>> B = gk.from_numpy(np.random.randn(1024, 1024).astype(np.float32) * 0.1)
        >>> C = gk.ops.matmul_fp8_sm100(A, B)
    """
    from pygpukit.core.dtypes import float32

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8_sm100 requires 2D arrays, got {a.ndim}D for first argument")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8_sm100 requires 2D arrays, got {b.ndim}D for second argument")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul_fp8_sm100 dimension mismatch: {a.shape} @ {b.shape} "
            f"(inner dimensions {a.shape[1]} and {b.shape[0]} must match)"
        )

    if a.dtype != float32 or b.dtype != float32:
        raise ValueError("matmul_fp8_sm100 requires float32 inputs")

    if not fp8_sm100_available():
        raise RuntimeError(
            "FP8 SM100 GEMM is not available. Requires SM100+ GPU and CUTLASS SM100 support."
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_fp8_sm100_native(a, b, out=out)
    else:
        raise RuntimeError("FP8 SM100 GEMM requires native backend")


def _matmul_fp8_sm100_native(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ implementation of FP8 GEMM for SM100."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays
    a_native = a._get_native()
    b_native = b._get_native()

    # Allocate output if needed
    if out is None:
        M, K = a.shape
        N = b.shape[1]
        out_native = native.empty([M, N], native.DataType.Float32)
        out = GPUArray._wrap_native(out_native)
    else:
        out_native = out._get_native()

    # Call FP8 GEMM
    native.gemm_fp8_sm100(a_native, b_native, out_native)

    return out


def matmul_fp8_sm120(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 matrix multiplication for SM120 (Blackwell GeForce).

    This function takes FP32 inputs, internally quantizes them to FP8,
    performs the GEMM using CUTLASS FP8 kernels with BF16 accumulation,
    and returns the result as FP32.

    Args:
        a: First input array (M x K), FP32.
        b: Second input array (K x N), FP32.
        out: Optional output array (M x N), FP32. If provided, result is
            written to this array instead of allocating a new one.

    Returns:
        The result GPUArray (M x N), FP32.

    Raises:
        ValueError: If arrays are not 2D, not FP32, or dimensions don't match.
        RuntimeError: If FP8 SM120 GEMM is not available or kernel fails.

    Example:
        >>> import pygpukit as gk
        >>> A = gk.from_numpy(np.random.randn(1024, 1024).astype(np.float32) * 0.1)
        >>> B = gk.from_numpy(np.random.randn(1024, 1024).astype(np.float32) * 0.1)
        >>> C = gk.ops.matmul_fp8_sm120(A, B)
    """
    from pygpukit.core.dtypes import float32

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8_sm120 requires 2D arrays, got {a.ndim}D for first argument")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8_sm120 requires 2D arrays, got {b.ndim}D for second argument")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul_fp8_sm120 dimension mismatch: {a.shape} @ {b.shape} "
            f"(inner dimensions {a.shape[1]} and {b.shape[0]} must match)"
        )

    if a.dtype != float32 or b.dtype != float32:
        raise ValueError("matmul_fp8_sm120 requires float32 inputs")

    if not fp8_sm120_available():
        raise RuntimeError(
            "FP8 SM120 GEMM is not available. Requires SM120+ GPU and CUTLASS SM120 support."
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_fp8_sm120_native(a, b, out=out)
    else:
        raise RuntimeError("FP8 SM120 GEMM requires native backend")


def _matmul_fp8_sm120_native(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ implementation of FP8 GEMM for SM120."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays
    a_native = a._get_native()
    b_native = b._get_native()

    # Allocate output if needed
    if out is None:
        M, K = a.shape
        N = b.shape[1]
        out_native = native.empty([M, N], native.DataType.Float32)
        out = GPUArray._wrap_native(out_native)
    else:
        out_native = out._get_native()

    # Call FP8 GEMM
    native.gemm_fp8_sm120(a_native, b_native, out_native)

    return out


def matmul_fp8_sm90(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 matrix multiplication for SM90 (Hopper).

    This function takes FP32 inputs, internally quantizes them to FP8 with
    per-tensor scaling, performs the GEMM using CUTLASS FP8 kernels,
    and returns the result as FP32.

    Args:
        a: First input array (M x K), FP32.
        b: Second input array (K x N), FP32.
        out: Optional output array (M x N), FP32. If provided, result is
            written to this array instead of allocating a new one.

    Returns:
        The result GPUArray (M x N), FP32.

    Raises:
        ValueError: If arrays are not 2D, not FP32, or dimensions don't match.
        RuntimeError: If FP8 SM90 GEMM is not available or kernel fails.

    Example:
        >>> import pygpukit as gk
        >>> A = gk.from_numpy(np.random.randn(1024, 1024).astype(np.float32) * 0.1)
        >>> B = gk.from_numpy(np.random.randn(1024, 1024).astype(np.float32) * 0.1)
        >>> C = gk.ops.matmul_fp8_sm90(A, B)
    """
    from pygpukit.core.dtypes import float32

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8_sm90 requires 2D arrays, got {a.ndim}D for first argument")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8_sm90 requires 2D arrays, got {b.ndim}D for second argument")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul_fp8_sm90 dimension mismatch: {a.shape} @ {b.shape} "
            f"(inner dimensions {a.shape[1]} and {b.shape[0]} must match)"
        )

    if a.dtype != float32 or b.dtype != float32:
        raise ValueError("matmul_fp8_sm90 requires float32 inputs")

    if not fp8_sm90_available():
        raise RuntimeError(
            "FP8 SM90 GEMM is not available. Requires SM90+ GPU and CUTLASS SM90 support."
        )

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_fp8_sm90_native(a, b, out=out)
    else:
        raise RuntimeError("FP8 SM90 GEMM requires native backend")


def _matmul_fp8_sm90_native(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ implementation of FP8 GEMM for SM90."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays
    a_native = a._get_native()
    b_native = b._get_native()

    # Allocate output if needed
    if out is None:
        M, K = a.shape
        N = b.shape[1]
        out_native = native.empty([M, N], native.DataType.Float32)
        out = GPUArray._wrap_native(out_native)
    else:
        out_native = out._get_native()

    # Call FP8 GEMM
    native.gemm_fp8_sm90(a_native, b_native, out_native)

    return out


def nvf4_bf16_sm120_available() -> bool:
    """Check if NVF4 (4-bit) BF16 GEMM is available on SM120 (Blackwell GeForce).

    This variant uses NVF4 (4-bit float) for 2x memory bandwidth compared to FP8,
    making it ideal for memory-bound LLM inference workloads.

    Returns:
        True if NVF4 BF16 SM120 GEMM is available, False otherwise.
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.nvf4_bf16_sm120_available()
    else:
        return False


def matmul_nvf4_bf16_sm120(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """NVF4 (4-bit) GEMM with BF16 input/output for SM120 (Blackwell GeForce).

    This variant uses NVF4 (float_e2m1_t, 4-bit) for the internal computation,
    providing 2x memory bandwidth compared to FP8. Ideal for memory-bound
    LLM inference workloads.

    Data flow: BF16 input -> NVF4 quantize with block scaling -> GEMM -> BF16 output

    Args:
        a: First input array (M x K), BF16.
        b: Second input array (K x N), BF16.
        out: Optional output array (M x N), BF16.

    Returns:
        The result GPUArray (M x N), BF16.

    Raises:
        ValueError: If arrays are not 2D, not BF16, or dimensions don't match.
        RuntimeError: If NVF4 BF16 SM120 GEMM is not available.
    """
    from pygpukit.core.dtypes import bfloat16

    if a.ndim != 2:
        raise ValueError(f"matmul_nvf4_bf16_sm120 requires 2D arrays, got {a.ndim}D")
    if b.ndim != 2:
        raise ValueError(f"matmul_nvf4_bf16_sm120 requires 2D arrays, got {b.ndim}D")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul_nvf4_bf16_sm120 dimension mismatch: {a.shape} @ {b.shape}")

    if a.dtype != bfloat16 or b.dtype != bfloat16:
        raise ValueError("matmul_nvf4_bf16_sm120 requires bfloat16 inputs")

    if not nvf4_bf16_sm120_available():
        raise RuntimeError("NVF4 BF16 SM120 GEMM is not available. Requires SM120+ GPU.")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        return _matmul_nvf4_bf16_sm120_native(a, b, out=out)
    else:
        raise RuntimeError("NVF4 BF16 SM120 GEMM requires native backend")


def _matmul_nvf4_bf16_sm120_native(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Native C++ implementation of NVF4 BF16 GEMM for SM120."""
    from pygpukit.core.backend import get_native_module

    native = get_native_module()

    # Get native arrays
    a_native = a._get_native()
    b_native = b._get_native()

    # Allocate output if needed
    if out is None:
        M, K = a.shape
        N = b.shape[1]
        out_native = native.empty([M, N], native.DataType.BFloat16)
        out = GPUArray._wrap_native(out_native)
    else:
        out_native = out._get_native()

    # Call NVF4 BF16 GEMM
    native.gemm_nvf4_bf16_sm120(a_native, b_native, out_native)

    return out


# ============================================================================
# GEMV Operations (M=1 special case)
# ============================================================================


def gemv_nvf4_available() -> bool:
    """Check if NVF4 GEMV is available (SM120+).

    Returns:
        True if NVF4 GEMV is available on current GPU.
    """
    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.gemv_nvf4_available()
    else:
        return False


def nvf4_get_sizes(K: int, N: int) -> tuple[int, int]:
    """Get buffer sizes for NVF4-quantized weights.

    Args:
        K: Inner dimension (input features).
        N: Output dimension (output features).

    Returns:
        Tuple of (data_size, scale_size) in bytes.
        - data_size: Size for packed NVF4 weights [K/2, N]
        - scale_size: Size for UE4M3 scale factors [K/32, N]

    Note:
        NVF4 provides 4x compression vs BF16:
        - BF16 weight size: K * N * 2 bytes
        - NVF4 total size: K/2 * N + K/32 * N bytes
    """
    data_size = (K // 2) * N
    scale_size = ((K + 31) // 32) * N
    return data_size, scale_size


def quantize_bf16_to_nvf4(
    input: GPUArray,
    out_data: GPUArray,
    out_scale: GPUArray,
) -> None:
    """Quantize BF16 weights to NVF4 format with block scaling.

    This quantizes BF16 weights to 4-bit NVF4 format with UE4M3 scale factors.
    Each 32-element block shares one scale factor.

    Args:
        input: BF16 weight matrix [K, N].
        out_data: Pre-allocated buffer for packed NVF4 data [K/2, N] (uint8).
        out_scale: Pre-allocated buffer for scale factors [K/32, N] (uint8).

    Raises:
        ValueError: If input is not 2D BF16, or buffers have wrong size.
        RuntimeError: If NVF4 is not available.

    Note:
        NVF4 values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} and negatives.
        Block size: 32 elements per scale factor.
    """
    from pygpukit.core.dtypes import bfloat16

    if input.ndim != 2:
        raise ValueError(f"quantize_bf16_to_nvf4 requires 2D input, got {input.ndim}D")

    if input.dtype != bfloat16:
        raise ValueError(f"quantize_bf16_to_nvf4 requires bfloat16 input, got {input.dtype}")

    if not gemv_nvf4_available():
        raise RuntimeError("NVF4 quantization not available. Requires SM120+ GPU.")

    K, N = input.shape
    expected_data_size, expected_scale_size = nvf4_get_sizes(K, N)

    # Validate buffer sizes (count elements)
    actual_data_size = (
        out_data.shape[0] * out_data.shape[1] if out_data.ndim == 2 else out_data.size
    )
    actual_scale_size = (
        out_scale.shape[0] * out_scale.shape[1] if out_scale.ndim == 2 else out_scale.size
    )

    if actual_data_size < expected_data_size:
        raise ValueError(f"out_data buffer too small: {actual_data_size} < {expected_data_size}")
    if actual_scale_size < expected_scale_size:
        raise ValueError(f"out_scale buffer too small: {actual_scale_size} < {expected_scale_size}")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        input_native = input._get_native()
        data_native = out_data._get_native()
        scale_native = out_scale._get_native()
        native.quantize_bf16_to_nvf4(input_native, data_native, scale_native)


def gemv_nvf4_bf16(
    a: GPUArray,
    b_data: GPUArray,
    b_scale: GPUArray,
    *,
    out: GPUArray | None = None,
    alpha: float = 1.0,
) -> GPUArray:
    """NVF4 GEMV: C[N] = alpha * A[K] @ B[K,N] (NVF4 quantized).

    This performs matrix-vector multiplication where the weight matrix B
    is pre-quantized to NVF4 format with block scaling.

    Args:
        a: Input vector [K], BF16.
        b_data: Packed NVF4 weight data [K/2, N], uint8.
        b_scale: UE4M3 scale factors [K/32, N], uint8.
        out: Optional output vector [N], BF16.
        alpha: Scaling factor (default 1.0).

    Returns:
        Output vector [N], BF16.

    Raises:
        ValueError: If shapes or dtypes don't match.
        RuntimeError: If NVF4 GEMV is not available.

    Note:
        For LLM inference decode path (M=1), NVF4 provides 4x bandwidth
        reduction vs BF16, which is critical for memory-bound workloads.
    """
    from pygpukit.core.dtypes import bfloat16

    if a.ndim != 1:
        raise ValueError(f"gemv_nvf4_bf16 requires 1D input vector, got {a.ndim}D")

    if a.dtype != bfloat16:
        raise ValueError(f"gemv_nvf4_bf16 requires bfloat16 input, got {a.dtype}")

    if not gemv_nvf4_available():
        raise RuntimeError("NVF4 GEMV not available. Requires SM120+ GPU.")

    # Infer N from b_data shape: [K/2, N]
    if b_data.ndim == 2:
        N = b_data.shape[1]
    else:
        raise ValueError(f"b_data must be 2D [K/2, N], got {b_data.ndim}D")

    # Validate output
    if out is not None:
        if out.shape != (N,):
            raise ValueError(f"out shape {out.shape} does not match expected ({N},)")
        if out.dtype != bfloat16:
            raise ValueError(f"out dtype {out.dtype} must be bfloat16")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        a_native = a._get_native()
        data_native = b_data._get_native()
        scale_native = b_scale._get_native()

        if out is None:
            out_native = native.empty([N], native.DataType.BFloat16)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemv_nvf4_bf16(a_native, data_native, scale_native, out_native, alpha)

        return out
    else:
        raise RuntimeError("NVF4 GEMV requires native backend")


def gemv_bf16(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> GPUArray:
    """BF16 GEMV: C[N] = alpha * A[K] @ B[K,N] + beta * C[N].

    Standard BF16 matrix-vector multiplication without quantization.

    Args:
        a: Input vector [K], BF16.
        b: Weight matrix [K, N], BF16 (row-major).
        out: Optional output vector [N], BF16.
        alpha: Scaling factor for A @ B (default 1.0).
        beta: Scaling factor for existing C (default 0.0).

    Returns:
        Output vector [N], BF16.

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    from pygpukit.core.dtypes import bfloat16

    if a.ndim != 1:
        raise ValueError(f"gemv_bf16 requires 1D input vector, got {a.ndim}D")

    if b.ndim != 2:
        raise ValueError(f"gemv_bf16 requires 2D weight matrix, got {b.ndim}D")

    if a.dtype != bfloat16 or b.dtype != bfloat16:
        raise ValueError("gemv_bf16 requires bfloat16 inputs")

    K = a.shape[0]
    if b.shape[0] != K:
        raise ValueError(f"gemv_bf16 dimension mismatch: A[{K}] vs B[{b.shape[0]}, {b.shape[1]}]")

    N = b.shape[1]

    # Validate output
    if out is not None:
        if out.shape != (N,):
            raise ValueError(f"out shape {out.shape} does not match expected ({N},)")
        if out.dtype != bfloat16:
            raise ValueError(f"out dtype {out.dtype} must be bfloat16")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        a_native = a._get_native()
        b_native = b._get_native()

        if out is None:
            out_native = native.empty([N], native.DataType.BFloat16)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemv_bf16(a_native, b_native, out_native, alpha, beta)

        return out
    else:
        # CPU fallback
        a_np: np.ndarray[np.floating] = a.to_numpy().astype(np.float32)
        b_np: np.ndarray[np.floating] = b.to_numpy().astype(np.float32)
        result: np.ndarray[np.floating] = alpha * (a_np @ b_np)
        if out is not None:
            result = result + beta * out.to_numpy().astype(np.float32)
        return from_numpy(result.astype(np.float16).view(np.uint16).astype(np.uint16))


# Flag to track if FP8 LUT has been initialized
_FP8_LUT_INITIALIZED = False


def fp8_init_lut() -> None:
    """Initialize FP8 E4M3 lookup table for dequantization.

    Note: LUT is defined as __device__ __constant__ in C++ and initialized
    at compile time, so this function is a no-op. Kept for API compatibility.
    """
    global _FP8_LUT_INITIALIZED
    if _FP8_LUT_INITIALIZED:
        return
    # LUT is already initialized in constant memory at compile time
    _FP8_LUT_INITIALIZED = True


def gemv_fp8_bf16(
    a: GPUArray,
    b_nk: GPUArray,
    b_scale: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Optimized FP8 GEMV: C[N] = A[K] @ B[N,K]^T.

    W8A16 GEMV: FP8 weights with BF16 activation and output.
    Uses warp-level reduction, shared memory, and vectorized loads.

    Args:
        a: Activation vector [K], BF16.
        b_nk: FP8 E4M3 weight matrix [N, K], stored as uint8.
        b_scale: Block-wise scale factors [N/128, K/128], BF16.
        out: Optional output vector [N], BF16.

    Returns:
        Output vector [N], BF16.

    Note:
        Weight layout is [N, K] (row = output dimension).
        Use original weight tensor directly (no transpose needed).
    """
    from pygpukit.core.dtypes import bfloat16, uint8

    if a.ndim != 1:
        raise ValueError(f"gemv_fp8_bf16 requires 1D input vector, got {a.ndim}D")

    if b_nk.ndim != 2:
        raise ValueError(f"gemv_fp8_bf16 requires 2D weight matrix, got {b_nk.ndim}D")

    if a.dtype != bfloat16:
        raise ValueError(f"gemv_fp8_bf16 requires bfloat16 activation, got {a.dtype}")

    if b_nk.dtype != uint8:
        raise ValueError(f"gemv_fp8_bf16 requires uint8 (FP8) weights, got {b_nk.dtype}")

    if b_scale.dtype != bfloat16:
        raise ValueError(f"gemv_fp8_bf16 requires bfloat16 scale, got {b_scale.dtype}")

    K = a.shape[0]
    N = b_nk.shape[0]  # [N, K] layout

    if b_nk.shape[1] != K:
        raise ValueError(f"gemv_fp8_bf16 dimension mismatch: A[{K}] vs B[{N}, {b_nk.shape[1]}]")

    # Validate output
    if out is not None:
        if out.shape != (N,):
            raise ValueError(f"out shape {out.shape} does not match expected ({N},)")
        if out.dtype != bfloat16:
            raise ValueError(f"out dtype {out.dtype} must be bfloat16")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        a_native = a._get_native()
        b_nk_native = b_nk._get_native()
        b_scale_native = b_scale._get_native()

        if out is None:
            out_native = native.empty([N], native.DataType.BFloat16)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemv_fp8_bf16_opt(a_native, b_nk_native, b_scale_native, out_native)

        return out
    else:
        raise NotImplementedError("FP8 GEMV requires native GPU backend")


def gemv_fp8_bf16_batched(
    a: GPUArray,
    b_nk: GPUArray,
    b_scale: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Optimized batched FP8 GEMV: C[M,N] = A[M,K] @ B[N,K]^T.

    W8A16 GEMM for M>1: FP8 weights with BF16 activation and output.
    Uses warp-level reduction, shared memory, and vectorized loads.

    Args:
        a: Activation matrix [M, K], BF16.
        b_nk: FP8 E4M3 weight matrix [N, K], stored as uint8.
        b_scale: Block-wise scale factors [N/128, K/128], BF16.
        out: Optional output matrix [M, N], BF16.

    Returns:
        Output matrix [M, N], BF16.

    Note:
        Weight layout is [N, K] (row = output dimension).
        Use original weight tensor directly (no transpose needed).
    """
    from pygpukit.core.dtypes import bfloat16, uint8

    if a.ndim != 2:
        raise ValueError(f"gemv_fp8_bf16_batched requires 2D input matrix, got {a.ndim}D")

    if b_nk.ndim != 2:
        raise ValueError(f"gemv_fp8_bf16_batched requires 2D weight matrix, got {b_nk.ndim}D")

    if a.dtype != bfloat16:
        raise ValueError(f"gemv_fp8_bf16_batched requires bfloat16 activation, got {a.dtype}")

    if b_nk.dtype != uint8:
        raise ValueError(f"gemv_fp8_bf16_batched requires uint8 (FP8) weights, got {b_nk.dtype}")

    if b_scale.dtype != bfloat16:
        raise ValueError(f"gemv_fp8_bf16_batched requires bfloat16 scale, got {b_scale.dtype}")

    M = a.shape[0]
    K = a.shape[1]
    N = b_nk.shape[0]  # [N, K] layout

    if b_nk.shape[1] != K:
        raise ValueError(
            f"gemv_fp8_bf16_batched dimension mismatch: A[{M},{K}] vs B[{N},{b_nk.shape[1]}]"
        )

    # Validate output
    if out is not None:
        if out.shape != (M, N):
            raise ValueError(f"out shape {out.shape} does not match expected ({M}, {N})")
        if out.dtype != bfloat16:
            raise ValueError(f"out dtype {out.dtype} must be bfloat16")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        a_native = a._get_native()
        b_nk_native = b_nk._get_native()
        b_scale_native = b_scale._get_native()

        if out is None:
            out_native = native.empty([M, N], native.DataType.BFloat16)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemv_fp8_bf16_opt_batched(a_native, b_nk_native, b_scale_native, out_native)

        return out
    else:
        raise NotImplementedError("FP8 batched GEMV requires native GPU backend")


def w8a16_gemm_sm120(
    a: GPUArray,
    b_fp8: GPUArray,
    b_scale: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """W8A16 GEMM for SM120: C[M,N] = A[M,K] @ dequant(B_fp8[K,N]).

    FP8 weight x BF16 activation -> BF16 output.
    Uses TensorCore GEMM with online FP8 dequantization.
    More efficient than batched GEMV for M > 1.

    Args:
        a: Activation matrix [M, K], BF16.
        b_fp8: FP8 E4M3 weight matrix [K, N], stored as uint8.
        b_scale: Block-wise scale factors [K/128, N/128], BF16.
        out: Optional output matrix [M, N], BF16.

    Returns:
        Output matrix [M, N], BF16.
    """
    from pygpukit.core.dtypes import bfloat16, uint8

    if a.ndim != 2:
        raise ValueError(f"w8a16_gemm_sm120 requires 2D input matrix, got {a.ndim}D")

    if b_fp8.ndim != 2:
        raise ValueError(f"w8a16_gemm_sm120 requires 2D weight matrix, got {b_fp8.ndim}D")

    if a.dtype != bfloat16:
        raise ValueError(f"w8a16_gemm_sm120 requires bfloat16 activation, got {a.dtype}")

    if b_fp8.dtype != uint8:
        raise ValueError(f"w8a16_gemm_sm120 requires uint8 (FP8) weights, got {b_fp8.dtype}")

    if b_scale.dtype != bfloat16:
        raise ValueError(f"w8a16_gemm_sm120 requires bfloat16 scale, got {b_scale.dtype}")

    M = a.shape[0]
    K = a.shape[1]
    if b_fp8.shape[0] != K:
        raise ValueError(
            f"w8a16_gemm_sm120 dimension mismatch: A[{M},{K}] vs B[{b_fp8.shape[0]}, {b_fp8.shape[1]}]"
        )

    N = b_fp8.shape[1]

    # Validate output
    if out is not None:
        if out.shape != (M, N):
            raise ValueError(f"out shape {out.shape} does not match expected ({M}, {N})")
        if out.dtype != bfloat16:
            raise ValueError(f"out dtype {out.dtype} must be bfloat16")

    # Initialize LUT if not already done
    fp8_init_lut()

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        a_native = a._get_native()
        b_fp8_native = b_fp8._get_native()
        b_scale_native = b_scale._get_native()

        if out is None:
            out_native = native.empty([M, N], native.DataType.BFloat16)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.w8a16_gemm_sm120(a_native, b_fp8_native, b_scale_native, out_native)

        return out
    else:
        raise NotImplementedError("W8A16 GEMM requires native GPU backend with SM120")


# Track if grouped GEMM LUT is initialized
_grouped_gemm_lut_initialized = False


def grouped_gemm_init_lut() -> None:
    """Initialize FP8->BF16 LUT for grouped GEMM.

    This must be called once before using grouped_gemm_fp8_bf16.
    """
    global _grouped_gemm_lut_initialized
    if _grouped_gemm_lut_initialized:
        return

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        native.grouped_gemm_init_lut()
        _grouped_gemm_lut_initialized = True
    else:
        raise NotImplementedError("Grouped GEMM requires native GPU backend")


def grouped_gemm_fp8_bf16(
    a: GPUArray,
    b_stacked: GPUArray,
    b_scale: GPUArray,
    row_expert_ids: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Grouped GEMM for MoE: C = A @ B_stacked with per-row expert IDs.

    Each row has an associated expert ID, and the kernel dispatches to the
    correct expert's weights for each row.

    Args:
        a: Input tokens [M, K], BF16.
        b_stacked: Stacked expert weights [num_experts, N, K], FP8 (uint8).
        b_scale: Block-wise scales [num_experts, N/128, K/128], BF16.
        row_expert_ids: Expert ID for each row [M], int32.
        out: Optional output tensor [M, N], BF16.

    Returns:
        Output tensor [M, N], BF16.
    """
    from pygpukit.core.dtypes import bfloat16, int32, uint8

    if a.ndim != 2:
        raise ValueError(f"grouped_gemm_fp8_bf16 requires 2D input, got {a.ndim}D")

    if b_stacked.ndim != 3:
        raise ValueError(f"grouped_gemm_fp8_bf16 requires 3D weight, got {b_stacked.ndim}D")

    if a.dtype != bfloat16:
        raise ValueError(f"grouped_gemm_fp8_bf16 requires bfloat16 input, got {a.dtype}")

    if b_stacked.dtype != uint8:
        raise ValueError(
            f"grouped_gemm_fp8_bf16 requires uint8 (FP8) weights, got {b_stacked.dtype}"
        )

    if b_scale.dtype != bfloat16:
        raise ValueError(f"grouped_gemm_fp8_bf16 requires bfloat16 scale, got {b_scale.dtype}")

    if row_expert_ids.dtype != int32:
        raise ValueError(
            f"grouped_gemm_fp8_bf16 requires int32 row_expert_ids, got {row_expert_ids.dtype}"
        )

    M = a.shape[0]
    K = a.shape[1]
    N = b_stacked.shape[1]

    if b_stacked.shape[2] != K:
        raise ValueError(
            f"grouped_gemm_fp8_bf16: K mismatch A[{M},{K}] vs B[...{N},{b_stacked.shape[2]}]"
        )

    if row_expert_ids.shape[0] != M:
        raise ValueError(
            f"grouped_gemm_fp8_bf16: row_expert_ids size {row_expert_ids.shape[0]} != M ({M})"
        )

    # Validate output
    if out is not None:
        if out.shape != (M, N):
            raise ValueError(f"out shape {out.shape} does not match expected ({M}, {N})")
        if out.dtype != bfloat16:
            raise ValueError(f"out dtype {out.dtype} must be bfloat16")

    # Initialize LUT if not already done
    grouped_gemm_init_lut()

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        a_native = a._get_native()
        b_stacked_native = b_stacked._get_native()
        b_scale_native = b_scale._get_native()
        row_expert_ids_native = row_expert_ids._get_native()

        if out is None:
            out_native = native.empty([M, N], native.DataType.BFloat16)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.grouped_gemm_fp8_bf16(
            a_native, b_stacked_native, b_scale_native, out_native, row_expert_ids_native
        )

        return out
    else:
        raise NotImplementedError("Grouped GEMM requires native GPU backend")


def fp8_get_sizes(K: int, N: int) -> tuple[int, int, int]:
    """Get scale tensor dimensions for FP8 block quantization.

    Args:
        K: Input dimension.
        N: Output dimension.

    Returns:
        (scale_K, scale_N, scale_size_bytes): Scale tensor dimensions
        for 128x128 block quantization.
    """
    scale_k = (K + 127) // 128
    scale_n = (N + 127) // 128
    scale_size = scale_k * scale_n * 2  # BF16 = 2 bytes
    return scale_k, scale_n, scale_size


# ============================================================================
# FP8 Operations
# ============================================================================


def matmul_fp8(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 matrix multiplication with automatic backend selection.

    This function takes FP32 inputs, internally quantizes them to FP8,
    performs the GEMM using the best available CUTLASS FP8 kernel,
    and returns the result as FP32.

    Backend priority:
    - SM120 (Blackwell GeForce): blockwise scaling (when CUTLASS bug #2902 is fixed)
    - SM90 (Hopper): per-tensor scaling

    Args:
        a: First input array (M x K), FP32.
        b: Second input array (K x N), FP32.
        out: Optional output array (M x N), FP32. If provided, result is
            written to this array instead of allocating a new one.

    Returns:
        The result GPUArray (M x N), FP32.

    Raises:
        ValueError: If arrays are not 2D, not FP32, or dimensions don't match.
        RuntimeError: If no FP8 GEMM backend is available.

    Example:
        >>> import pygpukit as gk
        >>> A = gk.from_numpy(np.random.randn(1024, 1024).astype(np.float32) * 0.1)
        >>> B = gk.from_numpy(np.random.randn(1024, 1024).astype(np.float32) * 0.1)
        >>> C = gk.ops.matmul_fp8(A, B)
    """
    from pygpukit.core.dtypes import float32

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8 requires 2D arrays, got {a.ndim}D for first argument")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8 requires 2D arrays, got {b.ndim}D for second argument")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul_fp8 dimension mismatch: {a.shape} @ {b.shape} "
            f"(inner dimensions {a.shape[1]} and {b.shape[0]} must match)"
        )

    if a.dtype != float32 or b.dtype != float32:
        raise ValueError("matmul_fp8 requires float32 inputs")

    if not fp8_available():
        raise RuntimeError("FP8 GEMM is not available. Requires SM90+ GPU and CUTLASS support.")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        # Get native arrays
        a_native = a._get_native()
        b_native = b._get_native()

        # Allocate output if needed
        if out is None:
            M, K = a.shape
            N = b.shape[1]
            out_native = native.empty([M, N], native.DataType.Float32)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        # Call auto-dispatch FP8 GEMM
        native.gemm_fp8(a_native, b_native, out_native)

        return out
    else:
        raise RuntimeError("FP8 GEMM requires native backend")
