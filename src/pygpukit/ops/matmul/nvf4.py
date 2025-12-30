"""NVF4 (4-bit float) operations.

NVF4 provides 4x memory bandwidth compared to BF16.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend

from .availability import gemv_nvf4_available, nvf4_bf16_sm120_available


def nvf4_get_sizes(K: int, N: int) -> tuple[int, int]:
    """Get buffer sizes for NVF4-quantized weights.

    Args:
        K: Inner dimension (input features).
        N: Output dimension (output features).

    Returns:
        Tuple of (data_size, scale_size) in bytes.
    """
    data_size = (K // 2) * N
    scale_size = ((K + 31) // 32) * N
    return data_size, scale_size


gemv_nvf4_get_sizes = nvf4_get_sizes


def quantize_bf16_to_nvf4(
    input: GPUArray,
    out_data: GPUArray,
    out_scale: GPUArray,
) -> None:
    """Quantize BF16 weights to NVF4 format with block scaling.

    Args:
        input: BF16 weight matrix [K, N].
        out_data: Pre-allocated buffer for packed NVF4 data [K/2, N] (uint8).
        out_scale: Pre-allocated buffer for scale factors [K/32, N] (uint8).
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


def matmul_nvf4_bf16_sm120(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """NVF4 (4-bit) GEMM with BF16 input/output for SM120.

    Data flow: BF16 input -> NVF4 quantize with block scaling -> GEMM -> BF16 output
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
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        a_native = a._get_native()
        b_native = b._get_native()

        if out is None:
            M, K = a.shape
            N = b.shape[1]
            out_native = native.empty([M, N], native.DataType.BFloat16)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemm_nvf4_bf16_sm120(a_native, b_native, out_native)
        return out
    else:
        raise RuntimeError("NVF4 BF16 SM120 GEMM requires native backend")


gemm_nvf4_bf16_sm120 = matmul_nvf4_bf16_sm120


def gemv_nvf4_bf16(
    a: GPUArray,
    b_data: GPUArray,
    b_scale: GPUArray,
    *,
    out: GPUArray | None = None,
    alpha: float = 1.0,
) -> GPUArray:
    """NVF4 GEMV: C[N] = alpha * A[K] @ B[K,N] (NVF4 quantized).

    Args:
        a: Input vector [K], BF16.
        b_data: Packed NVF4 weight data [K/2, N], uint8.
        b_scale: UE4M3 scale factors [K/32, N], uint8.
        out: Optional output vector [N], BF16.
        alpha: Scaling factor (default 1.0).

    Returns:
        Output vector [N], BF16.
    """
    from pygpukit.core.dtypes import bfloat16

    if a.ndim != 1:
        raise ValueError(f"gemv_nvf4_bf16 requires 1D input vector, got {a.ndim}D")
    if a.dtype != bfloat16:
        raise ValueError(f"gemv_nvf4_bf16 requires bfloat16 input, got {a.dtype}")
    if not gemv_nvf4_available():
        raise RuntimeError("NVF4 GEMV not available. Requires SM120+ GPU.")

    if b_data.ndim == 2:
        N = b_data.shape[1]
    else:
        raise ValueError(f"b_data must be 2D [K/2, N], got {b_data.ndim}D")

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

        native.gemv_nvf4_bf16_sm120(a_native, data_native, scale_native, out_native, alpha)
        return out
    else:
        raise RuntimeError("NVF4 GEMV requires native backend")


gemv_nvf4_bf16_sm120 = gemv_nvf4_bf16


__all__ = [
    "nvf4_get_sizes",
    "gemv_nvf4_get_sizes",
    "quantize_bf16_to_nvf4",
    "matmul_nvf4_bf16_sm120",
    "gemm_nvf4_bf16_sm120",
    "gemv_nvf4_bf16",
    "gemv_nvf4_bf16_sm120",
]
