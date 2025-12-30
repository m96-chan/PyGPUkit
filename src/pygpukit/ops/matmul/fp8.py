"""FP8 GEMM operations.

FP8 matrix multiplication for SM90/SM100/SM120.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend

from .availability import (
    fp8_available,
    fp8_fp8_sm120_available,
    fp8_sm90_available,
    fp8_sm100_available,
    fp8_sm120_available,
)


def matmul_fp8(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 matrix multiplication with automatic backend selection.

    Takes FP32 inputs, internally quantizes to FP8, performs GEMM,
    and returns FP32 result.
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
        a_native = a._get_native()
        b_native = b._get_native()

        if out is None:
            M, K = a.shape
            N = b.shape[1]
            out_native = native.empty([M, N], native.DataType.Float32)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemm_fp8(a_native, b_native, out_native)
        return out
    else:
        raise RuntimeError("FP8 GEMM requires native backend")


def matmul_fp8_sm90(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 matrix multiplication for SM90 (Hopper)."""
    from pygpukit.core.dtypes import float32

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8_sm90 requires 2D arrays, got {a.ndim}D for first argument")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8_sm90 requires 2D arrays, got {b.ndim}D for second argument")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul_fp8_sm90 dimension mismatch: {a.shape} @ {b.shape}")

    if a.dtype != float32 or b.dtype != float32:
        raise ValueError("matmul_fp8_sm90 requires float32 inputs")

    if not fp8_sm90_available():
        raise RuntimeError("FP8 SM90 GEMM is not available.")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        a_native = a._get_native()
        b_native = b._get_native()

        if out is None:
            M, K = a.shape
            N = b.shape[1]
            out_native = native.empty([M, N], native.DataType.Float32)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemm_fp8_f32_sm90(a_native, b_native, out_native)
        return out
    else:
        raise RuntimeError("FP8 SM90 GEMM requires native backend")


gemm_fp8_f32_sm90 = matmul_fp8_sm90


def matmul_fp8_sm100(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 matrix multiplication for SM100 (Blackwell datacenter)."""
    from pygpukit.core.dtypes import float32

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8_sm100 requires 2D arrays, got {a.ndim}D")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8_sm100 requires 2D arrays, got {b.ndim}D")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul_fp8_sm100 dimension mismatch: {a.shape} @ {b.shape}")

    if a.dtype != float32 or b.dtype != float32:
        raise ValueError("matmul_fp8_sm100 requires float32 inputs")

    if not fp8_sm100_available():
        raise RuntimeError("FP8 SM100 GEMM is not available.")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        a_native = a._get_native()
        b_native = b._get_native()

        if out is None:
            M, K = a.shape
            N = b.shape[1]
            out_native = native.empty([M, N], native.DataType.Float32)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemm_fp8_f32_sm100(a_native, b_native, out_native)
        return out
    else:
        raise RuntimeError("FP8 SM100 GEMM requires native backend")


gemm_fp8_f32_sm100 = matmul_fp8_sm100


def matmul_fp8_sm120(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """FP8 matrix multiplication for SM120 (Blackwell GeForce)."""
    from pygpukit.core.dtypes import float32

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8_sm120 requires 2D arrays, got {a.ndim}D")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8_sm120 requires 2D arrays, got {b.ndim}D")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul_fp8_sm120 dimension mismatch: {a.shape} @ {b.shape}")

    if a.dtype != float32 or b.dtype != float32:
        raise ValueError("matmul_fp8_sm120 requires float32 inputs")

    if not fp8_sm120_available():
        raise RuntimeError("FP8 SM120 GEMM is not available.")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        a_native = a._get_native()
        b_native = b._get_native()

        if out is None:
            M, K = a.shape
            N = b.shape[1]
            out_native = native.empty([M, N], native.DataType.Float32)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemm_fp8_f32_sm120(a_native, b_native, out_native)
        return out
    else:
        raise RuntimeError("FP8 SM120 GEMM requires native backend")


gemm_fp8_f32_sm120 = matmul_fp8_sm120


def matmul_fp8_fp8_sm120(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Pure FP8 I/O matrix multiplication for SM120 (Blackwell GeForce).

    Takes FP8 E4M3 inputs directly (no conversion from FP32).
    """
    from pygpukit.core.dtypes import uint8

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8_fp8_sm120 requires 2D arrays, got {a.ndim}D")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8_fp8_sm120 requires 2D arrays, got {b.ndim}D")

    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul_fp8_fp8_sm120 dimension mismatch: {a.shape} @ {b.shape}")

    if a.dtype != uint8 or b.dtype != uint8:
        raise ValueError("matmul_fp8_fp8_sm120 requires uint8 inputs (FP8 E4M3)")

    if not fp8_fp8_sm120_available():
        raise RuntimeError("Pure FP8 SM120 GEMM is not available.")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        a_native = a._get_native()
        b_native = b._get_native()

        if out is None:
            M, K = a.shape
            N = b.shape[1]
            out_native = native.empty([M, N], native.DataType.UInt8)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemm_fp8_fp8_sm120(a_native, b_native, out_native)
        return out
    else:
        raise RuntimeError("Pure FP8 SM120 GEMM requires native backend")


gemm_fp8_fp8_sm120 = matmul_fp8_fp8_sm120


def fp8_fp8_get_scale_sizes(M: int, N: int, K: int) -> tuple[int, int]:
    """Get scale factor sizes for FP8 blockwise GEMM."""
    backend = get_backend()
    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.gemm_fp8_fp8_get_scale_sizes(M, N, K)
    return (0, 0)


gemm_fp8_fp8_get_scale_sizes = fp8_fp8_get_scale_sizes


def matmul_fp8_fp8_blockwise_sm120(
    a: GPUArray,
    b: GPUArray,
    scale_a: GPUArray,
    scale_b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Blockwise scaled FP8 I/O matrix multiplication for SM120."""
    from pygpukit.core.dtypes import float32, uint8

    if a.ndim != 2:
        raise ValueError(f"matmul_fp8_fp8_blockwise_sm120 requires 2D arrays, got {a.ndim}D")
    if b.ndim != 2:
        raise ValueError(f"matmul_fp8_fp8_blockwise_sm120 requires 2D arrays, got {b.ndim}D")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"matmul_fp8_fp8_blockwise_sm120 dimension mismatch: {a.shape} @ {b.shape}"
        )

    if a.dtype != uint8 or b.dtype != uint8:
        raise ValueError("matmul_fp8_fp8_blockwise_sm120 requires uint8 inputs (FP8)")

    if scale_a.dtype != float32 or scale_b.dtype != float32:
        raise ValueError("matmul_fp8_fp8_blockwise_sm120 requires float32 scale factors")

    if not fp8_fp8_sm120_available():
        raise RuntimeError("FP8 blockwise SM120 GEMM is not available.")

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        a_native = a._get_native()
        b_native = b._get_native()
        scale_a_native = scale_a._get_native()
        scale_b_native = scale_b._get_native()

        if out is None:
            M, K = a.shape
            N = b.shape[1]
            out_native = native.empty([M, N], native.DataType.UInt8)
            out = GPUArray._wrap_native(out_native)
        else:
            out_native = out._get_native()

        native.gemm_fp8_fp8_blockwise_sm120(
            a_native, b_native, out_native, scale_a_native, scale_b_native
        )
        return out
    else:
        raise RuntimeError("FP8 blockwise SM120 GEMM requires native backend")


gemm_fp8_fp8_blockwise_sm120 = matmul_fp8_fp8_blockwise_sm120


def fp8_get_sizes(K: int, N: int) -> tuple[int, int, int]:
    """Get scale tensor dimensions for FP8 block quantization."""
    scale_k = (K + 127) // 128
    scale_n = (N + 127) // 128
    scale_size = scale_k * scale_n * 2
    return scale_k, scale_n, scale_size


# LUT initialization
_FP8_LUT_INITIALIZED = False


def fp8_init_lut() -> None:
    """Initialize FP8 E4M3 lookup table for dequantization."""
    global _FP8_LUT_INITIALIZED
    if _FP8_LUT_INITIALIZED:
        return
    _FP8_LUT_INITIALIZED = True


__all__ = [
    "matmul_fp8",
    "matmul_fp8_sm90",
    "matmul_fp8_sm100",
    "matmul_fp8_sm120",
    "matmul_fp8_fp8_sm120",
    "matmul_fp8_fp8_blockwise_sm120",
    "fp8_fp8_get_scale_sizes",
    "fp8_get_sizes",
    "fp8_init_lut",
    # Aliases
    "gemm_fp8_f32_sm90",
    "gemm_fp8_f32_sm100",
    "gemm_fp8_f32_sm120",
    "gemm_fp8_fp8_sm120",
    "gemm_fp8_fp8_blockwise_sm120",
    "gemm_fp8_fp8_get_scale_sizes",
]
