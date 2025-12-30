"""GEMV (Matrix-Vector) operations.

Optimized GEMV for LLM decode (M=1 case).
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend
from pygpukit.core.factory import from_numpy


def gemv_bf16(
    a: GPUArray,
    b: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """BF16 GEMV: C[N] = A[K] @ B[N,K]^T.

    Optimized BF16 matrix-vector multiplication with B[N,K] layout.
    """
    from pygpukit.core.dtypes import bfloat16

    if a.ndim != 1:
        raise ValueError(f"gemv_bf16 requires 1D input vector, got {a.ndim}D")
    if b.ndim != 2:
        raise ValueError(f"gemv_bf16 requires 2D weight matrix, got {b.ndim}D")
    if a.dtype != bfloat16 or b.dtype != bfloat16:
        raise ValueError("gemv_bf16 requires bfloat16 inputs")

    K = a.shape[0]
    N = b.shape[0]

    if b.shape[1] != K:
        raise ValueError(f"gemv_bf16 dimension mismatch: A[{K}] vs B[{N}, {b.shape[1]}]")

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

        native.gemv_bf16_bf16_sm120(a_native, b_native, out_native)
        return out
    else:
        a_np: np.ndarray = a.to_numpy().astype(np.float32)
        b_np: np.ndarray = b.to_numpy().astype(np.float32)
        result: np.ndarray = b_np @ a_np
        return from_numpy(result.astype(np.float16).view(np.uint16).astype(np.uint16))


gemv_bf16_bf16_sm120 = gemv_bf16


def gemv_fp8_bf16(
    a: GPUArray,
    b_nk: GPUArray,
    b_scale: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Optimized FP8 GEMV: C[N] = A[K] @ B[N,K]^T.

    W8A16 GEMV: FP8 weights with BF16 activation and output.
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
    N = b_nk.shape[0]

    if b_nk.shape[1] != K:
        raise ValueError(f"gemv_fp8_bf16 dimension mismatch: A[{K}] vs B[{N}, {b_nk.shape[1]}]")

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

        native.gemv_fp8_bf16_sm120(a_native, b_nk_native, b_scale_native, out_native)
        return out
    else:
        raise NotImplementedError("FP8 GEMV requires native GPU backend")


gemv_fp8_bf16_sm120 = gemv_fp8_bf16


def gemv_fp8_bf16_batched(
    a: GPUArray,
    b_nk: GPUArray,
    b_scale: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Optimized batched FP8 GEMV: C[M,N] = A[M,K] @ B[N,K]^T.

    W8A16 GEMM for M>1: FP8 weights with BF16 activation and output.
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
    N = b_nk.shape[0]

    if b_nk.shape[1] != K:
        raise ValueError(
            f"gemv_fp8_bf16_batched dimension mismatch: A[{M},{K}] vs B[{N},{b_nk.shape[1]}]"
        )

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

        native.gemv_fp8_bf16_batched_sm120(a_native, b_nk_native, b_scale_native, out_native)
        return out
    else:
        raise NotImplementedError("FP8 batched GEMV requires native GPU backend")


gemv_fp8_bf16_batched_sm120 = gemv_fp8_bf16_batched


__all__ = [
    "gemv_bf16",
    "gemv_bf16_bf16_sm120",
    "gemv_fp8_bf16",
    "gemv_fp8_bf16_sm120",
    "gemv_fp8_bf16_batched",
    "gemv_fp8_bf16_batched_sm120",
]
