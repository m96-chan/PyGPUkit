"""W8A16 GEMM operations.

Weight 8-bit (FP8), Activation 16-bit (BF16) GEMM.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend

# Flag to track if W8A16 GEMM LUT has been initialized
_W8A16_GEMM_LUT_INITIALIZED = False


def w8a16_gemm_init_lut() -> None:
    """Initialize FP8->F32 LUT for W8A16 GEMM.

    This uses runtime initialization to avoid symbol conflicts with the GEMV LUT.
    Must be called before using w8a16_gemm_sm120.
    """
    global _W8A16_GEMM_LUT_INITIALIZED
    if _W8A16_GEMM_LUT_INITIALIZED:
        return

    backend = get_backend()

    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        native.gemm_w8a16_init_lut()
        _W8A16_GEMM_LUT_INITIALIZED = True


gemm_w8a16_init_lut = w8a16_gemm_init_lut


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

    # Initialize W8A16 GEMM LUT (runtime initialization to avoid symbol conflicts)
    w8a16_gemm_init_lut()

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

        native.gemm_w8a16_bf16_sm120(a_native, b_fp8_native, b_scale_native, out_native)

        return out
    else:
        raise NotImplementedError("W8A16 GEMM requires native GPU backend with SM120")


gemm_w8a16_bf16_sm120 = w8a16_gemm_sm120


__all__ = [
    "w8a16_gemm_init_lut",
    "gemm_w8a16_init_lut",
    "w8a16_gemm_sm120",
    "gemm_w8a16_bf16_sm120",
]
