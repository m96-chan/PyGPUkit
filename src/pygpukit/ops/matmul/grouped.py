"""Grouped GEMM operations for MoE (Mixture of Experts).

Grouped GEMM with per-row expert dispatching.
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import NativeBackend, get_backend

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

        native.grouped_gemm_fp8_bf16_sm120(
            a_native, b_stacked_native, b_scale_native, out_native, row_expert_ids_native
        )

        return out
    else:
        raise NotImplementedError("Grouped GEMM requires native GPU backend")


grouped_gemm_fp8_bf16_sm120 = grouped_gemm_fp8_bf16


__all__ = [
    "grouped_gemm_init_lut",
    "grouped_gemm_fp8_bf16",
    "grouped_gemm_fp8_bf16_sm120",
]
