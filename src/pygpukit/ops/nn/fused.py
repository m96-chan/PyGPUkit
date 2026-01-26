"""Fused NN operations for improved performance.

Provides fused kernels that combine multiple operations into one kernel launch:
- rmsnorm_residual: Fused RMSNorm + Residual addition
- swiglu: Fused SiLU(gate) * up
- geglu: Fused GELU(gate) * up
"""

from __future__ import annotations

from pygpukit.core.array import GPUArray
from pygpukit.core.backend import get_native_module
from pygpukit.ops._common import _validate_float_dtype


def rmsnorm_residual(
    input: GPUArray,
    residual: GPUArray,
    gamma: GPUArray,
    eps: float = 1e-5,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Fused RMSNorm + Residual addition.

    Computes: y = rmsnorm(x + residual) * gamma

    This fuses two operations into a single kernel:
    1. Residual addition: z = x + residual
    2. RMSNorm: y = z / sqrt(mean(z^2) + eps) * gamma

    Performance benefit: ~1.5-2x vs separate kernels (fewer memory round-trips).

    Args:
        input: Input tensor of shape [batch, features].
        residual: Residual tensor of shape [batch, features].
        gamma: Scale parameter of shape [features].
        eps: Epsilon for numerical stability (default: 1e-5).
        out: Optional output buffer for CUDA Graph capture.

    Returns:
        Normalized output of shape [batch, features].

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(input, "rmsnorm_residual")

    if input.ndim != 2:
        raise ValueError("rmsnorm_residual: input must be 2D [batch, features]")
    if input.shape != residual.shape:
        raise ValueError("rmsnorm_residual: input and residual shape mismatch")
    if gamma.ndim != 1 or gamma.shape[0] != input.shape[1]:
        raise ValueError("rmsnorm_residual: gamma must be 1D with size == features")
    if input.dtype != residual.dtype or input.dtype != gamma.dtype:
        raise ValueError("rmsnorm_residual: all inputs must have same dtype")

    native = get_native_module()
    input_native = input._get_native()
    residual_native = residual._get_native()
    gamma_native = gamma._get_native()

    if out is not None:
        if out.shape != input.shape or out.dtype != input.dtype:
            raise ValueError("rmsnorm_residual: output shape/dtype mismatch")
        out_native = out._get_native()
        native.rmsnorm_residual_(input_native, residual_native, gamma_native, out_native, eps)
        return out
    else:
        result_native = native.rmsnorm_residual(
            input_native, residual_native, gamma_native, eps
        )
        return GPUArray._wrap_native(result_native)


def swiglu(
    gate_proj: GPUArray,
    up_proj: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Fused SwiGLU activation.

    Computes: y = silu(gate_proj) * up_proj

    Where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Used in Qwen, LLaMA3, Mistral FFN layers:
        FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    This kernel fuses the element-wise SiLU and multiply after the projections.

    Performance benefit: ~2x vs separate silu + multiply kernels.

    Args:
        gate_proj: Gate projection tensor (any shape, typically [batch, features]).
        up_proj: Up projection tensor (same shape as gate_proj).
        out: Optional output buffer for CUDA Graph capture.

    Returns:
        Output tensor of same shape as inputs.

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(gate_proj, "swiglu")

    if gate_proj.shape != up_proj.shape:
        raise ValueError("swiglu: gate_proj and up_proj shape mismatch")
    if gate_proj.dtype != up_proj.dtype:
        raise ValueError("swiglu: gate_proj and up_proj dtype mismatch")

    native = get_native_module()
    gate_native = gate_proj._get_native()
    up_native = up_proj._get_native()

    if out is not None:
        if out.shape != gate_proj.shape or out.dtype != gate_proj.dtype:
            raise ValueError("swiglu: output shape/dtype mismatch")
        out_native = out._get_native()
        native.swiglu_(gate_native, up_native, out_native)
        return out
    else:
        result_native = native.swiglu(gate_native, up_native)
        return GPUArray._wrap_native(result_native)


def geglu(
    gate_proj: GPUArray,
    up_proj: GPUArray,
    *,
    out: GPUArray | None = None,
) -> GPUArray:
    """Fused GeGLU activation.

    Computes: y = gelu(gate_proj) * up_proj

    Where gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    GELU variant of gated linear unit, used in some transformer architectures.

    Args:
        gate_proj: Gate projection tensor (any shape, typically [batch, features]).
        up_proj: Up projection tensor (same shape as gate_proj).
        out: Optional output buffer for CUDA Graph capture.

    Returns:
        Output tensor of same shape as inputs.

    Raises:
        ValueError: If shapes or dtypes don't match.
    """
    _validate_float_dtype(gate_proj, "geglu")

    if gate_proj.shape != up_proj.shape:
        raise ValueError("geglu: gate_proj and up_proj shape mismatch")
    if gate_proj.dtype != up_proj.dtype:
        raise ValueError("geglu: gate_proj and up_proj dtype mismatch")

    native = get_native_module()
    gate_native = gate_proj._get_native()
    up_native = up_proj._get_native()

    if out is not None:
        if out.shape != gate_proj.shape or out.dtype != gate_proj.dtype:
            raise ValueError("geglu: output shape/dtype mismatch")
        out_native = out._get_native()
        native.geglu_(gate_native, up_native, out_native)
        return out
    else:
        result_native = native.geglu(gate_native, up_native)
        return GPUArray._wrap_native(result_native)


__all__ = [
    "rmsnorm_residual",
    "swiglu",
    "geglu",
]
