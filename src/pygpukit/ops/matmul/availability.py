"""Availability check functions for GEMM/GEMV operations.

All *_available() functions to check GPU capability.
"""

from __future__ import annotations

from pygpukit.core.backend import NativeBackend, get_backend


def fp8_available() -> bool:
    """Check if FP8 GEMM is available (any backend)."""
    backend = get_backend()
    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return (
            native.gemm_fp8_f32_sm90_available()
            or native.gemm_fp8_f32_sm100_available()
            or native.gemm_fp8_f32_sm120_available()
        )
    return False


gemm_fp8_available = fp8_available


def fp8_sm90_available() -> bool:
    """Check if FP8 GEMM is available on SM90 (Hopper)."""
    backend = get_backend()
    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.gemm_fp8_f32_sm90_available()
    return False


gemm_fp8_f32_sm90_available = fp8_sm90_available


def fp8_sm100_available() -> bool:
    """Check if FP8 GEMM is available on SM100 (Blackwell datacenter)."""
    backend = get_backend()
    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.gemm_fp8_f32_sm100_available()
    return False


gemm_fp8_f32_sm100_available = fp8_sm100_available


def fp8_sm120_available() -> bool:
    """Check if FP8 GEMM is available on SM120 (Blackwell GeForce)."""
    backend = get_backend()
    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.gemm_fp8_f32_sm120_available()
    return False


gemm_fp8_f32_sm120_available = fp8_sm120_available


def fp8_fp8_sm120_available() -> bool:
    """Check if Pure FP8 I/O GEMM is available on SM120."""
    backend = get_backend()
    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.gemm_fp8_fp8_sm120_available()
    return False


gemm_fp8_fp8_sm120_available = fp8_fp8_sm120_available


def nvf4_bf16_sm120_available() -> bool:
    """Check if NVF4 (4-bit) BF16 GEMM is available on SM120."""
    backend = get_backend()
    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.gemm_nvf4_bf16_sm120_available()
    return False


gemm_nvf4_bf16_sm120_available = nvf4_bf16_sm120_available


def gemv_nvf4_available() -> bool:
    """Check if NVF4 GEMV is available (SM120+)."""
    backend = get_backend()
    if isinstance(backend, NativeBackend) and backend.is_available():
        from pygpukit.core.backend import get_native_module

        native = get_native_module()
        return native.gemv_nvf4_bf16_sm120_available()
    return False


gemv_nvf4_bf16_sm120_available = gemv_nvf4_available


__all__ = [
    "fp8_available",
    "gemm_fp8_available",
    "fp8_sm90_available",
    "gemm_fp8_f32_sm90_available",
    "fp8_sm100_available",
    "gemm_fp8_f32_sm100_available",
    "fp8_sm120_available",
    "gemm_fp8_f32_sm120_available",
    "fp8_fp8_sm120_available",
    "gemm_fp8_fp8_sm120_available",
    "nvf4_bf16_sm120_available",
    "gemm_nvf4_bf16_sm120_available",
    "gemv_nvf4_available",
    "gemv_nvf4_bf16_sm120_available",
]
