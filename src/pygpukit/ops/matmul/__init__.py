"""Matrix multiplication operations for GPUArrays.

This module provides various GEMM (General Matrix Multiply) and GEMV
(General Matrix-Vector) operations optimized for different GPU architectures
and data types.

Corresponds to native/ops/matmul/.
"""

from __future__ import annotations

# Availability checks
from .availability import (
    fp8_available,
    fp8_fp8_sm120_available,
    fp8_sm90_available,
    fp8_sm100_available,
    fp8_sm120_available,
    gemm_fp8_available,
    gemm_fp8_f32_sm90_available,
    gemm_fp8_f32_sm100_available,
    gemm_fp8_f32_sm120_available,
    gemm_fp8_fp8_sm120_available,
    gemm_nvf4_bf16_sm120_available,
    gemv_nvf4_available,
    gemv_nvf4_bf16_sm120_available,
    nvf4_bf16_sm120_available,
)

# FP8 GEMM operations
from .fp8 import (
    fp8_fp8_get_scale_sizes,
    fp8_get_sizes,
    fp8_init_lut,
    gemm_fp8_f32_sm90,
    gemm_fp8_f32_sm100,
    gemm_fp8_f32_sm120,
    gemm_fp8_fp8_blockwise_sm120,
    gemm_fp8_fp8_get_scale_sizes,
    gemm_fp8_fp8_sm120,
    matmul_fp8,
    matmul_fp8_fp8_blockwise_sm120,
    matmul_fp8_fp8_sm120,
    matmul_fp8_sm90,
    matmul_fp8_sm100,
    matmul_fp8_sm120,
)

# GEMV operations
from .gemv import (
    gemv_bf16,
    gemv_bf16_bf16_sm120,
    gemv_fp8_bf16,
    gemv_fp8_bf16_batched,
    gemv_fp8_bf16_batched_sm120,
    gemv_fp8_bf16_sm120,
)

# Generic matmul operations
from .generic import (
    batched_matmul,
    linear_bias_gelu,
    matmul,
    transpose,
)

# Grouped GEMM for MoE
from .grouped import (
    grouped_gemm_fp8_bf16,
    grouped_gemm_fp8_bf16_sm120,
    grouped_gemm_init_lut,
)

# NVF4 (4-bit) operations
from .nvf4 import (
    gemm_nvf4_bf16_sm120,
    gemv_nvf4_bf16,
    gemv_nvf4_bf16_sm120,
    gemv_nvf4_get_sizes,
    matmul_nvf4_bf16_sm120,
    nvf4_get_sizes,
    quantize_bf16_to_nvf4,
)

# W8A16 GEMM operations
from .w8a16 import (
    gemm_w8a16_bf16_sm120,
    gemm_w8a16_init_lut,
    w8a16_gemm_init_lut,
    w8a16_gemm_sm120,
)

__all__ = [
    # Generic operations
    "matmul",
    "batched_matmul",
    "transpose",
    "linear_bias_gelu",
    # Availability checks
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
    # FP8 GEMM operations
    "matmul_fp8",
    "matmul_fp8_sm90",
    "matmul_fp8_sm100",
    "matmul_fp8_sm120",
    "matmul_fp8_fp8_sm120",
    "matmul_fp8_fp8_blockwise_sm120",
    "fp8_fp8_get_scale_sizes",
    "fp8_get_sizes",
    "fp8_init_lut",
    # FP8 aliases
    "gemm_fp8_f32_sm90",
    "gemm_fp8_f32_sm100",
    "gemm_fp8_f32_sm120",
    "gemm_fp8_fp8_sm120",
    "gemm_fp8_fp8_blockwise_sm120",
    "gemm_fp8_fp8_get_scale_sizes",
    # NVF4 (4-bit) operations
    "nvf4_get_sizes",
    "gemv_nvf4_get_sizes",
    "quantize_bf16_to_nvf4",
    "matmul_nvf4_bf16_sm120",
    "gemm_nvf4_bf16_sm120",
    "gemv_nvf4_bf16",
    "gemv_nvf4_bf16_sm120",
    # GEMV operations
    "gemv_bf16",
    "gemv_bf16_bf16_sm120",
    "gemv_fp8_bf16",
    "gemv_fp8_bf16_sm120",
    "gemv_fp8_bf16_batched",
    "gemv_fp8_bf16_batched_sm120",
    # W8A16 GEMM operations
    "w8a16_gemm_init_lut",
    "gemm_w8a16_init_lut",
    "w8a16_gemm_sm120",
    "gemm_w8a16_bf16_sm120",
    # Grouped GEMM (MoE)
    "grouped_gemm_init_lut",
    "grouped_gemm_fp8_bf16",
    "grouped_gemm_fp8_bf16_sm120",
]
