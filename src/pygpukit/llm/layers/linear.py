"""Linear layer implementations for PyGPUkit LLM.

Provides:
- LinearBF16: Dense layer with BF16 weights
- LinearFP8: Dense layer with FP8 weights (online dequantization)
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.dtypes import bfloat16 as dt_bfloat16
from pygpukit.core.factory import from_numpy, zeros
from pygpukit.ops.basic import (
    bias_add_inplace,
    gemv_bf16,
    gemv_fp8_bf16,
    matmul,
    transpose,
    w8a16_gemm_sm120,
)


class LinearBF16:
    """BF16 Linear layer: y = xW^T + b

    Weights are stored as [out_features, in_features] (PyTorch convention).

    For M=1 (single token decode), uses custom GEMV kernel which is 4-6x faster
    than cuBLASLt matmul. Automatically falls back to matmul for batch > 1.
    """

    # Class-level flag to enable/disable GEMV optimization
    _use_gemv: bool = True

    def __init__(self, weight: GPUArray, bias: GPUArray | None = None):
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {weight.ndim}D")
        self.weight = weight
        self.bias = bias
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]
        self._weight_t: GPUArray | None = None

    def __call__(self, x: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
        """Forward pass: y = xW^T + b

        Args:
            x: Input tensor [batch, in_features]
            out: Optional output buffer [batch, out_features]. If provided,
                 result is written in-place (for CUDA Graph capture).
        """
        if x.ndim != 2:
            raise ValueError(f"input must be 2D [batch, in_features], got {x.ndim}D")
        if x.shape[1] != self.in_features:
            raise ValueError(f"input features {x.shape[1]} != weight {self.in_features}")

        if self._weight_t is None:
            self._weight_t = transpose(self.weight)

        # Use GEMV for M=1 with BF16 (1.3-2.4x faster than matmul)
        # Skip GEMV when out is provided (CUDA Graph mode) - GEMV allocates internally
        use_gemv = (
            LinearBF16._use_gemv
            and x.shape[0] == 1
            and x.dtype == dt_bfloat16
            and out is None  # GEMV allocates, not compatible with CUDA Graph
        )

        if use_gemv:
            # GEMV path for M=1 decode
            from pygpukit.core.backend import get_native_module

            native = get_native_module()
            x_1d = x.view((self.in_features,))

            # Use optimized kernel (SM80+) with B[N,K] layout
            if native.gemv_bf16_opt_available():
                y_1d = zeros((self.out_features,), dtype="bfloat16")
                # gemv_bf16_opt: A[K] @ B[N,K]^T -> C[N]
                native.gemv_bf16_opt_sm120(
                    x_1d._get_native(),
                    self.weight._get_native(),  # [N, K] - no transpose
                    y_1d._get_native(),
                )
            else:
                # Fallback: old kernel with B[K,N] layout
                y_1d = gemv_bf16(x_1d, self._weight_t)

            y = y_1d.view((1, self.out_features))
        else:
            # Standard matmul path
            y = matmul(x, self._weight_t, out=out)

        if self.bias is not None:
            bias_add_inplace(y, self.bias)

        return y


# Backward compatibility alias
Linear = LinearBF16


class LinearFP8:
    """FP8 Linear layer with online dequantization: y = x @ dequant(W)^T + b

    Stores weights in FP8 E4M3 format with block-wise scaling factors.
    Dequantizes on-the-fly during forward pass using CUDA kernel.

    Memory savings: 50% vs BF16 (1 byte vs 2 bytes per weight + small scale overhead)

    For M=1 (single token decode), uses FP8 GEMV kernel with online dequantization.
    For larger batches, falls back to CPU dequantization + GPU matmul.
    """

    # Class-level flag to enable/disable GEMV optimization
    _use_gemv: bool = True

    # FP8 E4M3 to float32 lookup table (for CPU fallback)
    _FP8_TABLE: np.ndarray | None = None

    @classmethod
    def _get_fp8_table(cls) -> np.ndarray:
        """Build FP8 E4M3 to float32 conversion lookup table."""
        if cls._FP8_TABLE is not None:
            return cls._FP8_TABLE

        table = np.zeros(256, dtype=np.float32)
        for i in range(256):
            sign = (i >> 7) & 1
            exp = (i >> 3) & 0xF
            mant = i & 0x7

            if exp == 0xF and mant == 0x7:
                table[i] = np.nan
            elif exp == 0:
                value = (mant / 8.0) * (2.0**-6)
                table[i] = -value if sign else value
            else:
                value = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
                table[i] = -value if sign else value

        cls._FP8_TABLE = table
        return table

    def __init__(
        self,
        weight_fp8: GPUArray,  # [out_features, in_features] as uint8
        scale_inv: GPUArray,  # [out_features // block_h, in_features // block_w] as bf16
        bias: GPUArray | None = None,
        block_size: tuple[int, int] = (128, 128),
    ):
        if weight_fp8.ndim != 2:
            raise ValueError(f"weight must be 2D, got {weight_fp8.ndim}D")
        self.weight_fp8 = weight_fp8
        self.scale_inv = scale_inv
        self.bias = bias
        self.block_size = block_size
        self.out_features = weight_fp8.shape[0]
        self.in_features = weight_fp8.shape[1]

        # Transposed weight for GEMV: [in_features, out_features]
        # FP8 GEMV expects B[K,N] where K=in_features, N=out_features
        self._weight_fp8_t: GPUArray | None = None
        self._scale_inv_t: GPUArray | None = None

        # Cached dequantized weight for fallback (lazy initialization)
        self._weight_dequant: GPUArray | None = None
        self._weight_dequant_t: GPUArray | None = None

    def _ensure_transposed_fp8(self) -> None:
        """Ensure transposed FP8 weight is available for GEMV."""
        if self._weight_fp8_t is None:
            # Transpose weight: [out, in] -> [in, out]
            self._weight_fp8_t = transpose(self.weight_fp8)
            # Transpose scale: [out/128, in/128] -> [in/128, out/128]
            self._scale_inv_t = transpose(self.scale_inv)

    def _dequantize_cpu(self) -> np.ndarray:
        """Dequantize FP8 weight to float32 on CPU."""
        table = self._get_fp8_table()

        # Get FP8 bytes
        fp8_np = self.weight_fp8.to_numpy()
        if fp8_np.dtype != np.uint8:
            fp8_np = fp8_np.view(np.uint8)

        # Convert to float32
        f32 = table[fp8_np.ravel()].reshape(fp8_np.shape)

        # Get scale_inv (bf16 as uint16)
        scale_np = self.scale_inv.to_numpy()
        if scale_np.dtype == np.uint16:
            scale_f32 = np.empty(scale_np.shape, dtype=np.float32)
            scale_f32.view(np.uint32)[:] = scale_np.astype(np.uint32) << 16
        else:
            scale_f32 = scale_np.astype(np.float32)

        # Apply block-wise scaling
        H, W = f32.shape
        block_h, block_w = self.block_size
        num_blocks_h = H // block_h
        num_blocks_w = W // block_w

        f32_reshaped = f32.reshape(num_blocks_h, block_h, num_blocks_w, block_w)
        scale_expanded = scale_f32[:, np.newaxis, :, np.newaxis]
        f32_scaled = f32_reshaped * scale_expanded

        return f32_scaled.reshape(H, W)

    def _ensure_dequantized(self) -> None:
        """Ensure dequantized weight is available (lazy init, for fallback)."""
        if self._weight_dequant is None:
            # Dequantize on CPU and upload to GPU
            weight_f32 = self._dequantize_cpu()

            # Convert to BF16
            uint32_view = weight_f32.view(np.uint32)
            weight_bf16 = ((uint32_view + 0x7FFF + ((uint32_view >> 16) & 1)) >> 16).astype(
                np.uint16
            )

            self._weight_dequant = from_numpy(weight_bf16)
            self._weight_dequant_t = transpose(self._weight_dequant)

    def __call__(self, x: GPUArray, *, out: GPUArray | None = None) -> GPUArray:
        """Forward pass with online dequantization.

        For M=1 (single token), uses FP8 GEMV kernel with online dequantization.
        For M>1, uses batched FP8 GEMV kernel.
        """
        if x.ndim != 2:
            raise ValueError(f"input must be 2D [batch, in_features], got {x.ndim}D")
        if x.shape[1] != self.in_features:
            raise ValueError(f"input features {x.shape[1]} != weight {self.in_features}")

        M = x.shape[0]

        if M == 1 and self._use_gemv:
            # M=1 path: Use FP8 GEMV kernel with B[N,K] layout (no transpose needed)
            x_1d = x.view((self.in_features,))

            if out is not None:
                out_1d = out.view((self.out_features,))
                gemv_fp8_bf16(x_1d, self.weight_fp8, self.scale_inv, out=out_1d)
                y = out
            else:
                y_1d = gemv_fp8_bf16(x_1d, self.weight_fp8, self.scale_inv)
                y = y_1d.view((1, self.out_features))
        else:
            # M>1 path: Use W8A16 GEMM with FP8 TensorCore (requires transposed weights)
            self._ensure_transposed_fp8()
            y = w8a16_gemm_sm120(x, self._weight_fp8_t, self._scale_inv_t, out=out)

        if self.bias is not None:
            bias_add_inplace(y, self.bias)

        return y


__all__ = [
    "LinearBF16",
    "LinearFP8",
    "Linear",
]
