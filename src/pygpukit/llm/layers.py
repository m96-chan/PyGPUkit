"""Neural network layer implementations for PyGPUkit LLM.

Provides:
- LinearBF16: Dense layer with BF16 weights
- LinearFP8: Dense layer with FP8 weights (online dequantization)
- Norm: RMSNorm and LayerNorm
- Attention: Multi-head attention with RoPE, GQA, QK-Norm, KV cache
- MLP: Feed-forward network (GELU/SwiGLU)
- TransformerBlock: Attention + MLP with residual connections
- RoPE utilities: precompute_freqs_cis, apply_rotary_pos_emb_numpy
- Repack utilities: repack_weight, repack_linear, repack_norm
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.dtypes import bfloat16 as dt_bfloat16
from pygpukit.core.dtypes import float16 as dt_float16
from pygpukit.core.factory import from_numpy, zeros
from pygpukit.ops.basic import (
    add,
    bias_add_inplace,
    concat_axis0,
    copy_to,
    gelu,
    gemv_bf16,
    gemv_fp8_bf16,
    kv_cache_prefill_gqa,
    kv_cache_update_gqa,
    layernorm,
    matmul,
    mul,
    repeat_interleave_axis1,
    reshape_copy,
    rmsnorm,
    rope_inplace,
    sdpa_causal,
    sdpa_causal_fixed_cache,
    silu,
    slice_rows_range_ptr,
    split_qkv_batch,
    transpose,
    transpose_3d_021,
    w8a16_gemm_sm120,
)

if TYPE_CHECKING:
    from pygpukit.llm.buffers import DecodeBuffers
    from pygpukit.llm.config import TransformerConfig


# =============================================================================
# Common Building Blocks
# =============================================================================


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
            # GEMV path: zero-copy view to 1D, call gemv_bf16, view back to 2D
            x_1d = x.view((self.in_features,))
            y_1d = gemv_bf16(x_1d, self._weight_t)

            if out is not None:
                # Copy to output buffer
                copy_to(y_1d.view((1, self.out_features)), out)
                y = out
            else:
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


class Norm:
    """Unified normalization layer supporting RMSNorm and LayerNorm."""

    def __init__(
        self,
        weight: GPUArray,
        bias: GPUArray | None = None,
        norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
        eps: float = 1e-5,
    ):
        self.weight = weight
        self.bias = bias
        self.norm_type = norm_type
        self.eps = eps

    def __call__(self, x: GPUArray) -> GPUArray:
        if self.norm_type == "rmsnorm":
            return rmsnorm(x, self.weight, self.eps)
        else:
            if self.bias is None:
                raise ValueError("LayerNorm requires bias")
            return layernorm(x, self.weight, self.bias, self.eps)


# =============================================================================
# Weight Repacking - Fix GPU memory placement for optimal performance
# =============================================================================


def repack_weight(weight: GPUArray) -> GPUArray:
    """Repack a weight tensor into a new contiguous GPU buffer.

    This fixes performance issues caused by fragmented GPU memory allocation.
    Weights allocated later during model loading may end up in suboptimal
    memory regions, causing 7x slower matmul performance.

    Args:
        weight: Original weight tensor on GPU

    Returns:
        New GPUArray with same data in freshly allocated contiguous memory
    """
    # Copy to CPU, then back to GPU to get fresh allocation
    # This ensures the new buffer is allocated contiguously
    weight_np = weight.to_numpy()
    return from_numpy(weight_np)


def repack_linear(linear: LinearBF16) -> None:
    """Repack a LinearBF16 layer's weight in-place.

    Args:
        linear: LinearBF16 layer to repack
    """
    linear.weight = repack_weight(linear.weight)
    # Clear transpose cache - will be regenerated on first use
    linear._weight_t = None
    if linear.bias is not None:
        linear.bias = repack_weight(linear.bias)


def repack_norm(norm: Norm) -> None:
    """Repack a Norm layer's weight in-place.

    Args:
        norm: Norm layer to repack
    """
    norm.weight = repack_weight(norm.weight)
    if norm.bias is not None:
        norm.bias = repack_weight(norm.bias)


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================


def precompute_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute rotary embedding cos/sin tables."""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    cos = np.concatenate([cos, cos], axis=-1)
    sin = np.concatenate([sin, sin], axis=-1)
    return cos, sin


def apply_rotary_pos_emb_numpy(
    q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply rotary position embeddings to Q and K (numpy version)."""

    def rotate_half(x: np.ndarray) -> np.ndarray:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return np.concatenate([-x2, x1], axis=-1)

    cos = cos[:, np.newaxis, :]
    sin = sin[:, np.newaxis, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Unified Attention
# =============================================================================


class Attention:
    """Unified attention with Hybrid CPU/GPU execution.

    Supports:
    - Multi-Head Attention (MHA): num_kv_heads == num_heads
    - Grouped Query Attention (GQA): num_kv_heads < num_heads
    - RoPE: enabled via config.use_rope
    - QK Norm: optional normalization of Q and K (Qwen3 style)
    - Hybrid execution: CPU for seq_len=1, GPU for longer sequences
    - FP8 quantized weights via LinearFP8
    """

    def __init__(
        self,
        q_proj: GPUArray | LinearBF16 | LinearFP8,
        k_proj: GPUArray | LinearBF16 | LinearFP8,
        v_proj: GPUArray | LinearBF16 | LinearFP8,
        o_proj: GPUArray | LinearBF16 | LinearFP8,
        config: TransformerConfig,
        q_bias: GPUArray | None = None,
        k_bias: GPUArray | None = None,
        v_bias: GPUArray | None = None,
        o_bias: GPUArray | None = None,
        q_norm: Norm | None = None,
        k_norm: Norm | None = None,
    ):
        # Accept either GPUArray (wrapped in LinearBF16) or pre-built LinearBF16/LinearFP8
        def wrap_linear(
            proj: GPUArray | LinearBF16 | LinearFP8, bias: GPUArray | None
        ) -> LinearBF16 | LinearFP8:
            if isinstance(proj, (LinearBF16, LinearFP8)):
                return proj
            return LinearBF16(proj, bias)

        self.q_proj = wrap_linear(q_proj, q_bias)
        self.k_proj = wrap_linear(k_proj, k_bias)
        self.v_proj = wrap_linear(v_proj, v_bias)
        self.o_proj = wrap_linear(o_proj, o_bias)

        # QK Norm (Qwen3 style)
        self.q_norm = q_norm
        self.k_norm = k_norm

        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        assert config.num_kv_heads is not None  # Set in __post_init__
        self.num_kv_heads: int = config.num_kv_heads
        self.num_kv_groups = config.num_kv_groups

        # Store dimensions for QKV split
        self.q_dim = self.num_heads * self.head_dim
        self.k_dim = self.num_kv_heads * self.head_dim
        self.v_dim = self.num_kv_heads * self.head_dim

        # Create fused QKV projection (reduces 3 matmuls to 1)
        # Skip fusion for FP8 (LinearFP8 can't be concatenated)
        self.qkv_proj: LinearBF16 | None = None
        if not isinstance(self.q_proj, LinearFP8):
            # Extract weights from LinearBF16 for concatenation
            q_weight = self.q_proj.weight if isinstance(self.q_proj, LinearBF16) else q_proj
            k_weight = self.k_proj.weight if isinstance(self.k_proj, LinearBF16) else k_proj
            v_weight = self.v_proj.weight if isinstance(self.v_proj, LinearBF16) else v_proj
            qkv_weight = concat_axis0(concat_axis0(q_weight, k_weight), v_weight)
            self.qkv_proj = LinearBF16(qkv_weight, None)

        # Precompute RoPE if enabled
        self._cos: np.ndarray | None
        self._sin: np.ndarray | None
        if config.use_rope:
            self._cos, self._sin = precompute_freqs_cis(
                self.head_dim, config.max_position_embeddings, config.rope_theta
            )
        else:
            self._cos, self._sin = None, None

        # Fixed-length KV cache for CUDA Graph (initialized on first use)
        self._k_cache: GPUArray | None = None
        self._v_cache: GPUArray | None = None
        self._max_cache_len: int = 0

        # Lookahead KV tracking for Jacobi decoding
        self._confirmed_pos: int = 0
        self._logical_pos: int = 0

    def init_fixed_cache(self, max_seq_len: int, dtype: str = "float16") -> None:
        """Initialize fixed-length KV cache for CUDA Graph capture.

        Args:
            max_seq_len: Maximum sequence length to support.
            dtype: Data type for cache (float16/bfloat16/float32).
        """
        cache_shape = (self.num_heads, max_seq_len, self.head_dim)
        if dtype == "float16":
            np_dtype = np.float16
        elif dtype == "bfloat16":
            np_dtype = np.uint16  # bf16 stored as uint16
        else:
            np_dtype = np.float32
        self._k_cache = from_numpy(np.zeros(cache_shape, dtype=np_dtype))
        self._v_cache = from_numpy(np.zeros(cache_shape, dtype=np_dtype))
        self._max_cache_len = max_seq_len
        self._confirmed_pos = 0
        self._logical_pos = 0

    # =========================================================================
    # Lookahead KV Cache Management (for Jacobi Decoding)
    # =========================================================================

    def set_confirmed_pos(self, pos: int) -> None:
        """Set the confirmed position (e.g., after prefill)."""
        assert 0 <= pos <= self._max_cache_len, f"Invalid pos {pos}"
        self._confirmed_pos = pos
        self._logical_pos = pos

    def reset_lookahead(self) -> None:
        """Reset lookahead pointer to confirmed position."""
        self._logical_pos = self._confirmed_pos

    def commit_lookahead(self, n_accepted: int) -> None:
        """Commit accepted tokens by advancing confirmed_pos."""
        new_pos = self._confirmed_pos + n_accepted
        assert new_pos <= self._max_cache_len, f"Commit exceeds cache: {new_pos}"
        self._confirmed_pos = new_pos
        self._logical_pos = new_pos

    def get_confirmed_pos(self) -> int:
        """Get current confirmed position."""
        return self._confirmed_pos

    def __call__(
        self,
        x: GPUArray,
        position_ids: list[int] | None = None,
        past_kv: tuple | None = None,
        use_cache: bool = False,
    ) -> tuple[GPUArray, tuple | None]:
        """Forward pass with hybrid CPU/GPU attention.

        Args:
            x: Input tensor [seq_len, hidden_size]
            position_ids: Position IDs for RoPE (auto-generated if None)
            past_kv: Tuple of (past_k, past_v) numpy arrays
            use_cache: Whether to return KV cache

        Returns:
            Tuple of (output, present_kv)
        """
        seq_len = x.shape[0]

        if position_ids is None:
            position_ids = list(range(seq_len))

        return self._forward_gpu(x, position_ids, past_kv, use_cache)

    def _forward_gpu(
        self,
        x: GPUArray,
        position_ids: list[int],
        past_kv: tuple | None,
        use_cache: bool,
    ) -> tuple[GPUArray, tuple | None]:
        """GPU path for long sequences (prefill)."""
        seq_len = x.shape[0]

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = reshape_copy(q, (seq_len, self.num_heads, self.head_dim))
        k = reshape_copy(k, (seq_len, self.num_kv_heads, self.head_dim))
        v = reshape_copy(v, (seq_len, self.num_kv_heads, self.head_dim))

        # QK Norm (Qwen3 style)
        if self.q_norm is not None:
            q_shape = (seq_len, self.num_heads, self.head_dim)
            q_2d = reshape_copy(q, (seq_len * self.num_heads, self.head_dim))
            q_2d = self.q_norm(q_2d)
            q = reshape_copy(q_2d, q_shape)
        if self.k_norm is not None:
            k_shape = (seq_len, self.num_kv_heads, self.head_dim)
            k_2d = reshape_copy(k, (seq_len * self.num_kv_heads, self.head_dim))
            k_2d = self.k_norm(k_2d)
            k = reshape_copy(k_2d, k_shape)

        # Apply RoPE on GPU
        if self.config.use_rope:
            assert self._cos is not None and self._sin is not None
            from pygpukit.ops.basic import rope_inplace_f32table

            q_dtype = q.dtype
            cos_f32 = from_numpy(self._cos[position_ids].astype(np.float32))
            sin_f32 = from_numpy(self._sin[position_ids].astype(np.float32))
            if q_dtype in (dt_float16, dt_bfloat16):
                # Use f32 tables directly for higher precision (no intermediate alloc)
                rope_inplace_f32table(q, k, cos_f32, sin_f32)
            else:
                rope_inplace(q, k, cos_f32, sin_f32)

        # GPU KV Cache
        if past_kv is not None:
            past_k, past_v = past_kv
            if isinstance(past_k, GPUArray):
                k = concat_axis0(past_k, k)
                v = concat_axis0(past_v, v)
            else:
                k_np = k.to_numpy()
                v_np = v.to_numpy()
                k_np = np.concatenate([past_k, k_np], axis=0)
                v_np = np.concatenate([past_v, v_np], axis=0)
                k = from_numpy(k_np)
                v = from_numpy(v_np)

        present_kv = (k, v) if use_cache else None

        # Expand for GQA on GPU
        if self.num_kv_groups > 1:
            k_expanded = repeat_interleave_axis1(k, self.num_kv_groups)
            v_expanded = repeat_interleave_axis1(v, self.num_kv_groups)
        else:
            k_expanded = k
            v_expanded = v

        # GPU SDPA
        q_t = transpose_3d_021(q)
        k_t = transpose_3d_021(k_expanded)
        v_t = transpose_3d_021(v_expanded)

        attn_output = sdpa_causal(q_t, k_t, v_t)

        attn_output = transpose_3d_021(attn_output)
        attn_output = reshape_copy(attn_output, (seq_len, self.num_heads * self.head_dim))

        return self.o_proj(attn_output), present_kv

    def forward_fixed_cache(
        self,
        x: GPUArray,
        position: int,
        context_len: int,
        *,
        out: GPUArray | None = None,
    ) -> GPUArray:
        """Forward pass using fixed-length KV cache (for CUDA Graph decode).

        Args:
            x: Input tensor [1, hidden_size] - single token
            position: Current position in sequence (for RoPE and cache update)
            context_len: Total context length (prefill + decoded so far)
            out: Optional pre-allocated output buffer

        Returns:
            Output tensor [1, hidden_size]
        """
        assert self._k_cache is not None, "Call init_fixed_cache first"
        assert x.shape[0] == 1, "forward_fixed_cache expects single token"

        if self.qkv_proj is not None:
            # Fused QKV projection (faster for non-FP8)
            qkv = self.qkv_proj(x)
            q_2d = qkv.narrow(0, self.q_dim)
            k_2d = qkv.narrow(self.q_dim, self.k_dim)
            v_2d = qkv.narrow(self.q_dim + self.k_dim, self.v_dim)

            # Apply biases separately
            if self.q_proj.bias is not None:
                bias_add_inplace(q_2d, self.q_proj.bias)
            if self.k_proj.bias is not None:
                bias_add_inplace(k_2d, self.k_proj.bias)
            if self.v_proj.bias is not None:
                bias_add_inplace(v_2d, self.v_proj.bias)
        else:
            # Separate projections (for FP8)
            q_2d = self.q_proj(x)
            k_2d = self.k_proj(x)
            v_2d = self.v_proj(x)

        # Zero-copy reshape
        q = q_2d.view((1, self.num_heads, self.head_dim))
        k = k_2d.view((1, self.num_kv_heads, self.head_dim))
        v = v_2d.view((1, self.num_kv_heads, self.head_dim))

        # QK Norm
        if self.q_norm is not None:
            q_flat = q.view((self.num_heads, self.head_dim))
            q_normed = self.q_norm(q_flat)
            q = q_normed.view((1, self.num_heads, self.head_dim))
        if self.k_norm is not None:
            k_flat = k.view((self.num_kv_heads, self.head_dim))
            k_normed = self.k_norm(k_flat)
            k = k_normed.view((1, self.num_kv_heads, self.head_dim))

        q_dtype = q.dtype

        # Apply RoPE
        if self.config.use_rope and self._cos is not None and self._sin is not None:
            from pygpukit.ops.basic import rope_inplace_f32table

            cos_f32 = from_numpy(self._cos[position : position + 1].astype(np.float32))
            sin_f32 = from_numpy(self._sin[position : position + 1].astype(np.float32))
            if q_dtype in (dt_float16, dt_bfloat16):
                rope_inplace_f32table(q, k, cos_f32, sin_f32)
            else:
                rope_inplace(q, k, cos_f32, sin_f32)

        # Update KV cache
        kv_cache_update_gqa(k, self._k_cache, self.num_heads, position)
        kv_cache_update_gqa(v, self._v_cache, self.num_heads, position)

        q_t = q.view((self.num_heads, 1, self.head_dim))

        # Allocate output buffer if needed
        if out is None:
            if q_dtype == dt_float16:
                out_np_dtype = np.float16
            elif q_dtype == dt_bfloat16:
                out_np_dtype = np.uint16
            else:
                out_np_dtype = np.float32
            attn_out = from_numpy(np.zeros((self.num_heads, 1, self.head_dim), dtype=out_np_dtype))
        else:
            attn_out = out

        sdpa_causal_fixed_cache(q_t, self._k_cache, self._v_cache, attn_out, context_len)

        attn_output = attn_out.view((1, self.num_heads * self.head_dim))
        return self.o_proj(attn_output)

    def forward_fixed_cache_batch(
        self,
        x: GPUArray,
        start_position: int,
        context_len: int,
    ) -> GPUArray:
        """Forward pass for batch decode using fixed-length KV cache.

        Processes multiple tokens at once for speculative decoding verification.
        """
        assert self._k_cache is not None, "Call init_fixed_cache first"
        seq_len = x.shape[0]

        if seq_len == 1:
            return self.forward_fixed_cache(x, start_position, context_len)

        if self.qkv_proj is not None:
            # Fused QKV projection (faster for non-FP8)
            qkv = self.qkv_proj(x)
            qkv_np = qkv.to_numpy()
            q_np = qkv_np[:, : self.q_dim]
            k_np = qkv_np[:, self.q_dim : self.q_dim + self.k_dim]
            v_np = qkv_np[:, self.q_dim + self.k_dim :]

            # Apply biases
            if self.q_proj.bias is not None:
                q_np = q_np + self.q_proj.bias.to_numpy()
            if self.k_proj.bias is not None:
                k_np = k_np + self.k_proj.bias.to_numpy()
            if self.v_proj.bias is not None:
                v_np = v_np + self.v_proj.bias.to_numpy()

            q_2d = from_numpy(q_np.astype(qkv_np.dtype))
            k_2d = from_numpy(k_np.astype(qkv_np.dtype))
            v_2d = from_numpy(v_np.astype(qkv_np.dtype))
        else:
            # Separate projections (for FP8)
            q_2d = self.q_proj(x)
            k_2d = self.k_proj(x)
            v_2d = self.v_proj(x)

        q = reshape_copy(q_2d, (seq_len, self.num_heads, self.head_dim))
        k = reshape_copy(k_2d, (seq_len, self.num_kv_heads, self.head_dim))
        v = reshape_copy(v_2d, (seq_len, self.num_kv_heads, self.head_dim))

        # QK Norm
        if self.q_norm is not None:
            q_flat = reshape_copy(q, (seq_len * self.num_heads, self.head_dim))
            q_normed = self.q_norm(q_flat)
            q = reshape_copy(q_normed, (seq_len, self.num_heads, self.head_dim))
        if self.k_norm is not None:
            k_flat = reshape_copy(k, (seq_len * self.num_kv_heads, self.head_dim))
            k_normed = self.k_norm(k_flat)
            k = reshape_copy(k_normed, (seq_len, self.num_kv_heads, self.head_dim))

        q_dtype = q.dtype

        # RoPE
        if self.config.use_rope and self._cos is not None and self._sin is not None:
            from pygpukit.ops.basic import rope_inplace_f32table

            end_pos = start_position + seq_len
            cos_f32 = from_numpy(self._cos[start_position:end_pos].astype(np.float32))
            sin_f32 = from_numpy(self._sin[start_position:end_pos].astype(np.float32))
            if q_dtype in (dt_float16, dt_bfloat16):
                rope_inplace_f32table(q, k, cos_f32, sin_f32)
            else:
                rope_inplace(q, k, cos_f32, sin_f32)

        # Update KV cache
        kv_cache_prefill_gqa(k, self._k_cache, self.num_heads, start_position)
        kv_cache_prefill_gqa(v, self._v_cache, self.num_heads, start_position)

        q_t = transpose_3d_021(q)
        # Allocate attn_out with matching dtype
        if q_dtype == dt_float16:
            out_np_dtype = np.float16
        elif q_dtype == dt_bfloat16:
            out_np_dtype = np.uint16  # bfloat16 stored as uint16
        else:
            out_np_dtype = np.float32
        attn_out = from_numpy(
            np.zeros((self.num_heads, seq_len, self.head_dim), dtype=out_np_dtype)
        )

        sdpa_causal_fixed_cache(q_t, self._k_cache, self._v_cache, attn_out, context_len)

        attn_output = transpose_3d_021(attn_out)
        attn_output = reshape_copy(attn_output, (seq_len, self.num_heads * self.head_dim))
        return self.o_proj(attn_output)

    def forward_fixed_cache_batch_zero_alloc(
        self,
        x: GPUArray,
        start_position: int,
        context_len: int,
        buffers: DecodeBuffers,
        rope_cos_gpu: GPUArray | None,
        rope_sin_gpu: GPUArray | None,
        start_pos_buf: GPUArray,
    ) -> GPUArray:
        """Zero-allocation forward pass for batch decode using fixed-length KV cache.

        This version uses pre-allocated buffers for all operations, making it
        compatible with CUDA Graph capture. No memory allocations occur.
        """
        assert self._k_cache is not None, "Call init_fixed_cache first"
        seq_len = x.shape[0]

        q_out = buffers.q_batch.view((seq_len, self.num_heads, self.head_dim))
        k_out = buffers.k_batch.view((seq_len, self.num_kv_heads, self.head_dim))
        v_out = buffers.v_batch.view((seq_len, self.num_kv_heads, self.head_dim))

        if self.qkv_proj is not None:
            # Fused QKV projection into pre-allocated buffer
            qkv_out = buffers.qkv_proj_out_batch.slice_rows(seq_len)
            self.qkv_proj(x, out=qkv_out)

            # Split QKV
            split_qkv_batch(qkv_out, q_out, k_out, v_out, self.q_dim, self.k_dim, self.v_dim)

            # Apply biases
            if self.q_proj.bias is not None:
                q_out_2d = q_out.view((seq_len, self.q_dim))
                bias_add_inplace(q_out_2d, self.q_proj.bias)
            if self.k_proj.bias is not None:
                k_out_2d = k_out.view((seq_len, self.k_dim))
                bias_add_inplace(k_out_2d, self.k_proj.bias)
            if self.v_proj.bias is not None:
                v_out_2d = v_out.view((seq_len, self.v_dim))
                bias_add_inplace(v_out_2d, self.v_proj.bias)
        else:
            # Separate projections (for FP8 - allocates, not zero-alloc)
            q_2d = self.q_proj(x)
            k_2d = self.k_proj(x)
            v_2d = self.v_proj(x)
            copy_to(reshape_copy(q_2d, (seq_len, self.num_heads, self.head_dim)), q_out)
            copy_to(reshape_copy(k_2d, (seq_len, self.num_kv_heads, self.head_dim)), k_out)
            copy_to(reshape_copy(v_2d, (seq_len, self.num_kv_heads, self.head_dim)), v_out)

        # QK Norm
        if self.q_norm is not None and buffers.q_flat_batch is not None:
            q_flat = buffers.q_flat_batch.slice_rows(seq_len * self.num_heads)
            copy_to(q_out.view((seq_len * self.num_heads, self.head_dim)), q_flat)
            rmsnorm(q_flat, self.q_norm.weight, self.q_norm.eps, out=q_flat)
            copy_to(q_flat.view((seq_len, self.num_heads, self.head_dim)), q_out)

        if self.k_norm is not None and buffers.k_flat_batch is not None:
            k_flat = buffers.k_flat_batch.slice_rows(seq_len * self.num_kv_heads)
            copy_to(k_out.view((seq_len * self.num_kv_heads, self.head_dim)), k_flat)
            rmsnorm(k_flat, self.k_norm.weight, self.k_norm.eps, out=k_flat)
            copy_to(k_flat.view((seq_len, self.num_kv_heads, self.head_dim)), k_out)

        # RoPE
        if self.config.use_rope and rope_cos_gpu is not None and rope_sin_gpu is not None:
            cos_out = buffers.cos_batch.slice_rows(seq_len)
            sin_out = buffers.sin_batch.slice_rows(seq_len)
            slice_rows_range_ptr(rope_cos_gpu, cos_out, start_pos_buf, seq_len)
            slice_rows_range_ptr(rope_sin_gpu, sin_out, start_pos_buf, seq_len)
            rope_inplace(q_out, k_out, cos_out, sin_out)

        # Update KV cache
        kv_cache_prefill_gqa(k_out, self._k_cache, self.num_heads, start_position)
        kv_cache_prefill_gqa(v_out, self._v_cache, self.num_heads, start_position)

        # Transpose Q for SDPA
        q_t_out = buffers.q_t_batch.view((self.num_heads, seq_len, self.head_dim))
        transpose_3d_021(q_out, out=q_t_out)

        # SDPA
        attn_out = buffers.attn_out_batch.view((self.num_heads, seq_len, self.head_dim))
        sdpa_causal_fixed_cache(q_t_out, self._k_cache, self._v_cache, attn_out, context_len)

        # Transpose output
        attn_out_t = buffers.attn_out_t_batch.view((seq_len, self.num_heads, self.head_dim))
        transpose_3d_021(attn_out, out=attn_out_t)

        attn_out_2d = attn_out_t.view((seq_len, self.num_heads * self.head_dim))

        # O projection
        o_out = buffers.o_proj_out_batch.slice_rows(seq_len)
        self.o_proj(attn_out_2d, out=o_out)

        return o_out


# =============================================================================
# Unified MLP
# =============================================================================


class MLP:
    """Unified MLP supporting GELU and SwiGLU activations.

    GELU (GPT-2 style):
        fc1 -> GELU -> fc2

    SwiGLU (LLaMA style):
        gate_proj -> SiLU -> * up_proj -> down_proj

    Supports FP8 quantized weights via LinearFP8.
    """

    def __init__(
        self,
        config: TransformerConfig,
        # GELU path weights (GPUArray or LinearBF16/LinearFP8)
        fc1_weight: GPUArray | LinearBF16 | LinearFP8 | None = None,
        fc1_bias: GPUArray | None = None,
        fc2_weight: GPUArray | LinearBF16 | LinearFP8 | None = None,
        fc2_bias: GPUArray | None = None,
        # SwiGLU path weights (GPUArray or LinearBF16/LinearFP8)
        gate_proj: GPUArray | LinearBF16 | LinearFP8 | None = None,
        up_proj: GPUArray | LinearBF16 | LinearFP8 | None = None,
        down_proj: GPUArray | LinearBF16 | LinearFP8 | None = None,
    ):
        self.config = config
        self.activation = config.activation

        # Helper to wrap GPUArray in LinearBF16, or use pre-built LinearBF16/LinearFP8
        def wrap_linear(
            proj: GPUArray | LinearBF16 | LinearFP8 | None, bias: GPUArray | None = None
        ) -> LinearBF16 | LinearFP8 | None:
            if proj is None:
                return None
            if isinstance(proj, (LinearBF16, LinearFP8)):
                return proj
            return LinearBF16(proj, bias)

        if config.activation == "gelu":
            if fc1_weight is None or fc2_weight is None:
                raise ValueError("GELU MLP requires fc1_weight and fc2_weight")
            self.fc1 = wrap_linear(fc1_weight, fc1_bias)
            self.fc2 = wrap_linear(fc2_weight, fc2_bias)
        else:  # silu (SwiGLU)
            if gate_proj is None or up_proj is None or down_proj is None:
                raise ValueError("SwiGLU MLP requires gate_proj, up_proj, down_proj")

            self.gate_proj = wrap_linear(gate_proj)
            self.up_proj = wrap_linear(up_proj)
            self.down_proj = wrap_linear(down_proj)

            # Get intermediate size from the projection
            if isinstance(gate_proj, (LinearBF16, LinearFP8)):
                self.intermediate_size = gate_proj.out_features
            else:
                self.intermediate_size = gate_proj.shape[0]

            # Fused gate_up projection only for non-FP8 (GPUArray) weights
            # FP8 weights can't be concatenated trivially
            if isinstance(gate_proj, GPUArray) and isinstance(up_proj, GPUArray):
                gate_up_weight = concat_axis0(gate_proj, up_proj)
                self.gate_up_proj: LinearBF16 | None = LinearBF16(gate_up_weight, None)
            else:
                self.gate_up_proj = None

    def __call__(self, x: GPUArray) -> GPUArray:
        if self.activation == "gelu":
            h = self.fc1(x)
            h = gelu(h)
            return self.fc2(h)
        else:
            gate = silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(mul(gate, up))


# =============================================================================
# Mixture of Experts Layer
# =============================================================================


class MoELayer:
    """Mixture of Experts layer for Mixtral-style models.

    Architecture:
        1. Router: hidden -> [num_experts] logits
        2. Top-K selection with softmax
        3. Expert FFN (SwiGLU) for each selected expert
        4. Weighted combination of expert outputs

    Supports FP8 quantized expert weights via LinearFP8.
    """

    def __init__(
        self,
        config: TransformerConfig,
        gate_weight: GPUArray,  # [num_experts, hidden_size] - router
        expert_weights: list,  # [(gate, up, down), ...] - GPUArray or LinearBF16/LinearFP8
    ):
        self.config = config
        self.num_experts = config.num_experts or len(expert_weights)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size or config.intermediate_size

        # Router (gate) projection
        self.gate = LinearBF16(gate_weight)

        # Expert FFNs
        self.experts: list[MLP] = []
        for gate_proj, up_proj, down_proj in expert_weights:
            expert = MLP(
                config,
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
            )
            self.experts.append(expert)

        # Check if all experts use FP8 weights for grouped GEMM optimization
        self._use_grouped_gemm = False
        self._stacked_gate_weight: GPUArray | None = None
        self._stacked_gate_scale: GPUArray | None = None
        self._stacked_up_weight: GPUArray | None = None
        self._stacked_up_scale: GPUArray | None = None
        self._stacked_down_weight: GPUArray | None = None
        self._stacked_down_scale: GPUArray | None = None

        # Check if first expert uses FP8 - use grouped GEMM v2 for optimization
        if len(self.experts) > 0 and isinstance(self.experts[0].gate_proj, LinearFP8):
            self._stack_fp8_weights()

    # Profiling flag (set to True to enable timing)
    _profile: bool = True
    _profile_count: int = 0

    def _stack_fp8_weights(self) -> None:
        """Stack FP8 expert weights for grouped GEMM optimization."""
        # Collect weights from all experts
        gate_weights = []
        gate_scales = []
        up_weights = []
        up_scales = []
        down_weights = []
        down_scales = []

        for expert in self.experts:
            if not isinstance(expert.gate_proj, LinearFP8):
                return  # Not all experts are FP8, abort

            gate_weights.append(expert.gate_proj.weight_fp8)
            gate_scales.append(expert.gate_proj.scale_inv)
            up_weights.append(expert.up_proj.weight_fp8)
            up_scales.append(expert.up_proj.scale_inv)
            down_weights.append(expert.down_proj.weight_fp8)
            down_scales.append(expert.down_proj.scale_inv)

        # Stack weights: [num_experts, N, K]
        # gate_proj: [intermediate_size, hidden_size] -> stacked [num_experts, intermediate_size, hidden_size]
        # Each weight is [N, K], stack along new axis 0

        def stack_arrays_fast(arrays: list[GPUArray]) -> GPUArray:
            """Stack arrays along new axis 0 using single allocation + cudaMemcpy."""
            from pygpukit.core.backend import get_native_module

            native = get_native_module()

            # Get shape info from first array
            first = arrays[0]
            num_arrays = len(arrays)
            inner_shape = first.shape  # [N, K] or [N/128, K/128]

            # Calculate strides (nbytes is property, not method)
            bytes_per_array = first._get_native().nbytes

            # Allocate output: [num_arrays, *inner_shape]
            out_shape = [num_arrays] + list(inner_shape)
            out_native = native.empty(out_shape, first._get_native().dtype)
            out = GPUArray._wrap_native(out_native)

            # Copy each array to its slice using cuMemcpy
            for i, arr in enumerate(arrays):
                offset_bytes = i * bytes_per_array
                native.memcpy_device_to_device_offset(
                    arr._get_native(),
                    out._get_native(),
                    0,  # src offset
                    offset_bytes,  # dst offset
                    bytes_per_array,
                )

            return out

        self._stacked_gate_weight = stack_arrays_fast(gate_weights)
        self._stacked_gate_scale = stack_arrays_fast(gate_scales)
        self._stacked_up_weight = stack_arrays_fast(up_weights)
        self._stacked_up_scale = stack_arrays_fast(up_scales)
        self._stacked_down_weight = stack_arrays_fast(down_weights)
        self._stacked_down_scale = stack_arrays_fast(down_scales)

        self._use_grouped_gemm = True
        print(f"[MoE] Stacked {self.num_experts} expert weights for grouped GEMM")

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass through MoE layer.

        Args:
            x: Input tensor [batch, seq, hidden_size] or [seq, hidden_size]

        Returns:
            Output tensor with same shape as input
        """
        import time

        from pygpukit.core.backend import get_native_module

        native = get_native_module()

        profile = self._profile and MoELayer._profile_count < 3
        if profile:
            native.device_synchronize()
            t0 = time.perf_counter()

        original_shape = x.shape
        # Flatten to [num_tokens, hidden_size]
        if len(original_shape) == 3:
            batch, seq, hidden = original_shape
            num_tokens = batch * seq
            x = x.reshape(num_tokens, hidden)
        else:
            num_tokens, hidden = original_shape

        k = self.num_experts_per_tok

        # Step 1: Compute router logits
        router_logits = self.gate(x)  # [num_tokens, num_experts]
        if profile:
            native.device_synchronize()
            t1 = time.perf_counter()

        # Step 2: Top-K selection
        router_weights = zeros((num_tokens, k), dtype=x.dtype)
        expert_indices = zeros((num_tokens, k), dtype="int32")
        native.moe_topk_with_indices(
            router_logits._get_native(),
            router_weights._get_native(),
            expert_indices._get_native(),
            k,
        )

        # Step 3: Softmax over selected experts
        native.moe_softmax_topk(router_weights._get_native(), k)

        # Step 4: Compute permutation for efficient expert dispatch
        expert_counts = zeros((self.num_experts,), dtype="int32")
        expert_offsets = zeros((self.num_experts + 1,), dtype="int32")
        permute_indices = zeros((num_tokens * k,), dtype="int32")
        reverse_perm = zeros((num_tokens * k,), dtype="int32")
        native.moe_compute_permutation(
            expert_indices._get_native(),
            expert_counts._get_native(),
            expert_offsets._get_native(),
            permute_indices._get_native(),
            reverse_perm._get_native(),
            self.num_experts,
            k,
        )

        # Step 5: Gather hidden states for experts
        gathered = zeros((num_tokens * k, hidden), dtype=x.dtype)
        native.moe_gather(
            x._get_native(),
            permute_indices._get_native(),
            gathered._get_native(),
            k,
        )
        if profile:
            native.device_synchronize()
            t2 = time.perf_counter()

        # Step 6: Run experts
        if self._use_grouped_gemm:
            # Use grouped GEMM for all experts in single kernel launches
            from pygpukit.ops.matmul import grouped_gemm_fp8_bf16

            # Create row_expert_ids from expert_offsets
            M_total = num_tokens * k
            row_expert_ids = zeros((M_total,), dtype="int32")
            native.moe_expand_expert_offsets(
                expert_offsets._get_native(),
                row_expert_ids._get_native(),
                self.num_experts,
            )

            # gate_proj: gathered[M_total, hidden] @ gate_weight[experts, inter, hidden]^T
            gate_out = grouped_gemm_fp8_bf16(
                gathered,
                self._stacked_gate_weight,
                self._stacked_gate_scale,
                row_expert_ids,
            )

            # up_proj: gathered[M_total, hidden] @ up_weight[experts, inter, hidden]^T
            up_out = grouped_gemm_fp8_bf16(
                gathered,
                self._stacked_up_weight,
                self._stacked_up_scale,
                row_expert_ids,
            )

            # SiLU(gate) * up
            intermediate = mul(silu(gate_out), up_out)

            # down_proj: intermediate[M_total, inter] @ down_weight[experts, hidden, inter]^T
            expert_outputs = grouped_gemm_fp8_bf16(
                intermediate,
                self._stacked_down_weight,
                self._stacked_down_scale,
                row_expert_ids,
            )
        else:
            # Fallback: Run experts sequentially
            # Get expert counts on CPU for loop
            expert_counts_cpu = expert_counts.to_numpy()
            expert_offsets_cpu = expert_offsets.to_numpy()

            # Build list of (expert_id, start, count) for non-empty experts
            expert_tasks = []
            for e in range(self.num_experts):
                start = int(expert_offsets_cpu[e])
                count = int(expert_counts_cpu[e])
                if count > 0:
                    expert_tasks.append((e, start, count))

            def run_expert(task: tuple) -> GPUArray:
                e, start, count = task
                expert_input = gathered[start : start + count]
                return self.experts[e](expert_input)

            # Run experts sequentially
            expert_output_list = [run_expert(task) for task in expert_tasks]

            # Concatenate all expert outputs on GPU
            from functools import reduce

            expert_outputs = reduce(concat_axis0, expert_output_list)

        if profile:
            native.device_synchronize()
            t3 = time.perf_counter()

        # Step 7: Scatter and combine outputs
        output = zeros((num_tokens, hidden), dtype=x.dtype)
        native.moe_scatter(
            expert_outputs._get_native(),
            router_weights._get_native(),
            reverse_perm._get_native(),
            output._get_native(),
            k,
        )
        if profile:
            native.device_synchronize()
            t4 = time.perf_counter()
            MoELayer._profile_count += 1
            print(
                f"[MoE Profile] router={t1 - t0:.3f}s, routing={t2 - t1:.3f}s, experts={t3 - t2:.3f}s, scatter={t4 - t3:.3f}s"
            )

        # Reshape back
        if len(original_shape) == 3:
            output = output.reshape(*original_shape)

        return output

    def forward_zero_alloc(
        self,
        x: GPUArray,
        router_logits: GPUArray,
        router_weights: GPUArray,
        expert_indices: GPUArray,
        expert_counts: GPUArray,
        expert_offsets: GPUArray,
        permute_indices: GPUArray,
        reverse_perm: GPUArray,
        row_expert_ids: GPUArray,
        gathered: GPUArray,
        gate_out: GPUArray,
        up_out: GPUArray,
        intermediate: GPUArray,
        expert_outputs: GPUArray,
        output: GPUArray,
    ) -> GPUArray:
        """Zero-allocation forward pass for CUDA Graph support.

        This method uses pre-allocated buffers from DecodeBuffers to avoid
        any memory allocations during forward pass, enabling CUDA Graph capture.

        Args:
            x: Input tensor [1, hidden_size]
            router_logits: Pre-allocated [1, num_experts]
            router_weights: Pre-allocated [1, k]
            expert_indices: Pre-allocated [1, k] int32
            expert_counts: Pre-allocated [num_experts] int32
            expert_offsets: Pre-allocated [num_experts + 1] int32
            permute_indices: Pre-allocated [k] int32
            reverse_perm: Pre-allocated [k] int32
            row_expert_ids: Pre-allocated [k] int32
            gathered: Pre-allocated [k, hidden_size]
            gate_out: Pre-allocated [k, moe_intermediate_size]
            up_out: Pre-allocated [k, moe_intermediate_size]
            intermediate: Pre-allocated [k, moe_intermediate_size]
            expert_outputs: Pre-allocated [k, hidden_size]
            output: Pre-allocated [1, hidden_size]

        Returns:
            The output tensor (same as output parameter)
        """
        from pygpukit.core.backend import get_native_module
        from pygpukit.ops.elementwise import mul
        from pygpukit.ops.matmul import grouped_gemm_fp8_bf16
        from pygpukit.ops.nn import silu

        native = get_native_module()

        k = self.num_experts_per_tok

        # Step 1: Router forward (gate projection)
        self.gate(x, out=router_logits)

        # Step 2: Top-K selection (writes to router_weights and expert_indices)
        native.moe_topk_with_indices(
            router_logits._get_native(),
            router_weights._get_native(),
            expert_indices._get_native(),
            k,
        )

        # Step 3: Softmax over selected experts (in-place)
        native.moe_softmax_topk(router_weights._get_native(), k)

        # Step 4: Compute permutation
        native.moe_compute_permutation(
            expert_indices._get_native(),
            expert_counts._get_native(),
            expert_offsets._get_native(),
            permute_indices._get_native(),
            reverse_perm._get_native(),
            self.num_experts,
            k,
        )

        # Step 5: Gather hidden states
        native.moe_gather(
            x._get_native(),
            permute_indices._get_native(),
            gathered._get_native(),
            k,
        )

        # Step 6: Create row_expert_ids for grouped GEMM
        native.moe_expand_expert_offsets(
            expert_offsets._get_native(),
            row_expert_ids._get_native(),
            self.num_experts,
        )

        # Step 7: Expert computation with grouped GEMM
        # gate_proj: gathered[k, hidden] @ gate_weight[experts, inter, hidden]^T
        grouped_gemm_fp8_bf16(
            gathered,
            self._stacked_gate_weight,
            self._stacked_gate_scale,
            row_expert_ids,
            out=gate_out,
        )

        # up_proj: gathered[k, hidden] @ up_weight[experts, inter, hidden]^T
        grouped_gemm_fp8_bf16(
            gathered,
            self._stacked_up_weight,
            self._stacked_up_scale,
            row_expert_ids,
            out=up_out,
        )

        # SiLU(gate) * up -> intermediate
        silu(gate_out, out=intermediate)
        mul(intermediate, up_out, out=intermediate)

        # down_proj: intermediate[k, inter] @ down_weight[experts, hidden, inter]^T
        grouped_gemm_fp8_bf16(
            intermediate,
            self._stacked_down_weight,
            self._stacked_down_scale,
            row_expert_ids,
            out=expert_outputs,
        )

        # Step 8: Scatter and combine outputs
        native.moe_scatter(
            expert_outputs._get_native(),
            router_weights._get_native(),
            reverse_perm._get_native(),
            output._get_native(),
            k,
        )

        return output


# =============================================================================
# Unified TransformerBlock
# =============================================================================


class TransformerBlock:
    """Unified transformer block.

    Structure:
        Norm -> Attention -> Residual
        Norm -> MLP/MoE -> Residual
    """

    def __init__(
        self,
        attn_norm: Norm,
        attn: Attention,
        mlp_norm: Norm,
        mlp: MLP | MoELayer,
    ):
        self.attn_norm = attn_norm
        self.attn = attn
        self.mlp_norm = mlp_norm
        self.mlp = mlp  # Can be MLP or MoELayer

    def __call__(
        self,
        x: GPUArray,
        position_ids: list[int] | None = None,
        past_kv: tuple | None = None,
        use_cache: bool = False,
    ) -> tuple[GPUArray, tuple | None]:
        # Attention block
        residual = x
        x = self.attn_norm(x)
        attn_out, present_kv = self.attn(x, position_ids, past_kv, use_cache)
        x = add(residual, attn_out)

        # MLP block
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = add(residual, x)

        return x, present_kv
