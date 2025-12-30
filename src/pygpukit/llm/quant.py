"""Quantization configuration and utilities for PyGPUkit LLM.

Provides:
- FP8QuantConfig: FP8 quantization configuration
- QATQuantConfig: QAT (Quantization-Aware Training) configuration
- PruningConfig: Pruning configuration
- SparsityConfig: Sparsity pattern configuration
- ModelOptimizationInfo: Combined optimization information
- FP8 dequantization utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray
    from pygpukit.llm.safetensors import SafeTensorsFile, ShardedSafeTensorsFile


# =============================================================================
# FP8 Quantization Support
# =============================================================================


@dataclass
class FP8QuantConfig:
    """FP8 quantization configuration from HuggingFace config.json."""

    quant_method: str  # "fp8"
    fmt: str  # "e4m3" or "e5m2"
    weight_block_size: tuple[int, int]  # e.g., (128, 128)
    modules_to_not_convert: list[str]  # List of module name patterns to skip

    @classmethod
    def from_config(cls, config: dict) -> FP8QuantConfig | None:
        """Parse quantization config from HF config.json."""
        qc = config.get("quantization_config")
        if qc is None or qc.get("quant_method") != "fp8":
            return None

        block_size = qc.get("weight_block_size", [128, 128])
        return cls(
            quant_method="fp8",
            fmt=qc.get("fmt", "e4m3"),
            weight_block_size=(block_size[0], block_size[1]),
            modules_to_not_convert=qc.get("modules_to_not_convert", []),
        )


# =============================================================================
# QAT/QAD Quantization Support (Issue #115)
# =============================================================================


@dataclass
class QATQuantConfig:
    """QAT (Quantization-Aware Training) configuration.

    Supports models trained with:
    - NVIDIA TensorRT Model Optimizer
    - HuggingFace Optimum
    - PyTorch Quantization

    Reference:
    - https://nvidia.github.io/TensorRT-Model-Optimizer/
    - https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/
    """

    quant_method: str  # "qat", "modelopt", "nvfp4", etc.
    quant_algo: str  # "FP8", "INT8", "NVFP4", "W8A8", etc.
    group_size: int  # Block/group size for quantization
    kv_cache_quant_algo: str | None  # KV cache quantization (optional)
    exclude_modules: list[str]  # Modules to skip quantization
    producer: str | None  # Tool that produced the checkpoint (e.g., "modelopt")
    producer_version: str | None  # Version of the producer tool

    @classmethod
    def from_config(cls, config: dict) -> QATQuantConfig | None:
        """Parse QAT config from HF config.json or hf_quant_config.json."""
        # Check for TensorRT Model Optimizer format (hf_quant_config.json style)
        if "producer" in config and "quantization" in config:
            producer_info = config.get("producer", {})
            quant_info = config.get("quantization", {})
            return cls(
                quant_method="modelopt",
                quant_algo=quant_info.get("quant_algo", "unknown"),
                group_size=quant_info.get("group_size", 128),
                kv_cache_quant_algo=quant_info.get("kv_cache_quant_algo"),
                exclude_modules=quant_info.get("exclude_modules", []),
                producer=producer_info.get("name"),
                producer_version=producer_info.get("version"),
            )

        # Check for HF quantization_config with QAT method
        qc = config.get("quantization_config")
        if qc is None:
            return None

        quant_method = qc.get("quant_method", "")
        # QAT methods: "qat", "awq", "gptq", etc. (exclude "fp8" which is handled separately)
        qat_methods = {"qat", "awq", "gptq", "bnb", "modelopt"}
        if quant_method not in qat_methods:
            return None

        return cls(
            quant_method=quant_method,
            quant_algo=qc.get("quant_algo", qc.get("bits", "unknown")),
            group_size=qc.get("group_size", qc.get("block_size", 128)),
            kv_cache_quant_algo=qc.get("kv_cache_quant_algo"),
            exclude_modules=qc.get("modules_to_not_convert", []),
            producer=None,
            producer_version=None,
        )


# =============================================================================
# Pruning Support (Issue #115)
# =============================================================================


@dataclass
class PruningConfig:
    """Pruning configuration for structurally smaller models.

    Supports models pruned with:
    - NVIDIA TensorRT Model Optimizer
    - HuggingFace nn_pruning
    - Neural Compressor

    Reference:
    - https://github.com/huggingface/nn_pruning
    - https://github.com/NVIDIA/TensorRT-Model-Optimizer
    """

    pruning_method: str  # "magnitude", "movement", "structured", "unstructured"
    sparsity: float  # Target sparsity (0.0 to 1.0)
    pruned_heads: dict[int, list[int]] | None  # Layer -> pruned head indices
    is_structured: bool  # True if structured pruning (removes entire heads/neurons)

    @classmethod
    def from_config(cls, config: dict) -> PruningConfig | None:
        """Parse pruning config from HF config.json."""
        # Check for pruned_heads (HuggingFace standard)
        pruned_heads = config.get("pruned_heads")
        if pruned_heads:
            # Convert string keys to int if needed
            if isinstance(pruned_heads, dict):
                pruned_heads = {int(k): v for k, v in pruned_heads.items()}
            return cls(
                pruning_method="structured",
                sparsity=0.0,  # Unknown from config alone
                pruned_heads=pruned_heads,
                is_structured=True,
            )

        # Check for pruning_config section
        pc = config.get("pruning_config")
        if pc is None:
            return None

        return cls(
            pruning_method=pc.get("pruning_type", pc.get("method", "unknown")),
            sparsity=pc.get("target_sparsity", pc.get("sparsity", 0.0)),
            pruned_heads=pc.get("pruned_heads"),
            is_structured=pc.get("is_structured", pc.get("structured", False)),
        )


# =============================================================================
# Sparsity Pattern Support (Issue #115)
# =============================================================================


@dataclass
class SparsityConfig:
    """Sparsity pattern configuration for sparse tensor operations.

    Supports:
    - 2:4 structured sparsity (Ampere+)
    - Block sparsity patterns
    - Custom sparsity masks

    Reference:
    - https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
    """

    pattern: str  # "2:4", "4:8", "block", "unstructured"
    block_size: tuple[int, int] | None  # For block sparsity
    density: float  # Non-zero ratio (1 - sparsity)

    @classmethod
    def from_config(cls, config: dict) -> SparsityConfig | None:
        """Parse sparsity config from HF config.json."""
        sc = config.get("sparsity_config")
        if sc is None:
            # Check for sparsity in quantization_config
            qc = config.get("quantization_config", {})
            sparsity_pattern = qc.get("sparsity_pattern")
            if sparsity_pattern:
                return cls(
                    pattern=sparsity_pattern,
                    block_size=None,
                    density=1.0 - qc.get("sparsity", 0.5),
                )
            return None

        pattern = sc.get("pattern", sc.get("sparsity_pattern", "unknown"))
        block_size = sc.get("block_size")
        if block_size and isinstance(block_size, list):
            block_size = tuple(block_size)

        return cls(
            pattern=pattern,
            block_size=block_size,
            density=sc.get("density", 1.0 - sc.get("sparsity", 0.0)),
        )

    def is_2_4_sparse(self) -> bool:
        """Check if this is 2:4 structured sparsity (Ampere+ TensorCore)."""
        return self.pattern == "2:4"


# =============================================================================
# Model Optimization Info (Issue #115)
# =============================================================================


@dataclass
class ModelOptimizationInfo:
    """Combined optimization information for a model.

    Aggregates all optimization techniques applied to the model:
    - Quantization (FP8, QAT, etc.)
    - Pruning (structured, unstructured)
    - Sparsity (2:4, block)
    """

    fp8_config: FP8QuantConfig | None
    qat_config: QATQuantConfig | None
    pruning_config: PruningConfig | None
    sparsity_config: SparsityConfig | None

    @classmethod
    def from_config(cls, config: dict) -> ModelOptimizationInfo:
        """Parse all optimization configs from config.json."""
        return cls(
            fp8_config=FP8QuantConfig.from_config(config),
            qat_config=QATQuantConfig.from_config(config),
            pruning_config=PruningConfig.from_config(config),
            sparsity_config=SparsityConfig.from_config(config),
        )

    def has_any_optimization(self) -> bool:
        """Check if any optimization is applied."""
        return any(
            [
                self.fp8_config,
                self.qat_config,
                self.pruning_config,
                self.sparsity_config,
            ]
        )

    def summary(self) -> str:
        """Return a summary string of optimizations."""
        parts = []
        if self.fp8_config:
            parts.append(f"FP8({self.fp8_config.fmt})")
        if self.qat_config:
            parts.append(f"QAT({self.qat_config.quant_algo})")
        if self.pruning_config:
            parts.append(f"Pruned({self.pruning_config.pruning_method})")
        if self.sparsity_config:
            parts.append(f"Sparse({self.sparsity_config.pattern})")
        return ", ".join(parts) if parts else "None"


# =============================================================================
# FP8 E4M3 Conversion Utilities
# =============================================================================

# FP8 E4M3 to float32 lookup table (256 entries)
# Format: 1 sign bit, 4 exponent bits, 3 mantissa bits
# Special values: NaN (0x7F/0xFF), no infinity
_FP8_E4M3_TO_F32_TABLE: np.ndarray | None = None


def _get_fp8_e4m3_table() -> np.ndarray:
    """Build FP8 E4M3 to float32 conversion lookup table."""
    global _FP8_E4M3_TO_F32_TABLE
    if _FP8_E4M3_TO_F32_TABLE is not None:
        return _FP8_E4M3_TO_F32_TABLE

    table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        # Extract components
        sign = (i >> 7) & 1
        exp = (i >> 3) & 0xF  # 4 exponent bits
        mant = i & 0x7  # 3 mantissa bits

        if exp == 0xF and mant == 0x7:
            # NaN (0x7F and 0xFF)
            table[i] = np.nan
        elif exp == 0:
            # Subnormal (exponent = 0)
            # Value = (-1)^sign * 2^(-6) * (0.mantissa)
            value = (mant / 8.0) * (2.0**-6)
            table[i] = -value if sign else value
        else:
            # Normal
            # Value = (-1)^sign * 2^(exp-7) * (1.mantissa)
            value = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
            table[i] = -value if sign else value

    _FP8_E4M3_TO_F32_TABLE = table
    return table


def dequantize_fp8_e4m3_block(
    fp8_bytes: np.ndarray,
    scale_inv: np.ndarray,
    block_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Dequantize FP8 E4M3 weight with block-wise scaling.

    Args:
        fp8_bytes: Raw FP8 data as uint8 array, shape [H, W]
        scale_inv: Inverse scale factors, shape [H//block_h, W//block_w]
        block_size: Block size for quantization (default 128x128)

    Returns:
        Dequantized float32 array, shape [H, W]
    """
    # Convert FP8 bytes to float32 using lookup table
    table = _get_fp8_e4m3_table()
    f32 = table[fp8_bytes.ravel()].reshape(fp8_bytes.shape)

    # Apply block-wise scaling
    H, W = f32.shape
    block_h, block_w = block_size

    # Ensure scale_inv is float32 for computation
    if scale_inv.dtype != np.float32:
        # BF16 stored as uint16 -> convert to float32
        if scale_inv.dtype == np.uint16:
            scale_f32 = np.empty(scale_inv.shape, dtype=np.float32)
            scale_f32.view(np.uint32)[:] = scale_inv.astype(np.uint32) << 16
        else:
            scale_f32 = scale_inv.astype(np.float32)
    else:
        scale_f32 = scale_inv

    # Apply scaling per block using broadcasting
    num_blocks_h = H // block_h
    num_blocks_w = W // block_w

    # Reshape for vectorized block scaling
    f32_reshaped = f32.reshape(num_blocks_h, block_h, num_blocks_w, block_w)
    scale_expanded = scale_f32[:, np.newaxis, :, np.newaxis]
    f32_scaled = f32_reshaped * scale_expanded
    result = f32_scaled.reshape(H, W)

    return result


def is_fp8_weight(tensor_name: str, tensor_names: list[str]) -> bool:
    """Check if a weight tensor has an FP8 scale tensor."""
    scale_name = tensor_name + "_scale_inv"
    return scale_name in tensor_names


def load_fp8_weight_direct(
    st: SafeTensorsFile | ShardedSafeTensorsFile,
    weight_name: str,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[GPUArray, GPUArray]:
    """Load FP8 weight directly without dequantization.

    Returns:
        (weight_fp8, scale_inv) tuple:
        - weight_fp8: [out_features, in_features] as uint8
        - scale_inv: [out/block_h, in/block_w] as bf16
    """
    from pygpukit.core.factory import from_numpy
    from pygpukit.llm.safetensors import Dtype

    # Load FP8 weight as uint8
    info = st.tensor_info(weight_name)
    data = st.tensor_bytes(weight_name)
    fp8_bytes = np.frombuffer(data, dtype=np.uint8).reshape(info.shape).copy()
    weight_fp8 = from_numpy(fp8_bytes)

    # Load scale_inv tensor
    scale_name = weight_name + "_scale_inv"
    scale_info = st.tensor_info(scale_name)
    scale_data = st.tensor_bytes(scale_name)

    # scale_inv is typically bfloat16
    if scale_info.dtype == Dtype.BFloat16:
        scale_inv = np.frombuffer(scale_data, dtype=np.uint16).reshape(scale_info.shape).copy()
    else:
        # Convert float32 to bfloat16
        scale_f32 = np.frombuffer(scale_data, dtype=np.float32).reshape(scale_info.shape)
        uint32_view = scale_f32.view(np.uint32)
        scale_inv = ((uint32_view + 0x7FFF + ((uint32_view >> 16) & 1)) >> 16).astype(np.uint16)

    scale_inv_gpu = from_numpy(scale_inv)

    return weight_fp8, scale_inv_gpu


__all__ = [
    # Quantization configs
    "FP8QuantConfig",
    "QATQuantConfig",
    "PruningConfig",
    "SparsityConfig",
    "ModelOptimizationInfo",
    # FP8 utilities
    "dequantize_fp8_e4m3_block",
    "is_fp8_weight",
    "load_fp8_weight_direct",
]
