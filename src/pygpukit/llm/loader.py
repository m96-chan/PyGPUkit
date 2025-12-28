"""Model loading utilities for PyGPUkit LLM.

Provides:
- load_model_from_safetensors: Generic model loader with auto-detection
- load_gpt2_from_safetensors: GPT-2 specific loader
- load_llama_from_safetensors: LLaMA specific loader
- load_qwen3_from_safetensors: Qwen3 specific loader
- repack_model_weights: Optimize GPU memory placement
- FP8 dequantization: Block-wise FP8 E4M3 to BF16/FP16 conversion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.dtypes import bfloat16 as dt_bfloat16
from pygpukit.core.dtypes import float16 as dt_float16
from pygpukit.core.dtypes import float32 as dt_float32
from pygpukit.core.factory import empty, from_numpy
from pygpukit.llm.config import (
    GPT2_SPEC,
    LLAMA_SPEC,
    MIXTRAL_SPEC,
    QWEN3_SPEC,
    ModelSpec,
    TransformerConfig,
    detect_model_spec,
)
from pygpukit.llm.layers import (
    MLP,
    Attention,
    LinearBF16,
    LinearFP8,
    MoELayer,
    Norm,
    TransformerBlock,
)

if TYPE_CHECKING:
    from pygpukit.llm import SafeTensorsFile, ShardedSafeTensorsFile
    from pygpukit.llm.model import CausalTransformerModel


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
    from pygpukit.llm import Dtype

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


# =============================================================================
# Legacy Loaders (convenience wrappers)
# =============================================================================


def load_gpt2_from_safetensors(
    model_path: str,
    dtype: str = "float32",
) -> CausalTransformerModel:
    """Load GPT-2 model from safetensors file.

    Args:
        model_path: Path to model.safetensors
        dtype: Weight dtype ("float32" or "float16")

    Returns:
        CausalTransformerModel instance
    """
    return load_model_from_safetensors(model_path, dtype=dtype, spec=GPT2_SPEC)


def load_llama_from_safetensors(
    model_path: str,
    dtype: str = "float32",
) -> CausalTransformerModel:
    """Load Llama model from safetensors file.

    Args:
        model_path: Path to model.safetensors
        dtype: Weight dtype ("float32" or "float16")

    Returns:
        CausalTransformerModel instance
    """
    return load_model_from_safetensors(model_path, dtype=dtype, spec=LLAMA_SPEC)


def load_qwen3_from_safetensors(
    model_path: str,
    dtype: str = "float32",
) -> CausalTransformerModel:
    """Load Qwen3 model from safetensors file.

    Args:
        model_path: Path to model.safetensors or model.safetensors.index.json
        dtype: Weight dtype ("float32" or "float16")

    Returns:
        CausalTransformerModel instance
    """
    return load_model_from_safetensors(model_path, dtype=dtype, spec=QWEN3_SPEC)


def load_mixtral_from_safetensors(
    model_path: str,
    dtype: str = "bfloat16",
) -> CausalTransformerModel:
    """Load Mixtral MoE model from safetensors file.

    Args:
        model_path: Path to model.safetensors or model.safetensors.index.json
        dtype: Weight dtype ("float32", "float16", or "bfloat16")

    Returns:
        CausalTransformerModel instance with MoELayer blocks
    """
    return load_model_from_safetensors(model_path, dtype=dtype, spec=MIXTRAL_SPEC)


# =============================================================================
# Model Weight Repacking
# =============================================================================


def repack_model_weights(model: CausalTransformerModel) -> None:
    """Repack all model weights into contiguous GPU memory.

    This fixes severe performance regression (7x slowdown) caused by
    fragmented GPU memory allocation during model loading. Weights
    allocated later end up in suboptimal memory regions.

    The repacking is done in two phases:
    1. Convert ALL weights to numpy (freeing GPU memory)
    2. Reallocate ALL weights fresh in contiguous memory

    Args:
        model: CausalTransformerModel to repack in-place

    Note:
        MoE models are currently skipped (not repacked) due to different
        weight structure. This will be addressed in a future update.
    """
    import gc

    # Skip repacking for MoE models (different weight structure)
    if model.blocks and isinstance(model.blocks[0].mlp, MoELayer):
        return

    # Phase 1: Collect all weights as numpy arrays
    numpy_cache: dict[int, dict] = {}
    dummy_arrays: list[GPUArray] = []

    # Embedding
    embed_np = model.embed_tokens.to_numpy()
    model.embed_tokens = None  # type: ignore

    # Position embedding
    pos_embed_np = None
    if model.position_embed is not None:
        pos_embed_np = model.position_embed.to_numpy()
        model.position_embed = None

    # lm_head
    lm_head_np = None
    if model._lm_head is not None:
        lm_head_np = model._lm_head.to_numpy()
        model._lm_head = None

    # Final norm
    final_norm_weight_np = model.final_norm.weight.to_numpy()
    final_norm_bias_np = None
    if model.final_norm.bias is not None:
        final_norm_bias_np = model.final_norm.bias.to_numpy()
    model.final_norm.weight = None  # type: ignore
    model.final_norm.bias = None

    # All blocks
    for i, block in enumerate(model.blocks):
        numpy_cache[i] = {}

        # Attention norms
        numpy_cache[i]["attn_norm_w"] = block.attn_norm.weight.to_numpy()
        numpy_cache[i]["attn_norm_b"] = (
            block.attn_norm.bias.to_numpy() if block.attn_norm.bias is not None else None
        )
        block.attn_norm.weight = None  # type: ignore
        block.attn_norm.bias = None

        numpy_cache[i]["mlp_norm_w"] = block.mlp_norm.weight.to_numpy()
        numpy_cache[i]["mlp_norm_b"] = (
            block.mlp_norm.bias.to_numpy() if block.mlp_norm.bias is not None else None
        )
        block.mlp_norm.weight = None  # type: ignore
        block.mlp_norm.bias = None

        # Attention projections
        attn = block.attn
        numpy_cache[i]["q_w"] = attn.q_proj.weight.to_numpy()
        numpy_cache[i]["q_b"] = (
            attn.q_proj.bias.to_numpy() if attn.q_proj.bias is not None else None
        )
        attn.q_proj.weight = None  # type: ignore
        attn.q_proj.bias = None
        attn.q_proj._weight_t = None

        numpy_cache[i]["k_w"] = attn.k_proj.weight.to_numpy()
        numpy_cache[i]["k_b"] = (
            attn.k_proj.bias.to_numpy() if attn.k_proj.bias is not None else None
        )
        attn.k_proj.weight = None  # type: ignore
        attn.k_proj.bias = None
        attn.k_proj._weight_t = None

        numpy_cache[i]["v_w"] = attn.v_proj.weight.to_numpy()
        numpy_cache[i]["v_b"] = (
            attn.v_proj.bias.to_numpy() if attn.v_proj.bias is not None else None
        )
        attn.v_proj.weight = None  # type: ignore
        attn.v_proj.bias = None
        attn.v_proj._weight_t = None

        numpy_cache[i]["o_w"] = attn.o_proj.weight.to_numpy()
        numpy_cache[i]["o_b"] = (
            attn.o_proj.bias.to_numpy() if attn.o_proj.bias is not None else None
        )
        attn.o_proj.weight = None  # type: ignore
        attn.o_proj.bias = None
        attn.o_proj._weight_t = None

        # QK norms
        if attn.q_norm is not None:
            numpy_cache[i]["q_norm_w"] = attn.q_norm.weight.to_numpy()
            numpy_cache[i]["q_norm_b"] = (
                attn.q_norm.bias.to_numpy() if attn.q_norm.bias is not None else None
            )
            attn.q_norm.weight = None  # type: ignore
            attn.q_norm.bias = None
        if attn.k_norm is not None:
            numpy_cache[i]["k_norm_w"] = attn.k_norm.weight.to_numpy()
            numpy_cache[i]["k_norm_b"] = (
                attn.k_norm.bias.to_numpy() if attn.k_norm.bias is not None else None
            )
            attn.k_norm.weight = None  # type: ignore
            attn.k_norm.bias = None

        # MLP projections
        mlp = block.mlp
        if mlp.activation == "gelu":
            numpy_cache[i]["fc1_w"] = mlp.fc1.weight.to_numpy()
            numpy_cache[i]["fc1_b"] = mlp.fc1.bias.to_numpy() if mlp.fc1.bias is not None else None
            mlp.fc1.weight = None  # type: ignore
            mlp.fc1.bias = None
            mlp.fc1._weight_t = None

            numpy_cache[i]["fc2_w"] = mlp.fc2.weight.to_numpy()
            numpy_cache[i]["fc2_b"] = mlp.fc2.bias.to_numpy() if mlp.fc2.bias is not None else None
            mlp.fc2.weight = None  # type: ignore
            mlp.fc2.bias = None
            mlp.fc2._weight_t = None
        else:  # SwiGLU
            numpy_cache[i]["gate_w"] = mlp.gate_proj.weight.to_numpy()
            numpy_cache[i]["gate_b"] = (
                mlp.gate_proj.bias.to_numpy() if mlp.gate_proj.bias is not None else None
            )
            mlp.gate_proj.weight = None  # type: ignore
            mlp.gate_proj.bias = None
            mlp.gate_proj._weight_t = None

            numpy_cache[i]["up_w"] = mlp.up_proj.weight.to_numpy()
            numpy_cache[i]["up_b"] = (
                mlp.up_proj.bias.to_numpy() if mlp.up_proj.bias is not None else None
            )
            mlp.up_proj.weight = None  # type: ignore
            mlp.up_proj.bias = None
            mlp.up_proj._weight_t = None

            numpy_cache[i]["down_w"] = mlp.down_proj.weight.to_numpy()
            numpy_cache[i]["down_b"] = (
                mlp.down_proj.bias.to_numpy() if mlp.down_proj.bias is not None else None
            )
            mlp.down_proj.weight = None  # type: ignore
            mlp.down_proj.bias = None
            mlp.down_proj._weight_t = None

    # Force garbage collection to free GPU memory
    gc.collect()

    # Allocate dummy arrays to fill the freed memory space
    dummy_size = 1024 * 1024 * 512  # 512M elements = 1GB for FP16
    try:
        for _ in range(16):  # Allocate ~16GB of dummy memory
            dummy = from_numpy(np.zeros(dummy_size, dtype=np.float16))
            dummy_arrays.append(dummy)
    except Exception:
        pass  # Continue with whatever dummy memory we could allocate

    # Phase 2: Reallocate all weights fresh (REVERSE order for memory optimization)
    for i in reversed(range(len(model.blocks))):
        block = model.blocks[i]
        cache = numpy_cache[i]

        # Attention norms
        block.attn_norm.weight = from_numpy(cache["attn_norm_w"])
        if cache["attn_norm_b"] is not None:
            block.attn_norm.bias = from_numpy(cache["attn_norm_b"])

        block.mlp_norm.weight = from_numpy(cache["mlp_norm_w"])
        if cache["mlp_norm_b"] is not None:
            block.mlp_norm.bias = from_numpy(cache["mlp_norm_b"])

        # Attention projections
        attn = block.attn
        attn.q_proj.weight = from_numpy(cache["q_w"])
        if cache["q_b"] is not None:
            attn.q_proj.bias = from_numpy(cache["q_b"])

        attn.k_proj.weight = from_numpy(cache["k_w"])
        if cache["k_b"] is not None:
            attn.k_proj.bias = from_numpy(cache["k_b"])

        attn.v_proj.weight = from_numpy(cache["v_w"])
        if cache["v_b"] is not None:
            attn.v_proj.bias = from_numpy(cache["v_b"])

        attn.o_proj.weight = from_numpy(cache["o_w"])
        if cache["o_b"] is not None:
            attn.o_proj.bias = from_numpy(cache["o_b"])

        # QK norms
        if "q_norm_w" in cache:
            attn.q_norm.weight = from_numpy(cache["q_norm_w"])
            if cache["q_norm_b"] is not None:
                attn.q_norm.bias = from_numpy(cache["q_norm_b"])
        if "k_norm_w" in cache:
            attn.k_norm.weight = from_numpy(cache["k_norm_w"])
            if cache["k_norm_b"] is not None:
                attn.k_norm.bias = from_numpy(cache["k_norm_b"])

        # MLP projections
        mlp = block.mlp
        if mlp.activation == "gelu":
            mlp.fc1.weight = from_numpy(cache["fc1_w"])
            if cache["fc1_b"] is not None:
                mlp.fc1.bias = from_numpy(cache["fc1_b"])

            mlp.fc2.weight = from_numpy(cache["fc2_w"])
            if cache["fc2_b"] is not None:
                mlp.fc2.bias = from_numpy(cache["fc2_b"])
        else:  # SwiGLU
            mlp.gate_proj.weight = from_numpy(cache["gate_w"])
            if cache["gate_b"] is not None:
                mlp.gate_proj.bias = from_numpy(cache["gate_b"])

            mlp.up_proj.weight = from_numpy(cache["up_w"])
            if cache["up_b"] is not None:
                mlp.up_proj.bias = from_numpy(cache["up_b"])

            mlp.down_proj.weight = from_numpy(cache["down_w"])
            if cache["down_b"] is not None:
                mlp.down_proj.bias = from_numpy(cache["down_b"])

        # Clear this block's cache immediately
        del numpy_cache[i]

    # Final norm
    model.final_norm.weight = from_numpy(final_norm_weight_np)
    if final_norm_bias_np is not None:
        model.final_norm.bias = from_numpy(final_norm_bias_np)

    # lm_head
    if lm_head_np is not None:
        model._lm_head = from_numpy(lm_head_np)

    # Embedding and position embedding last
    model.embed_tokens = from_numpy(embed_np)
    del embed_np

    if pos_embed_np is not None:
        model.position_embed = from_numpy(pos_embed_np)
        del pos_embed_np

    # Clear any cached transposes
    if hasattr(model, "_lm_head_t_cache"):
        delattr(model, "_lm_head_t_cache")

    # Free dummy arrays
    del dummy_arrays
    gc.collect()


# =============================================================================
# Generic Model Loader using ModelSpec
# =============================================================================


def load_model_from_safetensors(
    model_path: str,
    dtype: str = "float32",
    spec: ModelSpec | None = None,
    repack_weights: bool = True,
) -> CausalTransformerModel:
    """Load model from safetensors file using ModelSpec abstraction.

    Automatically detects model type (GPT-2, LLaMA, Qwen3) from tensor names
    and loads using the appropriate ModelSpec configuration.

    Args:
        model_path: Path to model.safetensors or model.safetensors.index.json
        dtype: Weight dtype ("float32", "float16", or "bfloat16")
        spec: Optional ModelSpec to use (auto-detected if None)
        repack_weights: Whether to repack weights for optimal memory placement

    Returns:
        CausalTransformerModel instance

    Example:
        # Auto-detect model type
        model = load_model_from_safetensors("/path/to/model.safetensors")

        # Explicit model type
        model = load_model_from_safetensors("/path/to/model.safetensors", spec=LLAMA_SPEC)
    """
    # Import here to avoid circular import
    from pygpukit.llm import Dtype, load_safetensors
    from pygpukit.llm.model import CausalTransformerModel

    st = load_safetensors(model_path)

    # Try to import direct mmap-to-GPU transfer function
    use_direct_transfer = False
    try:
        from pygpukit._native_loader import get_native_module

        _native = get_native_module()
        memcpy_ptr_to_device = getattr(_native, "memcpy_ptr_to_device", None)
        if memcpy_ptr_to_device is None:
            raise AttributeError("memcpy_ptr_to_device not found")

        first_tensor = st.tensor_names[0]
        st.tensor_data_ptr(first_tensor)
        use_direct_transfer = True
    except (ImportError, AttributeError):
        pass

    # Map dtype string to numpy dtype and native dtype
    if dtype == "float16":
        target_np_dtype = np.float16
        target_dtype_id = Dtype.Float16
        target_dt = dt_float16
    elif dtype == "bfloat16":
        target_np_dtype = np.uint16  # bf16 stored as uint16
        target_dtype_id = Dtype.BFloat16
        target_dt = dt_bfloat16
    else:  # float32
        target_np_dtype = np.float32
        target_dtype_id = Dtype.Float32
        target_dt = dt_float32

    # Detect model type if not specified
    if spec is None:
        spec = detect_model_spec(st.tensor_names)

    # Detect FP8 quantization from config.json
    fp8_config: FP8QuantConfig | None = None
    try:
        import json
        from pathlib import Path

        model_path_obj = Path(model_path)
        if model_path_obj.name.endswith(".index.json"):
            config_path = model_path_obj.parent / "config.json"
        else:
            config_path = model_path_obj.parent / "config.json"

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                hf_config = json.load(f)
            fp8_config = FP8QuantConfig.from_config(hf_config)
            if fp8_config is not None:
                print(
                    f"[FP8] Detected FP8 quantization: {fp8_config.fmt}, block_size={fp8_config.weight_block_size}"
                )
    except Exception:
        pass

    # Helper to check if a weight is FP8 quantized
    def is_fp8_weight(name: str) -> bool:
        scale_inv_name = name + "_scale_inv"
        return fp8_config is not None and scale_inv_name in st.tensor_names

    # Helper to load linear layer (returns LinearBF16 or LinearFP8)
    def load_linear(
        weight_name: str,
        bias_name: str | None = None,
        do_transpose: bool = False,
    ) -> LinearBF16 | LinearFP8:
        """Load a linear layer, using LinearFP8 for FP8 weights."""
        if is_fp8_weight(weight_name):
            # FP8 path: load as LinearFP8 without dequantization
            weight_fp8, scale_inv = load_fp8_weight_direct(
                st,
                weight_name,
                fp8_config.weight_block_size,  # type: ignore
            )
            # Load bias if specified (bias is not quantized)
            bias = None
            if bias_name and bias_name in st.tensor_names:
                bias = load_tensor(bias_name)
            return LinearFP8(weight_fp8, scale_inv, bias, fp8_config.weight_block_size)  # type: ignore
        else:
            # Standard path: load as LinearBF16
            weight = load_tensor(weight_name, do_transpose)
            bias = None
            if bias_name and bias_name in st.tensor_names:
                bias = load_tensor(bias_name)
            return LinearBF16(weight, bias)

    # Helper to load tensor with dtype conversion (no FP8 dequant - use load_linear for weights)
    def load_tensor(name: str, do_transpose: bool = False) -> GPUArray:
        info = st.tensor_info(name)

        # Direct mmap-to-GPU transfer for matching dtypes
        if use_direct_transfer and not do_transpose and info.dtype == target_dtype_id:
            ptr, size_bytes = st.tensor_data_ptr(name)
            gpu_arr = empty(info.shape, target_dt)
            memcpy_ptr_to_device(gpu_arr._native, ptr, size_bytes)
            return gpu_arr

        # Fallback: load via numpy with dtype conversion
        data = st.tensor_bytes(name)
        src_dtype_id = info.dtype

        if src_dtype_id == Dtype.BFloat16:
            arr = np.frombuffer(data, dtype=np.uint16).reshape(info.shape)
            if target_dtype_id == Dtype.BFloat16:
                arr = arr.copy()
            else:
                arr_f32 = np.empty(arr.shape, dtype=np.float32)
                arr_f32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
                arr = arr_f32.astype(target_np_dtype)
        else:
            dtype_map = {
                Dtype.Float32: np.float32,
                Dtype.Float16: np.float16,
                3: np.float64,
            }
            np_src_dtype = dtype_map.get(src_dtype_id, np.float32)
            arr = np.frombuffer(data, dtype=np_src_dtype).reshape(info.shape).copy()

            if target_dtype_id == Dtype.BFloat16:
                arr_f32 = arr.astype(np.float32)
                uint32_view = arr_f32.view(np.uint32)
                arr = ((uint32_view + 0x7FFF + ((uint32_view >> 16) & 1)) >> 16).astype(np.uint16)
            else:
                arr = arr.astype(target_np_dtype)

        if do_transpose and arr.ndim == 2:
            arr = arr.T.copy()

        return from_numpy(arr)

    def try_load(name: str | None, do_transpose: bool = False) -> GPUArray | None:
        if name is None or name not in st.tensor_names:
            return None
        return load_tensor(name, do_transpose)

    def layer_name(pattern: str | None, layer: int) -> str | None:
        if pattern is None:
            return None
        return pattern.format(layer=layer)

    def required_name(pattern: str, layer: int) -> str:
        """Get layer name for a required pattern (never None)."""
        return pattern.format(layer=layer)

    # Auto-detect config from tensor shapes
    embed_info = st.tensor_info(spec.embed_tokens)
    vocab_size = embed_info.shape[0]
    hidden_size = embed_info.shape[1]

    # Count layers
    num_layers = 0
    while required_name(spec.q_proj, num_layers) in st.tensor_names:
        num_layers += 1

    # Detect num_heads and num_kv_heads from projection shapes
    q_info = st.tensor_info(required_name(spec.q_proj, 0))
    q_dim = q_info.shape[0]
    head_dim = 64  # Default

    # Try to get head_dim from q_norm if present (Qwen3)
    if spec.use_qk_norm and spec.q_norm is not None:
        q_norm_name = required_name(spec.q_norm, 0)
        if q_norm_name in st.tensor_names:
            q_norm_info = st.tensor_info(q_norm_name)
            head_dim = q_norm_info.shape[0]
    else:
        # For models without q_norm, detect head_dim from tensor shapes
        for hd in [128, 64, 256]:
            if q_dim % hd == 0 and hidden_size % hd == 0:
                potential_num_heads = q_dim // hd
                if 4 <= potential_num_heads <= 128:
                    head_dim = hd
                    break

    num_heads = q_dim // head_dim

    # For GQA models, detect num_kv_heads
    num_kv_heads = num_heads
    if not spec.qkv_combined:
        k_info = st.tensor_info(required_name(spec.k_proj, 0))
        num_kv_heads = k_info.shape[0] // head_dim

    # Detect intermediate_size
    intermediate_size = 4 * hidden_size
    if spec.activation == "silu" and spec.gate_proj is not None:
        gate_info = st.tensor_info(required_name(spec.gate_proj, 0))
        intermediate_size = gate_info.shape[0]
    elif spec.activation == "gelu" and spec.fc1 is not None:
        fc1_info = st.tensor_info(required_name(spec.fc1, 0))
        intermediate_size = fc1_info.shape[0]

    # Build TransformerConfig
    explicit_head_dim = None
    if head_dim != hidden_size // num_heads:
        explicit_head_dim = head_dim

    # Try to read rope_theta, norm_eps, and MoE params from config.json
    rope_theta = spec.default_rope_theta
    norm_eps = spec.default_norm_eps
    num_experts: int | None = None
    num_experts_per_tok = 2
    moe_intermediate_size: int | None = None
    try:
        import json
        from pathlib import Path

        model_path_obj = Path(model_path)
        if model_path_obj.name.endswith(".index.json"):
            config_path = model_path_obj.parent / "config.json"
        else:
            config_path = model_path_obj.parent / "config.json"

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                hf_config = json.load(f)
            if "rope_theta" in hf_config:
                rope_theta = float(hf_config["rope_theta"])
            if "rms_norm_eps" in hf_config:
                norm_eps = float(hf_config["rms_norm_eps"])
            # MoE parameters (Mixtral uses num_local_experts, Qwen3-MoE uses num_experts)
            if "num_local_experts" in hf_config:
                num_experts = int(hf_config["num_local_experts"])
            elif "num_experts" in hf_config:
                num_experts = int(hf_config["num_experts"])
            if "num_experts_per_tok" in hf_config:
                num_experts_per_tok = int(hf_config["num_experts_per_tok"])
            if "moe_intermediate_size" in hf_config:
                moe_intermediate_size = int(hf_config["moe_intermediate_size"])
    except Exception:
        pass  # Use defaults

    transformer_config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        _head_dim=explicit_head_dim,
        norm_type=spec.norm_type,
        activation=spec.activation,
        use_rope=spec.use_rope,
        causal=True,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
    )

    # Load embeddings
    embed_tokens = load_tensor(spec.embed_tokens)
    position_embed = try_load(spec.position_embed) if spec.use_position_embed else None

    # Load blocks
    blocks = []
    for layer_idx in range(num_layers):
        # Attention norm (required)
        attn_norm_weight = load_tensor(required_name(spec.attn_norm, layer_idx))
        attn_norm_bias = try_load(layer_name(spec.attn_norm_bias, layer_idx))
        attn_norm = Norm(attn_norm_weight, attn_norm_bias, spec.norm_type, spec.default_norm_eps)

        # QK Norm (Qwen3, optional)
        q_norm_layer = None
        k_norm_layer = None
        if spec.use_qk_norm:
            q_norm_weight = try_load(layer_name(spec.q_norm, layer_idx))
            k_norm_weight = try_load(layer_name(spec.k_norm, layer_idx))
            if q_norm_weight is not None:
                q_norm_layer = Norm(q_norm_weight, None, spec.norm_type, spec.default_norm_eps)
            if k_norm_weight is not None:
                k_norm_layer = Norm(k_norm_weight, None, spec.norm_type, spec.default_norm_eps)

        # Attention projections
        if spec.qkv_combined:
            # GPT-2 style: combined QKV tensor needs to be split
            c_attn_weight = load_tensor(
                required_name(spec.q_proj, layer_idx), do_transpose=spec.weight_transpose
            )
            c_attn_bias = try_load(layer_name(spec.q_bias, layer_idx))

            # Split combined QKV
            c_attn_np = c_attn_weight.to_numpy()
            q_weight = from_numpy(c_attn_np[:hidden_size].copy().astype(target_np_dtype))
            k_weight = from_numpy(
                c_attn_np[hidden_size : 2 * hidden_size].copy().astype(target_np_dtype)
            )
            v_weight = from_numpy(c_attn_np[2 * hidden_size :].copy().astype(target_np_dtype))

            q_bias, k_bias, v_bias = None, None, None
            if c_attn_bias is not None:
                c_attn_bias_np = c_attn_bias.to_numpy()
                q_bias = from_numpy(c_attn_bias_np[:hidden_size].copy().astype(target_np_dtype))
                k_bias = from_numpy(
                    c_attn_bias_np[hidden_size : 2 * hidden_size].copy().astype(target_np_dtype)
                )
                v_bias = from_numpy(
                    c_attn_bias_np[2 * hidden_size :].copy().astype(target_np_dtype)
                )

            o_weight = load_tensor(
                required_name(spec.o_proj, layer_idx), do_transpose=spec.weight_transpose
            )
            o_bias = try_load(layer_name(spec.o_bias, layer_idx))

            attn = Attention(
                q_weight,
                k_weight,
                v_weight,
                o_weight,
                transformer_config,
                q_bias,
                k_bias,
                v_bias,
                o_bias,
                q_norm_layer,
                k_norm_layer,
            )
        else:
            # Separate Q, K, V projections (LLaMA/Qwen3 style)
            # Use load_linear to get LinearBF16 or LinearFP8 depending on quantization
            q_proj = load_linear(
                required_name(spec.q_proj, layer_idx),
                layer_name(spec.q_bias, layer_idx),
            )
            k_proj = load_linear(
                required_name(spec.k_proj, layer_idx),
                layer_name(spec.k_bias, layer_idx),
            )
            v_proj = load_linear(
                required_name(spec.v_proj, layer_idx),
                layer_name(spec.v_bias, layer_idx),
            )
            o_proj = load_linear(
                required_name(spec.o_proj, layer_idx),
                layer_name(spec.o_bias, layer_idx),
            )

            attn = Attention(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                transformer_config,
                q_norm=q_norm_layer,
                k_norm=k_norm_layer,
            )

        # MLP norm (required)
        mlp_norm_weight = load_tensor(required_name(spec.mlp_norm, layer_idx))
        mlp_norm_bias = try_load(layer_name(spec.mlp_norm_bias, layer_idx))
        mlp_norm = Norm(mlp_norm_weight, mlp_norm_bias, spec.norm_type, spec.default_norm_eps)

        # MLP or MoE
        mlp: MLP | MoELayer
        if spec.is_moe and num_experts is not None:
            # MoE: Load router gate and all experts
            def expert_name(pattern: str, layer: int, expert: int) -> str:
                return pattern.format(layer=layer, expert=expert)

            # Router gate: [hidden_size, num_experts]
            gate_weight = load_tensor(required_name(spec.moe_gate, layer_idx))

            # Load all expert weights (using load_linear for FP8 support)
            expert_weights: list = []
            for expert_idx in range(num_experts):
                exp_gate = load_linear(expert_name(spec.expert_gate_proj, layer_idx, expert_idx))
                exp_up = load_linear(expert_name(spec.expert_up_proj, layer_idx, expert_idx))
                exp_down = load_linear(expert_name(spec.expert_down_proj, layer_idx, expert_idx))
                expert_weights.append((exp_gate, exp_up, exp_down))

            mlp = MoELayer(transformer_config, gate_weight, expert_weights)
        elif spec.activation == "gelu" and spec.fc1 is not None and spec.fc2 is not None:
            fc1_weight = load_tensor(
                required_name(spec.fc1, layer_idx), do_transpose=spec.weight_transpose
            )
            fc1_bias = try_load(layer_name(spec.fc1_bias, layer_idx))
            fc2_weight = load_tensor(
                required_name(spec.fc2, layer_idx), do_transpose=spec.weight_transpose
            )
            fc2_bias = try_load(layer_name(spec.fc2_bias, layer_idx))
            mlp = MLP(
                transformer_config,
                fc1_weight=fc1_weight,
                fc1_bias=fc1_bias,
                fc2_weight=fc2_weight,
                fc2_bias=fc2_bias,
            )
        elif spec.gate_proj is not None and spec.up_proj is not None and spec.down_proj is not None:
            # SwiGLU - use load_linear for FP8 support
            gate_proj_linear = load_linear(required_name(spec.gate_proj, layer_idx))
            up_proj_linear = load_linear(required_name(spec.up_proj, layer_idx))
            down_proj_linear = load_linear(required_name(spec.down_proj, layer_idx))
            mlp = MLP(
                transformer_config,
                gate_proj=gate_proj_linear,
                up_proj=up_proj_linear,
                down_proj=down_proj_linear,
            )
        else:
            raise ValueError(f"ModelSpec {spec.name} has invalid MLP configuration")

        block = TransformerBlock(attn_norm, attn, mlp_norm, mlp)
        blocks.append(block)

    # Final norm
    final_norm_weight = load_tensor(spec.final_norm)
    final_norm_bias = try_load(spec.final_norm_bias)
    final_norm = Norm(final_norm_weight, final_norm_bias, spec.norm_type, spec.default_norm_eps)

    # LM head
    lm_head = None
    if spec.lm_head is not None and spec.lm_head in st.tensor_names:
        lm_head = load_tensor(spec.lm_head)

    model = CausalTransformerModel(
        transformer_config,
        embed_tokens,
        blocks,
        final_norm,
        lm_head,
        position_embed,
        spec,
    )
    if repack_weights:
        repack_model_weights(model)
    return model
