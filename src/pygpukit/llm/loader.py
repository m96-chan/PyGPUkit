"""Model loading utilities for PyGPUkit LLM.

Provides:
- load_model_from_safetensors: Generic model loader with auto-detection
- load_gpt2_from_safetensors: GPT-2 specific loader
- load_llama_from_safetensors: LLaMA specific loader
- load_qwen3_from_safetensors: Qwen3 specific loader
- load_mixtral_from_safetensors: Mixtral MoE specific loader
"""

from __future__ import annotations

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

# Re-export quantization configs and utilities from quant module
from pygpukit.llm.quant import (
    FP8QuantConfig,
    ModelOptimizationInfo,
    PruningConfig,
    QATQuantConfig,
    SparsityConfig,
    load_fp8_weight_direct,
)

# Re-export repack function
from pygpukit.llm.repack import repack_model_weights

if TYPE_CHECKING:
    from pygpukit.llm.model import CausalTransformerModel


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
    from pygpukit.llm.model import CausalTransformerModel
    from pygpukit.llm.safetensors import Dtype, load_safetensors

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


__all__ = [
    # Main loaders
    "load_model_from_safetensors",
    "load_gpt2_from_safetensors",
    "load_llama_from_safetensors",
    "load_qwen3_from_safetensors",
    "load_mixtral_from_safetensors",
    # Weight repacking
    "repack_model_weights",
    # Quantization configs (re-exported)
    "FP8QuantConfig",
    "QATQuantConfig",
    "PruningConfig",
    "SparsityConfig",
    "ModelOptimizationInfo",
]
