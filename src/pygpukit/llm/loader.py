"""Model loading utilities for PyGPUkit LLM.

Provides:
- load_model_from_safetensors: Generic model loader with auto-detection
- load_gpt2_from_safetensors: GPT-2 specific loader
- load_llama_from_safetensors: LLaMA specific loader
- load_qwen3_from_safetensors: Qwen3 specific loader
- repack_model_weights: Optimize GPU memory placement
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
    QWEN3_SPEC,
    ModelSpec,
    TransformerConfig,
    detect_model_spec,
)
from pygpukit.llm.layers import MLP, Attention, Norm, TransformerBlock

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
    """
    import gc

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

    # Helper to load tensor with dtype conversion
    def load_tensor(name: str, do_transpose: bool = False) -> GPUArray:
        info = st.tensor_info(name)

        # Direct mmap-to-GPU transfer for matching dtypes
        if use_direct_transfer and not do_transpose and info.dtype == target_dtype_id:
            ptr, size_bytes = st.tensor_data_ptr(name)
            gpu_arr = empty(info.shape, target_dt)
            memcpy_ptr_to_device(gpu_arr._array, ptr, size_bytes)
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

    # Try to read rope_theta and norm_eps from config.json
    rope_theta = spec.default_rope_theta
    norm_eps = spec.default_norm_eps
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
            q_weight = load_tensor(required_name(spec.q_proj, layer_idx))
            k_weight = load_tensor(required_name(spec.k_proj, layer_idx))
            v_weight = load_tensor(required_name(spec.v_proj, layer_idx))
            o_weight = load_tensor(required_name(spec.o_proj, layer_idx))

            q_bias = try_load(layer_name(spec.q_bias, layer_idx))
            k_bias = try_load(layer_name(spec.k_bias, layer_idx))
            v_bias = try_load(layer_name(spec.v_bias, layer_idx))
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

        # MLP norm (required)
        mlp_norm_weight = load_tensor(required_name(spec.mlp_norm, layer_idx))
        mlp_norm_bias = try_load(layer_name(spec.mlp_norm_bias, layer_idx))
        mlp_norm = Norm(mlp_norm_weight, mlp_norm_bias, spec.norm_type, spec.default_norm_eps)

        # MLP
        if spec.activation == "gelu" and spec.fc1 is not None and spec.fc2 is not None:
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
            # SwiGLU
            gate_proj = load_tensor(required_name(spec.gate_proj, layer_idx))
            up_proj = load_tensor(required_name(spec.up_proj, layer_idx))
            down_proj = load_tensor(required_name(spec.down_proj, layer_idx))
            mlp = MLP(
                transformer_config,
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
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
