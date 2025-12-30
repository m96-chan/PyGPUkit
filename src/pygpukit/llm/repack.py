"""Model weight repacking for PyGPUkit LLM.

Provides memory optimization by repacking weights into contiguous GPU memory
to fix performance regression from fragmented allocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.factory import from_numpy
from pygpukit.llm.layers import MoELayer

if TYPE_CHECKING:
    from pygpukit.llm.model import CausalTransformerModel


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

    from pygpukit.core.array import GPUArray

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


__all__ = [
    "repack_model_weights",
]
