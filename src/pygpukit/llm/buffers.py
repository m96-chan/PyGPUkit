"""Pre-allocated buffers for CUDA Graph support.

Provides:
- DecodeBuffers: Buffers for allocation-free decode steps (seq_len=1)
- PrefillBuffers: Buffers for allocation-free prefill phase (variable seq_len)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import zeros

if TYPE_CHECKING:
    from pygpukit.llm.config import TransformerConfig


# =============================================================================
# Decode Buffers for CUDA Graph Support
# =============================================================================


@dataclass
class DecodeBuffers:
    """Pre-allocated buffers for allocation-free decode steps.

    These buffers are layer-shared (reused across all layers in a single decode step)
    since layers are processed sequentially. This eliminates all memory allocations
    during decode, enabling CUDA Graph capture.

    Buffer shapes (for Qwen3-8B example):
    - hidden: [1, 4096] - layer input/output
    - qkv_proj_out: [1, 6144] - Fused QKV projection output (q_dim + k_dim + v_dim)
    - q_proj_out: [1, 4096] - Q projection output (2D) - DEPRECATED, kept for compat
    - k_proj_out, v_proj_out: [1, 1024] - K/V projection outputs (2D) - DEPRECATED
    - o_proj_out: [1, 4096] - O projection output (2D)
    - q: [1, 32, 128] - query after reshape (3D)
    - k, v: [1, 8, 128] - key/value after reshape (3D)
    - attn_out: [32, 1, 128] - SDPA output (transposed format)
    - gate_up_out: [1, 24576] - Fused gate_up projection output (2 * intermediate_size)
    - mlp_gate, mlp_up: [1, 12288] - MLP intermediates (views into gate_up_out)
    - cos, sin: [1, 128] - RoPE tables
    - embed_out: [1, 4096] - embedding lookup output
    """

    # Main computation buffers
    hidden: GPUArray  # [1, hidden_size]
    q: GPUArray  # [1, num_heads, head_dim]
    k: GPUArray  # [1, num_kv_heads, head_dim]
    v: GPUArray  # [1, num_kv_heads, head_dim]
    attn_out: GPUArray  # [num_heads, 1, head_dim]
    mlp_gate: GPUArray  # [1, intermediate_size]
    mlp_up: GPUArray  # [1, intermediate_size]
    mlp_down: GPUArray  # [1, hidden_size] - down projection output

    # Projection output buffers (2D, for matmul out=)
    q_proj_out: GPUArray  # [1, num_heads * head_dim]
    k_proj_out: GPUArray  # [1, num_kv_heads * head_dim]
    v_proj_out: GPUArray  # [1, num_kv_heads * head_dim]
    o_proj_out: GPUArray  # [1, hidden_size]

    # Transposed Q buffer for SDPA
    q_t: GPUArray  # [num_heads, 1, head_dim]

    # RoPE buffers
    cos: GPUArray  # [1, head_dim]
    sin: GPUArray  # [1, head_dim]

    # Embedding output
    embed_out: GPUArray  # [1, hidden_size]

    # Temporary buffers for intermediate computations
    residual: GPUArray  # [1, hidden_size]
    norm_out: GPUArray  # [1, hidden_size]

    # For QK norm (Qwen3)
    q_2d: GPUArray | None = None  # [num_heads, head_dim] - rmsnorm output
    k_2d: GPUArray | None = None  # [num_kv_heads, head_dim] - rmsnorm output
    q_flat: GPUArray | None = None  # [num_heads, head_dim] - rmsnorm input
    k_flat: GPUArray | None = None  # [num_kv_heads, head_dim] - rmsnorm input

    # GPU position buffer for CUDA Graph replay (int32)
    position_buf: GPUArray | None = None  # [1] int32

    # Fused projection buffers (for reduced matmul count)
    # Used with GPUArray.narrow() for zero-copy splitting:
    # - qkv_proj_out: Single matmul replaces 3 (Q, K, V projections)
    # - gate_up_out: Single matmul replaces 2 (gate, up projections)
    qkv_proj_out: GPUArray | None = None  # [1, q_dim + k_dim + v_dim]
    gate_up_out: GPUArray | None = None  # [1, 2 * intermediate_size]

    # Pre-cached narrow views (created once, reused every forward)
    q_view: GPUArray | None = None  # view of qkv_proj_out[0:q_dim]
    k_view: GPUArray | None = None  # view of qkv_proj_out[q_dim:q_dim+k_dim]
    v_view: GPUArray | None = None  # view of qkv_proj_out[q_dim+k_dim:]
    gate_view: GPUArray | None = None  # view of gate_up_out[0:intermediate_size]
    up_view: GPUArray | None = None  # view of gate_up_out[intermediate_size:]

    # Logits buffer for CUDA Graph (lm_head projection output)
    logits: GPUArray | None = None  # [1, vocab_size]

    # Sampling buffers for CUDA Graph
    sampled_token: GPUArray | None = None  # [1] int32 - sampled token ID
    random_val: GPUArray | None = None  # [1] float32 - random value for sampling

    # Input token ID buffer for CUDA Graph replay
    token_id_buf: GPUArray | None = None  # [1] int32 - input token ID

    # Context length buffer for CUDA Graph replay (for SDPA)
    context_len_buf: GPUArray | None = None  # [1] int32 - context length

    # =========================================================================
    # MoE Decode Buffers (for zero-allocation MoE decode)
    # =========================================================================
    moe_num_experts: int = 0  # 0 means MoE buffers not allocated
    moe_num_experts_per_tok: int = 0  # k (top-k experts per token)
    moe_intermediate_size: int = 0  # MoE intermediate size

    # Router outputs
    moe_router_logits: GPUArray | None = None  # [1, num_experts]
    moe_router_weights: GPUArray | None = None  # [1, k]
    moe_expert_indices: GPUArray | None = None  # [1, k] int32

    # Permutation buffers
    moe_expert_counts: GPUArray | None = None  # [num_experts] int32
    moe_expert_offsets: GPUArray | None = None  # [num_experts + 1] int32
    moe_permute_indices: GPUArray | None = None  # [k] int32
    moe_reverse_perm: GPUArray | None = None  # [k] int32
    moe_row_expert_ids: GPUArray | None = None  # [k] int32

    # Expert computation buffers
    moe_gathered: GPUArray | None = None  # [k, hidden_size]
    moe_gate_out: GPUArray | None = None  # [k, moe_intermediate_size]
    moe_up_out: GPUArray | None = None  # [k, moe_intermediate_size]
    moe_intermediate: GPUArray | None = None  # [k, moe_intermediate_size]
    moe_expert_outputs: GPUArray | None = None  # [k, hidden_size]
    moe_output: GPUArray | None = None  # [1, hidden_size]

    # =========================================================================
    # Batch Decode Buffers (for zero-allocation batch verify, max_batch tokens)
    # =========================================================================
    max_batch_size: int = 0  # 0 means batch buffers not allocated

    # Batch input/output
    hidden_batch: GPUArray | None = None  # [max_batch, hidden_size]
    residual_batch: GPUArray | None = None  # [max_batch, hidden_size]
    norm_out_batch: GPUArray | None = None  # [max_batch, hidden_size]

    # Batch QKV projection
    qkv_proj_out_batch: GPUArray | None = None  # [max_batch, q_dim + k_dim + v_dim]

    # Batch Q/K/V after split (3D for attention)
    q_batch: GPUArray | None = None  # [max_batch, num_heads, head_dim]
    k_batch: GPUArray | None = None  # [max_batch, num_kv_heads, head_dim]
    v_batch: GPUArray | None = None  # [max_batch, num_kv_heads, head_dim]

    # Batch Q transposed for SDPA
    q_t_batch: GPUArray | None = None  # [num_heads, max_batch, head_dim]

    # Batch attention output
    attn_out_batch: GPUArray | None = None  # [num_heads, max_batch, head_dim]
    attn_out_t_batch: GPUArray | None = None  # [max_batch, num_heads, head_dim]

    # Batch O projection output
    o_proj_out_batch: GPUArray | None = None  # [max_batch, hidden_size]

    # Batch MLP
    gate_up_out_batch: GPUArray | None = None  # [max_batch, 2 * intermediate_size]
    mlp_down_batch: GPUArray | None = None  # [max_batch, hidden_size]

    # Batch RoPE
    cos_batch: GPUArray | None = None  # [max_batch, head_dim]
    sin_batch: GPUArray | None = None  # [max_batch, head_dim]

    # Batch logits (for verify)
    logits_batch: GPUArray | None = None  # [max_batch, vocab_size]

    # Batch QK norm (Qwen3)
    q_flat_batch: GPUArray | None = None  # [max_batch * num_heads, head_dim]
    k_flat_batch: GPUArray | None = None  # [max_batch * num_kv_heads, head_dim]

    # Batch CUDA Graph buffers (for graph capture/replay)
    token_ids_batch_buf: GPUArray | None = None  # [max_batch] int32 - batch token IDs
    start_position_batch_buf: GPUArray | None = None  # [1] int32 - start position

    @classmethod
    def allocate(
        cls,
        config: TransformerConfig,
        dtype: str = "float16",
        use_qk_norm: bool = False,
        vocab_size: int | None = None,
        max_batch_size: int = 0,
        moe_config: dict | None = None,
    ) -> DecodeBuffers:
        """Allocate all decode buffers.

        Args:
            config: Model configuration
            dtype: Data type for buffers
            use_qk_norm: Whether to allocate QK norm buffers (Qwen3)
            vocab_size: Vocabulary size for logits buffer (optional, for CUDA Graph)
            max_batch_size: Maximum batch size for batch decode (0 = no batch buffers)
            moe_config: MoE configuration dict with keys:
                - num_experts: Number of experts (e.g., 128)
                - num_experts_per_tok: Top-k experts per token (e.g., 8)
                - moe_intermediate_size: MoE intermediate size (e.g., 768)
        """
        assert config.num_kv_heads is not None
        assert config.intermediate_size is not None

        hidden = zeros((1, config.hidden_size), dtype=dtype)
        q = zeros((1, config.num_heads, config.head_dim), dtype=dtype)
        k = zeros((1, config.num_kv_heads, config.head_dim), dtype=dtype)
        v = zeros((1, config.num_kv_heads, config.head_dim), dtype=dtype)
        attn_out = zeros((config.num_heads, 1, config.head_dim), dtype=dtype)
        mlp_gate = zeros((1, config.intermediate_size), dtype=dtype)
        mlp_up = zeros((1, config.intermediate_size), dtype=dtype)
        mlp_down = zeros((1, config.hidden_size), dtype=dtype)

        # Projection output buffers (2D for matmul out=)
        q_proj_out = zeros((1, config.num_heads * config.head_dim), dtype=dtype)
        k_proj_out = zeros((1, config.num_kv_heads * config.head_dim), dtype=dtype)
        v_proj_out = zeros((1, config.num_kv_heads * config.head_dim), dtype=dtype)
        o_proj_out = zeros((1, config.hidden_size), dtype=dtype)

        # Transposed Q buffer for SDPA
        q_t = zeros((config.num_heads, 1, config.head_dim), dtype=dtype)

        cos = zeros((1, config.head_dim), dtype=dtype)
        sin = zeros((1, config.head_dim), dtype=dtype)

        embed_out = zeros((1, config.hidden_size), dtype=dtype)
        residual = zeros((1, config.hidden_size), dtype=dtype)
        norm_out = zeros((1, config.hidden_size), dtype=dtype)

        # QK norm buffers
        q_2d = None
        k_2d = None
        q_flat = None
        k_flat = None
        if use_qk_norm:
            q_2d = zeros((config.num_heads, config.head_dim), dtype=dtype)
            k_2d = zeros((config.num_kv_heads, config.head_dim), dtype=dtype)
            q_flat = zeros((config.num_heads, config.head_dim), dtype=dtype)
            k_flat = zeros((config.num_kv_heads, config.head_dim), dtype=dtype)

        # GPU position buffer for CUDA Graph replay
        position_buf = zeros((1,), dtype="int32")

        # Fused projection buffers
        q_dim = config.num_heads * config.head_dim
        k_dim = config.num_kv_heads * config.head_dim
        v_dim = config.num_kv_heads * config.head_dim
        qkv_proj_out = zeros((1, q_dim + k_dim + v_dim), dtype=dtype)
        gate_up_out = zeros((1, 2 * config.intermediate_size), dtype=dtype)

        # Pre-create narrow views (avoids object creation overhead in forward loop)
        q_view = qkv_proj_out.narrow(0, q_dim)
        k_view = qkv_proj_out.narrow(q_dim, k_dim)
        v_view = qkv_proj_out.narrow(q_dim + k_dim, v_dim)
        gate_view = gate_up_out.narrow(0, config.intermediate_size)
        up_view = gate_up_out.narrow(config.intermediate_size, config.intermediate_size)

        # Logits buffer for CUDA Graph (optional)
        logits_buf = None
        sampled_token_buf = None
        random_val_buf = None
        token_id_buf = None
        context_len_buf = None
        if vocab_size is not None:
            logits_buf = zeros((1, vocab_size), dtype=dtype)
            sampled_token_buf = zeros((1,), dtype="int32")
            random_val_buf = zeros((1,), dtype="float32")
            token_id_buf = zeros((1,), dtype="int32")
            context_len_buf = zeros((1,), dtype="int32")

        # Batch decode buffers (optional, for zero-allocation batch verify)
        hidden_batch = None
        residual_batch = None
        norm_out_batch = None
        qkv_proj_out_batch = None
        q_batch = None
        k_batch = None
        v_batch = None
        q_t_batch = None
        attn_out_batch = None
        attn_out_t_batch = None
        o_proj_out_batch = None
        gate_up_out_batch = None
        mlp_down_batch = None
        cos_batch = None
        sin_batch = None
        logits_batch = None
        q_flat_batch = None
        k_flat_batch = None

        if max_batch_size > 0:
            hidden_batch = zeros((max_batch_size, config.hidden_size), dtype=dtype)
            residual_batch = zeros((max_batch_size, config.hidden_size), dtype=dtype)
            norm_out_batch = zeros((max_batch_size, config.hidden_size), dtype=dtype)
            qkv_proj_out_batch = zeros((max_batch_size, q_dim + k_dim + v_dim), dtype=dtype)
            q_batch = zeros((max_batch_size, config.num_heads, config.head_dim), dtype=dtype)
            k_batch = zeros((max_batch_size, config.num_kv_heads, config.head_dim), dtype=dtype)
            v_batch = zeros((max_batch_size, config.num_kv_heads, config.head_dim), dtype=dtype)
            q_t_batch = zeros((config.num_heads, max_batch_size, config.head_dim), dtype=dtype)
            attn_out_batch = zeros((config.num_heads, max_batch_size, config.head_dim), dtype=dtype)
            attn_out_t_batch = zeros(
                (max_batch_size, config.num_heads, config.head_dim), dtype=dtype
            )
            o_proj_out_batch = zeros((max_batch_size, config.hidden_size), dtype=dtype)
            gate_up_out_batch = zeros((max_batch_size, 2 * config.intermediate_size), dtype=dtype)
            mlp_down_batch = zeros((max_batch_size, config.hidden_size), dtype=dtype)
            cos_batch = zeros((max_batch_size, config.head_dim), dtype=dtype)
            sin_batch = zeros((max_batch_size, config.head_dim), dtype=dtype)

            if vocab_size is not None:
                logits_batch = zeros((max_batch_size, vocab_size), dtype=dtype)

            if use_qk_norm:
                q_flat_batch = zeros(
                    (max_batch_size * config.num_heads, config.head_dim), dtype=dtype
                )
                k_flat_batch = zeros(
                    (max_batch_size * config.num_kv_heads, config.head_dim), dtype=dtype
                )

        # Batch CUDA Graph buffers (allocated if max_batch_size > 0)
        token_ids_batch_buf = None
        start_position_batch_buf = None
        if max_batch_size > 0:
            token_ids_batch_buf = zeros((max_batch_size,), dtype="int32")
            start_position_batch_buf = zeros((1,), dtype="int32")

        # MoE buffers (allocated if moe_config is provided)
        moe_num_experts = 0
        moe_num_experts_per_tok = 0
        moe_intermediate_size = 0
        moe_router_logits = None
        moe_router_weights = None
        moe_expert_indices = None
        moe_expert_counts = None
        moe_expert_offsets = None
        moe_permute_indices = None
        moe_reverse_perm = None
        moe_row_expert_ids = None
        moe_gathered = None
        moe_gate_out = None
        moe_up_out = None
        moe_intermediate = None
        moe_expert_outputs = None
        moe_output = None

        if moe_config is not None:
            moe_num_experts = moe_config["num_experts"]
            moe_num_experts_per_tok = moe_config["num_experts_per_tok"]
            moe_intermediate_size = moe_config["moe_intermediate_size"]
            moe_k = moe_num_experts_per_tok

            # Router outputs
            moe_router_logits = zeros((1, moe_num_experts), dtype=dtype)
            moe_router_weights = zeros((1, moe_k), dtype=dtype)
            moe_expert_indices = zeros((1, moe_k), dtype="int32")

            # Permutation buffers
            moe_expert_counts = zeros((moe_num_experts,), dtype="int32")
            moe_expert_offsets = zeros((moe_num_experts + 1,), dtype="int32")
            moe_permute_indices = zeros((moe_k,), dtype="int32")
            moe_reverse_perm = zeros((moe_k,), dtype="int32")
            moe_row_expert_ids = zeros((moe_k,), dtype="int32")

            # Expert computation buffers
            moe_gathered = zeros((moe_k, config.hidden_size), dtype=dtype)
            moe_gate_out = zeros((moe_k, moe_intermediate_size), dtype=dtype)
            moe_up_out = zeros((moe_k, moe_intermediate_size), dtype=dtype)
            moe_intermediate = zeros((moe_k, moe_intermediate_size), dtype=dtype)
            moe_expert_outputs = zeros((moe_k, config.hidden_size), dtype=dtype)
            moe_output = zeros((1, config.hidden_size), dtype=dtype)

        return cls(
            hidden=hidden,
            q=q,
            k=k,
            v=v,
            attn_out=attn_out,
            mlp_gate=mlp_gate,
            mlp_up=mlp_up,
            mlp_down=mlp_down,
            q_proj_out=q_proj_out,
            k_proj_out=k_proj_out,
            v_proj_out=v_proj_out,
            o_proj_out=o_proj_out,
            q_t=q_t,
            cos=cos,
            sin=sin,
            embed_out=embed_out,
            residual=residual,
            norm_out=norm_out,
            q_2d=q_2d,
            k_2d=k_2d,
            q_flat=q_flat,
            k_flat=k_flat,
            position_buf=position_buf,
            qkv_proj_out=qkv_proj_out,
            gate_up_out=gate_up_out,
            q_view=q_view,
            k_view=k_view,
            v_view=v_view,
            gate_view=gate_view,
            up_view=up_view,
            logits=logits_buf,
            sampled_token=sampled_token_buf,
            random_val=random_val_buf,
            token_id_buf=token_id_buf,
            context_len_buf=context_len_buf,
            # Batch decode buffers
            max_batch_size=max_batch_size,
            hidden_batch=hidden_batch,
            residual_batch=residual_batch,
            norm_out_batch=norm_out_batch,
            qkv_proj_out_batch=qkv_proj_out_batch,
            q_batch=q_batch,
            k_batch=k_batch,
            v_batch=v_batch,
            q_t_batch=q_t_batch,
            attn_out_batch=attn_out_batch,
            attn_out_t_batch=attn_out_t_batch,
            o_proj_out_batch=o_proj_out_batch,
            gate_up_out_batch=gate_up_out_batch,
            mlp_down_batch=mlp_down_batch,
            cos_batch=cos_batch,
            sin_batch=sin_batch,
            logits_batch=logits_batch,
            q_flat_batch=q_flat_batch,
            k_flat_batch=k_flat_batch,
            token_ids_batch_buf=token_ids_batch_buf,
            start_position_batch_buf=start_position_batch_buf,
            # MoE buffers
            moe_num_experts=moe_num_experts,
            moe_num_experts_per_tok=moe_num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            moe_router_logits=moe_router_logits,
            moe_router_weights=moe_router_weights,
            moe_expert_indices=moe_expert_indices,
            moe_expert_counts=moe_expert_counts,
            moe_expert_offsets=moe_expert_offsets,
            moe_permute_indices=moe_permute_indices,
            moe_reverse_perm=moe_reverse_perm,
            moe_row_expert_ids=moe_row_expert_ids,
            moe_gathered=moe_gathered,
            moe_gate_out=moe_gate_out,
            moe_up_out=moe_up_out,
            moe_intermediate=moe_intermediate,
            moe_expert_outputs=moe_expert_outputs,
            moe_output=moe_output,
        )


# =============================================================================
# Prefill Buffers
# =============================================================================


@dataclass
class PrefillBuffers:
    """Pre-allocated buffers for allocation-free prefill phase.

    Unlike DecodeBuffers (seq_len=1), PrefillBuffers handles variable-length
    sequences up to max_seq_len. Buffers are allocated once and reused.

    Buffer shapes (for Qwen3-8B with max_seq_len=512):
    - hidden: [max_seq_len, hidden_size] - layer input/output
    - q_proj_out: [max_seq_len, num_heads * head_dim] - Q projection (2D)
    - k_proj_out: [max_seq_len, num_kv_heads * head_dim] - K projection (2D)
    - v_proj_out: [max_seq_len, num_kv_heads * head_dim] - V projection (2D)
    - o_proj_out: [max_seq_len, hidden_size] - O projection (2D)
    - q: [max_seq_len, num_heads, head_dim] - Q after reshape (3D)
    - k: [max_seq_len, num_kv_heads, head_dim] - K after reshape (3D)
    - v: [max_seq_len, num_kv_heads, head_dim] - V after reshape (3D)
    - q_t: [num_heads, max_seq_len, head_dim] - Q transposed for SDPA
    - k_t: [num_heads, max_seq_len, head_dim] - K transposed (GQA-expanded)
    - v_t: [num_heads, max_seq_len, head_dim] - V transposed (GQA-expanded)
    - attn_out: [num_heads, max_seq_len, head_dim] - SDPA output
    - attn_out_t: [max_seq_len, num_heads, head_dim] - attention transposed back
    - mlp_gate: [max_seq_len, intermediate_size] - MLP gate output
    - mlp_up: [max_seq_len, intermediate_size] - MLP up output
    - mlp_down: [max_seq_len, hidden_size] - MLP down output
    - residual: [max_seq_len, hidden_size] - residual connection
    - norm_out: [max_seq_len, hidden_size] - normalization output
    """

    max_seq_len: int

    # Main computation buffers
    hidden: GPUArray  # [max_seq_len, hidden_size]
    q: GPUArray  # [max_seq_len, num_heads, head_dim]
    k: GPUArray  # [max_seq_len, num_kv_heads, head_dim]
    v: GPUArray  # [max_seq_len, num_kv_heads, head_dim]

    # Projection outputs (2D for matmul)
    q_proj_out: GPUArray  # [max_seq_len, num_heads * head_dim]
    k_proj_out: GPUArray  # [max_seq_len, num_kv_heads * head_dim]
    v_proj_out: GPUArray  # [max_seq_len, num_kv_heads * head_dim]
    o_proj_out: GPUArray  # [max_seq_len, hidden_size]

    # Transposed buffers for SDPA (GQA-expanded for K, V)
    q_t: GPUArray  # [num_heads, max_seq_len, head_dim]
    k_t: GPUArray  # [num_heads, max_seq_len, head_dim]
    v_t: GPUArray  # [num_heads, max_seq_len, head_dim]

    # Attention output
    attn_out: GPUArray  # [num_heads, max_seq_len, head_dim]
    attn_out_t: GPUArray  # [max_seq_len, num_heads, head_dim]
    attn_out_2d: GPUArray  # [max_seq_len, num_heads * head_dim]

    # MLP buffers
    mlp_gate: GPUArray  # [max_seq_len, intermediate_size]
    mlp_up: GPUArray  # [max_seq_len, intermediate_size]
    mlp_down: GPUArray  # [max_seq_len, hidden_size]

    # RoPE buffers
    cos: GPUArray  # [max_seq_len, head_dim]
    sin: GPUArray  # [max_seq_len, head_dim]

    # Temporary buffers
    residual: GPUArray  # [max_seq_len, hidden_size]
    norm_out: GPUArray  # [max_seq_len, hidden_size]

    # QK Norm buffers (optional, for Qwen3)
    q_2d: GPUArray | None = None  # [max_seq_len * num_heads, head_dim]
    k_2d: GPUArray | None = None  # [max_seq_len * num_kv_heads, head_dim]

    @classmethod
    def allocate(
        cls,
        config: TransformerConfig,
        max_seq_len: int,
        dtype: str = "float16",
        use_qk_norm: bool = False,
    ) -> PrefillBuffers:
        """Allocate all prefill buffers.

        Args:
            config: Model configuration
            max_seq_len: Maximum sequence length for prefill
            dtype: Data type for buffers
            use_qk_norm: Whether to allocate QK norm buffers (Qwen3)
        """
        assert config.num_kv_heads is not None
        assert config.intermediate_size is not None

        # Main buffers
        hidden = zeros((max_seq_len, config.hidden_size), dtype=dtype)
        q = zeros((max_seq_len, config.num_heads, config.head_dim), dtype=dtype)
        k = zeros((max_seq_len, config.num_kv_heads, config.head_dim), dtype=dtype)
        v = zeros((max_seq_len, config.num_kv_heads, config.head_dim), dtype=dtype)

        # Projection outputs (2D)
        q_proj_out = zeros((max_seq_len, config.num_heads * config.head_dim), dtype=dtype)
        k_proj_out = zeros((max_seq_len, config.num_kv_heads * config.head_dim), dtype=dtype)
        v_proj_out = zeros((max_seq_len, config.num_kv_heads * config.head_dim), dtype=dtype)
        o_proj_out = zeros((max_seq_len, config.hidden_size), dtype=dtype)

        # Transposed buffers (GQA-expanded for K, V)
        q_t = zeros((config.num_heads, max_seq_len, config.head_dim), dtype=dtype)
        k_t = zeros((config.num_heads, max_seq_len, config.head_dim), dtype=dtype)
        v_t = zeros((config.num_heads, max_seq_len, config.head_dim), dtype=dtype)

        # Attention output buffers
        attn_out = zeros((config.num_heads, max_seq_len, config.head_dim), dtype=dtype)
        attn_out_t = zeros((max_seq_len, config.num_heads, config.head_dim), dtype=dtype)
        attn_out_2d = zeros((max_seq_len, config.num_heads * config.head_dim), dtype=dtype)

        # MLP buffers
        mlp_gate = zeros((max_seq_len, config.intermediate_size), dtype=dtype)
        mlp_up = zeros((max_seq_len, config.intermediate_size), dtype=dtype)
        mlp_down = zeros((max_seq_len, config.hidden_size), dtype=dtype)

        # RoPE buffers
        cos = zeros((max_seq_len, config.head_dim), dtype=dtype)
        sin = zeros((max_seq_len, config.head_dim), dtype=dtype)

        # Temporary buffers
        residual = zeros((max_seq_len, config.hidden_size), dtype=dtype)
        norm_out = zeros((max_seq_len, config.hidden_size), dtype=dtype)

        # QK Norm buffers (Qwen3)
        q_2d = None
        k_2d = None
        if use_qk_norm:
            q_2d = zeros((max_seq_len * config.num_heads, config.head_dim), dtype=dtype)
            k_2d = zeros((max_seq_len * config.num_kv_heads, config.head_dim), dtype=dtype)

        return cls(
            max_seq_len=max_seq_len,
            hidden=hidden,
            q=q,
            k=k,
            v=v,
            q_proj_out=q_proj_out,
            k_proj_out=k_proj_out,
            v_proj_out=v_proj_out,
            o_proj_out=o_proj_out,
            q_t=q_t,
            k_t=k_t,
            v_t=v_t,
            attn_out=attn_out,
            attn_out_t=attn_out_t,
            attn_out_2d=attn_out_2d,
            mlp_gate=mlp_gate,
            mlp_up=mlp_up,
            mlp_down=mlp_down,
            cos=cos,
            sin=sin,
            residual=residual,
            norm_out=norm_out,
            q_2d=q_2d,
            k_2d=k_2d,
        )
