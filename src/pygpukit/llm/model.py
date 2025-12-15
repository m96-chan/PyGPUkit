"""LLM model components for PyGPUkit.

Provides transformer building blocks for GPT-2 style models:
- MLP block (fc1 -> gelu -> fc2)
- TransformerBlock (ln -> mlp -> residual)
- GPT2Model (embedding -> blocks -> lm_head)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy, zeros
from pygpukit.ops.basic import add, gelu, layernorm, matmul

if TYPE_CHECKING:
    import numpy as np


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model.

    GPT-2 Small defaults:
        vocab_size=50257, n_embd=768, n_layer=12, n_head=12
    """

    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_positions: int = 1024
    layer_norm_eps: float = 1e-5

    @property
    def n_inner(self) -> int:
        """Inner dimension of MLP (4 * n_embd)."""
        return 4 * self.n_embd


class Linear:
    """Linear layer: y = xW^T + b

    For MVP, we store weight as [out_features, in_features] and transpose
    during forward pass using simple element access.
    """

    def __init__(
        self,
        weight: GPUArray,
        bias: GPUArray | None = None,
    ):
        """Initialize Linear layer.

        Args:
            weight: Weight matrix [out_features, in_features]
            bias: Optional bias vector [out_features]
        """
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {weight.ndim}D")
        self.weight = weight  # [out_features, in_features]
        self.bias = bias
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: y = xW^T + b

        Args:
            x: Input tensor [batch, in_features]

        Returns:
            Output tensor [batch, out_features]
        """
        if x.ndim != 2:
            raise ValueError(f"input must be 2D [batch, in_features], got {x.ndim}D")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"input features {x.shape[1]} doesn't match weight {self.in_features}"
            )

        # For MVP: transpose weight and use matmul
        # x: [batch, in_features]
        # weight: [out_features, in_features]
        # We need: y = x @ weight.T = x @ [in_features, out_features]

        # Simple approach: transpose weight to CPU, create new GPU array
        # This is not optimal but works for MVP
        weight_np = self.weight.to_numpy()
        weight_t = from_numpy(weight_np.T.copy())  # [in_features, out_features]

        # y = x @ weight_t: [batch, in_features] @ [in_features, out_features] = [batch, out_features]
        y = matmul(x, weight_t)

        if self.bias is not None:
            # Add bias: y[i, j] += bias[j]
            # For now, do this on CPU as we don't have broadcast add
            y_np = y.to_numpy()
            bias_np = self.bias.to_numpy()
            y_np += bias_np
            y = from_numpy(y_np)

        return y


class MLP:
    """MLP block for GPT-2.

    Structure: fc1 -> gelu -> fc2
    fc1: [n_embd] -> [n_inner]
    fc2: [n_inner] -> [n_embd]
    """

    def __init__(
        self,
        c_fc_weight: GPUArray,
        c_fc_bias: GPUArray | None,
        c_proj_weight: GPUArray,
        c_proj_bias: GPUArray | None,
    ):
        """Initialize MLP block.

        Args:
            c_fc_weight: First linear weight [n_inner, n_embd]
            c_fc_bias: First linear bias [n_inner]
            c_proj_weight: Second linear weight [n_embd, n_inner]
            c_proj_bias: Second linear bias [n_embd]
        """
        self.c_fc = Linear(c_fc_weight, c_fc_bias)
        self.c_proj = Linear(c_proj_weight, c_proj_bias)

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: fc1 -> gelu -> fc2

        Args:
            x: Input tensor [batch, n_embd]

        Returns:
            Output tensor [batch, n_embd]
        """
        h = self.c_fc(x)
        h = gelu(h)
        h = self.c_proj(h)
        return h


class LayerNorm:
    """Layer normalization with learnable parameters."""

    def __init__(
        self,
        weight: GPUArray,
        bias: GPUArray,
        eps: float = 1e-5,
    ):
        """Initialize LayerNorm.

        Args:
            weight: Scale parameter (gamma) [features]
            bias: Shift parameter (beta) [features]
            eps: Epsilon for numerical stability
        """
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass.

        Args:
            x: Input tensor [batch, features]

        Returns:
            Normalized tensor [batch, features]
        """
        return layernorm(x, self.weight, self.bias, self.eps)


class TransformerBlock:
    """Transformer block (MLP only, no attention for MVP).

    Structure: ln -> mlp -> residual
    """

    def __init__(
        self,
        ln_weight: GPUArray,
        ln_bias: GPUArray,
        mlp: MLP,
        eps: float = 1e-5,
    ):
        """Initialize TransformerBlock.

        Args:
            ln_weight: LayerNorm weight [n_embd]
            ln_bias: LayerNorm bias [n_embd]
            mlp: MLP block
            eps: LayerNorm epsilon
        """
        self.ln = LayerNorm(ln_weight, ln_bias, eps)
        self.mlp = mlp

    def __call__(self, x: GPUArray) -> GPUArray:
        """Forward pass: ln -> mlp -> residual

        Args:
            x: Input tensor [batch, n_embd]

        Returns:
            Output tensor [batch, n_embd]
        """
        # LayerNorm
        h = self.ln(x)
        # MLP
        h = self.mlp(h)
        # Residual
        return add(x, h)


class GPT2Model:
    """GPT-2 model (MLP only, no attention for MVP).

    Structure:
    - Token embedding
    - Position embedding
    - Transformer blocks (MLP only)
    - Final LayerNorm
    - LM head (tied to embedding)
    """

    def __init__(
        self,
        config: GPT2Config,
        wte: GPUArray,  # Token embedding [vocab_size, n_embd]
        wpe: GPUArray,  # Position embedding [n_positions, n_embd]
        blocks: list[TransformerBlock],
        ln_f_weight: GPUArray,
        ln_f_bias: GPUArray,
    ):
        """Initialize GPT-2 model.

        Args:
            config: Model configuration
            wte: Token embedding weights [vocab_size, n_embd]
            wpe: Position embedding weights [n_positions, n_embd]
            blocks: List of transformer blocks
            ln_f_weight: Final LayerNorm weight
            ln_f_bias: Final LayerNorm bias
        """
        self.config = config
        self.wte = wte
        self.wpe = wpe
        self.blocks = blocks
        self.ln_f = LayerNorm(ln_f_weight, ln_f_bias, config.layer_norm_eps)

    def __call__(self, input_ids: list[int], position_ids: list[int] | None = None) -> GPUArray:
        """Forward pass.

        Args:
            input_ids: Token IDs [seq_len]
            position_ids: Optional position IDs [seq_len]

        Returns:
            Hidden states [seq_len, n_embd]
        """
        import numpy as np

        seq_len = len(input_ids)

        if position_ids is None:
            position_ids = list(range(seq_len))

        # Get embeddings by indexing (CPU for MVP)
        wte_np = self.wte.to_numpy()
        wpe_np = self.wpe.to_numpy()

        # Token embeddings: select rows from wte
        token_embeds = wte_np[input_ids]  # [seq_len, n_embd]

        # Position embeddings: select rows from wpe
        pos_embeds = wpe_np[position_ids]  # [seq_len, n_embd]

        # Combine embeddings
        hidden = from_numpy((token_embeds + pos_embeds).astype(np.float32))

        # Apply transformer blocks
        for block in self.blocks:
            hidden = block(hidden)

        # Final LayerNorm
        hidden = self.ln_f(hidden)

        return hidden

    def lm_head(self, hidden: GPUArray) -> GPUArray:
        """Compute logits from hidden states.

        Args:
            hidden: Hidden states [seq_len, n_embd]

        Returns:
            Logits [seq_len, vocab_size]
        """
        # LM head is tied to embedding weights
        # logits = hidden @ wte.T
        wte_np = self.wte.to_numpy()
        hidden_np = hidden.to_numpy()
        logits = hidden_np @ wte_np.T
        return from_numpy(logits.astype(hidden_np.dtype))

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> list[int]:
        """Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = greedy argmax)

        Returns:
            List of all token IDs (input + generated)
        """
        import numpy as np

        tokens = list(input_ids)

        for _ in range(max_new_tokens):
            # Truncate to max context length
            context = tokens[-self.config.n_positions:]

            # Forward pass
            hidden = self(context)

            # Get logits for last position
            logits = self.lm_head(hidden)
            logits_np = logits.to_numpy()
            last_logits = logits_np[-1]  # [vocab_size]

            # Apply temperature
            if temperature != 1.0:
                last_logits = last_logits / temperature

            # Greedy decoding (argmax)
            next_token = int(np.argmax(last_logits))
            tokens.append(next_token)

            # Stop at EOS (50256 for GPT-2)
            if next_token == 50256:
                break

        return tokens


def load_gpt2_from_safetensors(
    model_path: str,
    config: GPT2Config | None = None,
) -> GPT2Model:
    """Load GPT-2 model from safetensors file.

    Note: This is an MVP that only loads MLP weights (no attention).
    The model will not produce coherent text without attention.

    Args:
        model_path: Path to model.safetensors file
        config: Model configuration (defaults to GPT-2 small)

    Returns:
        GPT2Model instance
    """
    from pygpukit.llm import SafeTensorsFile

    if config is None:
        config = GPT2Config()

    st = SafeTensorsFile(model_path)

    # Helper to load tensor
    def load_tensor(name: str) -> GPUArray:
        data = st.tensor_bytes(name)
        info = st.tensor_info(name)

        import numpy as np

        # Determine numpy dtype
        dtype_map = {
            0: np.float32,  # Float32
            1: np.float16,  # Float16
            2: np.float32,  # BFloat16 -> convert to float32 for now
            3: np.float64,  # Float64
        }
        np_dtype = dtype_map.get(info.dtype, np.float32)

        # Create numpy array from bytes
        arr = np.frombuffer(data, dtype=np_dtype).reshape(info.shape)
        return from_numpy(arr.copy())

    # Load embeddings
    wte = load_tensor("wte.weight")
    wpe = load_tensor("wpe.weight")

    # Load blocks
    blocks = []
    for i in range(config.n_layer):
        prefix = f"h.{i}."

        # Check if MLP weights exist
        mlp_c_fc_w_name = f"{prefix}mlp.c_fc.weight"
        if mlp_c_fc_w_name not in st.tensor_names:
            # Skip blocks without MLP (shouldn't happen for GPT-2)
            continue

        # LayerNorm 2 (before MLP in GPT-2)
        ln_2_w = load_tensor(f"{prefix}ln_2.weight")
        ln_2_b = load_tensor(f"{prefix}ln_2.bias")

        # MLP
        mlp_c_fc_w = load_tensor(f"{prefix}mlp.c_fc.weight")
        mlp_c_fc_b = load_tensor(f"{prefix}mlp.c_fc.bias")
        mlp_c_proj_w = load_tensor(f"{prefix}mlp.c_proj.weight")
        mlp_c_proj_b = load_tensor(f"{prefix}mlp.c_proj.bias")

        mlp = MLP(mlp_c_fc_w, mlp_c_fc_b, mlp_c_proj_w, mlp_c_proj_b)
        block = TransformerBlock(ln_2_w, ln_2_b, mlp, config.layer_norm_eps)
        blocks.append(block)

    # Final LayerNorm
    ln_f_w = load_tensor("ln_f.weight")
    ln_f_b = load_tensor("ln_f.bias")

    return GPT2Model(config, wte, wpe, blocks, ln_f_w, ln_f_b)
