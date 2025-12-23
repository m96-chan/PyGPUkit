"""Whisper model loader for SafeTensors format.

Loads Whisper models from HuggingFace format (SafeTensors) and maps
tensor names to PyGPUkit internal structure.

Tensor naming convention in HuggingFace Whisper:
    model.encoder.conv1.weight
    model.encoder.conv2.weight
    model.encoder.embed_positions.weight
    model.encoder.layers.{i}.self_attn.{k,v,q,out}_proj.{weight,bias}
    model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
    model.encoder.layers.{i}.fc1.{weight,bias}
    model.encoder.layers.{i}.fc2.{weight,bias}
    model.encoder.layers.{i}.final_layer_norm.{weight,bias}
    model.encoder.layer_norm.{weight,bias}
    model.decoder.embed_tokens.weight
    model.decoder.embed_positions.weight
    model.decoder.layers.{i}.self_attn.{k,v,q,out}_proj.{weight,bias}
    model.decoder.layers.{i}.self_attn_layer_norm.{weight,bias}
    model.decoder.layers.{i}.encoder_attn.{k,v,q,out}_proj.{weight,bias}
    model.decoder.layers.{i}.encoder_attn_layer_norm.{weight,bias}
    model.decoder.layers.{i}.fc1.{weight,bias}
    model.decoder.layers.{i}.fc2.{weight,bias}
    model.decoder.layers.{i}.final_layer_norm.{weight,bias}
    model.decoder.layer_norm.{weight,bias}
    proj_out.weight (output projection, may be tied to embed_tokens)
"""

import os
from typing import Optional

import numpy as np

from .config import WhisperConfig


def _bfloat16_to_float32(data: bytes, shape: tuple) -> np.ndarray:
    """Convert raw bfloat16 bytes to float32 numpy array.

    bfloat16 is the upper 16 bits of float32, so we just need to
    shift left by 16 bits and view as float32.

    Args:
        data: Raw bytes in bfloat16 format
        shape: Target tensor shape

    Returns:
        float32 numpy array
    """
    # Read as uint16
    bf16 = np.frombuffer(data, dtype=np.uint16)
    # Pad with zeros to create float32 (bfloat16 is upper 16 bits)
    f32_int = bf16.astype(np.uint32) << 16
    # View as float32
    f32 = f32_int.view(np.float32)
    return f32.reshape(shape)


def load_safetensors(file_path: str) -> dict[str, np.ndarray]:
    """Load tensors from SafeTensors file.

    Args:
        file_path: Path to .safetensors file

    Returns:
        Dictionary mapping tensor names to numpy arrays (float32)

    Note:
        bfloat16 tensors are automatically converted to float32 since
        numpy doesn't natively support bfloat16.
    """
    try:
        from safetensors import safe_open
    except ImportError as err:
        raise ImportError(
            "safetensors is required to load models. Install with: pip install safetensors"
        ) from err

    tensors = {}

    # Check if any tensor is bfloat16 by trying to load
    has_bfloat16 = False
    with safe_open(file_path, framework="numpy") as f:
        for key in f.keys():
            try:
                tensors[key] = f.get_tensor(key)
            except TypeError as e:
                if "bfloat16" in str(e):
                    has_bfloat16 = True
                    break
                raise

    # If bfloat16 detected, reload with raw bytes conversion
    if has_bfloat16:
        import json
        import struct

        tensors = {}

        # Read safetensors header to get tensor info
        with open(file_path, "rb") as f:
            # First 8 bytes: header size (uint64 little-endian)
            header_size = struct.unpack("<Q", f.read(8))[0]
            # Read header JSON
            header_json = f.read(header_size).decode("utf-8")
            header = json.loads(header_json)
            # Data starts after header
            data_start = 8 + header_size

            for key, info in header.items():
                if key == "__metadata__":
                    continue

                dtype = info["dtype"]
                shape = info["shape"]
                offsets = info["data_offsets"]
                start, end = offsets

                # Seek to tensor data
                f.seek(data_start + start)
                raw_data = f.read(end - start)

                if dtype == "BF16":
                    tensors[key] = _bfloat16_to_float32(raw_data, tuple(shape))
                elif dtype == "F32":
                    tensors[key] = np.frombuffer(raw_data, dtype=np.float32).reshape(shape)
                elif dtype == "F16":
                    tensors[key] = (
                        np.frombuffer(raw_data, dtype=np.float16).reshape(shape).astype(np.float32)
                    )
                elif dtype == "I64":
                    tensors[key] = np.frombuffer(raw_data, dtype=np.int64).reshape(shape)
                elif dtype == "I32":
                    tensors[key] = np.frombuffer(raw_data, dtype=np.int32).reshape(shape)
                else:
                    raise ValueError(f"Unsupported dtype: {dtype} for tensor {key}")

    return tensors


def download_model(model_id: str, cache_dir: Optional[str] = None) -> str:
    """Download model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "kotoba-tech/kotoba-whisper-v2.0")
        cache_dir: Optional cache directory

    Returns:
        Path to downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise ImportError(
            "huggingface_hub is required to download models. "
            "Install with: pip install huggingface_hub"
        ) from err

    model_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer.*", "vocab.*", "merges.txt"],
    )

    return model_path


class WhisperWeights:
    """Container for Whisper model weights.

    Organizes weights into encoder and decoder components with proper
    tensor mapping from HuggingFace format.
    """

    def __init__(self, config: WhisperConfig):
        self.config = config

        # Encoder weights
        self.encoder_conv1_weight: Optional[np.ndarray] = None
        self.encoder_conv1_bias: Optional[np.ndarray] = None
        self.encoder_conv2_weight: Optional[np.ndarray] = None
        self.encoder_conv2_bias: Optional[np.ndarray] = None
        self.encoder_embed_positions: Optional[np.ndarray] = None
        self.encoder_layers: list = []
        self.encoder_layer_norm_weight: Optional[np.ndarray] = None
        self.encoder_layer_norm_bias: Optional[np.ndarray] = None

        # Decoder weights
        self.decoder_embed_tokens: Optional[np.ndarray] = None
        self.decoder_embed_positions: Optional[np.ndarray] = None
        self.decoder_layers: list = []
        self.decoder_layer_norm_weight: Optional[np.ndarray] = None
        self.decoder_layer_norm_bias: Optional[np.ndarray] = None
        self.proj_out_weight: Optional[np.ndarray] = None

    @classmethod
    def from_safetensors(
        cls, model_path: str, config: Optional[WhisperConfig] = None
    ) -> "WhisperWeights":
        """Load weights from SafeTensors file or directory.

        Args:
            model_path: Path to .safetensors file or model directory
            config: Optional model config (will load from model_path if not provided)

        Returns:
            WhisperWeights instance with loaded tensors
        """
        # Resolve paths
        if os.path.isdir(model_path):
            safetensors_path = os.path.join(model_path, "model.safetensors")
            config_path = os.path.join(model_path, "config.json")
        else:
            safetensors_path = model_path
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

        # Load config if not provided
        if config is None:
            if os.path.exists(config_path):
                config = WhisperConfig.from_json(config_path)
            else:
                raise ValueError(f"Config not provided and config.json not found at {config_path}")

        # Load tensors
        tensors = load_safetensors(safetensors_path)

        # Create weights instance and populate
        weights = cls(config)
        weights._load_encoder_weights(tensors)
        weights._load_decoder_weights(tensors)

        return weights

    def _load_encoder_weights(self, tensors: dict[str, np.ndarray]) -> None:
        """Load encoder weights from tensor dictionary."""
        # Conv layers
        self.encoder_conv1_weight = tensors.get("model.encoder.conv1.weight")
        self.encoder_conv1_bias = tensors.get("model.encoder.conv1.bias")
        self.encoder_conv2_weight = tensors.get("model.encoder.conv2.weight")
        self.encoder_conv2_bias = tensors.get("model.encoder.conv2.bias")

        # Positional embeddings
        self.encoder_embed_positions = tensors.get("model.encoder.embed_positions.weight")

        # Final layer norm
        self.encoder_layer_norm_weight = tensors.get("model.encoder.layer_norm.weight")
        self.encoder_layer_norm_bias = tensors.get("model.encoder.layer_norm.bias")

        # Encoder layers
        self.encoder_layers = []
        for i in range(self.config.encoder_layers):
            layer = self._load_encoder_layer(tensors, i)
            self.encoder_layers.append(layer)

    def _load_encoder_layer(self, tensors: dict[str, np.ndarray], layer_idx: int) -> dict:
        """Load weights for a single encoder layer."""
        prefix = f"model.encoder.layers.{layer_idx}"

        return {
            # Self attention
            "self_attn_q_weight": tensors.get(f"{prefix}.self_attn.q_proj.weight"),
            "self_attn_q_bias": tensors.get(f"{prefix}.self_attn.q_proj.bias"),
            "self_attn_k_weight": tensors.get(f"{prefix}.self_attn.k_proj.weight"),
            "self_attn_k_bias": tensors.get(f"{prefix}.self_attn.k_proj.bias"),
            "self_attn_v_weight": tensors.get(f"{prefix}.self_attn.v_proj.weight"),
            "self_attn_v_bias": tensors.get(f"{prefix}.self_attn.v_proj.bias"),
            "self_attn_out_weight": tensors.get(f"{prefix}.self_attn.out_proj.weight"),
            "self_attn_out_bias": tensors.get(f"{prefix}.self_attn.out_proj.bias"),
            # Self attention layer norm
            "self_attn_layer_norm_weight": tensors.get(f"{prefix}.self_attn_layer_norm.weight"),
            "self_attn_layer_norm_bias": tensors.get(f"{prefix}.self_attn_layer_norm.bias"),
            # FFN
            "fc1_weight": tensors.get(f"{prefix}.fc1.weight"),
            "fc1_bias": tensors.get(f"{prefix}.fc1.bias"),
            "fc2_weight": tensors.get(f"{prefix}.fc2.weight"),
            "fc2_bias": tensors.get(f"{prefix}.fc2.bias"),
            # Final layer norm
            "final_layer_norm_weight": tensors.get(f"{prefix}.final_layer_norm.weight"),
            "final_layer_norm_bias": tensors.get(f"{prefix}.final_layer_norm.bias"),
        }

    def _load_decoder_weights(self, tensors: dict[str, np.ndarray]) -> None:
        """Load decoder weights from tensor dictionary."""
        # Embeddings
        self.decoder_embed_tokens = tensors.get("model.decoder.embed_tokens.weight")
        self.decoder_embed_positions = tensors.get("model.decoder.embed_positions.weight")

        # Final layer norm
        self.decoder_layer_norm_weight = tensors.get("model.decoder.layer_norm.weight")
        self.decoder_layer_norm_bias = tensors.get("model.decoder.layer_norm.bias")

        # Output projection (may be tied to embed_tokens)
        self.proj_out_weight = tensors.get("proj_out.weight")
        if self.proj_out_weight is None:
            # Tied weights - use embed_tokens
            self.proj_out_weight = self.decoder_embed_tokens

        # Decoder layers
        self.decoder_layers = []
        for i in range(self.config.decoder_layers):
            layer = self._load_decoder_layer(tensors, i)
            self.decoder_layers.append(layer)

    def _load_decoder_layer(self, tensors: dict[str, np.ndarray], layer_idx: int) -> dict:
        """Load weights for a single decoder layer."""
        prefix = f"model.decoder.layers.{layer_idx}"

        return {
            # Self attention
            "self_attn_q_weight": tensors.get(f"{prefix}.self_attn.q_proj.weight"),
            "self_attn_q_bias": tensors.get(f"{prefix}.self_attn.q_proj.bias"),
            "self_attn_k_weight": tensors.get(f"{prefix}.self_attn.k_proj.weight"),
            "self_attn_k_bias": tensors.get(f"{prefix}.self_attn.k_proj.bias"),
            "self_attn_v_weight": tensors.get(f"{prefix}.self_attn.v_proj.weight"),
            "self_attn_v_bias": tensors.get(f"{prefix}.self_attn.v_proj.bias"),
            "self_attn_out_weight": tensors.get(f"{prefix}.self_attn.out_proj.weight"),
            "self_attn_out_bias": tensors.get(f"{prefix}.self_attn.out_proj.bias"),
            # Self attention layer norm
            "self_attn_layer_norm_weight": tensors.get(f"{prefix}.self_attn_layer_norm.weight"),
            "self_attn_layer_norm_bias": tensors.get(f"{prefix}.self_attn_layer_norm.bias"),
            # Cross attention (encoder_attn)
            "cross_attn_q_weight": tensors.get(f"{prefix}.encoder_attn.q_proj.weight"),
            "cross_attn_q_bias": tensors.get(f"{prefix}.encoder_attn.q_proj.bias"),
            "cross_attn_k_weight": tensors.get(f"{prefix}.encoder_attn.k_proj.weight"),
            "cross_attn_k_bias": tensors.get(f"{prefix}.encoder_attn.k_proj.bias"),
            "cross_attn_v_weight": tensors.get(f"{prefix}.encoder_attn.v_proj.weight"),
            "cross_attn_v_bias": tensors.get(f"{prefix}.encoder_attn.v_proj.bias"),
            "cross_attn_out_weight": tensors.get(f"{prefix}.encoder_attn.out_proj.weight"),
            "cross_attn_out_bias": tensors.get(f"{prefix}.encoder_attn.out_proj.bias"),
            # Cross attention layer norm
            "cross_attn_layer_norm_weight": tensors.get(f"{prefix}.encoder_attn_layer_norm.weight"),
            "cross_attn_layer_norm_bias": tensors.get(f"{prefix}.encoder_attn_layer_norm.bias"),
            # FFN
            "fc1_weight": tensors.get(f"{prefix}.fc1.weight"),
            "fc1_bias": tensors.get(f"{prefix}.fc1.bias"),
            "fc2_weight": tensors.get(f"{prefix}.fc2.weight"),
            "fc2_bias": tensors.get(f"{prefix}.fc2.bias"),
            # Final layer norm
            "final_layer_norm_weight": tensors.get(f"{prefix}.final_layer_norm.weight"),
            "final_layer_norm_bias": tensors.get(f"{prefix}.final_layer_norm.bias"),
        }

    def summary(self) -> str:
        """Generate a summary of loaded weights."""
        lines = [
            "WhisperWeights Summary:",
            f"  Config: {self.config.d_model}d, {self.config.encoder_layers}enc, {self.config.decoder_layers}dec",
            "  Encoder:",
            f"    - Conv1: {self.encoder_conv1_weight.shape if self.encoder_conv1_weight is not None else 'None'}",
            f"    - Conv2: {self.encoder_conv2_weight.shape if self.encoder_conv2_weight is not None else 'None'}",
            f"    - Layers: {len(self.encoder_layers)}",
            "  Decoder:",
            f"    - Embed tokens: {self.decoder_embed_tokens.shape if self.decoder_embed_tokens is not None else 'None'}",
            f"    - Layers: {len(self.decoder_layers)}",
        ]
        return "\n".join(lines)


def load_whisper_model(
    model_path_or_id: str,
    cache_dir: Optional[str] = None,
) -> tuple[WhisperConfig, WhisperWeights]:
    """Load Whisper model configuration and weights.

    Args:
        model_path_or_id: Local path or HuggingFace model ID
        cache_dir: Optional cache directory for downloads

    Returns:
        Tuple of (WhisperConfig, WhisperWeights)

    Example:
        >>> config, weights = load_whisper_model("kotoba-tech/kotoba-whisper-v2.0")
        >>> print(config)
        >>> print(weights.summary())
    """
    # Check if it's a local path
    if os.path.exists(model_path_or_id):
        model_path = model_path_or_id
    else:
        # Download from HuggingFace
        model_path = download_model(model_path_or_id, cache_dir)

    # Load config
    config = WhisperConfig.from_pretrained(model_path)

    # Load weights
    weights = WhisperWeights.from_safetensors(model_path, config)

    return config, weights


__all__ = [
    "load_safetensors",
    "download_model",
    "WhisperWeights",
    "load_whisper_model",
]
