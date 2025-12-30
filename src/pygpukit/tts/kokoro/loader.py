"""Model loading utilities for Kokoro TTS.

Handles loading weights from SafeTensors or PyTorch (.pth) format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy

if TYPE_CHECKING:
    pass


def _download_model(repo_id: str, local_dir: Path | None = None) -> Path:
    """Download model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "hexgrad/Kokoro-82M")
        local_dir: Local directory to save files (optional)

    Returns:
        Path to downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=["*.json", "*.pth", "*.safetensors", "voices/*.pt"],
            )
        )
    except ImportError as err:
        raise ImportError(
            "huggingface_hub is required to download models. "
            "Install with: pip install huggingface_hub"
        ) from err


def _load_pytorch_weights(path: Path) -> dict[str, np.ndarray]:
    """Load weights from PyTorch .pth file.

    Args:
        path: Path to .pth file

    Returns:
        Dictionary mapping weight names to numpy arrays
    """
    try:
        import torch

        # Load with CPU mapping
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

        # Convert to numpy
        weights = {}
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                weights[name] = tensor.numpy()
            elif isinstance(tensor, np.ndarray):
                weights[name] = tensor

        return weights
    except ImportError as err:
        raise ImportError(
            "PyTorch is required to load .pth files. Install with: pip install torch"
        ) from err


def _load_safetensors_weights(path: Path) -> dict[str, np.ndarray]:
    """Load weights from SafeTensors file.

    Args:
        path: Path to .safetensors file

    Returns:
        Dictionary mapping weight names to numpy arrays
    """
    try:
        from safetensors import safe_open

        weights = {}
        with safe_open(path, framework="numpy") as f:
            for name in f.keys():
                weights[name] = f.get_tensor(name)

        return weights
    except ImportError as err:
        raise ImportError(
            "safetensors is required to load .safetensors files. "
            "Install with: pip install safetensors"
        ) from err


def _convert_to_gpu(
    weights: dict[str, np.ndarray],
    dtype: str = "bfloat16",
) -> dict[str, GPUArray]:
    """Convert numpy weights to GPUArrays.

    Args:
        weights: Dictionary of numpy arrays
        dtype: Target dtype ("bfloat16" or "float32")

    Returns:
        Dictionary of GPUArrays
    """
    gpu_weights = {}
    for name, array in weights.items():
        # Convert to float32 first if needed
        if array.dtype not in (np.float32, np.float16):
            array = array.astype(np.float32)

        # Create GPUArray
        gpu_array = from_numpy(array)

        # Cast to target dtype if needed
        if dtype == "bfloat16" and array.dtype == np.float32:
            # Cast on GPU
            from pygpukit.ops.tensor import cast_f32_to_bf16

            gpu_array = cast_f32_to_bf16(gpu_array)

        gpu_weights[name] = gpu_array

    return gpu_weights


def load_voice_embedding(
    voice_path: Path,
) -> GPUArray:
    """Load speaker/voice embedding from .pt file.

    Args:
        voice_path: Path to voice .pt file

    Returns:
        GPUArray containing voice embedding
    """
    try:
        import torch

        voice_data = torch.load(voice_path, map_location="cpu", weights_only=False)

        # Voice files contain style embedding tensor
        if isinstance(voice_data, torch.Tensor):
            embedding = voice_data.numpy()
        elif isinstance(voice_data, dict) and "style" in voice_data:
            embedding = voice_data["style"].numpy()
        else:
            raise ValueError(f"Unexpected voice file format: {type(voice_data)}")

        return from_numpy(embedding.astype(np.float32))
    except ImportError as err:
        raise ImportError(
            "PyTorch is required to load voice files. Install with: pip install torch"
        ) from err


def list_available_voices(model_path: Path) -> list[str]:
    """List available voice embeddings in model directory.

    Args:
        model_path: Path to model directory

    Returns:
        List of voice names (without .pt extension)
    """
    voices_dir = model_path / "voices"
    if not voices_dir.exists():
        return []

    voices = []
    for pt_file in voices_dir.glob("*.pt"):
        voices.append(pt_file.stem)

    return sorted(voices)


def load_kokoro_weights(
    model_path: str | Path,
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> tuple[dict[str, GPUArray], dict[str, Any]]:
    """Load Kokoro model weights and config.

    Args:
        model_path: Path to model directory or HuggingFace repo ID
        dtype: Weight dtype ("bfloat16" or "float32")
        device: Target device (currently only "cuda" supported)

    Returns:
        Tuple of (weights dict, config dict)
    """
    model_path = Path(model_path)

    # Download from HuggingFace if needed
    if not model_path.exists():
        model_path = _download_model(str(model_path))

    # Find weight file
    safetensors_path = model_path / "kokoro-v1_0.safetensors"
    pth_path = model_path / "kokoro-v1_0.pth"

    if safetensors_path.exists():
        weights = _load_safetensors_weights(safetensors_path)
    elif pth_path.exists():
        weights = _load_pytorch_weights(pth_path)
    else:
        # Try other common names
        for pattern in ["*.safetensors", "*.pth"]:
            files = list(model_path.glob(pattern))
            if files:
                if pattern == "*.safetensors":
                    weights = _load_safetensors_weights(files[0])
                else:
                    weights = _load_pytorch_weights(files[0])
                break
        else:
            raise FileNotFoundError(
                f"No weight file found in {model_path}. "
                "Expected kokoro-v1_0.safetensors or kokoro-v1_0.pth"
            )

    # Load config
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)
    else:
        config_dict = {}

    # Convert to GPU
    gpu_weights = _convert_to_gpu(weights, dtype=dtype)

    return gpu_weights, config_dict


def get_weight_info(weights: dict[str, GPUArray | np.ndarray]) -> dict[str, dict[str, Any]]:
    """Get information about model weights.

    Args:
        weights: Dictionary of weights

    Returns:
        Dictionary with shape and dtype info for each weight
    """
    info = {}
    for name, w in weights.items():
        if isinstance(w, GPUArray):
            info[name] = {
                "shape": w.shape,
                "dtype": str(w.dtype),
                "size_mb": w.nbytes / (1024 * 1024),
            }
        elif isinstance(w, np.ndarray):
            info[name] = {
                "shape": w.shape,
                "dtype": str(w.dtype),
                "size_mb": w.nbytes / (1024 * 1024),
            }
    return info


def print_weight_summary(weights: dict[str, GPUArray | np.ndarray]) -> None:
    """Print summary of model weights.

    Args:
        weights: Dictionary of weights
    """
    info = get_weight_info(weights)

    total_params = 0
    total_size_mb = 0.0

    print("=" * 60)
    print("Kokoro Model Weight Summary")
    print("=" * 60)

    # Group by prefix
    prefixes: dict[str, list[str]] = {}
    for name in sorted(info.keys()):
        prefix = name.split(".")[0] if "." in name else name
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(name)

    for prefix, names in prefixes.items():
        prefix_params = 0
        prefix_size = 0.0

        for name in names:
            shape = info[name]["shape"]
            params = 1
            for dim in shape:
                params *= dim
            prefix_params += params
            prefix_size += info[name]["size_mb"]

        print(f"{prefix}: {prefix_params:,} params ({prefix_size:.2f} MB)")
        total_params += prefix_params
        total_size_mb += prefix_size

    print("-" * 60)
    print(f"Total: {total_params:,} params ({total_size_mb:.2f} MB)")
    print("=" * 60)


__all__ = [
    "load_kokoro_weights",
    "load_voice_embedding",
    "list_available_voices",
    "get_weight_info",
    "print_weight_summary",
]
