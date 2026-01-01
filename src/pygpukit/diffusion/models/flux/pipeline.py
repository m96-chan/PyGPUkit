"""FLUX generation pipeline.

End-to-end text-to-image generation using FLUX transformer.
Uses external text encoders (CLIP + T5) and VAE from transformers/diffusers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pygpukit.core.factory import from_numpy
from pygpukit.diffusion.models.flux.embeddings import prepare_image_ids, prepare_text_ids
from pygpukit.diffusion.models.flux.model import FluxTransformer
from pygpukit.diffusion.models.flux.scheduler import (
    FlowMatchEulerScheduler,
    FlowMatchEulerSchedulerConfig,
)

if TYPE_CHECKING:
    from PIL import Image


class FluxPipeline:
    """FLUX text-to-image generation pipeline.

    This pipeline uses:
    - Our FluxTransformer implementation
    - External CLIP text encoder (from transformers)
    - External T5 text encoder (from transformers)
    - External VAE (from diffusers)

    Example:
        >>> pipe = FluxPipeline.from_pretrained("F:/ImageGenerate/flux1-schnell-full")
        >>> image = pipe("A cute orange cat sitting on grass", num_steps=4)
        >>> image.save("cat.png")
    """

    def __init__(
        self,
        transformer: FluxTransformer,
        scheduler: FlowMatchEulerScheduler,
        vae: object,  # AutoencoderKL from diffusers
        text_encoder: object,  # CLIPTextModel
        text_encoder_2: object,  # T5EncoderModel
        tokenizer: object,  # CLIPTokenizer
        tokenizer_2: object,  # T5Tokenizer
    ):
        """Initialize pipeline.

        Args:
            transformer: FLUX transformer model.
            scheduler: Flow matching scheduler.
            vae: VAE for latent encoding/decoding.
            text_encoder: CLIP text encoder.
            text_encoder_2: T5 text encoder.
            tokenizer: CLIP tokenizer.
            tokenizer_2: T5 tokenizer.
        """
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        # VAE scaling factors for FLUX
        # FLUX VAE: 8x downsampling with 16 channels
        # Then 2x2 packing gives 16x effective downsampling with 64 channels
        self.vae_scale_factor = 16  # Effective downsampling after packing
        self.latent_channels = 64  # Channels after packing (16 * 2 * 2)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        dtype: str = "float32",
    ) -> FluxPipeline:
        """Load pipeline from pretrained model.

        Args:
            path: Path to model directory (HuggingFace cache or local).
            dtype: Model dtype.

        Returns:
            Loaded pipeline.
        """
        import torch
        from diffusers import AutoencoderKL
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

        path = Path(path)

        # Find the actual model path in HuggingFace cache structure
        cache_path = path / "models--black-forest-labs--FLUX.1-schnell"
        if cache_path.exists():
            snapshots = list((cache_path / "snapshots").iterdir())
            if snapshots:
                model_path = snapshots[0]
            else:
                model_path = path
        else:
            model_path = path

        torch_dtype = torch.float32 if dtype == "float32" else torch.float16

        # Load transformer (our implementation)
        transformer = FluxTransformer.from_safetensors(model_path / "transformer", dtype=dtype)

        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            model_path / "vae",
            torch_dtype=torch_dtype,
        )

        # Load text encoders
        text_encoder = CLIPTextModel.from_pretrained(
            model_path / "text_encoder",
            torch_dtype=torch_dtype,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path / "text_encoder_2",
            torch_dtype=torch_dtype,
        )

        # Load tokenizers
        tokenizer = CLIPTokenizer.from_pretrained(model_path / "tokenizer")
        tokenizer_2 = T5Tokenizer.from_pretrained(model_path / "tokenizer_2")

        # Create scheduler
        scheduler = FlowMatchEulerScheduler(FlowMatchEulerSchedulerConfig())

        return cls(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
        )

    def encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode text prompt using CLIP and T5.

        Args:
            prompt: Text prompt.
            max_sequence_length: Maximum T5 sequence length.

        Returns:
            Tuple of (pooled_clip_embedding, t5_embeddings).
        """
        import torch

        device = next(self.text_encoder.parameters()).device

        # CLIP encoding (for pooled embedding)
        clip_inputs = self.tokenizer(  # type: ignore[operator]
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}

        with torch.no_grad():
            clip_outputs = self.text_encoder(**clip_inputs)  # type: ignore[operator]
            pooled_embed = clip_outputs.pooler_output  # [1, 768]

        # T5 encoding (for sequence embedding)
        t5_inputs = self.tokenizer_2(  # type: ignore[operator]
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_inputs = {k: v.to(device) for k, v in t5_inputs.items()}

        with torch.no_grad():
            t5_outputs = self.text_encoder_2(**t5_inputs)  # type: ignore[operator]
            t5_embed = t5_outputs.last_hidden_state  # [1, seq_len, 4096]

        return pooled_embed.cpu().numpy(), t5_embed.cpu().numpy()

    def _unpack_latents(
        self,
        latents: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Unpack 64-channel packed latents to 16-channel VAE format.

        Args:
            latents: Packed latents [B, seq_len, 64].
            height: Target image height.
            width: Target image width.

        Returns:
            Unpacked latents [B, 16, H/8, W/8].
        """
        B = latents.shape[0]
        # FLUX uses 16x effective downsampling (8x VAE + 2x packing)
        h = height // 16
        w = width // 16

        # Reshape to spatial: [B, h, w, 64]
        latents = latents.reshape(B, h, w, 64)

        # Unpack: 64 -> 16 channels with 2x spatial expansion
        # [B, h, w, 64] -> [B, h, w, 16, 2, 2] -> [B, h*2, w*2, 16]
        latents = latents.reshape(B, h, w, 16, 2, 2)
        latents = latents.transpose(0, 1, 4, 2, 5, 3)  # [B, h, 2, w, 2, 16]
        latents = latents.reshape(B, h * 2, w * 2, 16)

        # Convert to NCHW: [B, 16, H/8, W/8]
        latents = latents.transpose(0, 3, 1, 2)

        return latents

    def decode_latents(self, latents: np.ndarray, height: int, width: int) -> np.ndarray:
        """Decode latents to image using VAE.

        Args:
            latents: Packed latent tensor [B, seq_len, 64].
            height: Target image height.
            width: Target image width.

        Returns:
            Decoded image [B, H, W, 3] in [0, 255] range.
        """
        import torch

        device = next(self.vae.parameters()).device

        # Unpack 64-channel to 16-channel VAE format
        latents = self._unpack_latents(latents, height, width)

        # Scale latents
        latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, "shift_factor"):
            latents = latents + self.vae.config.shift_factor

        latents_torch = torch.from_numpy(latents.astype(np.float32)).to(device)

        with torch.no_grad():
            image = self.vae.decode(latents_torch).sample

        # Convert to numpy and scale to [0, 255]
        image = image.cpu().numpy()
        image = (image + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image = image.transpose(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        return image

    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,  # schnell doesn't use guidance
        seed: int | None = None,
        max_sequence_length: int = 512,
    ) -> Image.Image:
        """Generate image from text prompt.

        Args:
            prompt: Text prompt.
            height: Image height (must be divisible by 16).
            width: Image width (must be divisible by 16).
            num_inference_steps: Number of denoising steps (4 for schnell).
            guidance_scale: Guidance scale (0.0 for schnell).
            seed: Random seed.
            max_sequence_length: Maximum T5 sequence length.

        Returns:
            Generated PIL Image.
        """
        from PIL import Image

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Validate dimensions
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Height and width must be divisible by 16, got {height}x{width}")

        # Encode prompt
        pooled_embed, t5_embed = self.encode_prompt(prompt, max_sequence_length)

        # Compute latent dimensions
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_seq_len = latent_h * latent_w

        # Prepare position IDs
        img_ids = prepare_image_ids(1, latent_h, latent_w)
        txt_ids = prepare_text_ids(1, t5_embed.shape[1])

        # Initialize latents with random noise
        latents = np.random.randn(1, latent_seq_len, self.latent_channels).astype(np.float32)

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for _i, t in enumerate(self.scheduler.timesteps):
            # Prepare timestep
            timestep = np.array([t], dtype=np.float32)

            # Forward pass through transformer
            noise_pred = self.transformer.forward(
                hidden_states=from_numpy(latents),
                encoder_hidden_states=from_numpy(t5_embed.astype(np.float32)),
                pooled_projections=from_numpy(pooled_embed.astype(np.float32)),
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
            ).to_numpy()

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents)

        # Decode latents to image
        image_np = self.decode_latents(latents, height, width)

        # Convert to PIL Image
        return Image.fromarray(image_np[0])


def generate(
    prompt: str,
    model_path: str = "F:/ImageGenerate/flux1-schnell-full",
    height: int = 512,
    width: int = 512,
    num_steps: int = 4,
    seed: int | None = None,
) -> Image.Image:
    """Quick generation function.

    Args:
        prompt: Text prompt.
        model_path: Path to FLUX model.
        height: Image height.
        width: Image width.
        num_steps: Number of inference steps.
        seed: Random seed.

    Returns:
        Generated PIL Image.
    """
    pipe = FluxPipeline.from_pretrained(model_path)
    return pipe(prompt, height=height, width=width, num_inference_steps=num_steps, seed=seed)


__all__ = ["FluxPipeline", "generate"]
