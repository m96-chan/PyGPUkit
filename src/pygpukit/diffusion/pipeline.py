"""Text-to-Image Pipeline for diffusion models.

Provides a unified interface for generating images from text prompts
using various diffusion models (SD3, Flux, PixArt).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.diffusion.config import (
    FLUX_DEV_SPEC,
    FLUX_SCHNELL_SPEC,
    PIXART_SIGMA_SPEC,
    SD3_MEDIUM_SPEC,
)
from pygpukit.diffusion.models.dit import DiT
from pygpukit.diffusion.models.vae import VAE
from pygpukit.diffusion.scheduler.euler import EulerDiscreteScheduler
from pygpukit.diffusion.scheduler.rectified_flow import FlowMatchingScheduler
from pygpukit.diffusion.text_encoders.clip import CLIPTextEncoder
from pygpukit.diffusion.text_encoders.t5 import T5Encoder

if TYPE_CHECKING:
    from PIL.Image import Image


class Text2ImagePipeline:
    """Unified Text-to-Image Pipeline.

    Supports multiple diffusion model architectures:
    - Stable Diffusion 3 (MMDiT)
    - Flux.1 (Schnell/Dev)
    - PixArt-Sigma

    Example:
        >>> pipe = Text2ImagePipeline.from_pretrained("F:/SD3/sd3-medium")
        >>> image = pipe("A photo of a cat", num_inference_steps=28)
        >>> image.save("cat.png")
    """

    def __init__(
        self,
        transformer: DiT,
        vae: VAE,
        text_encoder: CLIPTextEncoder | None = None,
        text_encoder_2: T5Encoder | None = None,
        scheduler: FlowMatchingScheduler | EulerDiscreteScheduler | None = None,
        model_type: Literal["sd3", "flux", "pixart"] = "sd3",
    ):
        """Initialize pipeline.

        Args:
            transformer: DiT/MMDiT model.
            vae: VAE for encoding/decoding.
            text_encoder: CLIP text encoder.
            text_encoder_2: T5 text encoder (for SD3/Flux).
            scheduler: Noise scheduler.
            model_type: Type of model.
        """
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.scheduler = scheduler or FlowMatchingScheduler()
        self.model_type = model_type

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        dtype: str = "float32",
        model_type: Literal["sd3", "flux", "pixart"] | None = None,
    ) -> Text2ImagePipeline:
        """Load pipeline from pretrained model.

        Args:
            model_path: Path to model directory.
            dtype: Weight dtype.
            model_type: Model type (auto-detected if None).

        Returns:
            Loaded pipeline.
        """
        model_path = Path(model_path)

        # Auto-detect model type
        if model_type is None:
            model_type = cls._detect_model_type(model_path)

        # Load components based on model type
        if model_type == "flux":
            return cls._load_flux(model_path, dtype)
        elif model_type == "sd3":
            return cls._load_sd3(model_path, dtype)
        elif model_type == "pixart":
            return cls._load_pixart(model_path, dtype)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _detect_model_type(path: Path) -> str:
        """Detect model type from directory structure."""
        # Check for Flux indicators
        if (path / "flux1-schnell.safetensors").exists():
            return "flux"
        if (path / "flux1-dev.safetensors").exists():
            return "flux"
        if any("flux" in f.name.lower() for f in path.glob("*.safetensors")):
            return "flux"

        # Check for SD3 indicators
        if (path / "sd3_medium.safetensors").exists():
            return "sd3"
        if any("sd3" in f.name.lower() for f in path.glob("*.safetensors")):
            return "sd3"

        # Check for PixArt indicators
        if any("pixart" in f.name.lower() for f in path.glob("*.safetensors")):
            return "pixart"

        # Default to SD3
        return "sd3"

    @classmethod
    def _load_flux(cls, path: Path, dtype: str) -> Text2ImagePipeline:
        """Load Flux model."""
        # Find transformer weights
        transformer_path = None
        for name in [
            "flux1-dev.safetensors",
            "flux1-schnell.safetensors",
            "transformer.safetensors",
        ]:
            if (path / name).exists():
                transformer_path = path / name
                break

        if transformer_path is None:
            transformer_path = path

        # Detect if Schnell or Dev
        is_schnell = "schnell" in str(transformer_path).lower()
        spec = FLUX_SCHNELL_SPEC if is_schnell else FLUX_DEV_SPEC

        # Load components
        transformer = DiT.from_safetensors(transformer_path, spec=spec, dtype=dtype)

        # VAE
        vae_path = path / "vae"
        if not vae_path.exists():
            vae_path = path
        vae = VAE.from_safetensors(vae_path, dtype=dtype)

        # Text encoders
        clip_path = path / "text_encoder"
        t5_path = path / "text_encoder_2"

        text_encoder = None
        text_encoder_2 = None

        if clip_path.exists():
            text_encoder = CLIPTextEncoder.from_safetensors(clip_path, dtype=dtype)
        if t5_path.exists():
            text_encoder_2 = T5Encoder.from_safetensors(t5_path, dtype=dtype)

        scheduler = FlowMatchingScheduler()

        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            scheduler=scheduler,
            model_type="flux",
        )

    @classmethod
    def _load_sd3(cls, path: Path, dtype: str) -> Text2ImagePipeline:
        """Load SD3 model."""
        transformer_path = None
        for name in ["sd3_medium.safetensors", "transformer.safetensors"]:
            if (path / name).exists():
                transformer_path = path / name
                break

        if transformer_path is None:
            transformer_path = path

        transformer = DiT.from_safetensors(transformer_path, spec=SD3_MEDIUM_SPEC, dtype=dtype)

        # VAE
        vae_path = path / "vae"
        if not vae_path.exists():
            vae_path = path
        vae = VAE.from_safetensors(vae_path, dtype=dtype)

        # Text encoders
        text_encoder = None
        text_encoder_2 = None

        clip_path = path / "text_encoder"
        if clip_path.exists():
            text_encoder = CLIPTextEncoder.from_safetensors(clip_path, dtype=dtype)

        t5_path = path / "text_encoder_3"
        if t5_path.exists():
            text_encoder_2 = T5Encoder.from_safetensors(t5_path, dtype=dtype)

        scheduler = FlowMatchingScheduler()

        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            scheduler=scheduler,
            model_type="sd3",
        )

    @classmethod
    def _load_pixart(cls, path: Path, dtype: str) -> Text2ImagePipeline:
        """Load PixArt model."""
        transformer = DiT.from_safetensors(path, spec=PIXART_SIGMA_SPEC, dtype=dtype)

        vae_path = path / "vae"
        if not vae_path.exists():
            vae_path = path
        vae = VAE.from_safetensors(vae_path, dtype=dtype)

        t5_path = path / "text_encoder"
        text_encoder_2 = None
        if t5_path.exists():
            text_encoder_2 = T5Encoder.from_safetensors(t5_path, dtype=dtype)

        scheduler = EulerDiscreteScheduler()

        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=None,
            text_encoder_2=text_encoder_2,
            scheduler=scheduler,
            model_type="pixart",
        )

    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        seed: int | None = None,
        output_type: Literal["pil", "latent", "array"] = "pil",
        callback: Any | None = None,
    ) -> Image | GPUArray | list[Image]:
        """Generate image from text prompt.

        Args:
            prompt: Text prompt(s).
            negative_prompt: Negative prompt(s) for CFG.
            height: Output image height.
            width: Output image width.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            output_type: Output format ("pil", "latent", "array").
            callback: Optional callback for progress.

        Returns:
            Generated image(s).
        """
        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Handle batch
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # Encode text
        prompt_embeds, pooled_embeds = self._encode_prompt(prompt)

        # Encode negative prompt for CFG
        if guidance_scale > 1.0 and negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            neg_embeds, neg_pooled = self._encode_prompt(negative_prompt)
        else:
            neg_embeds = None
            neg_pooled = None

        # Generate initial noise
        latent_channels = self.vae.spec.latent_channels
        latent_height = height // 8
        latent_width = width // 8

        latents = np.random.randn(batch_size, latent_channels, latent_height, latent_width).astype(
            np.float32
        )
        latents = from_numpy(latents)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Scale initial latents
        if hasattr(self.scheduler, "sigmas_inference"):
            sigma_max = self.scheduler.sigmas_inference[0]
            latents_np = latents.to_numpy() * sigma_max
            latents = from_numpy(latents_np.astype(np.float32))

        # Denoising loop
        timesteps = self.scheduler.timesteps
        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            if guidance_scale > 1.0 and neg_embeds is not None:
                latent_model_input = self._concat_latents(latents, latents)
                encoder_hidden = self._concat_embeds(neg_embeds, prompt_embeds)
                pooled = (
                    self._concat_embeds(neg_pooled, pooled_embeds)
                    if pooled_embeds is not None
                    else None
                )
            else:
                latent_model_input = latents
                encoder_hidden = prompt_embeds
                pooled = pooled_embeds

            # Predict noise/velocity
            noise_pred = self.transformer.forward(
                latent_model_input,
                timestep=float(t),
                encoder_hidden_states=encoder_hidden,
                pooled_projections=pooled,
                guidance=guidance_scale if self.model_type == "flux" else None,
            )

            # CFG
            if guidance_scale > 1.0 and neg_embeds is not None:
                noise_pred_uncond, noise_pred_text = self._split_pred(noise_pred)
                noise_pred = self._cfg_combine(noise_pred_uncond, noise_pred_text, guidance_scale)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents)

            # Callback
            if callback is not None:
                callback(i, len(timesteps), latents)

        # Decode latents
        if output_type == "latent":
            return latents

        image = self.vae.decode(latents)

        if output_type == "array":
            return image

        # Convert to PIL
        return self.vae.to_pil(image)

    def _encode_prompt(
        self,
        prompt: list[str],
    ) -> tuple[GPUArray, GPUArray | None]:
        """Encode text prompt to embeddings."""
        # Use T5 if available (SD3, Flux)
        if self.text_encoder_2 is not None:
            t5_embeds = self.text_encoder_2.encode(prompt)
            prompt_embeds = t5_embeds

            # Get pooled from CLIP if available
            pooled_embeds = None
            if self.text_encoder is not None:
                _, pooled_embeds = self.text_encoder.encode(prompt)

            return prompt_embeds, pooled_embeds

        # Use CLIP only
        if self.text_encoder is not None:
            prompt_embeds, pooled_embeds = self.text_encoder.encode(prompt)
            return prompt_embeds, pooled_embeds

        # Fallback: random embeddings (for testing)
        batch_size = len(prompt)
        hidden_size = self.transformer.spec.text_encoder_dim
        seq_len = 77

        np.random.seed(42)
        prompt_embeds = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32) * 0.02
        pooled_embeds = np.random.randn(batch_size, hidden_size).astype(np.float32) * 0.02

        return from_numpy(prompt_embeds), from_numpy(pooled_embeds)

    def _concat_latents(self, a: GPUArray, b: GPUArray) -> GPUArray:
        """Concatenate latents along batch dimension."""
        a_np = a.to_numpy()
        b_np = b.to_numpy()
        return from_numpy(np.concatenate([a_np, b_np], axis=0).astype(np.float32))

    def _concat_embeds(self, a: GPUArray, b: GPUArray) -> GPUArray:
        """Concatenate embeddings along batch dimension."""
        a_np = a.to_numpy()
        b_np = b.to_numpy()
        return from_numpy(np.concatenate([a_np, b_np], axis=0).astype(np.float32))

    def _split_pred(self, pred: GPUArray) -> tuple[GPUArray, GPUArray]:
        """Split prediction into unconditional and conditional parts."""
        pred_np = pred.to_numpy()
        batch_size = pred_np.shape[0] // 2
        return (
            from_numpy(pred_np[:batch_size].astype(np.float32)),
            from_numpy(pred_np[batch_size:].astype(np.float32)),
        )

    def _cfg_combine(
        self,
        uncond: GPUArray,
        cond: GPUArray,
        scale: float,
    ) -> GPUArray:
        """Combine predictions with classifier-free guidance."""
        u = uncond.to_numpy()
        c = cond.to_numpy()
        result = u + scale * (c - u)
        return from_numpy(result.astype(np.float32))

    @staticmethod
    def create_demo_pipeline(
        model_type: Literal["sd3", "flux", "pixart"] = "sd3",
    ) -> Text2ImagePipeline:
        """Create a demo pipeline with random weights for testing.

        This creates a pipeline that can generate (random) images
        without requiring actual model weights.

        Args:
            model_type: Type of model to simulate.

        Returns:
            Demo pipeline.
        """
        from pygpukit.diffusion.config import (
            FLUX_SCHNELL_SPEC,
            FLUX_VAE_SPEC,
            PIXART_SIGMA_SPEC,
            SD3_MEDIUM_SPEC,
            SD3_VAE_SPEC,
            SDXL_VAE_SPEC,
        )

        # Select transformer and VAE specs based on model type
        if model_type == "flux":
            spec = FLUX_SCHNELL_SPEC
            vae_spec = FLUX_VAE_SPEC
        elif model_type == "pixart":
            spec = PIXART_SIGMA_SPEC
            vae_spec = SDXL_VAE_SPEC  # PixArt uses 4-channel VAE like SDXL
        else:
            spec = SD3_MEDIUM_SPEC
            vae_spec = SD3_VAE_SPEC

        # Create components with empty weights
        transformer = DiT(spec=spec)
        vae = VAE(spec=vae_spec)
        text_encoder = CLIPTextEncoder()
        text_encoder_2 = T5Encoder()

        if model_type == "flux":
            scheduler = FlowMatchingScheduler()
        elif model_type == "pixart":
            scheduler = EulerDiscreteScheduler()
        else:
            scheduler = FlowMatchingScheduler()

        return Text2ImagePipeline(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            scheduler=scheduler,
            model_type=model_type,
        )


__all__ = ["Text2ImagePipeline"]
