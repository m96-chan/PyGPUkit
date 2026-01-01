"""Base scheduler class for diffusion models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy


class BaseScheduler(ABC):
    """Abstract base class for diffusion schedulers.

    A scheduler controls the noise schedule and sampling process
    for diffusion models.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        """Initialize scheduler.

        Args:
            num_train_timesteps: Number of training timesteps.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            beta_schedule: Schedule type ("linear", "scaled_linear", "cosine").
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

        # Will be set by set_timesteps
        self.timesteps: np.ndarray | None = None
        self.num_inference_steps: int = 0

        # Compute betas and alphas
        self._compute_schedule()

    def _compute_schedule(self) -> None:
        """Compute the noise schedule."""
        if self.beta_schedule == "linear":
            self.betas = np.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        elif self.beta_schedule == "scaled_linear":
            # Scaling used in SD/SDXL
            self.betas = (
                np.linspace(
                    self.beta_start**0.5,
                    self.beta_end**0.5,
                    self.num_train_timesteps,
                )
                ** 2
            )
        elif self.beta_schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            steps = self.num_train_timesteps + 1
            t = np.linspace(0, self.num_train_timesteps, steps)
            alpha_bar = np.cos((t / self.num_train_timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = np.clip(betas, 0.0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set the number of inference timesteps.

        Args:
            num_inference_steps: Number of steps for inference.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = np.arange(0, num_inference_steps) * step_ratio
        self.timesteps = np.flip(self.timesteps).copy()

    @abstractmethod
    def step(
        self,
        model_output: GPUArray,
        timestep: int,
        sample: GPUArray,
        **kwargs,
    ) -> GPUArray:
        """Perform one scheduler step.

        Args:
            model_output: Output from the denoising model.
            timestep: Current timestep.
            sample: Current noisy sample.

        Returns:
            Denoised sample for the next step.
        """
        pass

    def add_noise(
        self,
        original_samples: GPUArray,
        noise: GPUArray,
        timesteps: np.ndarray | int,
    ) -> GPUArray:
        """Add noise to samples at given timesteps.

        Args:
            original_samples: Clean samples.
            noise: Noise to add.
            timesteps: Timesteps at which to add noise.

        Returns:
            Noisy samples.
        """
        if isinstance(timesteps, int):
            timesteps = np.array([timesteps])

        x = original_samples.to_numpy()
        n = noise.to_numpy()

        # Get alpha_cumprod for timesteps
        sqrt_alpha_prod = np.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = np.sqrt(1.0 - self.alphas_cumprod[timesteps])

        # Reshape for broadcasting
        while sqrt_alpha_prod.ndim < x.ndim:
            sqrt_alpha_prod = sqrt_alpha_prod[..., np.newaxis]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., np.newaxis]

        noisy = sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * n

        return from_numpy(noisy.astype(x.dtype))

    def get_velocity(
        self,
        sample: GPUArray,
        noise: GPUArray,
        timesteps: np.ndarray | int,
    ) -> GPUArray:
        """Get velocity for v-prediction models.

        Args:
            sample: Clean sample.
            noise: Noise.
            timesteps: Timesteps.

        Returns:
            Velocity target.
        """
        if isinstance(timesteps, int):
            timesteps = np.array([timesteps])

        x = sample.to_numpy()
        n = noise.to_numpy()

        sqrt_alpha_prod = np.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = np.sqrt(1.0 - self.alphas_cumprod[timesteps])

        while sqrt_alpha_prod.ndim < x.ndim:
            sqrt_alpha_prod = sqrt_alpha_prod[..., np.newaxis]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., np.newaxis]

        velocity = sqrt_alpha_prod * n - sqrt_one_minus_alpha_prod * x

        return from_numpy(velocity.astype(x.dtype))


__all__ = ["BaseScheduler"]
