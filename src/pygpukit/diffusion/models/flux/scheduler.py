"""Flow Matching Euler scheduler for FLUX.

Implements the flow matching scheduler used by FLUX.1 models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlowMatchEulerSchedulerConfig:
    """Configuration for Flow Match Euler scheduler."""

    num_train_timesteps: int = 1000
    shift: float = 1.0
    use_dynamic_shifting: bool = False
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096


class FlowMatchEulerScheduler:
    """Flow Matching Euler Discrete Scheduler.

    This scheduler implements the flow matching objective used in FLUX.1.
    It's simpler than DDPM-based schedulers and only requires forward Euler steps.

    The flow is defined as:
        x_t = (1 - sigma) * x_0 + sigma * noise
    where sigma goes from 1 (pure noise) to 0 (clean image).

    The model predicts the velocity (dx/dt) and we integrate using Euler method.
    """

    def __init__(self, config: FlowMatchEulerSchedulerConfig | None = None):
        """Initialize scheduler.

        Args:
            config: Scheduler configuration.
        """
        self.config = config or FlowMatchEulerSchedulerConfig()

        self.num_inference_steps: int | None = None
        self.timesteps: np.ndarray | None = None
        self.sigmas: np.ndarray | None = None
        self._step_index: int | None = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        mu: float | None = None,
    ) -> None:
        """Set the discrete timesteps for inference.

        Args:
            num_inference_steps: Number of denoising steps.
            mu: Optional shift parameter for dynamic shifting.
        """
        self.num_inference_steps = num_inference_steps

        # Generate timesteps from 1.0 to 0.0 (sigma schedule)
        # FLUX uses linear spacing in sigma space
        timesteps = np.linspace(1.0, 0.0, num_inference_steps + 1)

        # Apply shift warping: sigma' = shift * sigma / (1 + (shift - 1) * sigma)
        # This controls the noise level distribution
        shift = self.config.shift
        if shift != 1.0:
            timesteps = shift * timesteps / (1.0 + (shift - 1.0) * timesteps)

        # Optional dynamic shifting based on image resolution
        if self.config.use_dynamic_shifting and mu is not None:
            timesteps = self._time_shift(mu, 1.0, timesteps)

        self.sigmas = timesteps.astype(np.float32)
        # Convert sigmas to timesteps (for model input, typically sigma * 1000)
        self.timesteps = (self.sigmas[:-1] * self.config.num_train_timesteps).astype(np.float32)

        self._step_index = 0

    def _time_shift(
        self,
        mu: float,
        sigma: float,
        t: np.ndarray,
    ) -> np.ndarray:
        """Apply exponential time shifting based on resolution.

        Args:
            mu: Resolution-dependent shift (computed from image_seq_len).
            sigma: Base sigma (typically 1.0).
            t: Timesteps to shift.

        Returns:
            Shifted timesteps.
        """
        return np.exp(mu) / (np.exp(mu) + (1 / t - 1) ** sigma)

    def compute_mu(self, image_seq_len: int) -> float:
        """Compute mu for dynamic shifting based on image size.

        Args:
            image_seq_len: Number of image tokens (height * width).

        Returns:
            Computed mu value.
        """
        # Linear interpolation between base_shift and max_shift
        # based on image sequence length
        m = (self.config.max_shift - self.config.base_shift) / (
            self.config.max_image_seq_len - self.config.base_image_seq_len
        )
        b = self.config.base_shift - m * self.config.base_image_seq_len
        mu = image_seq_len * m + b
        return mu

    @property
    def step_index(self) -> int | None:
        """Current step index."""
        return self._step_index

    def step(
        self,
        model_output: np.ndarray,
        timestep: float,
        sample: np.ndarray,
    ) -> np.ndarray:
        """Perform one denoising step.

        Args:
            model_output: Predicted velocity from the model [B, seq_len, channels].
            timestep: Current timestep (sigma value).
            sample: Current noisy sample [B, seq_len, channels].

        Returns:
            Denoised sample for the next step.
        """
        if self._step_index is None or self.sigmas is None:
            raise ValueError("Timesteps not set. Call set_timesteps() first.")

        # Get current and next sigma
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        # Euler step: x_{t+1} = x_t + (sigma_next - sigma) * model_output
        # Since sigma decreases, dt = sigma_next - sigma is negative
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output

        # Increment step index
        self._step_index += 1

        return prev_sample.astype(np.float32)

    def add_noise(
        self,
        original_samples: np.ndarray,
        noise: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        """Add noise to samples for a given timestep.

        Used for flow matching training or inpainting.

        Args:
            original_samples: Clean samples [B, seq_len, channels].
            noise: Noise to add [B, seq_len, channels].
            timestep: Sigma value (0 = clean, 1 = pure noise).

        Returns:
            Noisy samples.
        """
        # Flow matching interpolation: x_t = (1 - t) * x_0 + t * noise
        sigma = timestep
        noisy = (1.0 - sigma) * original_samples + sigma * noise
        return noisy.astype(np.float32)

    def scale_model_input(
        self,
        sample: np.ndarray,
        timestep: float | None = None,
    ) -> np.ndarray:
        """Scale model input (identity for flow matching).

        Args:
            sample: Input sample.
            timestep: Current timestep (unused).

        Returns:
            Unmodified sample.
        """
        # Flow matching doesn't require input scaling
        return sample


__all__ = ["FlowMatchEulerScheduler", "FlowMatchEulerSchedulerConfig"]
