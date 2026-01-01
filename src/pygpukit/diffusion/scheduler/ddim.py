"""DDIM Scheduler.

Denoising Diffusion Implicit Models scheduler for
deterministic sampling with fewer steps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.diffusion.scheduler.base import BaseScheduler


class DDIMScheduler(BaseScheduler):
    """DDIM Scheduler for diffusion models.

    Implements deterministic (eta=0) and stochastic (eta>0) sampling.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        eta: float = 0.0,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        """Initialize DDIM scheduler.

        Args:
            num_train_timesteps: Number of training timesteps.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            beta_schedule: Schedule type.
            prediction_type: What the model predicts ("epsilon", "v_prediction", "sample").
            eta: Stochasticity parameter (0 = deterministic DDIM).
            clip_sample: Whether to clip predicted x0.
            clip_sample_range: Range for clipping.
        """
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)
        self.prediction_type = prediction_type
        self.eta = eta
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set inference timesteps.

        Args:
            num_inference_steps: Number of inference steps.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = np.arange(0, num_inference_steps) * step_ratio
        self.timesteps = np.flip(self.timesteps).astype(np.int64).copy()

    def step(
        self,
        model_output: GPUArray,
        timestep: int,
        sample: GPUArray,
        **kwargs: Any,
    ) -> GPUArray:
        """Perform one DDIM step.

        Args:
            model_output: Model prediction.
            timestep: Current timestep.
            sample: Current noisy sample.
            **kwargs: Additional arguments (generator for stochastic sampling).

        Returns:
            Denoised sample for next step.
        """
        generator: np.random.Generator | None = kwargs.get("generator")
        # Find current and previous timesteps
        step_index = np.where(self.timesteps == timestep)[0][0]
        prev_timestep = (
            self.timesteps[step_index + 1] if step_index < len(self.timesteps) - 1 else 0
        )

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else 1.0

        x = sample.to_numpy()
        eps = model_output.to_numpy()

        # Convert to predicted x0
        if self.prediction_type == "epsilon":
            pred_x0 = (x - np.sqrt(1 - alpha_prod_t) * eps) / np.sqrt(alpha_prod_t)
        elif self.prediction_type == "v_prediction":
            pred_x0 = np.sqrt(alpha_prod_t) * x - np.sqrt(1 - alpha_prod_t) * eps
        elif self.prediction_type == "sample":
            pred_x0 = eps
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Clip predicted x0
        if self.clip_sample:
            pred_x0 = np.clip(pred_x0, -self.clip_sample_range, self.clip_sample_range)

        # Compute variance for stochastic sampling
        variance = (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        )
        std_dev_t = self.eta * np.sqrt(variance)

        # Direction pointing to x_t
        pred_epsilon = (x - np.sqrt(alpha_prod_t) * pred_x0) / np.sqrt(1 - alpha_prod_t)

        # Compute x_{t-1}
        pred_sample_direction = np.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * pred_epsilon
        x_prev = np.sqrt(alpha_prod_t_prev) * pred_x0 + pred_sample_direction

        # Add noise for stochastic sampling
        if self.eta > 0:
            if generator is None:
                noise = np.random.randn(*x.shape)
            else:
                noise = generator.standard_normal(x.shape)
            x_prev = x_prev + std_dev_t * noise

        return from_numpy(x_prev.astype(x.dtype))


__all__ = ["DDIMScheduler"]
