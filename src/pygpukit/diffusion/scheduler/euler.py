"""Euler Discrete Scheduler.

Implements the Euler method for diffusion sampling,
commonly used with SDXL and other models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy
from pygpukit.diffusion.scheduler.base import BaseScheduler


class EulerDiscreteScheduler(BaseScheduler):
    """Euler Discrete Scheduler for diffusion models.

    Implements the Euler method with optional "ancestral" sampling
    for stochastic generation.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
    ):
        """Initialize Euler scheduler.

        Args:
            num_train_timesteps: Number of training timesteps.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            beta_schedule: Schedule type.
            prediction_type: What the model predicts ("epsilon", "v_prediction", "sample").
            timestep_spacing: How to space timesteps ("leading", "trailing", "linspace").
        """
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule)
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing

        # Compute sigmas for Euler
        self._compute_sigmas()

        # Initialize sigmas_inference with default (will be updated by set_timesteps)
        self.sigmas_inference = self.sigmas.copy()
        self.init_noise_sigma = self.sigmas_inference[0]

    def _compute_sigmas(self) -> None:
        """Compute sigma values for Euler method."""
        self.sigmas = np.sqrt((1 - self.alphas_cumprod) / self.alphas_cumprod)
        self.sigmas = np.concatenate([self.sigmas, np.array([0.0])])

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set inference timesteps with sigma interpolation.

        Args:
            num_inference_steps: Number of inference steps.
        """
        self.num_inference_steps = num_inference_steps

        if self.timestep_spacing == "linspace":
            # Linspace from max to 0 (matches diffusers)
            timesteps = np.linspace(self.num_train_timesteps - 1, 0, num_inference_steps)
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // num_inference_steps
            timesteps = np.arange(0, num_inference_steps) * step_ratio
            timesteps = np.flip(timesteps)
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / num_inference_steps
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio))[
                :num_inference_steps
            ]
        else:
            raise ValueError(f"Unknown timestep_spacing: {self.timestep_spacing}")

        self.timesteps = timesteps.astype(np.float32).copy()

        # Interpolate sigmas for inference timesteps
        sigmas = np.interp(self.timesteps, np.arange(len(self.sigmas) - 1), self.sigmas[:-1])
        self.sigmas_inference = np.concatenate([sigmas, np.array([0.0])])

        # Store init_noise_sigma for compatibility
        self.init_noise_sigma = self.sigmas_inference[0]

    def step(
        self,
        model_output: GPUArray,
        timestep: int,
        sample: GPUArray,
        **kwargs: Any,
    ) -> GPUArray:
        """Perform one Euler step.

        Args:
            model_output: Model prediction (noise, v-pred, or x0).
            timestep: Current timestep.
            sample: Current noisy sample.
            **kwargs: Additional arguments (generator for reproducibility).

        Returns:
            Denoised sample for next step.
        """
        # Note: generator kwarg is ignored; Euler is deterministic
        # Find step index
        step_index = np.where(self.timesteps == timestep)[0]
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index[0]

        sigma = self.sigmas_inference[step_index]
        sigma_next = self.sigmas_inference[step_index + 1]

        x = sample.to_numpy()
        eps = model_output.to_numpy()

        # Convert prediction to x0 if needed
        if self.prediction_type == "epsilon":
            # epsilon prediction: x_t = x_0 * alpha + eps * sigma
            # x_0 = (x_t - eps * sigma) / alpha
            # For Euler: x_0 = x_t - sigma * eps (in sigma space)
            pred_x0 = x - sigma * eps
        elif self.prediction_type == "v_prediction":
            # v-prediction: v = alpha * eps - sigma * x_0
            # x_0 = (x_t - sigma * v) / (alpha + sigma)
            alpha = 1.0 / np.sqrt(1 + sigma**2)
            pred_x0 = alpha * x - sigma * alpha * eps
        elif self.prediction_type == "sample":
            pred_x0 = eps
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Euler step: x_{t-1} = x_0 + sigma_{t-1} * (x_t - x_0) / sigma_t
        if sigma > 0:
            derivative = (x - pred_x0) / sigma
            x_next = pred_x0 + sigma_next * derivative
        else:
            x_next = pred_x0

        return from_numpy(x_next.astype(x.dtype))

    def scale_model_input(
        self,
        sample: GPUArray,
        timestep: int,
    ) -> GPUArray:
        """Scale model input for sigma-scaled models.

        Some models expect inputs scaled by sigma.

        Args:
            sample: Input sample.
            timestep: Current timestep.

        Returns:
            Scaled sample.
        """
        step_index = np.where(self.timesteps == timestep)[0]
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index[0]

        sigma = self.sigmas_inference[step_index]
        scale = 1.0 / np.sqrt(sigma**2 + 1)

        x = sample.to_numpy()
        return from_numpy((x * scale).astype(x.dtype))


__all__ = ["EulerDiscreteScheduler"]
