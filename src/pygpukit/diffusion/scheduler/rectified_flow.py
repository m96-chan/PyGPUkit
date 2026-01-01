"""Rectified Flow (Flow Matching) Scheduler.

Used by Stable Diffusion 3 and Flux models.
Implements the flow matching formulation where the model
learns a velocity field between noise and data.
"""

from __future__ import annotations

import numpy as np

from pygpukit.core.array import GPUArray
from pygpukit.core.factory import from_numpy


class FlowMatchingScheduler:
    """Flow Matching (Rectified Flow) Scheduler.

    Used by SD3 and Flux models. The model predicts velocity
    in the flow from noise to data.

    Key difference from diffusion:
    - Instead of predicting noise, predicts velocity
    - Linear interpolation between noise and data
    - Simpler ODE formulation
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        """Initialize Flow Matching scheduler.

        Args:
            num_train_timesteps: Number of training timesteps.
            shift: Time shift parameter (SD3/Flux use resolution-based shift).
            base_shift: Base shift value.
            max_shift: Maximum shift value.
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.base_shift = base_shift
        self.max_shift = max_shift

        self.timesteps: np.ndarray | None = None
        self.sigmas: np.ndarray | None = None
        self.num_inference_steps: int = 0

    def set_timesteps(
        self,
        num_inference_steps: int,
        mu: float | None = None,
    ) -> None:
        """Set inference timesteps.

        Args:
            num_inference_steps: Number of inference steps.
            mu: Optional shift parameter (computed from resolution if None).
        """
        self.num_inference_steps = num_inference_steps

        # Compute timesteps (linearly spaced in [0, 1])
        timesteps = np.linspace(1.0, 0.0, num_inference_steps + 1)

        # Apply shift if specified
        if mu is not None:
            timesteps = self._time_shift(timesteps, mu)
        elif self.shift != 1.0:
            timesteps = self._time_shift(timesteps, self.shift)

        self.timesteps = timesteps[:-1]  # Remove final 0
        self.sigmas = timesteps  # sigmas = t for flow matching

    def _time_shift(self, t: np.ndarray, mu: float) -> np.ndarray:
        """Apply time shift for resolution-dependent sampling.

        Args:
            t: Timesteps in [0, 1].
            mu: Shift parameter.

        Returns:
            Shifted timesteps.
        """
        # SD3/Flux shift formula: t' = exp(mu) * t / (1 + (exp(mu) - 1) * t)
        exp_mu = np.exp(mu)
        return exp_mu * t / (1 + (exp_mu - 1) * t)

    def compute_shift(
        self,
        image_seq_len: int,
        base_seq_len: int = 256,
    ) -> float:
        """Compute resolution-based shift.

        SD3/Flux use larger shift for higher resolutions.

        Args:
            image_seq_len: Number of image patches (H/patch_size * W/patch_size).
            base_seq_len: Base sequence length for shift=base_shift.

        Returns:
            Shift parameter mu.
        """
        m = (self.max_shift - self.base_shift) / (1024 - 256)
        b = self.base_shift - m * 256
        return image_seq_len * m + b

    def step(
        self,
        model_output: GPUArray,
        timestep: float,
        sample: GPUArray,
        **kwargs,
    ) -> GPUArray:
        """Perform one flow matching step.

        The model predicts velocity v, and we integrate:
        x_{t-dt} = x_t - dt * v

        Args:
            model_output: Predicted velocity [B, ...].
            timestep: Current timestep t in [0, 1].
            sample: Current sample x_t.

        Returns:
            Sample at next timestep.
        """
        # Find step index
        step_index = np.where(np.isclose(self.timesteps, timestep))[0]
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index[0]

        t = self.sigmas[step_index]
        t_next = self.sigmas[step_index + 1]
        dt = t_next - t  # Note: dt is negative (t decreases)

        x = sample.to_numpy()
        v = model_output.to_numpy()

        # Euler step: x_next = x + dt * v
        x_next = x + dt * v

        return from_numpy(x_next.astype(x.dtype))

    def add_noise(
        self,
        original_samples: GPUArray,
        noise: GPUArray,
        timesteps: np.ndarray | float,
    ) -> GPUArray:
        """Add noise at given timestep using flow interpolation.

        For flow matching: x_t = (1 - t) * x_0 + t * noise

        Args:
            original_samples: Clean samples x_0.
            noise: Noise samples.
            timesteps: Timesteps in [0, 1].

        Returns:
            Noisy samples x_t.
        """
        x = original_samples.to_numpy()
        n = noise.to_numpy()

        if isinstance(timesteps, float):
            t = timesteps
        else:
            t = timesteps[0] if len(timesteps) > 0 else 0.0

        # Linear interpolation
        x_t = (1 - t) * x + t * n

        return from_numpy(x_t.astype(x.dtype))

    def get_velocity(
        self,
        sample: GPUArray,
        noise: GPUArray,
        timesteps: np.ndarray | float,
    ) -> GPUArray:
        """Compute velocity target for training.

        For flow matching: v = noise - sample

        Args:
            sample: Clean sample x_0.
            noise: Noise.
            timesteps: Timesteps (unused, included for API compatibility).

        Returns:
            Velocity target.
        """
        x = sample.to_numpy()
        n = noise.to_numpy()

        velocity = n - x

        return from_numpy(velocity.astype(x.dtype))

    def scale_noise(
        self,
        sample: GPUArray,
        timestep: float,
    ) -> GPUArray:
        """Scale sample for model input (identity for flow matching).

        Args:
            sample: Input sample.
            timestep: Current timestep.

        Returns:
            Scaled sample (unchanged for flow matching).
        """
        return sample


__all__ = ["FlowMatchingScheduler"]
