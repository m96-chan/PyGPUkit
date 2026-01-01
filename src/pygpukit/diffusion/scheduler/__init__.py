"""Diffusion schedulers for denoising.

Provides various scheduler implementations:
- BaseScheduler: Abstract base class
- EulerDiscreteScheduler: Euler method (SDXL, SD)
- DDIMScheduler: DDIM scheduler
- FlowMatchingScheduler: Rectified flow (SD3, Flux)
"""

from __future__ import annotations

from pygpukit.diffusion.scheduler.base import BaseScheduler
from pygpukit.diffusion.scheduler.ddim import DDIMScheduler
from pygpukit.diffusion.scheduler.euler import EulerDiscreteScheduler
from pygpukit.diffusion.scheduler.rectified_flow import FlowMatchingScheduler

__all__ = [
    "BaseScheduler",
    "EulerDiscreteScheduler",
    "DDIMScheduler",
    "FlowMatchingScheduler",
]
