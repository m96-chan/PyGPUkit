"""FLUX diffusion transformer for PyGPUkit.

FLUX.1 is a 12B parameter rectified flow transformer for text-to-image generation.
This implementation supports FLUX.1-schnell (distilled, 4-step).
"""

from __future__ import annotations

from pygpukit.diffusion.models.flux.model import FluxConfig, FluxTransformer
from pygpukit.diffusion.models.flux.pipeline import FluxPipeline, generate
from pygpukit.diffusion.models.flux.scheduler import (
    FlowMatchEulerScheduler,
    FlowMatchEulerSchedulerConfig,
)

__all__ = [
    "FluxTransformer",
    "FluxConfig",
    "FlowMatchEulerScheduler",
    "FlowMatchEulerSchedulerConfig",
    "FluxPipeline",
    "generate",
]
