"""PyGPUkit Pipeline Module.

Provides end-to-end pipelines connecting different modalities:
- LLM to TTS: Text generation to speech synthesis
"""

from __future__ import annotations

from pygpukit.pipeline.llm_tts import LLMToTTSPipeline, StreamingTTSCallback

__all__ = [
    "LLMToTTSPipeline",
    "StreamingTTSCallback",
]
