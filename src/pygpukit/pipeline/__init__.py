"""PyGPUkit Pipeline Module.

Provides end-to-end pipelines connecting different modalities:
- LLM to TTS: Text generation to speech synthesis
- Voice Pipeline: Full voice conversation (Audio → ASR → LLM → TTS)
"""

from __future__ import annotations

from pygpukit.pipeline.llm_tts import LLMToTTSPipeline, StreamingTTSCallback
from pygpukit.pipeline.voice import (
    AudioBuffer,
    ConversationTurn,
    PipelineState,
    PipelineStats,
    VADConfig,
    VADEvent,
    VADState,
    VoiceActivityDetector,
    VoicePipeline,
    VoicePipelineCallback,
    create_voice_pipeline,
)

__all__ = [
    # LLM to TTS
    "LLMToTTSPipeline",
    "StreamingTTSCallback",
    # Voice Pipeline
    "VoicePipeline",
    "VoicePipelineCallback",
    "create_voice_pipeline",
    # VAD
    "VoiceActivityDetector",
    "VADConfig",
    "VADEvent",
    "VADState",
    # Utilities
    "AudioBuffer",
    "ConversationTurn",
    "PipelineState",
    "PipelineStats",
]
