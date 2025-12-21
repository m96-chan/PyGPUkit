"""Decode strategies for LLM inference.

This module provides different decode strategies that can be used with
the CausalTransformerModel class.
"""

from __future__ import annotations

from pygpukit.llm.decode.base import DecodeStrategy
from pygpukit.llm.decode.batch import DecodeBatch
from pygpukit.llm.decode.jacobi import DecodeJacobi
from pygpukit.llm.decode.m1 import DecodeM1
from pygpukit.llm.decode.m1_graph import DecodeM1Graph
from pygpukit.llm.decode.speculative import DecodeSpeculative

__all__ = [
    "DecodeStrategy",
    "DecodeM1",
    "DecodeM1Graph",
    "DecodeBatch",
    "DecodeSpeculative",
    "DecodeJacobi",
]
