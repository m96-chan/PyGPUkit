"""Base class for decode strategies.

This module defines the abstract base class for all decode strategies.
Each strategy implements a specific decoding algorithm (M=1, batch,
speculative, jacobi, etc.) while sharing common infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygpukit.core.array import GPUArray
    from pygpukit.llm.buffers import DecodeBuffers
    from pygpukit.llm.model import CausalTransformerModel


class DecodeStrategy(ABC):
    """Abstract base class for decode strategies.

    A decode strategy encapsulates a specific decoding algorithm.
    The Model class owns the CUDA Graph state; strategies only decide
    how to use (or not use) that infrastructure.

    Attributes:
        model: Reference to the CausalTransformerModel (set at runtime).
    """

    def __init__(self) -> None:
        """Initialize the decode strategy."""
        self._model: CausalTransformerModel | None = None

    def bind(self, model: CausalTransformerModel) -> None:
        """Bind this strategy to a model.

        Args:
            model: The model to bind to.
        """
        self._model = model

    @property
    def model(self) -> CausalTransformerModel:
        """Get the bound model."""
        if self._model is None:
            raise RuntimeError("Strategy not bound to a model. Call bind() first.")
        return self._model

    @abstractmethod
    def step(
        self,
        token_id: int,
        position: int,
        context_len: int,
        buffers: DecodeBuffers,
    ) -> GPUArray:
        """Execute a single decode step.

        Args:
            token_id: Current token ID to process.
            position: Position in the sequence.
            context_len: Total context length (for KV cache attention).
            buffers: Pre-allocated decode buffers.

        Returns:
            Hidden states or logits depending on the strategy.
        """
        pass

    def init_graph(self, max_seq_len: int = 512) -> None:  # noqa: B027
        """Initialize CUDA Graph for this strategy.

        Override in subclasses that support CUDA Graph acceleration.
        Default implementation does nothing (no graph support).

        Args:
            max_seq_len: Maximum sequence length for KV cache.
        """
        pass

    def has_graph(self) -> bool:
        """Check if this strategy has a captured CUDA Graph.

        Returns:
            True if a graph is ready for replay.
        """
        return False
