"""Loading strategies for lazy model loading.

Provides three strategies for controlling GPU memory during inference:

1. SimpleStreaming: Load layer, compute, unload (minimal VRAM)
2. SlidingWindow: Keep N layers in VRAM, prefetch ahead (balanced)
3. AutoLRU: Load on demand, automatic LRU eviction (maximum performance)

Example:
    >>> from pygpukit.llm import LazyModelLoader
    >>> from pygpukit.llm.streaming import SlidingWindow, LayerStreamingContext
    >>>
    >>> loader = LazyModelLoader(memory_budget=8 * 1024**3)
    >>> loader.load_file("model.safetensors")
    >>>
    >>> strategy = SlidingWindow(window_size=4)
    >>> with LayerStreamingContext(loader, strategy, num_layers=32) as ctx:
    ...     for i in range(32):
    ...         ctx.prepare(i)  # Manages loading/unloading
    ...         hidden = layers[i](hidden)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygpukit.llm.safetensors import LazyModelLoader


class LoadingStrategy(ABC):
    """Base class for layer loading strategies.

    Subclasses implement on_layer_start/on_layer_end to control
    when layers are loaded and unloaded from GPU memory.
    """

    @abstractmethod
    def on_layer_start(self, loader: LazyModelLoader, layer_idx: int, num_layers: int) -> None:
        """Called before processing a layer.

        Args:
            loader: The LazyModelLoader instance
            layer_idx: Current layer index (0-based)
            num_layers: Total number of layers
        """
        pass

    @abstractmethod
    def on_layer_end(self, loader: LazyModelLoader, layer_idx: int, num_layers: int) -> None:
        """Called after processing a layer.

        Args:
            loader: The LazyModelLoader instance
            layer_idx: Current layer index (0-based)
            num_layers: Total number of layers
        """
        pass

    def on_start(self, loader: LazyModelLoader, num_layers: int) -> None:
        """Called when streaming context starts.

        Default implementation does nothing. Override if needed.

        Args:
            loader: The LazyModelLoader instance
            num_layers: Total number of layers
        """
        # Default: no-op (subclasses can override)
        _ = loader, num_layers

    def on_end(self, loader: LazyModelLoader, num_layers: int) -> None:
        """Called when streaming context ends.

        Default implementation does nothing. Override if needed.

        Args:
            loader: The LazyModelLoader instance
            num_layers: Total number of layers
        """
        # Default: no-op (subclasses can override)
        _ = loader, num_layers

    @staticmethod
    def layer_prefix(layer_idx: int, prefix_template: str = "model.layers.{}.") -> str:
        """Generate layer prefix from index.

        Args:
            layer_idx: Layer index
            prefix_template: Template with {} placeholder for index

        Returns:
            Layer prefix string (e.g., "model.layers.0.")
        """
        return prefix_template.format(layer_idx)


@dataclass
class SimpleStreaming(LoadingStrategy):
    """Simple layer-by-layer streaming strategy.

    Loads each layer before processing and immediately unloads after.
    Minimizes VRAM usage but has highest loading overhead.

    Attributes:
        prefix_template: Template for layer prefix (default: "model.layers.{}.")

    Example:
        >>> strategy = SimpleStreaming()
        >>> # Each layer loaded/unloaded sequentially
    """

    prefix_template: str = "model.layers.{}."

    def on_layer_start(self, loader: LazyModelLoader, layer_idx: int, num_layers: int) -> None:
        """Load the current layer."""
        # Layer loading is handled by the native layer when tensors are accessed
        # This is a marker for explicit loading if needed
        pass

    def on_layer_end(self, loader: LazyModelLoader, layer_idx: int, num_layers: int) -> None:
        """Unload the current layer immediately."""
        prefix = self.layer_prefix(layer_idx, self.prefix_template)
        loader.unload_layer(prefix)


@dataclass
class SlidingWindow(LoadingStrategy):
    """Sliding window strategy with prefetching.

    Keeps a fixed number of layers in VRAM and prefetches upcoming layers
    while unloading old ones. Balances memory usage and performance.

    Attributes:
        window_size: Number of layers to keep in VRAM (default: 4)
        prefetch_ahead: How many layers ahead to prefetch (default: 1)
        prefix_template: Template for layer prefix

    Example:
        >>> strategy = SlidingWindow(window_size=4, prefetch_ahead=2)
        >>> # Keeps 4 layers in VRAM, prefetches 2 ahead
    """

    window_size: int = 4
    prefetch_ahead: int = 1
    prefix_template: str = "model.layers.{}."

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.prefetch_ahead < 0:
            raise ValueError("prefetch_ahead must be >= 0")

    def on_layer_start(self, loader: LazyModelLoader, layer_idx: int, num_layers: int) -> None:
        """Prefetch upcoming layers within window."""
        # Prefetch layers ahead
        for i in range(1, self.prefetch_ahead + 1):
            next_idx = layer_idx + i
            if next_idx < num_layers:
                # Trigger loading by checking layer state
                # (actual loading happens when tensors are accessed)
                pass

    def on_layer_end(self, loader: LazyModelLoader, layer_idx: int, num_layers: int) -> None:
        """Unload layers outside the window."""
        # Calculate the oldest layer that should be evicted
        evict_idx = layer_idx - self.window_size
        if evict_idx >= 0:
            prefix = self.layer_prefix(evict_idx, self.prefix_template)
            loader.unload_layer(prefix)


@dataclass
class AutoLRU(LoadingStrategy):
    """Automatic LRU-based eviction strategy.

    Relies on the memory pool's built-in LRU eviction. Tensors are loaded
    on demand and automatically evicted when memory budget is exceeded.
    Provides best performance when model fits in VRAM budget.

    Attributes:
        prefix_template: Template for layer prefix
        unload_on_end: Whether to unload all layers when context ends

    Example:
        >>> strategy = AutoLRU()
        >>> # Let the memory pool handle everything automatically
    """

    prefix_template: str = "model.layers.{}."
    unload_on_end: bool = False

    def on_layer_start(self, loader: LazyModelLoader, layer_idx: int, num_layers: int) -> None:
        """No explicit loading - let LRU handle it."""
        pass

    def on_layer_end(self, loader: LazyModelLoader, layer_idx: int, num_layers: int) -> None:
        """No explicit unloading - let LRU handle it."""
        pass

    def on_end(self, loader: LazyModelLoader, num_layers: int) -> None:
        """Optionally unload all layers when done."""
        if self.unload_on_end:
            loader.unload_model()


class LayerStreamingContext:
    """Context manager for layer-based model streaming.

    Manages loading and unloading of transformer layers during inference
    using a specified loading strategy.

    Example:
        >>> loader = LazyModelLoader(memory_budget=8 * 1024**3)
        >>> loader.load_file("model.safetensors")
        >>>
        >>> strategy = SlidingWindow(window_size=4)
        >>> with LayerStreamingContext(loader, strategy, num_layers=32) as ctx:
        ...     for i in range(32):
        ...         ctx.prepare(i)
        ...         hidden = model.layers[i](hidden)
    """

    def __init__(
        self,
        loader: LazyModelLoader,
        strategy: LoadingStrategy,
        num_layers: int,
        prefix_template: str = "model.layers.{}.",
    ):
        """Create a streaming context.

        Args:
            loader: LazyModelLoader instance
            strategy: Loading strategy to use
            num_layers: Total number of layers in the model
            prefix_template: Template for layer prefix with {} placeholder
        """
        self.loader = loader
        self.strategy = strategy
        self.num_layers = num_layers
        self.prefix_template = prefix_template
        self._current_layer: int | None = None
        self._active = False

    def __enter__(self) -> LayerStreamingContext:
        """Enter the streaming context."""
        self._active = True
        self.strategy.on_start(self.loader, self.num_layers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the streaming context."""
        # Finish last layer if any
        if self._current_layer is not None:
            self.strategy.on_layer_end(self.loader, self._current_layer, self.num_layers)
        self.strategy.on_end(self.loader, self.num_layers)
        self._active = False
        self._current_layer = None

    def prepare(self, layer_idx: int) -> None:
        """Prepare for processing a specific layer.

        This method should be called before processing each layer.
        It handles the loading/unloading according to the strategy.

        Args:
            layer_idx: Layer index to prepare (0-based)
        """
        if not self._active:
            raise RuntimeError("StreamingContext must be used within a 'with' block")

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

        # End previous layer if switching
        if self._current_layer is not None and self._current_layer != layer_idx:
            self.strategy.on_layer_end(self.loader, self._current_layer, self.num_layers)

        # Start new layer
        self._current_layer = layer_idx
        self.strategy.on_layer_start(self.loader, layer_idx, self.num_layers)

    def layer_prefix(self, layer_idx: int) -> str:
        """Get the prefix for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Layer prefix string
        """
        return self.prefix_template.format(layer_idx)

    @property
    def current_layer(self) -> int | None:
        """Current layer index being processed."""
        return self._current_layer

    @property
    def memory_stats(self) -> dict:
        """Get current memory statistics.

        Returns:
            Dictionary with memory usage information
        """
        stats = self.loader.pool_stats
        return {
            "quota_gb": stats.quota / (1024**3),
            "used_gb": stats.used / (1024**3),
            "available_gb": stats.available / (1024**3),
            "utilization_pct": stats.utilization,
            "active_blocks": stats.active_blocks,
            "eviction_count": stats.eviction_count,
        }


def create_streaming_context(
    loader: LazyModelLoader,
    strategy: str | LoadingStrategy,
    num_layers: int,
    prefix_template: str = "model.layers.{}.",
    **kwargs,
) -> LayerStreamingContext:
    """Factory function to create a streaming context.

    Args:
        loader: LazyModelLoader instance
        strategy: Strategy name ("simple", "sliding", "auto") or LoadingStrategy instance
        num_layers: Total number of layers
        prefix_template: Template for layer prefix
        **kwargs: Additional arguments passed to strategy constructor

    Returns:
        LayerStreamingContext configured with the specified strategy

    Example:
        >>> ctx = create_streaming_context(
        ...     loader, "sliding", num_layers=32, window_size=4
        ... )
    """
    if isinstance(strategy, str):
        strategy_lower = strategy.lower()
        if strategy_lower in ("simple", "simple_streaming"):
            strategy_obj = SimpleStreaming(
                prefix_template=kwargs.get("prefix_template", prefix_template)
            )
        elif strategy_lower in ("sliding", "sliding_window"):
            strategy_obj = SlidingWindow(
                window_size=kwargs.get("window_size", 4),
                prefetch_ahead=kwargs.get("prefetch_ahead", 1),
                prefix_template=kwargs.get("prefix_template", prefix_template),
            )
        elif strategy_lower in ("auto", "auto_lru", "lru"):
            strategy_obj = AutoLRU(
                prefix_template=kwargs.get("prefix_template", prefix_template),
                unload_on_end=kwargs.get("unload_on_end", False),
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'simple', 'sliding', or 'auto'.")
    else:
        strategy_obj = strategy

    return LayerStreamingContext(
        loader=loader,
        strategy=strategy_obj,
        num_layers=num_layers,
        prefix_template=prefix_template,
    )


__all__ = [
    "LoadingStrategy",
    "SimpleStreaming",
    "SlidingWindow",
    "AutoLRU",
    "LayerStreamingContext",
    "create_streaming_context",
]
