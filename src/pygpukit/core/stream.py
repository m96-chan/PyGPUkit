"""CUDA Stream management."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from pygpukit.core.backend import get_backend


class StreamPriority(IntEnum):
    """Stream priority levels."""

    HIGH = 0
    LOW = 1


class Stream:
    """A CUDA stream for asynchronous operations.

    Streams allow for concurrent execution of GPU operations.
    Higher priority streams may preempt lower priority streams.
    """

    def __init__(self, stream_handle: Any, priority: StreamPriority) -> None:
        """Initialize a Stream.

        Args:
            stream_handle: Backend-specific stream handle.
            priority: Priority level of the stream.
        """
        self._handle = stream_handle
        self._priority = priority

    @property
    def handle(self) -> Any:
        """Return the backend-specific stream handle."""
        return self._handle

    @property
    def priority(self) -> StreamPriority:
        """Return the stream priority."""
        return self._priority

    def synchronize(self) -> None:
        """Wait for all operations on this stream to complete."""
        backend = get_backend()
        backend.stream_synchronize(self._handle)

    def __repr__(self) -> str:
        priority_name = "HIGH" if self._priority == StreamPriority.HIGH else "LOW"
        return f"Stream(priority={priority_name})"


class StreamManager:
    """Manager for CUDA streams.

    Provides stream pooling and lifecycle management.
    """

    def __init__(self) -> None:
        """Initialize the StreamManager."""
        self._streams: list[Stream] = []
        self._default_stream: Stream | None = None

    def create_stream(self, priority: str | StreamPriority = "low") -> Stream:
        """Create a new stream.

        Args:
            priority: Priority level ("high" or "low", or StreamPriority enum).

        Returns:
            A new Stream instance.
        """
        if isinstance(priority, str):
            priority = StreamPriority.HIGH if priority.lower() == "high" else StreamPriority.LOW

        backend = get_backend()
        handle = backend.create_stream(priority=int(priority))
        stream = Stream(handle, priority)
        self._streams.append(stream)
        return stream

    def destroy_stream(self, stream: Stream) -> None:
        """Destroy a stream.

        Args:
            stream: The stream to destroy.
        """
        if stream in self._streams:
            backend = get_backend()
            backend.destroy_stream(stream.handle)
            self._streams.remove(stream)

    def get_default_stream(self) -> Stream:
        """Get or create the default stream.

        Returns:
            The default stream (low priority).
        """
        if self._default_stream is None:
            self._default_stream = self.create_stream(StreamPriority.LOW)
        return self._default_stream

    def synchronize_all(self) -> None:
        """Synchronize all streams."""
        for stream in self._streams:
            stream.synchronize()

    def __del__(self) -> None:
        """Clean up all streams."""
        backend = get_backend()
        for stream in self._streams:
            try:
                backend.destroy_stream(stream.handle)
            except Exception:
                pass
        self._streams.clear()


# Global default stream manager
_stream_manager: StreamManager | None = None


def get_stream_manager() -> StreamManager:
    """Get the global stream manager."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager


def default_stream() -> Stream:
    """Get the default stream."""
    return get_stream_manager().get_default_stream()
