"""Tests for Stream Manager."""

from pygpukit.core.stream import Stream, StreamManager, StreamPriority, default_stream


class TestStreamPriority:
    """Tests for StreamPriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert StreamPriority.HIGH == 0
        assert StreamPriority.LOW == 1

    def test_priority_comparison(self):
        """Test that HIGH priority is numerically lower."""
        assert StreamPriority.HIGH < StreamPriority.LOW


class TestStream:
    """Tests for Stream class."""

    def test_stream_creation(self):
        """Test stream creation."""
        manager = StreamManager()
        stream = manager.create_stream(priority="high")

        assert stream is not None
        assert stream.priority == StreamPriority.HIGH

    def test_stream_repr(self):
        """Test stream repr."""
        manager = StreamManager()
        stream = manager.create_stream(priority="high")

        assert "HIGH" in repr(stream)

    def test_stream_synchronize(self):
        """Test stream synchronization."""
        manager = StreamManager()
        stream = manager.create_stream()
        stream.synchronize()  # Should not raise


class TestStreamManager:
    """Tests for StreamManager class."""

    def test_create_stream_low_priority(self):
        """Test creating low priority stream."""
        manager = StreamManager()
        stream = manager.create_stream(priority="low")

        assert stream.priority == StreamPriority.LOW

    def test_create_stream_high_priority(self):
        """Test creating high priority stream."""
        manager = StreamManager()
        stream = manager.create_stream(priority="high")

        assert stream.priority == StreamPriority.HIGH

    def test_create_stream_with_enum(self):
        """Test creating stream with enum priority."""
        manager = StreamManager()
        stream = manager.create_stream(priority=StreamPriority.HIGH)

        assert stream.priority == StreamPriority.HIGH

    def test_destroy_stream(self):
        """Test stream destruction."""
        manager = StreamManager()
        stream = manager.create_stream()
        manager.destroy_stream(stream)

        assert stream not in manager._streams

    def test_get_default_stream(self):
        """Test getting default stream."""
        manager = StreamManager()
        stream1 = manager.get_default_stream()
        stream2 = manager.get_default_stream()

        assert stream1 is stream2  # Same instance
        assert stream1.priority == StreamPriority.LOW

    def test_synchronize_all(self):
        """Test synchronizing all streams."""
        manager = StreamManager()
        manager.create_stream(priority="high")
        manager.create_stream(priority="low")

        manager.synchronize_all()  # Should not raise

    def test_multiple_streams(self):
        """Test creating multiple streams."""
        manager = StreamManager()
        streams = [manager.create_stream() for _ in range(5)]

        assert len(streams) == 5
        assert len(manager._streams) == 5


class TestDefaultStream:
    """Tests for default_stream function."""

    def test_default_stream(self):
        """Test getting default stream via function."""
        stream = default_stream()
        assert stream is not None
        assert isinstance(stream, Stream)
