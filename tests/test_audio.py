"""Tests for GPU audio processing operations."""

import numpy as np
import pytest

import pygpukit as gk
from pygpukit.ops import audio


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not gk.is_cuda_available():
        pytest.skip("CUDA not available")


class TestPcmConversion:
    """Tests for PCM to float conversion."""

    def test_int16_to_float32(self, skip_if_no_cuda):
        """Test int16 PCM to float32 conversion."""
        # Test values: 0, half max, half min, max
        pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        buf = audio.from_pcm(pcm, sample_rate=48000)

        assert buf.sample_rate == 48000
        assert buf.channels == 1

        result = buf.to_numpy()
        expected = np.array([0.0, 0.5, -0.5, 32767 / 32768.0], dtype=np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_float32_passthrough(self, skip_if_no_cuda):
        """Test float32 samples pass through unchanged."""
        samples = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        buf = audio.from_pcm(samples, sample_rate=16000)

        result = buf.to_numpy()
        np.testing.assert_allclose(result, samples, rtol=1e-6)

    def test_stereo_metadata(self, skip_if_no_cuda):
        """Test stereo audio metadata."""
        stereo = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        buf = audio.from_pcm(stereo, sample_rate=48000, channels=2)

        assert buf.channels == 2
        assert buf.sample_rate == 48000


class TestStereoToMono:
    """Tests for stereo to mono conversion."""

    def test_stereo_to_mono(self, skip_if_no_cuda):
        """Test stereo to mono conversion."""
        # Interleaved stereo: [L0, R0, L1, R1, L2, R2]
        stereo = np.array([1.0, 0.0, 0.0, 1.0, 0.5, 0.5], dtype=np.float32)
        buf = audio.from_pcm(stereo, sample_rate=48000, channels=2)

        mono = buf.to_mono()

        assert mono.channels == 1
        result = mono.to_numpy()
        expected = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mono_passthrough(self, skip_if_no_cuda):
        """Test mono audio passes through unchanged."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        buf = audio.from_pcm(samples, sample_rate=16000, channels=1)

        result_buf = buf.to_mono()

        # Should be the same object (no conversion needed)
        assert result_buf is buf


class TestNormalization:
    """Tests for audio normalization."""

    def test_peak_normalize(self, skip_if_no_cuda):
        """Test peak normalization."""
        samples = np.array([0.0, 0.25, -0.5, 0.25], dtype=np.float32)
        buf = audio.from_pcm(samples, sample_rate=16000)

        buf.normalize(mode="peak")

        result = buf.to_numpy()
        # Max abs was 0.5, so everything should be scaled by 2
        expected = np.array([0.0, 0.5, -1.0, 0.5], dtype=np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_rms_normalize(self, skip_if_no_cuda):
        """Test RMS normalization."""
        # Create a signal with known RMS
        samples = np.ones(1000, dtype=np.float32) * 0.1
        buf = audio.from_pcm(samples, sample_rate=16000)

        # Normalize to -20 dB (RMS = 0.1)
        buf.normalize(mode="rms", target_db=-20.0)

        result = buf.to_numpy()
        result_rms = np.sqrt(np.mean(result**2))

        # -20 dB = 10^(-20/20) = 0.1
        expected_rms = 0.1
        np.testing.assert_allclose(result_rms, expected_rms, rtol=0.01)


class TestResampling:
    """Tests for audio resampling."""

    def test_resample_48_to_16(self, skip_if_no_cuda):
        """Test 48kHz to 16kHz resampling."""
        # Create a simple signal at 48kHz
        n_samples = 4800  # 100ms at 48kHz
        samples = np.sin(np.linspace(0, 2 * np.pi * 10, n_samples)).astype(np.float32)

        buf = audio.from_pcm(samples, sample_rate=48000)
        resampled = buf.resample(16000)

        assert resampled.sample_rate == 16000
        # 3:1 decimation
        assert resampled.data.shape[0] == n_samples // 3

    def test_same_rate_passthrough(self, skip_if_no_cuda):
        """Test same sample rate passes through unchanged."""
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        buf = audio.from_pcm(samples, sample_rate=16000)

        result_buf = buf.resample(16000)

        # Should be the same object (no conversion needed)
        assert result_buf is buf


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_repr(self, skip_if_no_cuda):
        """Test AudioBuffer string representation."""
        samples = np.zeros(1000, dtype=np.float32)
        buf = audio.from_pcm(samples, sample_rate=48000, channels=2)

        repr_str = repr(buf)
        assert "1000" in repr_str
        assert "48000" in repr_str
        assert "2" in repr_str

    def test_fluent_api(self, skip_if_no_cuda):
        """Test fluent API chaining."""
        # Create stereo 48kHz audio
        stereo_48k = np.random.randn(9600).astype(np.float32) * 0.5
        buf = audio.from_pcm(stereo_48k, sample_rate=48000, channels=2)

        # Chain operations
        result = buf.to_mono().resample(16000).normalize()

        assert result.sample_rate == 16000
        assert result.channels == 1

        data = result.to_numpy()
        max_abs = np.max(np.abs(data))
        np.testing.assert_allclose(max_abs, 1.0, rtol=0.01)


class TestAudioRingBuffer:
    """Tests for AudioRingBuffer."""

    def test_ring_buffer_creation(self, skip_if_no_cuda):
        """Test ring buffer creation."""
        ring = audio.AudioRingBuffer(capacity=16000, sample_rate=16000)
        assert ring.capacity == 16000
        assert ring.sample_rate == 16000
        assert ring.samples_available == 0

    def test_ring_buffer_write_read(self, skip_if_no_cuda):
        """Test writing and reading from ring buffer."""
        ring = audio.AudioRingBuffer(capacity=1000, sample_rate=16000)

        # Write samples
        samples = np.arange(100, dtype=np.float32)
        ring.write(samples)

        assert ring.samples_available == 100

        # Read samples back
        result = ring.read(100)
        np.testing.assert_allclose(result.to_numpy(), samples, rtol=1e-5)

    def test_ring_buffer_wrap_around(self, skip_if_no_cuda):
        """Test ring buffer wrap-around behavior."""
        ring = audio.AudioRingBuffer(capacity=100, sample_rate=16000)

        # Write 150 samples (should wrap)
        samples1 = np.ones(80, dtype=np.float32)
        samples2 = np.ones(70, dtype=np.float32) * 2

        ring.write(samples1)
        ring.write(samples2)

        # Buffer should be full
        assert ring.samples_available == 100

    def test_ring_buffer_clear(self, skip_if_no_cuda):
        """Test clearing the ring buffer."""
        ring = audio.AudioRingBuffer(capacity=1000, sample_rate=16000)

        samples = np.ones(500, dtype=np.float32)
        ring.write(samples)

        ring.clear()
        assert ring.samples_available == 0


class TestAudioStream:
    """Tests for AudioStream."""

    def test_stream_creation(self, skip_if_no_cuda):
        """Test stream creation."""
        stream = audio.AudioStream(chunk_size=480, sample_rate=16000)
        assert stream.chunk_size == 480
        assert stream.hop_size == 240  # Default 50% overlap
        assert stream.sample_rate == 16000

    def test_stream_push_and_has_chunk(self, skip_if_no_cuda):
        """Test pushing audio and checking for chunks."""
        stream = audio.AudioStream(chunk_size=480, hop_size=240, sample_rate=16000)

        # No chunk initially
        assert not stream.has_chunk()

        # Push 480 samples (one full chunk)
        samples = np.random.randn(480).astype(np.float32)
        stream.push(samples)

        # Now we should have one chunk
        assert stream.has_chunk()

    def test_stream_pop_chunk(self, skip_if_no_cuda):
        """Test popping chunks from stream."""
        stream = audio.AudioStream(chunk_size=480, hop_size=240, sample_rate=16000)

        # Push enough for 2 chunks (480 + 240 = 720 samples)
        samples = np.random.randn(720).astype(np.float32)
        stream.push(samples)

        # Should have 2 chunks available
        assert stream.chunks_available == 2

        # Pop first chunk
        chunk1 = stream.pop_chunk(apply_window=False)
        assert chunk1.shape[0] == 480

        # Pop second chunk
        chunk2 = stream.pop_chunk(apply_window=False)
        assert chunk2.shape[0] == 480

    def test_stream_windowing(self, skip_if_no_cuda):
        """Test Hann windowing on chunks."""
        stream = audio.AudioStream(chunk_size=480, sample_rate=16000)

        # Push constant signal
        samples = np.ones(480, dtype=np.float32)
        stream.push(samples)

        # Pop with windowing
        chunk = stream.pop_chunk(apply_window=True)
        result = chunk.to_numpy()

        # Hann window should taper the edges
        assert result[0] < 0.1  # Near zero at start
        assert result[-1] < 0.1  # Near zero at end
        assert result[240] > 0.9  # Near 1 at center

    def test_stream_reset(self, skip_if_no_cuda):
        """Test resetting the stream."""
        stream = audio.AudioStream(chunk_size=480, sample_rate=16000)

        samples = np.random.randn(1000).astype(np.float32)
        stream.push(samples)

        stream.reset()
        assert not stream.has_chunk()
        assert stream.chunks_available == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
