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


class TestVAD:
    """Tests for Voice Activity Detection."""

    def test_vad_creation(self, skip_if_no_cuda):
        """Test VAD creation with default parameters."""
        vad = audio.VAD(sample_rate=16000)
        assert vad.sample_rate == 16000
        assert vad.frame_size == 320  # 20ms @ 16kHz
        assert vad.hop_size == 160  # 10ms @ 16kHz

    def test_vad_detect_silence(self, skip_if_no_cuda):
        """Test VAD on silence (should detect no speech)."""
        vad = audio.VAD(sample_rate=16000, energy_threshold=0.01)

        # Create silent audio (1 second)
        silence = np.zeros(16000, dtype=np.float32)
        buf = audio.from_pcm(silence, sample_rate=16000)

        segments = vad.detect(buf)
        assert len(segments) == 0

    def test_vad_detect_speech(self, skip_if_no_cuda):
        """Test VAD on synthetic speech-like signal."""
        vad = audio.VAD(sample_rate=16000, energy_threshold=0.05)

        # Create audio: silence + tone + silence
        # 0.5s silence + 0.5s tone + 0.5s silence
        silence1 = np.zeros(8000, dtype=np.float32)
        tone = np.sin(np.linspace(0, 2 * np.pi * 200, 8000)).astype(np.float32) * 0.5
        silence2 = np.zeros(8000, dtype=np.float32)

        samples = np.concatenate([silence1, tone, silence2])
        buf = audio.from_pcm(samples, sample_rate=16000)

        segments = vad.detect(buf)

        # Should detect one speech segment
        assert len(segments) >= 1

        # Speech should be roughly in the middle
        seg = segments[0]
        assert seg.start_time >= 0.3  # After first silence
        assert seg.end_time <= 1.2  # Before end

    def test_vad_get_frame_features(self, skip_if_no_cuda):
        """Test getting raw frame features."""
        vad = audio.VAD(sample_rate=16000)

        # Create 1 second of audio
        samples = np.random.randn(16000).astype(np.float32) * 0.1
        buf = audio.from_pcm(samples, sample_rate=16000)

        energy, zcr = vad.get_frame_features(buf)

        # Check output shapes
        # With 20ms frame and 10ms hop: (16000 - 320) / 160 + 1 = 99 frames
        expected_frames = (16000 - vad.frame_size) // vad.hop_size + 1
        assert energy.shape[0] == expected_frames
        assert zcr.shape[0] == expected_frames

        # Check value ranges
        energy_np = energy.to_numpy()
        zcr_np = zcr.to_numpy()

        assert np.all(energy_np >= 0)  # Energy is non-negative
        assert np.all(zcr_np >= 0)  # ZCR is non-negative
        assert np.all(zcr_np <= 1)  # ZCR is normalized to [0, 1]

    def test_vad_speech_segment_times(self, skip_if_no_cuda):
        """Test SpeechSegment time calculations."""
        seg = audio.SpeechSegment(
            start_sample=16000,
            end_sample=32000,
            start_time=1.0,
            end_time=2.0,
        )

        assert seg.start_sample == 16000
        assert seg.end_sample == 32000
        assert seg.start_time == 1.0
        assert seg.end_time == 2.0

    def test_vad_hangover(self, skip_if_no_cuda):
        """Test VAD hangover smoothing."""
        # Create VAD with different hangover settings
        vad_no_hangover = audio.VAD(sample_rate=16000, hangover_ms=0)
        vad_with_hangover = audio.VAD(sample_rate=16000, hangover_ms=100)

        # Short burst of sound
        silence1 = np.zeros(4000, dtype=np.float32)
        tone = np.sin(np.linspace(0, 2 * np.pi * 200, 1600)).astype(np.float32) * 0.5
        silence2 = np.zeros(4000, dtype=np.float32)

        samples = np.concatenate([silence1, tone, silence2])
        buf = audio.from_pcm(samples, sample_rate=16000)

        seg_no = vad_no_hangover.detect(buf)
        seg_with = vad_with_hangover.detect(buf)

        # Hangover should extend the speech region
        if len(seg_no) > 0 and len(seg_with) > 0:
            # With hangover, end time should be later or equal
            assert seg_with[0].end_time >= seg_no[0].end_time

    def test_vad_repr(self, skip_if_no_cuda):
        """Test VAD string representation."""
        vad = audio.VAD(sample_rate=16000, frame_ms=30, hop_ms=15)

        repr_str = repr(vad)
        assert "16000" in repr_str
        assert "VAD" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
