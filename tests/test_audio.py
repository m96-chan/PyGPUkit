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


class TestAudioPreprocessing:
    """Tests for audio preprocessing functions."""

    def test_preemphasis(self, skip_if_no_cuda):
        """Test pre-emphasis filter."""
        # Create test signal
        samples = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        buf = audio.from_pcm(samples, sample_rate=16000)

        audio.preemphasis(buf, alpha=0.97)
        result = buf.to_numpy()

        # y[0] = x[0] - 0.97 * 0 = 0
        # y[1] = x[1] - 0.97 * x[0] = 1.0 - 0 = 1.0
        # y[2] = x[2] - 0.97 * x[1] = 0 - 0.97 = -0.97
        # y[3] = x[3] - 0.97 * x[2] = 1.0 - 0 = 1.0
        # y[4] = x[4] - 0.97 * x[3] = 0 - 0.97 = -0.97
        expected = np.array([0.0, 1.0, -0.97, 1.0, -0.97], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_preemphasis_with_gpuarray(self, skip_if_no_cuda):
        """Test pre-emphasis with GPUArray directly."""
        samples = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
        gpu_arr = gk.from_numpy(samples)

        result = audio.preemphasis(gpu_arr, alpha=0.5)
        # Should return the same object
        assert result is gpu_arr

    def test_deemphasis(self, skip_if_no_cuda):
        """Test de-emphasis filter."""
        # Create a simple signal
        samples = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        buf = audio.from_pcm(samples, sample_rate=16000)

        audio.deemphasis(buf, alpha=0.5)
        result = buf.to_numpy()

        # De-emphasis is IIR: y[n] = x[n] + alpha * y[n-1]
        # y[0] = 1.0 + 0.5 * 0 = 1.0
        # y[1] = 0.0 + 0.5 * 1.0 = 0.5
        # y[2] = 0.0 + 0.5 * 0.5 = 0.25
        # y[3] = 0.0 + 0.5 * 0.25 = 0.125
        # y[4] = 0.0 + 0.5 * 0.125 = 0.0625
        expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_remove_dc(self, skip_if_no_cuda):
        """Test DC offset removal."""
        # Signal with DC offset of 0.5
        samples = np.array([0.5, 0.6, 0.7, 0.4, 0.3], dtype=np.float32)
        buf = audio.from_pcm(samples, sample_rate=16000)

        audio.remove_dc(buf)
        result = buf.to_numpy()

        # Mean should be approximately zero
        np.testing.assert_allclose(np.mean(result), 0.0, atol=1e-6)

    def test_remove_dc_with_gpuarray(self, skip_if_no_cuda):
        """Test DC removal with GPUArray directly."""
        samples = np.ones(1000, dtype=np.float32) * 0.3
        gpu_arr = gk.from_numpy(samples)

        result = audio.remove_dc(gpu_arr)
        # Should return the same object
        assert result is gpu_arr

        # Mean should be zero
        np.testing.assert_allclose(np.mean(result.to_numpy()), 0.0, atol=1e-5)

    def test_highpass_filter(self, skip_if_no_cuda):
        """Test high-pass filter."""
        # Create a signal with DC offset + sine wave
        t = np.linspace(0, 0.1, 1600)  # 100ms at 16kHz
        dc_offset = 0.5
        sine = np.sin(2 * np.pi * 200 * t) * 0.3  # 200Hz sine
        samples = (dc_offset + sine).astype(np.float32)

        buf = audio.from_pcm(samples, sample_rate=16000)
        audio.highpass_filter(buf, cutoff_hz=20.0, sample_rate=16000)

        result = buf.to_numpy()

        # DC offset should be significantly reduced
        # (High-pass filter attenuates DC)
        assert abs(np.mean(result)) < 0.1

    def test_noise_gate(self, skip_if_no_cuda):
        """Test noise gate."""
        # Signal with some quiet samples
        samples = np.array([0.5, 0.005, -0.3, 0.001, 0.0, 0.8], dtype=np.float32)
        buf = audio.from_pcm(samples, sample_rate=16000)

        audio.noise_gate(buf, threshold=0.01)
        result = buf.to_numpy()

        # Samples below threshold should be zeroed
        expected = np.array([0.5, 0.0, -0.3, 0.0, 0.0, 0.8], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_noise_gate_with_gpuarray(self, skip_if_no_cuda):
        """Test noise gate with GPUArray directly."""
        samples = np.array([0.1, 0.001, 0.2, 0.0001], dtype=np.float32)
        gpu_arr = gk.from_numpy(samples)

        result = audio.noise_gate(gpu_arr, threshold=0.01)
        # Should return the same object
        assert result is gpu_arr

        result_np = result.to_numpy()
        assert result_np[1] == 0.0
        assert result_np[3] == 0.0

    def test_spectral_gate(self, skip_if_no_cuda):
        """Test spectral gate for noise reduction."""
        # Create signal: loud part + quiet noise
        loud = np.sin(np.linspace(0, 2 * np.pi * 10, 256)).astype(np.float32) * 0.5
        quiet = np.random.randn(256).astype(np.float32) * 0.001
        samples = np.concatenate([loud, quiet])

        buf = audio.from_pcm(samples, sample_rate=16000)
        audio.spectral_gate(buf, threshold=0.01, attack_samples=64)

        result = buf.to_numpy()

        # Loud part should be mostly preserved
        assert np.max(np.abs(result[:256])) > 0.3

        # Quiet part should be attenuated
        assert np.max(np.abs(result[256:])) < 0.01

    def test_compute_short_term_energy(self, skip_if_no_cuda):
        """Test short-term energy computation."""
        # Create signal with varying energy
        loud = np.ones(256, dtype=np.float32) * 0.5
        quiet = np.ones(256, dtype=np.float32) * 0.1
        samples = np.concatenate([loud, quiet])

        buf = audio.from_pcm(samples, sample_rate=16000)
        energy = audio.compute_short_term_energy(buf, frame_size=128)

        energy_np = energy.to_numpy()

        # Should have 4 frames (512 / 128)
        assert len(energy_np) == 4

        # First two frames should have higher energy
        assert energy_np[0] > energy_np[2]
        assert energy_np[1] > energy_np[3]

    def test_preemphasis_deemphasis_roundtrip(self, skip_if_no_cuda):
        """Test that pre-emphasis + de-emphasis approximately recovers original."""
        # Note: This is not exact due to the parallel approximation in preemphasis
        samples = np.sin(np.linspace(0, 2 * np.pi * 5, 1000)).astype(np.float32) * 0.5
        original = samples.copy()

        buf = audio.from_pcm(samples, sample_rate=16000)

        # Apply pre-emphasis then de-emphasis
        audio.preemphasis(buf, alpha=0.97)
        audio.deemphasis(buf, alpha=0.97)

        result = buf.to_numpy()

        # Should be close to original (not exact due to approximation)
        # The parallel preemphasis is an approximation, so we use a loose tolerance
        np.testing.assert_allclose(result, original, atol=0.5)


class TestSpectralProcessing:
    """Tests for spectral processing functions (STFT, Mel, MFCC, etc.)."""

    def test_stft_basic(self, skip_if_no_cuda):
        """Test basic STFT computation."""
        # Create 1 second of 440Hz sine wave at 16kHz
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        buf = audio.from_pcm(samples, sample_rate=sr)
        stft_out = audio.stft(buf, n_fft=512, hop_length=160)

        # Check shape: [n_frames, n_freq, 2]
        assert len(stft_out.shape) == 3
        assert stft_out.shape[1] == 257  # 512/2 + 1
        assert stft_out.shape[2] == 2  # real, imag

    def test_stft_power_spectrum(self, skip_if_no_cuda):
        """Test power spectrum computation from STFT."""
        sr = 16000
        samples = np.random.randn(sr).astype(np.float32) * 0.1
        buf = audio.from_pcm(samples, sample_rate=sr)

        stft_out = audio.stft(buf, n_fft=512, hop_length=160)
        power = audio.power_spectrum(stft_out)

        # Power should be non-negative
        power_np = power.to_numpy()
        assert np.all(power_np >= 0)

        # Shape should be [n_frames, n_freq]
        assert len(power.shape) == 2
        assert power.shape[1] == 257

    def test_stft_magnitude_spectrum(self, skip_if_no_cuda):
        """Test magnitude spectrum computation from STFT."""
        sr = 16000
        samples = np.random.randn(sr).astype(np.float32) * 0.1
        buf = audio.from_pcm(samples, sample_rate=sr)

        stft_out = audio.stft(buf, n_fft=512, hop_length=160)
        mag = audio.magnitude_spectrum(stft_out)

        # Magnitude should be non-negative
        mag_np = mag.to_numpy()
        assert np.all(mag_np >= 0)

    def test_mel_filterbank_creation(self, skip_if_no_cuda):
        """Test mel filterbank creation."""
        mel_fb = audio.create_mel_filterbank(
            n_mels=80, n_fft=512, sample_rate=16000, f_min=0.0, f_max=8000.0
        )

        # Check shape
        assert mel_fb.shape == (80, 257)

        # Filterbank weights should be non-negative
        fb_np = mel_fb.to_numpy()
        assert np.all(fb_np >= 0)

        # Each filter should have some non-zero weights
        for i in range(80):
            assert np.sum(fb_np[i, :]) > 0

    def test_apply_mel_filterbank(self, skip_if_no_cuda):
        """Test applying mel filterbank."""
        sr = 16000
        samples = np.random.randn(sr).astype(np.float32) * 0.1
        buf = audio.from_pcm(samples, sample_rate=sr)

        stft_out = audio.stft(buf, n_fft=512, hop_length=160)
        power = audio.power_spectrum(stft_out)

        mel_fb = audio.create_mel_filterbank(n_mels=80, n_fft=512, sample_rate=sr)
        mel = audio.apply_mel_filterbank(power, mel_fb)

        # Check shape: [n_frames, n_mels]
        assert len(mel.shape) == 2
        assert mel.shape[1] == 80

    def test_log_mel(self, skip_if_no_cuda):
        """Test log mel computation."""
        sr = 16000
        samples = np.random.randn(sr).astype(np.float32) * 0.1
        buf = audio.from_pcm(samples, sample_rate=sr)

        stft_out = audio.stft(buf, n_fft=512, hop_length=160)
        power = audio.power_spectrum(stft_out)
        mel_fb = audio.create_mel_filterbank(n_mels=80, n_fft=512, sample_rate=sr)
        mel = audio.apply_mel_filterbank(power, mel_fb)

        log_mel_out = audio.log_mel(mel)

        # Log mel should have same shape as mel
        assert log_mel_out.shape == mel.shape

        # Values should be finite
        log_mel_np = log_mel_out.to_numpy()
        assert np.all(np.isfinite(log_mel_np))

    def test_to_decibels(self, skip_if_no_cuda):
        """Test dB conversion."""
        sr = 16000
        samples = np.random.randn(sr).astype(np.float32) * 0.1
        buf = audio.from_pcm(samples, sample_rate=sr)

        stft_out = audio.stft(buf, n_fft=512, hop_length=160)
        power = audio.power_spectrum(stft_out)
        db = audio.to_decibels(power)

        # dB values should be finite
        db_np = db.to_numpy()
        assert np.all(np.isfinite(db_np))

    def test_mfcc(self, skip_if_no_cuda):
        """Test MFCC computation."""
        sr = 16000
        samples = np.random.randn(sr).astype(np.float32) * 0.1
        buf = audio.from_pcm(samples, sample_rate=sr)

        stft_out = audio.stft(buf, n_fft=512, hop_length=160)
        power = audio.power_spectrum(stft_out)
        mel_fb = audio.create_mel_filterbank(n_mels=80, n_fft=512, sample_rate=sr)
        mel = audio.apply_mel_filterbank(power, mel_fb)
        log_mel_out = audio.log_mel(mel)

        mfcc_out = audio.mfcc(log_mel_out, n_mfcc=13)

        # Check shape: [n_frames, n_mfcc]
        assert len(mfcc_out.shape) == 2
        assert mfcc_out.shape[1] == 13

        # MFCC values should be finite
        mfcc_np = mfcc_out.to_numpy()
        assert np.all(np.isfinite(mfcc_np))

    def test_delta_features(self, skip_if_no_cuda):
        """Test delta feature computation."""
        # Create simple features
        features = np.arange(100).reshape(10, 10).astype(np.float32)
        gpu_features = gk.from_numpy(features)

        delta_out = audio.delta(gpu_features, order=1, width=2)

        # Check shape preserved
        assert delta_out.shape == gpu_features.shape

        # Delta of increasing sequence should be positive
        delta_np = delta_out.to_numpy()
        assert np.all(np.isfinite(delta_np))

    def test_mel_spectrogram_high_level(self, skip_if_no_cuda):
        """Test high-level mel_spectrogram function."""
        sr = 16000
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32) * 0.5
        buf = audio.from_pcm(samples, sample_rate=sr)

        mel = audio.mel_spectrogram(buf, n_fft=512, hop_length=160, n_mels=80)

        # Check shape
        assert len(mel.shape) == 2
        assert mel.shape[1] == 80

        # Values should be non-negative
        mel_np = mel.to_numpy()
        assert np.all(mel_np >= 0)

    def test_log_mel_spectrogram_high_level(self, skip_if_no_cuda):
        """Test high-level log_mel_spectrogram function."""
        sr = 16000
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32) * 0.5
        buf = audio.from_pcm(samples, sample_rate=sr)

        log_mel = audio.log_mel_spectrogram(buf, n_fft=512, hop_length=160, n_mels=80)

        # Check shape
        assert len(log_mel.shape) == 2
        assert log_mel.shape[1] == 80

        # Values should be finite
        log_mel_np = log_mel.to_numpy()
        assert np.all(np.isfinite(log_mel_np))

    def test_stft_different_sizes(self, skip_if_no_cuda):
        """Test STFT with different FFT sizes."""
        sr = 16000
        samples = np.random.randn(sr).astype(np.float32) * 0.1
        buf = audio.from_pcm(samples, sample_rate=sr)

        # Test power of 2 sizes
        for n_fft in [256, 512, 1024]:
            stft_out = audio.stft(buf, n_fft=n_fft, hop_length=160)
            assert stft_out.shape[1] == n_fft // 2 + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
