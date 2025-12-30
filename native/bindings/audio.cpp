/**
 * Audio processing operations: PCM conversion, resampling, spectral analysis, VAD, etc.
 */
#include "bindings_common.hpp"

void init_audio(py::module_& m) {
    // Basic audio operations
    m.def("audio_pcm_to_float32", &ops::audio::pcm_to_float32,
          py::arg("input"),
          "Convert int16 PCM samples to float32.\n"
          "Returns: GPUArray of float32 samples normalized to [-1.0, 1.0]");

    m.def("audio_stereo_to_mono", &ops::audio::stereo_to_mono,
          py::arg("input"),
          "Convert stereo audio to mono by averaging channels.");

    m.def("audio_normalize_peak", &ops::audio::normalize_peak,
          py::arg("input"),
          "Peak normalize audio to [-1.0, 1.0] range (in-place).");

    m.def("audio_normalize_rms", &ops::audio::normalize_rms,
          py::arg("input"), py::arg("target_db") = -20.0f,
          "RMS normalize audio to target dB level (in-place).");

    m.def("audio_resample", &ops::audio::resample,
          py::arg("input"), py::arg("src_rate"), py::arg("dst_rate"),
          "Resample audio from source to target sample rate.");

    // Streaming operations
    m.def("audio_ring_buffer_write", &ops::audio::ring_buffer_write,
          py::arg("input"), py::arg("ring_buffer"), py::arg("write_pos"),
          "Write samples to a ring buffer with wrap-around.");

    m.def("audio_ring_buffer_read", &ops::audio::ring_buffer_read,
          py::arg("ring_buffer"), py::arg("read_pos"), py::arg("num_samples"),
          "Read samples from a ring buffer (linearized).");

    m.def("audio_apply_hann_window", &ops::audio::apply_hann_window,
          py::arg("data"),
          "Apply Hann window to audio data (in-place).");

    m.def("audio_overlap_add", &ops::audio::overlap_add,
          py::arg("input"), py::arg("output"), py::arg("output_offset"),
          "Overlap-add: add windowed chunk to output buffer.");

    // VAD operations
    m.def("vad_compute_energy", &ops::audio::vad_compute_energy,
          py::arg("audio"), py::arg("frame_size"), py::arg("hop_size"),
          "Compute frame-level RMS energy for VAD.");

    m.def("vad_compute_zcr", &ops::audio::vad_compute_zcr,
          py::arg("audio"), py::arg("frame_size"), py::arg("hop_size"),
          "Compute frame-level zero-crossing rate for VAD.");

    m.def("vad_decide", &ops::audio::vad_decide,
          py::arg("frame_energy"), py::arg("frame_zcr"),
          py::arg("energy_threshold"), py::arg("zcr_low"), py::arg("zcr_high"),
          "Apply threshold-based VAD decision.");

    m.def("vad_apply_hangover", &ops::audio::vad_apply_hangover,
          py::arg("vad_input"), py::arg("hangover_frames"),
          "Apply hangover smoothing to VAD output.");

    m.def("vad_compute_noise_floor", &ops::audio::vad_compute_noise_floor,
          py::arg("frame_energy"),
          "Compute noise floor for adaptive thresholding.");

    // Preprocessing
    m.def("audio_preemphasis", &ops::audio::preemphasis,
          py::arg("input"), py::arg("alpha") = 0.97f,
          "Apply pre-emphasis filter (in-place).");

    m.def("audio_deemphasis", &ops::audio::deemphasis,
          py::arg("input"), py::arg("alpha") = 0.97f,
          "Apply de-emphasis filter (in-place).");

    m.def("audio_remove_dc", &ops::audio::remove_dc,
          py::arg("input"),
          "Remove DC offset from audio signal (in-place).");

    m.def("audio_highpass_filter", &ops::audio::highpass_filter,
          py::arg("input"), py::arg("cutoff_hz") = 20.0f, py::arg("sample_rate") = 16000,
          "Apply high-pass filter for DC removal (in-place).");

    m.def("audio_noise_gate", &ops::audio::noise_gate,
          py::arg("input"), py::arg("threshold") = 0.01f,
          "Apply simple noise gate (in-place).");

    m.def("audio_spectral_gate", &ops::audio::spectral_gate,
          py::arg("input"), py::arg("threshold") = 0.01f,
          py::arg("attack_samples") = 64, py::arg("release_samples") = 256,
          "Apply spectral gate for noise reduction (in-place).");

    m.def("audio_compute_short_term_energy", &ops::audio::compute_short_term_energy,
          py::arg("input"), py::arg("frame_size"),
          "Compute short-term energy for adaptive noise gating.");

    // Spectral processing
    m.def("audio_stft", &ops::audio::stft,
          py::arg("input"), py::arg("n_fft") = 400, py::arg("hop_length") = 160,
          py::arg("win_length") = -1, py::arg("center") = true,
          "Compute Short-Time Fourier Transform (STFT).");

    m.def("audio_power_spectrum", &ops::audio::power_spectrum,
          py::arg("stft_output"),
          "Compute power spectrogram from STFT output.");

    m.def("audio_magnitude_spectrum", &ops::audio::magnitude_spectrum,
          py::arg("stft_output"),
          "Compute magnitude spectrogram from STFT output.");

    m.def("audio_create_mel_filterbank", &ops::audio::create_mel_filterbank,
          py::arg("n_mels"), py::arg("n_fft"), py::arg("sample_rate"),
          py::arg("f_min") = 0.0f, py::arg("f_max") = -1.0f,
          "Create Mel filterbank matrix.");

    m.def("audio_apply_mel_filterbank", &ops::audio::apply_mel_filterbank,
          py::arg("spectrogram"), py::arg("mel_filterbank"),
          "Apply Mel filterbank to spectrogram.");

    m.def("audio_log_mel_spectrogram", &ops::audio::log_mel_spectrogram,
          py::arg("mel_spectrogram"), py::arg("eps") = 1e-10f,
          "Compute log-mel spectrogram.");

    m.def("audio_to_decibels", &ops::audio::to_decibels,
          py::arg("input"), py::arg("eps") = 1e-10f,
          "Convert to decibels.");

    m.def("audio_mfcc", &ops::audio::mfcc,
          py::arg("log_mel"), py::arg("n_mfcc") = 13,
          "Compute MFCC from log-mel spectrogram.");

    m.def("audio_delta_features", &ops::audio::delta_features,
          py::arg("features"), py::arg("order") = 1, py::arg("width") = 2,
          "Compute delta features.");

    m.def("audio_whisper_mel_spectrogram", &ops::audio::whisper_mel_spectrogram,
          py::arg("input"), py::arg("n_fft") = 400, py::arg("hop_length") = 160,
          py::arg("n_mels") = 80,
          "Compute Whisper-compatible log-mel spectrogram.");

    // Inverse STFT
    m.def("audio_istft", &ops::audio::istft,
          py::arg("stft_output"), py::arg("hop_length") = 160,
          py::arg("win_length") = -1, py::arg("center") = true,
          py::arg("length") = -1,
          "Compute Inverse STFT.");

    // Griffin-Lim
    m.def("audio_griffin_lim", &ops::audio::griffin_lim,
          py::arg("magnitude"), py::arg("n_iter") = 32,
          py::arg("hop_length") = 160, py::arg("win_length") = -1,
          "Griffin-Lim phase reconstruction algorithm.");

    // Pitch detection
    m.def("audio_autocorrelation", &ops::audio::autocorrelation,
          py::arg("input"), py::arg("max_lag"),
          "Compute autocorrelation of signal.");

    m.def("audio_detect_pitch_yin", &ops::audio::detect_pitch_yin,
          py::arg("input"), py::arg("sample_rate"),
          py::arg("f_min") = 50.0f, py::arg("f_max") = 2000.0f,
          py::arg("threshold") = 0.1f,
          "Detect pitch using YIN algorithm.");

    m.def("audio_detect_pitch_yin_frames", &ops::audio::detect_pitch_yin_frames,
          py::arg("input"), py::arg("sample_rate"),
          py::arg("frame_size"), py::arg("hop_size"),
          py::arg("f_min") = 50.0f, py::arg("f_max") = 2000.0f,
          py::arg("threshold") = 0.1f,
          "Detect pitch for multiple frames using YIN algorithm.");

    // Spectral features
    m.def("audio_spectral_centroid", &ops::audio::spectral_centroid,
          py::arg("spectrum"), py::arg("sample_rate"),
          "Compute spectral centroid.");

    m.def("audio_spectral_bandwidth", &ops::audio::spectral_bandwidth,
          py::arg("spectrum"), py::arg("centroids"),
          py::arg("sample_rate"), py::arg("p") = 2,
          "Compute spectral bandwidth.");

    m.def("audio_spectral_rolloff", &ops::audio::spectral_rolloff,
          py::arg("spectrum"), py::arg("sample_rate"),
          py::arg("roll_percent") = 0.85f,
          "Compute spectral rolloff point.");

    m.def("audio_spectral_flatness", &ops::audio::spectral_flatness,
          py::arg("spectrum"),
          "Compute spectral flatness.");

    m.def("audio_spectral_contrast", &ops::audio::spectral_contrast,
          py::arg("spectrum"), py::arg("n_bands") = 6,
          py::arg("alpha") = 0.02f,
          "Compute spectral contrast.");

    m.def("audio_zero_crossing_rate", &ops::audio::zero_crossing_rate,
          py::arg("input"), py::arg("frame_size"), py::arg("hop_size"),
          "Compute zero-crossing rate.");

    // CQT
    m.def("audio_cqt", &ops::audio::cqt,
          py::arg("input"), py::arg("sample_rate"),
          py::arg("hop_length") = 512, py::arg("f_min") = 32.7f,
          py::arg("n_bins") = 84, py::arg("bins_per_octave") = 12,
          "Compute Constant-Q Transform.");

    m.def("audio_cqt_magnitude", &ops::audio::cqt_magnitude,
          py::arg("cqt_output"),
          "Compute CQT magnitude spectrogram.");

    // Chromagram
    m.def("audio_chroma_stft", &ops::audio::chroma_stft,
          py::arg("spectrum"), py::arg("sample_rate"),
          py::arg("n_chroma") = 12, py::arg("tuning") = 0.0f,
          "Compute chromagram from STFT.");

    m.def("audio_chroma_cqt", &ops::audio::chroma_cqt,
          py::arg("cqt_mag"), py::arg("bins_per_octave") = 12,
          "Compute chromagram from CQT.");

    // HPSS
    m.def("audio_hpss", [](const GPUArray& stft_magnitude, int kernel_size,
                           float power, float margin) {
              auto [h, p] = ops::audio::hpss(stft_magnitude, kernel_size, power, margin);
              return py::make_tuple(std::move(h), std::move(p));
          },
          py::arg("stft_magnitude"), py::arg("kernel_size") = 31,
          py::arg("power") = 2.0f, py::arg("margin") = 1.0f,
          "Harmonic-percussive source separation.");

    m.def("audio_harmonic", &ops::audio::harmonic,
          py::arg("stft_magnitude"), py::arg("kernel_size") = 31,
          py::arg("power") = 2.0f, py::arg("margin") = 1.0f,
          "Get harmonic component from HPSS.");

    m.def("audio_percussive", &ops::audio::percussive,
          py::arg("stft_magnitude"), py::arg("kernel_size") = 31,
          py::arg("power") = 2.0f, py::arg("margin") = 1.0f,
          "Get percussive component from HPSS.");

    // Time stretch / Pitch shift
    m.def("audio_time_stretch", &ops::audio::time_stretch,
          py::arg("input"), py::arg("rate"),
          py::arg("n_fft") = 2048, py::arg("hop_length") = -1,
          "Time-stretch audio using phase vocoder.");

    m.def("audio_pitch_shift", &ops::audio::pitch_shift,
          py::arg("input"), py::arg("sample_rate"), py::arg("n_steps"),
          py::arg("n_fft") = 2048, py::arg("hop_length") = -1,
          "Pitch-shift audio by n_steps semitones.");
}
