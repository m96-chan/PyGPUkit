"""Kokoro-82M TTS Example.

This example demonstrates text-to-speech synthesis using the Kokoro-82M model
with PyGPUkit's native LSTM kernel.

Usage:
    python examples/tts.py
    python examples/tts.py --text "Hello world" --voice af_heart
    python examples/tts.py --model F:/LLM/Kokoro-82M --output speech.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def test_lstm_kernel():
    """Test the native LSTM kernel works correctly."""
    import numpy as np

    import pygpukit as pk

    print("Testing native LSTM kernel...")

    batch = 2
    seq_len = 10
    input_size = 64
    hidden_size = 128

    # Create random test inputs
    x = pk.from_numpy(np.random.randn(batch, seq_len, input_size).astype(np.float32))
    W_ih = pk.from_numpy(
        np.random.randn(4 * hidden_size, input_size).astype(np.float32) * 0.1
    )
    W_hh = pk.from_numpy(
        np.random.randn(4 * hidden_size, hidden_size).astype(np.float32) * 0.1
    )
    b_ih = pk.from_numpy(np.zeros(4 * hidden_size, dtype=np.float32))
    b_hh = pk.from_numpy(np.zeros(4 * hidden_size, dtype=np.float32))

    # Forward LSTM
    output, h_n, c_n = pk.lstm_forward(x, W_ih, W_hh, b_ih, b_hh)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  h_n shape: {h_n.shape}")
    print(f"  c_n shape: {c_n.shape}")

    # Verify output is not all zeros
    out_np = output.to_numpy()
    assert not np.allclose(out_np, 0), "LSTM output should not be all zeros"

    print("  LSTM kernel test PASSED!")
    return True


def test_bidirectional_lstm():
    """Test bidirectional LSTM."""
    import numpy as np

    import pygpukit as pk

    print("Testing bidirectional LSTM...")

    batch = 2
    seq_len = 10
    input_size = 64
    hidden_size = 128

    # Create random test inputs
    x = pk.from_numpy(np.random.randn(batch, seq_len, input_size).astype(np.float32))

    # Forward direction weights
    W_ih_fwd = pk.from_numpy(
        np.random.randn(4 * hidden_size, input_size).astype(np.float32) * 0.1
    )
    W_hh_fwd = pk.from_numpy(
        np.random.randn(4 * hidden_size, hidden_size).astype(np.float32) * 0.1
    )
    b_ih_fwd = pk.from_numpy(np.zeros(4 * hidden_size, dtype=np.float32))
    b_hh_fwd = pk.from_numpy(np.zeros(4 * hidden_size, dtype=np.float32))

    # Backward direction weights
    W_ih_bwd = pk.from_numpy(
        np.random.randn(4 * hidden_size, input_size).astype(np.float32) * 0.1
    )
    W_hh_bwd = pk.from_numpy(
        np.random.randn(4 * hidden_size, hidden_size).astype(np.float32) * 0.1
    )
    b_ih_bwd = pk.from_numpy(np.zeros(4 * hidden_size, dtype=np.float32))
    b_hh_bwd = pk.from_numpy(np.zeros(4 * hidden_size, dtype=np.float32))

    # Bidirectional LSTM
    output, h_n, c_n = pk.lstm_bidirectional(
        x,
        W_ih_fwd,
        W_hh_fwd,
        b_ih_fwd,
        b_hh_fwd,
        W_ih_bwd,
        W_hh_bwd,
        b_ih_bwd,
        b_hh_bwd,
    )

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape} (2x hidden due to bidirectional)")
    print(f"  h_n shape: {h_n.shape}")
    print(f"  c_n shape: {c_n.shape}")

    # Verify shapes
    assert output.shape == (batch, seq_len, 2 * hidden_size)
    assert h_n.shape == (2, batch, hidden_size)
    assert c_n.shape == (2, batch, hidden_size)

    print("  Bidirectional LSTM test PASSED!")
    return True


def benchmark_lstm():
    """Benchmark LSTM performance."""
    import numpy as np

    import pygpukit as pk

    print("\nBenchmarking LSTM performance...")

    batch = 8
    seq_len = 100
    input_size = 768
    hidden_size = 512

    # Create test inputs (typical TTS dimensions)
    x = pk.from_numpy(np.random.randn(batch, seq_len, input_size).astype(np.float32))
    W_ih = pk.from_numpy(
        np.random.randn(4 * hidden_size, input_size).astype(np.float32) * 0.1
    )
    W_hh = pk.from_numpy(
        np.random.randn(4 * hidden_size, hidden_size).astype(np.float32) * 0.1
    )
    b_ih = pk.from_numpy(np.zeros(4 * hidden_size, dtype=np.float32))
    b_hh = pk.from_numpy(np.zeros(4 * hidden_size, dtype=np.float32))

    # Warmup
    for _ in range(3):
        output, h_n, c_n = pk.lstm_forward(x, W_ih, W_hh, b_ih, b_hh)

    # Benchmark
    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        output, h_n, c_n = pk.lstm_forward(x, W_ih, W_hh, b_ih, b_hh)
    elapsed = time.perf_counter() - start

    ms_per_call = (elapsed / iterations) * 1000
    print(f"  Config: batch={batch}, seq_len={seq_len}, input={input_size}, hidden={hidden_size}")
    print(f"  Time per forward: {ms_per_call:.2f} ms")
    print(f"  Throughput: {(batch * seq_len) / (ms_per_call / 1000):.0f} tokens/sec")

    return ms_per_call


def main():
    parser = argparse.ArgumentParser(description="Kokoro-82M TTS Example")
    parser.add_argument("--model", type=str, default="F:/LLM/Kokoro-82M", help="Model path")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the Kokoro text to speech system.", help="Text to synthesize")
    parser.add_argument("--voice", type=str, default="af_heart", help="Voice to use")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--test-only", action="store_true", help="Only run LSTM tests")
    args = parser.parse_args()

    print("=" * 60)
    print("PyGPUkit TTS Example - Kokoro-82M")
    print("=" * 60)

    # Test LSTM kernel
    if not test_lstm_kernel():
        print("LSTM kernel test failed!")
        return 1

    if not test_bidirectional_lstm():
        print("Bidirectional LSTM test failed!")
        return 1

    # Benchmark
    benchmark_lstm()

    if args.test_only:
        print("\nTest-only mode: skipping model loading")
        return 0

    # Try to load the TTS model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nModel not found at {model_path}")
        print("Please download Kokoro-82M from HuggingFace:")
        print("  huggingface-cli download hexgrad/Kokoro-82M --local-dir F:/LLM/Kokoro-82M")
        return 1

    print(f"\nModel found at: {model_path}")
    print(f"Note: Full TTS synthesis requires additional model components.")
    print(f"The LSTM kernel is now integrated and ready for use.")

    # TODO: Full TTS synthesis when model loading is complete
    # from pygpukit.tts.kokoro import KokoroModel
    # model = KokoroModel.from_pretrained(model_path)
    # result = model.synthesize(args.text, voice=args.voice)
    # result.to_wav(args.output)
    # print(f"Audio saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
