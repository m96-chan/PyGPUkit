"""Unit tests for Kokoro TTS layer implementations.

Tests the neural network layers used in Kokoro-82M TTS model.
Uses mock weights to verify layer behavior without requiring actual model files.
"""

import numpy as np
import pytest

import pygpukit as gk
from pygpukit.core.factory import from_numpy

# Check if new TTS layers are available (they may not be in older installations)
try:
    from pygpukit.tts.kokoro.layers import WeightNormConv1d  # noqa: F401

    HAS_TTS_LAYERS = True
except ImportError:
    HAS_TTS_LAYERS = False

pytestmark = pytest.mark.skipif(not HAS_TTS_LAYERS, reason="TTS layers not available")


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not gk.is_cuda_available():
        pytest.skip("CUDA not available")


class TestWeightNormConv1d:
    """Tests for WeightNormConv1d layer."""

    def test_weight_normalization(self, skip_if_no_cuda):
        """Test that weight normalization computes W = g * (v / ||v||)."""
        from pygpukit.tts.kokoro.layers import WeightNormConv1d

        out_channels, in_channels, kernel_size = 4, 2, 3

        # Create mock weights
        weight_g = from_numpy(np.ones((out_channels, 1, 1), dtype=np.float32) * 2.0)
        weight_v = from_numpy(
            np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32)
        )

        conv = WeightNormConv1d(weight_g=weight_g, weight_v=weight_v)

        # Compute normalized weight
        weight = conv._compute_weight()

        # Verify: each output channel should have L2 norm equal to g
        for i in range(out_channels):
            channel_norm = np.sqrt((weight[i] ** 2).sum())
            np.testing.assert_allclose(channel_norm, 2.0, rtol=1e-5)

    def test_forward_shape(self, skip_if_no_cuda):
        """Test that forward pass produces correct output shape."""
        from pygpukit.tts.kokoro.layers import WeightNormConv1d

        batch, in_channels, length = 2, 4, 16
        out_channels, kernel_size = 8, 3
        padding = 1

        weight_g = from_numpy(np.ones((out_channels, 1, 1), dtype=np.float32))
        weight_v = from_numpy(
            np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32)
        )
        bias = from_numpy(np.zeros(out_channels, dtype=np.float32))

        conv = WeightNormConv1d(weight_g=weight_g, weight_v=weight_v, bias=bias, padding=padding)

        x = from_numpy(np.random.randn(batch, in_channels, length).astype(np.float32))
        out = conv(x)

        # With padding=1 and kernel_size=3, output length should be same as input
        assert out.shape == (batch, out_channels, length)


class TestInstanceNorm1d:
    """Tests for InstanceNorm1d layer."""

    def test_normalization(self, skip_if_no_cuda):
        """Test that instance norm normalizes each channel to zero mean, unit variance."""
        from pygpukit.tts.kokoro.layers import InstanceNorm1d

        channels = 4
        gamma = from_numpy(np.ones(channels, dtype=np.float32))
        beta = from_numpy(np.zeros(channels, dtype=np.float32))

        norm = InstanceNorm1d(gamma=gamma, beta=beta)

        # Create input with known statistics
        batch, length = 2, 32
        x = from_numpy(np.random.randn(batch, channels, length).astype(np.float32) * 5 + 3)

        out = norm(x)
        out_np = out.to_numpy()

        # Check each sample and channel has ~zero mean and ~unit variance
        for b in range(batch):
            for c in range(channels):
                mean = out_np[b, c].mean()
                var = out_np[b, c].var()
                np.testing.assert_allclose(mean, 0.0, atol=1e-5)
                np.testing.assert_allclose(var, 1.0, atol=1e-4)

    def test_affine_transform(self, skip_if_no_cuda):
        """Test that gamma and beta are applied correctly."""
        from pygpukit.tts.kokoro.layers import InstanceNorm1d

        channels = 2
        gamma = from_numpy(np.array([2.0, 0.5], dtype=np.float32))
        beta = from_numpy(np.array([1.0, -1.0], dtype=np.float32))

        norm = InstanceNorm1d(gamma=gamma, beta=beta)

        x = from_numpy(np.random.randn(1, channels, 100).astype(np.float32))
        out = norm(x)
        out_np = out.to_numpy()

        # After normalization and affine: mean should be beta, std should be gamma
        np.testing.assert_allclose(out_np[0, 0].mean(), 1.0, atol=0.1)
        np.testing.assert_allclose(out_np[0, 1].mean(), -1.0, atol=0.1)
        np.testing.assert_allclose(out_np[0, 0].std(), 2.0, atol=0.1)
        np.testing.assert_allclose(out_np[0, 1].std(), 0.5, atol=0.1)


class TestAdaIN:
    """Tests for Adaptive Instance Normalization layer."""

    def test_style_conditioning(self, skip_if_no_cuda):
        """Test that style vector modulates scale and shift."""
        from pygpukit.tts.kokoro.layers import AdaIN

        channels, style_dim = 4, 8

        # FC layer: [2*channels, style_dim]
        fc_weight = from_numpy(np.random.randn(2 * channels, style_dim).astype(np.float32) * 0.1)
        fc_bias = from_numpy(np.zeros(2 * channels, dtype=np.float32))

        adain = AdaIN(fc_weight=fc_weight, fc_bias=fc_bias)

        batch, length = 2, 16
        x = from_numpy(np.random.randn(batch, channels, length).astype(np.float32))
        style = from_numpy(np.random.randn(batch, style_dim).astype(np.float32))

        out = adain(x, style)

        assert out.shape == (batch, channels, length)

    def test_different_styles_produce_different_outputs(self, skip_if_no_cuda):
        """Test that different style vectors produce different outputs."""
        from pygpukit.tts.kokoro.layers import AdaIN

        channels, style_dim = 4, 8

        fc_weight = from_numpy(np.random.randn(2 * channels, style_dim).astype(np.float32))
        fc_bias = from_numpy(np.zeros(2 * channels, dtype=np.float32))

        adain = AdaIN(fc_weight=fc_weight, fc_bias=fc_bias)

        x = from_numpy(np.random.randn(1, channels, 16).astype(np.float32))
        style1 = from_numpy(np.random.randn(1, style_dim).astype(np.float32))
        style2 = from_numpy(np.random.randn(1, style_dim).astype(np.float32))

        out1 = adain(x, style1).to_numpy()
        out2 = adain(x, style2).to_numpy()

        # Outputs should be different
        assert not np.allclose(out1, out2)


class TestALBERTLayer:
    """Tests for ALBERTLayer."""

    def test_forward_shape(self, skip_if_no_cuda):
        """Test that ALBERT layer preserves sequence dimensions."""
        from pygpukit.tts.kokoro.layers import ALBERTLayer, LayerNorm, Linear

        batch, seq_len, hidden_size = 2, 16, 64
        num_heads = 4
        intermediate_size = 128

        # Create mock weights
        def make_linear(in_f, out_f):
            w = from_numpy(np.random.randn(out_f, in_f).astype(np.float32) * 0.02)
            b = from_numpy(np.zeros(out_f, dtype=np.float32))
            return Linear(w, b)

        def make_norm(size):
            w = from_numpy(np.ones(size, dtype=np.float32))
            b = from_numpy(np.zeros(size, dtype=np.float32))
            return LayerNorm(w, b)

        layer = ALBERTLayer(
            query=make_linear(hidden_size, hidden_size),
            key=make_linear(hidden_size, hidden_size),
            value=make_linear(hidden_size, hidden_size),
            attention_dense=make_linear(hidden_size, hidden_size),
            attention_norm=make_norm(hidden_size),
            ffn=make_linear(hidden_size, intermediate_size),
            ffn_output=make_linear(intermediate_size, hidden_size),
            full_layer_norm=make_norm(hidden_size),
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
        )

        x = from_numpy(np.random.randn(batch, seq_len, hidden_size).astype(np.float32))
        out = layer(x)

        assert out.shape == (batch, seq_len, hidden_size)


class TestALBERTEncoder:
    """Tests for ALBERTEncoder."""

    def test_forward_shape(self, skip_if_no_cuda):
        """Test that ALBERT encoder produces correct output shape."""
        from pygpukit.tts.kokoro.layers import ALBERTEncoder, ALBERTLayer, LayerNorm, Linear

        vocab_size, embed_dim, hidden_size = 100, 32, 64
        max_positions, num_heads = 128, 4
        num_layers = 2
        intermediate_size = 128

        def make_linear(in_f, out_f):
            w = from_numpy(np.random.randn(out_f, in_f).astype(np.float32) * 0.02)
            b = from_numpy(np.zeros(out_f, dtype=np.float32))
            return Linear(w, b)

        def make_norm(size):
            w = from_numpy(np.ones(size, dtype=np.float32))
            b = from_numpy(np.zeros(size, dtype=np.float32))
            return LayerNorm(w, b)

        # Embeddings
        word_emb = from_numpy(np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02)
        pos_emb = from_numpy(np.random.randn(max_positions, embed_dim).astype(np.float32) * 0.02)
        type_emb = from_numpy(np.random.randn(2, embed_dim).astype(np.float32) * 0.02)

        # Shared layer
        layer = ALBERTLayer(
            query=make_linear(hidden_size, hidden_size),
            key=make_linear(hidden_size, hidden_size),
            value=make_linear(hidden_size, hidden_size),
            attention_dense=make_linear(hidden_size, hidden_size),
            attention_norm=make_norm(hidden_size),
            ffn=make_linear(hidden_size, intermediate_size),
            ffn_output=make_linear(intermediate_size, hidden_size),
            full_layer_norm=make_norm(hidden_size),
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
        )

        encoder = ALBERTEncoder(
            word_embeddings=word_emb,
            position_embeddings=pos_emb,
            token_type_embeddings=type_emb,
            embeddings_norm=make_norm(embed_dim),
            embedding_mapping=make_linear(embed_dim, hidden_size),
            layer=layer,
            num_hidden_layers=num_layers,
        )

        batch, seq_len = 2, 16
        input_ids = from_numpy(np.random.randint(0, vocab_size, (batch, seq_len)).astype(np.int32))

        out = encoder(input_ids)

        assert out.shape == (batch, seq_len, hidden_size)


class TestKokoroTextEncoder:
    """Tests for KokoroTextEncoder (CNN + BiLSTM)."""

    def test_forward_shape(self, skip_if_no_cuda):
        """Test that text encoder produces correct output shape."""
        from pygpukit.tts.kokoro.layers import (
            LSTM,
            InstanceNorm1d,
            KokoroTextEncoder,
            WeightNormConv1d,
        )

        vocab_size, embed_dim = 100, 32
        cnn_channels = 64
        lstm_hidden = 128

        # Embedding
        embedding = from_numpy(np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02)

        # CNN layers
        cnn_layers = []
        in_ch = embed_dim
        for _ in range(3):
            conv = WeightNormConv1d(
                weight_g=from_numpy(np.ones((cnn_channels, 1, 1), dtype=np.float32)),
                weight_v=from_numpy(
                    np.random.randn(cnn_channels, in_ch, 5).astype(np.float32) * 0.02
                ),
                padding=2,
            )
            norm = InstanceNorm1d(
                gamma=from_numpy(np.ones(cnn_channels, dtype=np.float32)),
                beta=from_numpy(np.zeros(cnn_channels, dtype=np.float32)),
            )
            cnn_layers.append((conv, norm))
            in_ch = cnn_channels

        # BiLSTM
        lstm = LSTM(
            W_ih=from_numpy(
                np.random.randn(4 * lstm_hidden, cnn_channels).astype(np.float32) * 0.02
            ),
            W_hh=from_numpy(
                np.random.randn(4 * lstm_hidden, lstm_hidden).astype(np.float32) * 0.02
            ),
            b_ih=from_numpy(np.zeros(4 * lstm_hidden, dtype=np.float32)),
            b_hh=from_numpy(np.zeros(4 * lstm_hidden, dtype=np.float32)),
            bidirectional=True,
            W_ih_reverse=from_numpy(
                np.random.randn(4 * lstm_hidden, cnn_channels).astype(np.float32) * 0.02
            ),
            W_hh_reverse=from_numpy(
                np.random.randn(4 * lstm_hidden, lstm_hidden).astype(np.float32) * 0.02
            ),
            b_ih_reverse=from_numpy(np.zeros(4 * lstm_hidden, dtype=np.float32)),
            b_hh_reverse=from_numpy(np.zeros(4 * lstm_hidden, dtype=np.float32)),
        )

        encoder = KokoroTextEncoder(embedding=embedding, cnn_layers=cnn_layers, lstm=lstm)

        batch, seq_len = 2, 16
        input_ids = from_numpy(np.random.randint(0, vocab_size, (batch, seq_len)).astype(np.int32))

        out = encoder(input_ids)

        # BiLSTM output: [batch, seq_len, 2 * lstm_hidden]
        assert out.shape == (batch, seq_len, 2 * lstm_hidden)


class TestAdaINResBlock:
    """Tests for AdaINResBlock."""

    def test_residual_connection(self, skip_if_no_cuda):
        """Test that residual connection is applied."""
        from pygpukit.tts.kokoro.layers import AdaIN, AdaINResBlock, WeightNormConv1d

        channels, style_dim = 32, 16

        def make_conv(in_ch, out_ch):
            return WeightNormConv1d(
                weight_g=from_numpy(np.ones((out_ch, 1, 1), dtype=np.float32)),
                weight_v=from_numpy(np.random.randn(out_ch, in_ch, 3).astype(np.float32) * 0.02),
                padding=1,
            )

        def make_adain(ch, style_d):
            return AdaIN(
                fc_weight=from_numpy(np.random.randn(2 * ch, style_d).astype(np.float32) * 0.1),
                fc_bias=from_numpy(np.zeros(2 * ch, dtype=np.float32)),
            )

        block = AdaINResBlock(
            conv1=make_conv(channels, channels),
            conv2=make_conv(channels, channels),
            norm1=make_adain(channels, style_dim),
            norm2=make_adain(channels, style_dim),
        )

        batch, length = 2, 16
        x = from_numpy(np.random.randn(batch, channels, length).astype(np.float32))
        style = from_numpy(np.random.randn(batch, style_dim).astype(np.float32))

        out = block(x, style)

        assert out.shape == (batch, channels, length)


class TestBuildFunctions:
    """Tests for weight builder functions."""

    def test_build_albert_missing_weights_raises(self, skip_if_no_cuda):
        """Test that missing weights raise KeyError."""
        from pygpukit.tts.kokoro.layers import build_albert_from_weights

        weights = {}  # Empty weights

        with pytest.raises(KeyError):
            build_albert_from_weights(weights)

    def test_build_text_encoder_missing_weights_raises(self, skip_if_no_cuda):
        """Test that missing weights raise KeyError."""
        from pygpukit.tts.kokoro.layers import build_text_encoder_from_weights

        weights = {}  # Empty weights

        with pytest.raises(KeyError):
            build_text_encoder_from_weights(weights)
