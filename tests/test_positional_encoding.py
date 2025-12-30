"""Tests for positional encoding operations (PoPE, ALiBi, YaRN, NTK)."""

import numpy as np
import pytest

from pygpukit import from_numpy
from pygpukit.ops.nn import (
    alibi_compute_bias,
    alibi_init_slopes,
    pope_init_encoding,
    pope_inplace,
    rope_init_linear,
    rope_init_ntk_aware,
    rope_init_yarn,
)


class TestPoPE:
    """Test PoPE (Positional Encoding) operations."""

    def test_pope_init_encoding_shape(self):
        """Test that PoPE encoding has correct shape."""
        max_seq_len = 512
        head_dim = 128

        encoding = pope_init_encoding(max_seq_len, head_dim)

        assert encoding.shape == (max_seq_len, head_dim)
        assert str(encoding.dtype) == "float32"

    def test_pope_init_encoding_sinusoidal(self):
        """Test that PoPE encoding follows sinusoidal pattern."""
        max_seq_len = 64
        head_dim = 32

        encoding = pope_init_encoding(max_seq_len, head_dim)
        enc_np = encoding.to_numpy()

        # Position 0 should have sin(0) = 0 for even dims
        # and cos(0) = 1 for odd dims (approximately)
        # Due to frequency scaling, only low-frequency dims will be close to 0/1
        assert enc_np[0, 0] == pytest.approx(0.0, abs=0.01)  # sin(0) = 0
        assert enc_np[0, 1] == pytest.approx(1.0, abs=0.01)  # cos(0) = 1

    def test_pope_inplace(self):
        """Test PoPE in-place application."""
        seq_len = 4
        n_heads = 2
        head_dim = 8
        max_seq_len = 16

        q = from_numpy(np.ones((seq_len, n_heads, head_dim), dtype=np.float32))
        k = from_numpy(np.ones((seq_len, n_heads, head_dim), dtype=np.float32))
        encoding = pope_init_encoding(max_seq_len, head_dim)

        q_before = q.to_numpy().copy()
        k_before = k.to_numpy().copy()

        pope_inplace(q, k, encoding)

        q_after = q.to_numpy()
        k_after = k.to_numpy()

        # Values should be modified (encoding added)
        assert not np.allclose(q_after, q_before)
        assert not np.allclose(k_after, k_before)


class TestALiBi:
    """Test ALiBi (Attention with Linear Biases) operations."""

    def test_alibi_init_slopes_shape(self):
        """Test that ALiBi slopes have correct shape."""
        num_heads = 8

        slopes = alibi_init_slopes(num_heads)

        assert slopes.shape == (num_heads,)
        assert str(slopes.dtype) == "float32"

    def test_alibi_init_slopes_values(self):
        """Test that ALiBi slopes follow the formula m_h = 2^(-8*h/n)."""
        num_heads = 8

        slopes = alibi_init_slopes(num_heads)
        slopes_np = slopes.to_numpy()

        # Verify formula: m_h = 2^(-8 * (h+1) / num_heads)
        for h in range(num_heads):
            expected = 2 ** (-8 * (h + 1) / num_heads)
            assert slopes_np[h] == pytest.approx(expected, rel=1e-5)

    def test_alibi_compute_bias_shape(self):
        """Test that ALiBi bias has correct shape."""
        seq_len = 32
        num_heads = 4

        slopes = alibi_init_slopes(num_heads)
        bias = alibi_compute_bias(seq_len, num_heads, slopes)

        assert bias.shape == (num_heads, seq_len, seq_len)
        assert str(bias.dtype) == "float32"

    def test_alibi_compute_bias_causal(self):
        """Test that ALiBi bias is causal (upper triangular is -inf)."""
        seq_len = 8
        num_heads = 2

        slopes = alibi_init_slopes(num_heads)
        bias = alibi_compute_bias(seq_len, num_heads, slopes, causal=True)
        bias_np = bias.to_numpy()

        # Check that upper triangular (j > i) is very negative (causal mask)
        for h in range(num_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    if j > i:
                        assert bias_np[h, i, j] < -1e8  # Should be -inf or very negative

    def test_alibi_compute_bias_diagonal_zero(self):
        """Test that ALiBi bias diagonal is zero (distance = 0)."""
        seq_len = 8
        num_heads = 2

        slopes = alibi_init_slopes(num_heads)
        bias = alibi_compute_bias(seq_len, num_heads, slopes)
        bias_np = bias.to_numpy()

        # Diagonal should be 0 (distance = 0)
        for h in range(num_heads):
            for i in range(seq_len):
                assert bias_np[h, i, i] == pytest.approx(0.0, abs=1e-6)

    def test_alibi_compute_bias_linear_decrease(self):
        """Test that ALiBi bias decreases linearly with distance."""
        seq_len = 8
        num_heads = 4

        slopes = alibi_init_slopes(num_heads)
        bias = alibi_compute_bias(seq_len, num_heads, slopes)
        bias_np = bias.to_numpy()
        slopes_np = slopes.to_numpy()

        # Check that bias[h, i, j] = -slope * (i - j) for j <= i
        for h in range(num_heads):
            slope = slopes_np[h]
            for i in range(seq_len):
                for j in range(i + 1):  # Only lower triangular
                    expected = -slope * (i - j)
                    assert bias_np[h, i, j] == pytest.approx(expected, rel=1e-4)


class TestRoPEExtensions:
    """Test RoPE extension methods (NTK, YaRN, Linear)."""

    def test_rope_init_ntk_aware_shape(self):
        """Test that NTK-aware RoPE tables have correct shape."""
        max_seq_len = 512
        head_dim = 128

        cos, sin = rope_init_ntk_aware(max_seq_len, head_dim)

        assert cos.shape == (max_seq_len, head_dim)
        assert sin.shape == (max_seq_len, head_dim)
        assert str(cos.dtype) == "float32"
        assert str(sin.dtype) == "float32"

    def test_rope_init_ntk_aware_scale(self):
        """Test that NTK-aware scaling affects frequencies."""
        max_seq_len = 128
        head_dim = 64

        cos1, sin1 = rope_init_ntk_aware(max_seq_len, head_dim, scale=1.0)
        cos2, sin2 = rope_init_ntk_aware(max_seq_len, head_dim, scale=2.0)

        cos1_np = cos1.to_numpy()
        cos2_np = cos2.to_numpy()

        # With scale > 1, frequencies should be different
        assert not np.allclose(cos1_np, cos2_np)

    def test_rope_init_yarn_shape(self):
        """Test that YaRN RoPE tables have correct shape."""
        max_seq_len = 1024
        head_dim = 128

        cos, sin = rope_init_yarn(
            max_seq_len,
            head_dim,
            scale=2.0,
            original_max_len=512,
        )

        assert cos.shape == (max_seq_len, head_dim)
        assert sin.shape == (max_seq_len, head_dim)

    def test_rope_init_yarn_vs_linear(self):
        """Test that YaRN produces different results than linear interpolation."""
        max_seq_len = 256
        head_dim = 64
        scale = 2.0

        cos_yarn, sin_yarn = rope_init_yarn(max_seq_len, head_dim, scale=scale)
        cos_linear, sin_linear = rope_init_linear(max_seq_len, head_dim, scale=scale)

        # YaRN should produce different frequencies than linear
        assert not np.allclose(cos_yarn.to_numpy(), cos_linear.to_numpy())

    def test_rope_init_linear_shape(self):
        """Test that linear interpolation RoPE tables have correct shape."""
        max_seq_len = 512
        head_dim = 128

        cos, sin = rope_init_linear(max_seq_len, head_dim)

        assert cos.shape == (max_seq_len, head_dim)
        assert sin.shape == (max_seq_len, head_dim)

    def test_rope_tables_normalized(self):
        """Test that cos^2 + sin^2 = 1 for all positions and dimensions."""
        max_seq_len = 64
        head_dim = 32

        for init_fn in [rope_init_ntk_aware, rope_init_linear]:
            cos, sin = init_fn(max_seq_len, head_dim)
            cos_np = cos.to_numpy()
            sin_np = sin.to_numpy()

            # cos^2 + sin^2 should be ~1
            sum_sq = cos_np**2 + sin_np**2
            np.testing.assert_allclose(sum_sq, 1.0, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
