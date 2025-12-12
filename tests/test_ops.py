"""Tests for basic operations."""

import numpy as np
import pytest

import pygpukit as gp


class TestAddOperation:
    """Tests for add operation."""

    def test_add_same_shape(self):
        """Test adding arrays of same shape."""
        a = gp.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        b = gp.from_numpy(np.array([4, 5, 6], dtype=np.float32))

        c = gp.add(a, b)

        result = c.to_numpy()
        expected = np.array([5, 7, 9], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_add_2d_arrays(self):
        """Test adding 2D arrays."""
        a = gp.from_numpy(np.ones((3, 3), dtype=np.float32))
        b = gp.from_numpy(np.ones((3, 3), dtype=np.float32) * 2)

        c = gp.add(a, b)

        result = c.to_numpy()
        expected = np.ones((3, 3), dtype=np.float32) * 3
        np.testing.assert_array_almost_equal(result, expected)

    def test_add_preserves_dtype(self):
        """Test that add preserves data type."""
        a = gp.zeros((10,), dtype="float64")
        b = gp.ones((10,), dtype="float64")

        c = gp.add(a, b)

        assert c.dtype == gp.float64

    def test_add_different_shapes_raises(self):
        """Test that adding different shapes raises error."""
        a = gp.zeros((10,), dtype="float32")
        b = gp.zeros((20,), dtype="float32")

        with pytest.raises(ValueError, match="shape"):
            gp.add(a, b)

    def test_add_large_array(self):
        """Test adding large arrays."""
        size = 10000
        a_np = np.random.rand(size).astype(np.float32)
        b_np = np.random.rand(size).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.add(a, b)

        result = c.to_numpy()
        expected = a_np + b_np
        np.testing.assert_array_almost_equal(result, expected)


class TestMulOperation:
    """Tests for mul operation."""

    def test_mul_same_shape(self):
        """Test multiplying arrays of same shape."""
        a = gp.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        b = gp.from_numpy(np.array([4, 5, 6], dtype=np.float32))

        c = gp.mul(a, b)

        result = c.to_numpy()
        expected = np.array([4, 10, 18], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_mul_2d_arrays(self):
        """Test multiplying 2D arrays."""
        a = gp.from_numpy(np.ones((3, 3), dtype=np.float32) * 2)
        b = gp.from_numpy(np.ones((3, 3), dtype=np.float32) * 3)

        c = gp.mul(a, b)

        result = c.to_numpy()
        expected = np.ones((3, 3), dtype=np.float32) * 6
        np.testing.assert_array_almost_equal(result, expected)

    def test_mul_preserves_dtype(self):
        """Test that mul preserves data type."""
        a = gp.ones((10,), dtype="float64")
        b = gp.ones((10,), dtype="float64")

        c = gp.mul(a, b)

        assert c.dtype == gp.float64

    def test_mul_different_shapes_raises(self):
        """Test that multiplying different shapes raises error."""
        a = gp.zeros((10,), dtype="float32")
        b = gp.zeros((20,), dtype="float32")

        with pytest.raises(ValueError, match="shape"):
            gp.mul(a, b)

    def test_mul_by_zeros(self):
        """Test multiplying by zeros."""
        a = gp.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        b = gp.zeros((3,), dtype="float32")

        c = gp.mul(a, b)

        result = c.to_numpy()
        expected = np.zeros(3, dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


class TestMatmulOperation:
    """Tests for matmul operation."""

    def test_matmul_2d(self):
        """Test matrix multiplication of 2D arrays."""
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_non_square(self):
        """Test matrix multiplication of non-square matrices."""
        a_np = np.random.rand(3, 4).astype(np.float32)
        b_np = np.random.rand(4, 5).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        assert c.shape == (3, 5)
        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_identity(self):
        """Test multiplication with identity matrix."""
        a_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        eye_np = np.eye(3, dtype=np.float32)

        a = gp.from_numpy(a_np)
        eye = gp.from_numpy(eye_np)

        c = gp.matmul(a, eye)

        result = c.to_numpy()
        np.testing.assert_array_almost_equal(result, a_np)

    def test_matmul_incompatible_shapes_raises(self):
        """Test that incompatible shapes raise error."""
        a = gp.zeros((3, 4), dtype="float32")
        b = gp.zeros((5, 6), dtype="float32")

        with pytest.raises(ValueError, match="shape|dimension"):
            gp.matmul(a, b)

    def test_matmul_1d_raises(self):
        """Test that 1D arrays raise error."""
        a = gp.zeros((10,), dtype="float32")
        b = gp.zeros((10,), dtype="float32")

        with pytest.raises(ValueError, match="2D|dimension"):
            gp.matmul(a, b)

    def test_matmul_preserves_dtype(self):
        """Test that matmul preserves data type."""
        a = gp.ones((3, 3), dtype="float64")
        b = gp.ones((3, 3), dtype="float64")

        c = gp.matmul(a, b)

        assert c.dtype == gp.float64

    def test_matmul_large_matrices(self):
        """Test matmul with larger matrices."""
        a_np = np.random.rand(64, 64).astype(np.float32)
        b_np = np.random.rand(64, 64).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)


class TestMatmulTiled:
    """Tests for tiled matmul optimization (Issue #26).

    These tests verify correctness for various matrix sizes including:
    - Tile-aligned sizes (multiples of 16/32)
    - Non-aligned sizes
    - Large matrices
    """

    @pytest.mark.parametrize("size", [16, 32, 64, 128, 256])
    def test_matmul_tile_aligned_square(self, size: int):
        """Test matmul with tile-aligned square matrices."""
        np.random.seed(42)
        a_np = np.random.rand(size, size).astype(np.float32)
        b_np = np.random.rand(size, size).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    @pytest.mark.parametrize("size", [17, 33, 65, 100, 129, 200])
    def test_matmul_non_aligned_square(self, size: int):
        """Test matmul with non-tile-aligned square matrices."""
        np.random.seed(42)
        a_np = np.random.rand(size, size).astype(np.float32)
        b_np = np.random.rand(size, size).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    @pytest.mark.parametrize(
        "m,k,n",
        [
            (32, 64, 32),  # Aligned rectangular
            (64, 32, 128),  # Aligned rectangular
            (33, 65, 17),  # Non-aligned rectangular
            (100, 50, 75),  # Non-aligned rectangular
            (128, 256, 64),  # Large aligned
        ],
    )
    def test_matmul_rectangular(self, m: int, k: int, n: int):
        """Test matmul with rectangular matrices of various sizes."""
        np.random.seed(42)
        a_np = np.random.rand(m, k).astype(np.float32)
        b_np = np.random.rand(k, n).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        assert c.shape == (m, n)
        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_matmul_large_512(self):
        """Test matmul with 512x512 matrices (performance test)."""
        np.random.seed(42)
        a_np = np.random.rand(512, 512).astype(np.float32)
        b_np = np.random.rand(512, 512).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_matmul_float64_tiled(self):
        """Test tiled matmul with float64."""
        np.random.seed(42)
        a_np = np.random.rand(64, 64).astype(np.float64)
        b_np = np.random.rand(64, 64).astype(np.float64)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        assert c.dtype == gp.float64
        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_matmul_tall_matrix(self):
        """Test matmul with tall matrix (M >> N)."""
        np.random.seed(42)
        a_np = np.random.rand(256, 32).astype(np.float32)
        b_np = np.random.rand(32, 16).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        assert c.shape == (256, 16)
        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_matmul_wide_matrix(self):
        """Test matmul with wide matrix (N >> M)."""
        np.random.seed(42)
        a_np = np.random.rand(16, 32).astype(np.float32)
        b_np = np.random.rand(32, 256).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        assert c.shape == (16, 256)
        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_matmul_single_row_col(self):
        """Test matmul edge case: single row times single column."""
        np.random.seed(42)
        a_np = np.random.rand(1, 64).astype(np.float32)
        b_np = np.random.rand(64, 1).astype(np.float32)

        a = gp.from_numpy(a_np)
        b = gp.from_numpy(b_np)

        c = gp.matmul(a, b)

        assert c.shape == (1, 1)
        result = c.to_numpy()
        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)
