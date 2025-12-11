"""Tests for GPUArray."""

import numpy as np

from pygpukit.core.dtypes import float32, float64, int32, int64


class TestGPUArrayCreation:
    """Tests for GPUArray creation."""

    def test_zeros_creates_array(self):
        """Test that zeros creates an array filled with zeros."""
        import pygpukit as gp

        arr = gp.zeros((10, 10), dtype="float32")
        assert arr.shape == (10, 10)
        assert arr.dtype == float32
        assert arr.size == 100

        result = arr.to_numpy()
        np.testing.assert_array_equal(result, np.zeros((10, 10), dtype=np.float32))

    def test_ones_creates_array(self):
        """Test that ones creates an array filled with ones."""
        import pygpukit as gp

        arr = gp.ones((5, 5), dtype="float32")
        assert arr.shape == (5, 5)
        assert arr.dtype == float32

        result = arr.to_numpy()
        np.testing.assert_array_equal(result, np.ones((5, 5), dtype=np.float32))

    def test_empty_creates_array(self):
        """Test that empty creates an uninitialized array."""
        import pygpukit as gp

        arr = gp.empty((3, 3), dtype="float64")
        assert arr.shape == (3, 3)
        assert arr.dtype == float64
        assert arr.size == 9

    def test_from_numpy_creates_array(self):
        """Test that from_numpy creates array from NumPy array."""
        import pygpukit as gp

        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        arr = gp.from_numpy(np_arr)

        assert arr.shape == (2, 3)
        assert arr.dtype == float32

        result = arr.to_numpy()
        np.testing.assert_array_equal(result, np_arr)

    def test_zeros_with_different_dtypes(self):
        """Test zeros with different data types."""
        import pygpukit as gp

        for dtype_str, dtype_obj in [
            ("float32", float32),
            ("float64", float64),
            ("int32", int32),
            ("int64", int64),
        ]:
            arr = gp.zeros((4,), dtype=dtype_str)
            assert arr.dtype == dtype_obj

    def test_zeros_1d_array(self):
        """Test creating 1D array."""
        import pygpukit as gp

        arr = gp.zeros((100,), dtype="float32")
        assert arr.shape == (100,)
        assert arr.ndim == 1

    def test_zeros_3d_array(self):
        """Test creating 3D array."""
        import pygpukit as gp

        arr = gp.zeros((2, 3, 4), dtype="float32")
        assert arr.shape == (2, 3, 4)
        assert arr.ndim == 3
        assert arr.size == 24


class TestGPUArrayProperties:
    """Tests for GPUArray properties."""

    def test_shape_property(self):
        """Test shape property."""
        import pygpukit as gp

        arr = gp.zeros((5, 10, 15), dtype="float32")
        assert arr.shape == (5, 10, 15)

    def test_dtype_property(self):
        """Test dtype property."""
        import pygpukit as gp

        arr = gp.zeros((5,), dtype="int64")
        assert arr.dtype == int64

    def test_size_property(self):
        """Test size property."""
        import pygpukit as gp

        arr = gp.zeros((3, 4, 5), dtype="float32")
        assert arr.size == 60

    def test_ndim_property(self):
        """Test ndim property."""
        import pygpukit as gp

        arr1 = gp.zeros((10,), dtype="float32")
        assert arr1.ndim == 1

        arr2 = gp.zeros((10, 20), dtype="float32")
        assert arr2.ndim == 2

        arr3 = gp.zeros((2, 3, 4, 5), dtype="float32")
        assert arr3.ndim == 4

    def test_nbytes_property(self):
        """Test nbytes property."""
        import pygpukit as gp

        arr_f32 = gp.zeros((100,), dtype="float32")
        assert arr_f32.nbytes == 400  # 100 * 4 bytes

        arr_f64 = gp.zeros((100,), dtype="float64")
        assert arr_f64.nbytes == 800  # 100 * 8 bytes

    def test_itemsize_property(self):
        """Test itemsize property."""
        import pygpukit as gp

        arr_f32 = gp.zeros((10,), dtype="float32")
        assert arr_f32.itemsize == 4

        arr_i64 = gp.zeros((10,), dtype="int64")
        assert arr_i64.itemsize == 8


class TestGPUArrayDataTransfer:
    """Tests for data transfer between CPU and GPU."""

    def test_to_numpy_returns_copy(self):
        """Test that to_numpy returns a copy."""
        import pygpukit as gp

        np_arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        arr = gp.from_numpy(np_arr)

        result = arr.to_numpy()
        np.testing.assert_array_equal(result, np_arr)

        # Modify result should not affect original
        result[0] = 999
        result2 = arr.to_numpy()
        assert result2[0] == 1

    def test_from_numpy_copies_data(self):
        """Test that from_numpy copies data."""
        import pygpukit as gp

        np_arr = np.array([1, 2, 3], dtype=np.float32)
        arr = gp.from_numpy(np_arr)

        # Modify original should not affect GPU array
        np_arr[0] = 999

        result = arr.to_numpy()
        assert result[0] == 1

    def test_roundtrip_preserves_data(self):
        """Test that CPU->GPU->CPU roundtrip preserves data."""
        import pygpukit as gp

        original = np.random.rand(10, 10).astype(np.float32)
        arr = gp.from_numpy(original)
        result = arr.to_numpy()

        np.testing.assert_array_almost_equal(result, original)

    def test_roundtrip_different_shapes(self):
        """Test roundtrip with various shapes."""
        import pygpukit as gp

        shapes = [(100,), (10, 10), (5, 5, 4), (2, 3, 4, 5)]
        for shape in shapes:
            original = np.random.rand(*shape).astype(np.float32)
            arr = gp.from_numpy(original)
            result = arr.to_numpy().reshape(shape)
            np.testing.assert_array_almost_equal(result, original)


class TestGPUArrayMemoryManagement:
    """Tests for memory management."""

    def test_array_can_be_deleted(self):
        """Test that arrays can be deleted without error."""
        import pygpukit as gp

        arr = gp.zeros((100, 100), dtype="float32")
        del arr  # Should not raise

    def test_multiple_arrays_independent(self):
        """Test that multiple arrays are independent."""
        import pygpukit as gp

        arr1 = gp.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        arr2 = gp.from_numpy(np.array([4, 5, 6], dtype=np.float32))

        result1 = arr1.to_numpy()
        result2 = arr2.to_numpy()

        np.testing.assert_array_equal(result1, [1, 2, 3])
        np.testing.assert_array_equal(result2, [4, 5, 6])


class TestGPUArrayRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test repr output."""
        import pygpukit as gp

        arr = gp.zeros((10, 20), dtype="float32")
        repr_str = repr(arr)

        assert "GPUArray" in repr_str
        assert "(10, 20)" in repr_str
        assert "float32" in repr_str

    def test_str(self):
        """Test str output."""
        import pygpukit as gp

        arr = gp.zeros((5,), dtype="int32")
        str_output = str(arr)

        assert "GPUArray" in str_output
