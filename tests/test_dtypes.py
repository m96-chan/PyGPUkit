"""Tests for data types."""

import numpy as np
import pytest

from pygpukit.core.dtypes import DataType, DataTypeKind, float32, float64, int32, int64


class TestDataType:
    """Tests for DataType class."""

    def test_float32_properties(self):
        """Test float32 data type properties."""
        assert float32.kind == DataTypeKind.FLOAT32
        assert float32.itemsize == 4
        assert float32.name == "float32"

    def test_float64_properties(self):
        """Test float64 data type properties."""
        assert float64.kind == DataTypeKind.FLOAT64
        assert float64.itemsize == 8
        assert float64.name == "float64"

    def test_int32_properties(self):
        """Test int32 data type properties."""
        assert int32.kind == DataTypeKind.INT32
        assert int32.itemsize == 4
        assert int32.name == "int32"

    def test_int64_properties(self):
        """Test int64 data type properties."""
        assert int64.kind == DataTypeKind.INT64
        assert int64.itemsize == 8
        assert int64.name == "int64"

    def test_str_representation(self):
        """Test string representation."""
        assert str(float32) == "float32"
        assert str(int64) == "int64"

    def test_repr_representation(self):
        """Test repr representation."""
        assert repr(float32) == "DataType(float32)"
        assert repr(int64) == "DataType(int64)"

    def test_to_numpy_dtype(self):
        """Test conversion to NumPy dtype."""
        assert float32.to_numpy_dtype() == np.dtype(np.float32)
        assert float64.to_numpy_dtype() == np.dtype(np.float64)
        assert int32.to_numpy_dtype() == np.dtype(np.int32)
        assert int64.to_numpy_dtype() == np.dtype(np.int64)

    def test_from_numpy_dtype(self):
        """Test creation from NumPy dtype."""
        assert DataType.from_numpy_dtype(np.dtype(np.float32)) == float32
        assert DataType.from_numpy_dtype(np.dtype(np.float64)) == float64
        assert DataType.from_numpy_dtype(np.dtype(np.int32)) == int32
        assert DataType.from_numpy_dtype(np.dtype(np.int64)) == int64

    def test_from_numpy_dtype_unsupported(self):
        """Test that unsupported dtypes raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            DataType.from_numpy_dtype(np.dtype(np.complex64))

    def test_from_string(self):
        """Test creation from string."""
        assert DataType.from_string("float32") == float32
        assert DataType.from_string("float64") == float64
        assert DataType.from_string("int32") == int32
        assert DataType.from_string("int64") == int64

    def test_from_string_unsupported(self):
        """Test that unsupported strings raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype string"):
            DataType.from_string("complex64")

    def test_datatype_equality(self):
        """Test that data types can be compared for equality."""
        assert float32 == float32
        assert float32 != float64
        assert int32 != float32

    def test_datatype_immutable(self):
        """Test that data types are immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            float32.itemsize = 8
