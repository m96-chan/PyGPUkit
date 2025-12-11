"""Data type definitions for PyGPUkit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DataTypeKind(Enum):
    """Enumeration of supported data type kinds."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"


@dataclass(frozen=True)
class DataType:
    """Represents a data type for GPU arrays.

    Attributes:
        kind: The kind of data type.
        itemsize: Size in bytes of each element.
        name: Human-readable name of the type.
    """

    kind: DataTypeKind
    itemsize: int
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"DataType({self.name})"

    def to_numpy_dtype(self) -> Any:
        """Convert to NumPy dtype."""
        import numpy as np

        dtype_map = {
            DataTypeKind.FLOAT32: np.float32,
            DataTypeKind.FLOAT64: np.float64,
            DataTypeKind.INT32: np.int32,
            DataTypeKind.INT64: np.int64,
        }
        return np.dtype(dtype_map[self.kind])

    @staticmethod
    def from_numpy_dtype(dtype: Any) -> DataType:
        """Create DataType from NumPy dtype."""
        import numpy as np

        dtype = np.dtype(dtype)
        name = dtype.name

        if name == "float32":
            return float32
        elif name == "float64":
            return float64
        elif name == "int32":
            return int32
        elif name == "int64":
            return int64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def from_string(name: str) -> DataType:
        """Create DataType from string name."""
        type_map = {
            "float32": float32,
            "float64": float64,
            "int32": int32,
            "int64": int64,
        }
        if name not in type_map:
            raise ValueError(f"Unsupported dtype string: {name}")
        return type_map[name]


# Pre-defined data types
float32 = DataType(DataTypeKind.FLOAT32, 4, "float32")
float64 = DataType(DataTypeKind.FLOAT64, 8, "float64")
int32 = DataType(DataTypeKind.INT32, 4, "int32")
int64 = DataType(DataTypeKind.INT64, 8, "int64")
