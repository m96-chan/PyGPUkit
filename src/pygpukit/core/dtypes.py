"""Data type definitions for PyGPUkit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DataTypeKind(Enum):
    """Enumeration of supported data type kinds."""

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"


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
            DataTypeKind.FLOAT64: np.float64,
            DataTypeKind.FLOAT32: np.float32,
            DataTypeKind.FLOAT16: np.float16,
            DataTypeKind.BFLOAT16: np.uint16,  # NumPy has no native bfloat16
            DataTypeKind.INT64: np.int64,
            DataTypeKind.INT32: np.int32,
            DataTypeKind.INT16: np.int16,
            DataTypeKind.INT8: np.int8,
            DataTypeKind.UINT8: np.uint8,
            DataTypeKind.INT4: np.uint8,  # Int4 packed as uint8
        }
        return np.dtype(dtype_map[self.kind])

    @staticmethod
    def from_numpy_dtype(dtype: Any) -> DataType:
        """Create DataType from NumPy dtype."""
        import numpy as np

        dtype = np.dtype(dtype)
        name = dtype.name

        if name == "float64":
            return float64
        elif name == "float32":
            return float32
        elif name == "float16":
            return float16
        elif name == "uint16":
            # uint16 is used as storage for bfloat16
            return bfloat16
        elif name == "int64":
            return int64
        elif name == "int32":
            return int32
        elif name == "int16":
            return int16
        elif name == "int8":
            return int8
        elif name == "uint8":
            return uint8
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def from_string(name: str) -> DataType:
        """Create DataType from string name."""
        type_map = {
            "float64": float64,
            "float32": float32,
            "float16": float16,
            "bfloat16": bfloat16,
            "int64": int64,
            "int32": int32,
            "int16": int16,
            "int8": int8,
            "uint8": uint8,
            "int4": int4,
        }
        if name not in type_map:
            raise ValueError(f"Unsupported dtype string: {name}")
        return type_map[name]


# Pre-defined data types
float64 = DataType(DataTypeKind.FLOAT64, 8, "float64")
float32 = DataType(DataTypeKind.FLOAT32, 4, "float32")
float16 = DataType(DataTypeKind.FLOAT16, 2, "float16")
bfloat16 = DataType(DataTypeKind.BFLOAT16, 2, "bfloat16")
int64 = DataType(DataTypeKind.INT64, 8, "int64")
int32 = DataType(DataTypeKind.INT32, 4, "int32")
int16 = DataType(DataTypeKind.INT16, 2, "int16")
int8 = DataType(DataTypeKind.INT8, 1, "int8")
uint8 = DataType(DataTypeKind.UINT8, 1, "uint8")
int4 = DataType(DataTypeKind.INT4, 1, "int4")  # 2 values per byte
