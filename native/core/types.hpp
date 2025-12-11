#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace pygpukit {

// Data type enumeration
enum class DataType {
    Float32,
    Float64,
    Int32,
    Int64
};

// Get size in bytes for a data type
inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float64: return 8;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        default: throw std::runtime_error("Unknown dtype");
    }
}

// Get string name for a data type
inline std::string dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float64: return "float64";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        default: throw std::runtime_error("Unknown dtype");
    }
}

// Device pointer wrapper
using DevicePtr = void*;

// Error handling
class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& msg) : std::runtime_error(msg) {}
};

class NvrtcError : public std::runtime_error {
public:
    explicit NvrtcError(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace pygpukit
