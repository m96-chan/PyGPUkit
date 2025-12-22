#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace pygpukit {

// Data type enumeration
enum class DataType {
    Float64,
    Float32,
    Float16,    // FP16 (half precision)
    BFloat16,   // BF16 (bfloat16)
    Int64,
    Int32,
    Int16,      // Signed 16-bit integer (for audio PCM)
    Int8,       // Signed 8-bit integer (for quantization)
    UInt8,      // Unsigned 8-bit integer
    Int4,       // 4-bit integer (packed, 2 values per byte)
};

// Get size in bytes for a data type
// Note: Int4 returns 1 (stores 2 values per byte, handled specially)
inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::Float64: return 8;
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::BFloat16: return 2;
        case DataType::Int64: return 8;
        case DataType::Int32: return 4;
        case DataType::Int16: return 2;
        case DataType::Int8: return 1;
        case DataType::UInt8: return 1;
        case DataType::Int4: return 1;  // 2 values per byte
        default: throw std::runtime_error("Unknown dtype");
    }
}

// Get string name for a data type
inline std::string dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::Float64: return "float64";
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::BFloat16: return "bfloat16";
        case DataType::Int64: return "int64";
        case DataType::Int32: return "int32";
        case DataType::Int16: return "int16";
        case DataType::Int8: return "int8";
        case DataType::UInt8: return "uint8";
        case DataType::Int4: return "int4";
        default: throw std::runtime_error("Unknown dtype");
    }
}

// Device pointer wrapper
using DevicePtr = void*;

// Error handling

// NVRTC error codes (matches nvrtcResult + custom codes)
enum class NvrtcErrorCode {
    Success = 0,
    OutOfMemory = 1,
    ProgramCreationFailure = 2,
    InvalidInput = 3,
    InvalidProgram = 4,
    InvalidOption = 5,
    Compilation = 6,
    BuiltinOperationFailure = 7,
    NoNameExpressionsAfterCompilation = 8,
    NoLoweredNamesBeforeCompilation = 9,
    NameExpressionNotValid = 10,
    InternalError = 11,
    // Custom error codes (1000+)
    NotLoaded = 1000,           // NVRTC DLL not loaded
    PtxLoadFailed = 1001,       // cuModuleLoadData failed
    FunctionNotFound = 1002,    // cuModuleGetFunction failed
    LaunchFailed = 1003,        // cuLaunchKernel failed
};

// Get string name for error code
inline const char* nvrtc_error_name(NvrtcErrorCode code) {
    switch (code) {
        case NvrtcErrorCode::Success: return "Success";
        case NvrtcErrorCode::OutOfMemory: return "OutOfMemory";
        case NvrtcErrorCode::ProgramCreationFailure: return "ProgramCreationFailure";
        case NvrtcErrorCode::InvalidInput: return "InvalidInput";
        case NvrtcErrorCode::InvalidProgram: return "InvalidProgram";
        case NvrtcErrorCode::InvalidOption: return "InvalidOption";
        case NvrtcErrorCode::Compilation: return "Compilation";
        case NvrtcErrorCode::BuiltinOperationFailure: return "BuiltinOperationFailure";
        case NvrtcErrorCode::NoNameExpressionsAfterCompilation: return "NoNameExpressionsAfterCompilation";
        case NvrtcErrorCode::NoLoweredNamesBeforeCompilation: return "NoLoweredNamesBeforeCompilation";
        case NvrtcErrorCode::NameExpressionNotValid: return "NameExpressionNotValid";
        case NvrtcErrorCode::InternalError: return "InternalError";
        case NvrtcErrorCode::NotLoaded: return "NotLoaded";
        case NvrtcErrorCode::PtxLoadFailed: return "PtxLoadFailed";
        case NvrtcErrorCode::FunctionNotFound: return "FunctionNotFound";
        case NvrtcErrorCode::LaunchFailed: return "LaunchFailed";
        default: return "Unknown";
    }
}

class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& msg) : std::runtime_error(msg) {}
};

class NvrtcError : public std::runtime_error {
public:
    explicit NvrtcError(const std::string& msg)
        : std::runtime_error(msg)
        , code_(NvrtcErrorCode::InternalError)
        , log_() {}

    NvrtcError(const std::string& msg, NvrtcErrorCode code)
        : std::runtime_error(msg)
        , code_(code)
        , log_() {}

    NvrtcError(const std::string& msg, NvrtcErrorCode code, const std::string& log)
        : std::runtime_error(msg)
        , code_(code)
        , log_(log) {}

    NvrtcErrorCode code() const { return code_; }
    const std::string& log() const { return log_; }

private:
    NvrtcErrorCode code_;
    std::string log_;
};

} // namespace pygpukit
