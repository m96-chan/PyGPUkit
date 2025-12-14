#pragma once

// Dynamic NVRTC Loader
// Loads NVRTC at runtime without requiring link-time dependency.
// This allows the wheel to be self-contained and work without CUDA Toolkit.

#include <string>
#include <tuple>

namespace pygpukit {
namespace nvrtc {

// NVRTC result type (matches nvrtcResult)
enum class Result {
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
    NotLoaded = 1000,  // Custom: NVRTC not loaded
};

// Opaque program handle
using Program = void*;

// Initialize NVRTC loader (attempts to find and load NVRTC DLL/SO)
// Returns true if NVRTC was loaded successfully
bool initialize();

// Check if NVRTC is available
bool is_available();

// Get the path to the loaded NVRTC library (empty if not loaded)
std::string get_library_path();

// Get NVRTC version (returns {0,0} if not available)
std::tuple<int, int> get_version();

// NVRTC API wrappers
// These return Result::NotLoaded if NVRTC is not available

Result create_program(
    Program* prog,
    const char* src,
    const char* name,
    int num_headers,
    const char* const* headers,
    const char* const* include_names
);

Result destroy_program(Program* prog);

Result compile_program(
    Program prog,
    int num_options,
    const char* const* options
);

Result get_ptx_size(Program prog, size_t* ptx_size);

Result get_ptx(Program prog, char* ptx);

Result get_program_log_size(Program prog, size_t* log_size);

Result get_program_log(Program prog, char* log);

// Get error string for result code
const char* get_error_string(Result result);

} // namespace nvrtc
} // namespace pygpukit
