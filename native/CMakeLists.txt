cmake_minimum_required(VERSION 3.18)

# Check if we should skip native build (prebuilt modules are used)
if(DEFINED ENV{PYGPUKIT_SKIP_NATIVE_BUILD})
    message(STATUS "PYGPUKIT_SKIP_NATIVE_BUILD is set - skipping native module build")
    message(STATUS "Prebuilt native modules should be in src/pygpukit/")
    # Create a dummy project so cmake doesn't fail
    project(pygpukit_native_skip LANGUAGES NONE)
    return()
endif()

project(pygpukit_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# PyGPUkit v0.2.4+: Always build in driver-only mode for single-binary distribution
# Only nvcuda.dll (GPU driver) is required - no CUDA Toolkit needed at runtime
message(STATUS "Building in DRIVER-ONLY mode (single-binary distribution)")

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# CUTLASS (header-only library)
# Can be disabled via environment variable PYGPUKIT_DISABLE_CUTLASS=1
# Try multiple paths for CUTLASS (scikit-build-core may change working directory)
set(CUTLASS_DIR_CANDIDATES
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass"
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/cutlass"
    "${CMAKE_SOURCE_DIR}/../third_party/cutlass"
)
set(CUTLASS_FOUND FALSE)
foreach(CUTLASS_CANDIDATE ${CUTLASS_DIR_CANDIDATES})
    if(EXISTS "${CUTLASS_CANDIDATE}/include" AND NOT CUTLASS_FOUND)
        set(CUTLASS_DIR "${CUTLASS_CANDIDATE}")
        set(CUTLASS_FOUND TRUE)
    endif()
endforeach()

if(DEFINED ENV{PYGPUKIT_DISABLE_CUTLASS})
    message(STATUS "CUTLASS disabled via PYGPUKIT_DISABLE_CUTLASS environment variable")
    add_definitions(-DPYGPUKIT_HAS_CUTLASS=0)
elseif(CUTLASS_FOUND)
    message(STATUS "CUTLASS found at: ${CUTLASS_DIR}")
    include_directories(${CUTLASS_DIR}/include)
    include_directories(${CUTLASS_DIR}/tools/util/include)
    add_definitions(-DPYGPUKIT_HAS_CUTLASS=1)
    # Note: CUTLASS 3.x SM90+ features (Hopper/Blackwell) require SM90+ hardware
    # Disabled for now - will be enabled when SM90+ testing is available
    # add_definitions(-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1)
else()
    message(STATUS "CUTLASS not found at any of: ${CUTLASS_DIR_CANDIDATES}")
    message(STATUS "Using fallback kernels")
    add_definitions(-DPYGPUKIT_HAS_CUTLASS=0)
endif()

# Set default CUDA architectures if not specified
# PyGPUkit requires SM >= 80 (Ampere and newer)
# Older architectures (Pascal/Turing) are NOT supported
#
# Supported architectures:
# - SM 80 (A100): Ampere datacenter, 4-stage pipeline
# - SM 86 (RTX 30xx): Ampere consumer, 5-stage pipeline
# - SM 89 (RTX 40xx): Ada Lovelace, 6-stage pipeline
# - SM 90 (H100): Hopper, WGMMA/TMA
# - SM 100 (B100/B200): Blackwell datacenter
# - SM 120 (RTX 5090): Blackwell consumer (GeForce)
#
# For SM100+ (Blackwell), use CUDA 13.x and set CMAKE_CUDA_ARCHITECTURES env var
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Enable CUTLASS SM support based on target architectures
# _SUPPORTED macros enable host-side type definitions
# _ENABLED macros are auto-defined by CUTLASS based on __CUDA_ARCH__ during device compilation
string(FIND "${CMAKE_CUDA_ARCHITECTURES}" "90" SM90_POS)
string(FIND "${CMAKE_CUDA_ARCHITECTURES}" "100" SM100_POS)
string(FIND "${CMAKE_CUDA_ARCHITECTURES}" "120" SM120_POS)

# SM90 (Hopper) - FP8 GEMM with per-tensor scaling
# Also enable for SM100+ since they are forward compatible
if(NOT SM90_POS EQUAL -1 OR NOT SM100_POS EQUAL -1 OR NOT SM120_POS EQUAL -1)
    message(STATUS "Enabling CUTLASS SM90 (Hopper) support")
    add_definitions(-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1)
endif()

# SM100 (Blackwell datacenter)
# Also enable for SM120 since they are both Blackwell architecture
if(NOT SM100_POS EQUAL -1 OR NOT SM120_POS EQUAL -1)
    message(STATUS "Enabling CUTLASS SM100 (Blackwell datacenter) support")
    add_definitions(-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1)
endif()

# SM120 (Blackwell GeForce) - FP8 GEMM with blockwise scaling
# Note: Use 120a for full accelerated features (tensor cores, block-scaled MMA)
if(NOT SM120_POS EQUAL -1)
    message(STATUS "Enabling CUTLASS SM120 (Blackwell GeForce) support")
    add_definitions(-DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1)
    # For SM120a (full accelerated features), also enable feature macros
    string(FIND "${CMAKE_CUDA_ARCHITECTURES}" "120a" SM120A_POS)
    if(NOT SM120A_POS EQUAL -1)
        message(STATUS "  SM120a: Full accelerated features enabled")
    endif()
endif()

# Ampere-optimized compiler flags
# Add -v for verbose ptxas output to check register usage
# NOTE: Do NOT use -maxrregcount for CUTLASS - it needs many registers for optimal performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math --ptxas-options=-v -O3")

# Module name: can be overridden via PYGPUKIT_MODULE_SUFFIX for multi-CUDA builds
# E.g., PYGPUKIT_MODULE_SUFFIX=_cu129 produces _pygpukit_native_cu129
set(MODULE_SUFFIX "" CACHE STRING "Module name suffix (e.g., _cu129, _cu131)")
if(DEFINED ENV{PYGPUKIT_MODULE_SUFFIX})
    set(MODULE_SUFFIX $ENV{PYGPUKIT_MODULE_SUFFIX})
endif()

set(MODULE_NAME "_pygpukit_native${MODULE_SUFFIX}")
message(STATUS "Building native module: ${MODULE_NAME}")

# Build single pybind11 module with all sources
pybind11_add_module(${MODULE_NAME}
    # Core
    core/device.cpp
    core/device.cu
    core/memory.cpp
    core/memory.cu
    core/stream.cpp
    core/stream.cu
    core/event.cpp
    core/cuda_graph.cu
    # JIT
    jit/compiler.cpp
    jit/kernel.cpp
    jit/nvrtc_loader.cpp
    jit/cublaslt_loader.cpp
    # Ops - Modular structure
    ops/elementwise/elementwise.cu
    ops/unary/unary.cu
    ops/reduction/reduction.cu
    ops/matmul/matmul.cu
    ops/matmul/matmul_cutlass.cu
    # GEMM kernels
    ops/matmul/gemm/f32/f32/generic/f32_ampere.cu
    ops/matmul/gemm/fp8/f32/sm90/fp8_cutlass.cu
    ops/matmul/gemm/fp8/f32/sm100/fp8_blockwise.cu
    ops/matmul/gemm/fp8/bf16/sm120/fp8_blockwise.cu
    ops/matmul/gemm/fp8/bf16/sm120/w8a16_gemm.cu
    ops/matmul/gemm/fp8/bf16/sm120/w8a16_cutlass.cu
    ops/matmul/gemm/fp8/bf16/sm120/grouped_gemm.cu
    ops/matmul/gemm/fp8/fp8/sm120/fp8_cutlass.cu
    ops/matmul/gemm/fp8/fp8/sm120/fp8_cutlass_v2.cu
    ops/matmul/gemm/int8/int8/sm120/int8_native.cu
    ops/matmul/gemm/int4/int4/sm120/int4_via_int8.cu
    ops/matmul/gemm/nvf4/bf16/sm120/nvf4_cutlass.cu
    ops/matmul/gemm/nvf4/bf16/sm120/nvf4_nvf4_cutlass.cu
    # GEMV kernels
    ops/matmul/gemv/bf16/bf16/sm120/nvf4.cu
    ops/matmul/gemv/bf16/bf16/sm120/nvf4_kernels.cu
    ops/matmul/gemv/bf16/bf16/sm120/fp8_opt_kernels.cu
    ops/matmul/gemv/fp8/fp8/sm120/fp8_gemv.cu
    ops/matmul/gemv/nvf4/nvf4/sm120/nvf4_gemv.cu
    ops/matmul/gemv/int4/int4/sm120/int4_gemv.cu
    ops/nn/nn.cu
    ops/quantize/quantize.cu
    ops/attention/paged_attention.cu
    ops/batch/continuous_batching.cu
    ops/sampling/sampling.cu
    ops/audio/audio.cu
    ops/moe/moe.cu
    # Bindings
    bindings/module.cpp
    bindings/core_bindings.cpp
    bindings/jit_bindings.cpp
    bindings/ops_bindings.cpp
)

# Link only cuda_driver (no cudart, no nvrtc/cublasLt link-time dependency)
# NVRTC is loaded dynamically at runtime via nvrtc_loader.cpp
# cuBLASLt is loaded dynamically at runtime via cublaslt_loader.cpp
# FFT is implemented with custom Radix-2 kernel (no cuFFT dependency)
# This enables single-binary distribution that works with just GPU drivers
target_link_libraries(${MODULE_NAME} PRIVATE
    CUDA::cuda_driver
)

# IMPORTANT: Do NOT enable CUDA_SEPARABLE_COMPILATION
# It causes 15x performance degradation for CUTLASS kernels
# due to prevented inlining and indirect function calls
set_target_properties(${MODULE_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Install the module to the correct location for scikit-build-core
# scikit-build-core's wheel.install-dir already sets the base to pygpukit
install(TARGETS ${MODULE_NAME}
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
