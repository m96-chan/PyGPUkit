cmake_minimum_required(VERSION 3.18)
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

# Set default CUDA architectures if not specified
# PyGPUkit requires SM >= 80 (Ampere and newer)
# Older architectures (Pascal/Turing) are NOT supported
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Ampere-optimized compiler flags
# Add -v for verbose ptxas output to check register usage
# Limit registers to 128 to prevent spilling issues with WMMA kernels
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math --ptxas-options=-v -maxrregcount=128")

# Build single pybind11 module with all sources
pybind11_add_module(_pygpukit_native
    # Core
    core/device.cpp
    core/device.cu
    core/memory.cpp
    core/memory.cu
    core/stream.cpp
    core/stream.cu
    # JIT
    jit/compiler.cpp
    jit/kernel.cpp
    jit/nvrtc_loader.cpp
    # Ops
    ops/basic.cu
    # Bindings
    bindings/module.cpp
    bindings/core_bindings.cpp
    bindings/jit_bindings.cpp
    bindings/ops_bindings.cpp
)

# Link only cuda_driver (no cudart, no nvrtc link-time dependency)
# NVRTC is loaded dynamically at runtime via nvrtc_loader.cpp
# This enables single-binary distribution that works with just GPU drivers
target_link_libraries(_pygpukit_native PRIVATE
    CUDA::cuda_driver
)

set_target_properties(_pygpukit_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Install the module to the correct location for scikit-build-core
# scikit-build-core's wheel.install-dir already sets the base to pygpukit
install(TARGETS _pygpukit_native
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
