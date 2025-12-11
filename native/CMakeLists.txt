cmake_minimum_required(VERSION 3.18)
project(pygpukit_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Set default CUDA architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

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
    # Ops
    ops/basic.cu
    # Bindings
    bindings/module.cpp
    bindings/core_bindings.cpp
    bindings/jit_bindings.cpp
    bindings/ops_bindings.cpp
)

target_link_libraries(_pygpukit_native PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
)

set_target_properties(_pygpukit_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Install the module to the correct location for scikit-build-core
install(TARGETS _pygpukit_native
    LIBRARY DESTINATION pygpukit
    RUNTIME DESTINATION pygpukit
)
