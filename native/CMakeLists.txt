cmake_minimum_required(VERSION 3.18)
project(pygpukit_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Core library (CUDA backend)
add_library(pygpukit_core
    core/device.cpp
    core/device.cu
    core/memory.cpp
    core/memory.cu
    core/stream.cpp
    core/stream.cu
)

target_link_libraries(pygpukit_core
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(pygpukit_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# JIT library (NVRTC)
add_library(pygpukit_jit
    jit/compiler.cpp
    jit/kernel.cpp
)

target_link_libraries(pygpukit_jit
    CUDA::nvrtc
    CUDA::cuda_driver
    pygpukit_core
)

set_target_properties(pygpukit_jit PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Ops library (CUDA kernels)
add_library(pygpukit_ops
    ops/basic.cu
)

target_link_libraries(pygpukit_ops
    pygpukit_core
)

set_target_properties(pygpukit_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Python bindings
pybind11_add_module(_pygpukit_native
    bindings/module.cpp
    bindings/core_bindings.cpp
    bindings/jit_bindings.cpp
    bindings/ops_bindings.cpp
)

target_link_libraries(_pygpukit_native PRIVATE
    pygpukit_core
    pygpukit_jit
    pygpukit_ops
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install
install(TARGETS _pygpukit_native
    LIBRARY DESTINATION pygpukit
)
