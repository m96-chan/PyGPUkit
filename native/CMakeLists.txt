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

# CUTLASS (header-only library)
# Can be disabled via environment variable PYGPUKIT_DISABLE_CUTLASS=1
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass")
if(DEFINED ENV{PYGPUKIT_DISABLE_CUTLASS})
    message(STATUS "CUTLASS disabled via PYGPUKIT_DISABLE_CUTLASS environment variable")
    add_definitions(-DPYGPUKIT_HAS_CUTLASS=0)
elseif(EXISTS "${CUTLASS_DIR}/include")
    message(STATUS "CUTLASS found at: ${CUTLASS_DIR}")
    include_directories(${CUTLASS_DIR}/include)
    include_directories(${CUTLASS_DIR}/tools/util/include)
    add_definitions(-DPYGPUKIT_HAS_CUTLASS=1)
    # Note: CUTLASS 3.x SM90+ features (Hopper/Blackwell) require SM90+ hardware
    # Disabled for now - will be enabled when SM90+ testing is available
    # add_definitions(-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1)
else()
    message(STATUS "CUTLASS not found, using fallback kernels")
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
#
# For SM100+ (Blackwell), use CUDA 13.x and set CMAKE_CUDA_ARCHITECTURES env var
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Ampere-optimized compiler flags
# Add -v for verbose ptxas output to check register usage
# NOTE: Do NOT use -maxrregcount for CUTLASS - it needs many registers for optimal performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math --ptxas-options=-v -O3")

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
    # Ops - Modular structure
    ops/elementwise/elementwise.cu
    ops/unary/unary.cu
    ops/reduction/reduction.cu
    ops/matmul/matmul.cu
    ops/matmul/matmul_cutlass.cu
    ops/nn/nn.cu
    ops/quantize/quantize.cu
    ops/attention/paged_attention.cu
    ops/batch/continuous_batching.cu
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

# IMPORTANT: Do NOT enable CUDA_SEPARABLE_COMPILATION
# It causes 15x performance degradation for CUTLASS kernels
# due to prevented inlining and indirect function calls
set_target_properties(_pygpukit_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Install the module to the correct location for scikit-build-core
# scikit-build-core's wheel.install-dir already sets the base to pygpukit
install(TARGETS _pygpukit_native
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
