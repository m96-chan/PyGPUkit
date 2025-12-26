cmake_minimum_required(VERSION 3.18)
project(cutlass_79a_test CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 120a)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(test_79a ../third_party/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
target_include_directories(test_79a PRIVATE 
    ../third_party/cutlass/include
    ../third_party/cutlass/tools/util/include
)
target_compile_options(test_79a PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
