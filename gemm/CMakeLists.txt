cmake_minimum_required(VERSION 3.18)
if(EXISTS "/usr/bin/gcc-12")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-12")
endif()
project(CUDA_GEMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

include_directories(include src ../third-party/cutlass/include)

# Library
add_library(gemm_lib
    src/gemm_api.cpp
    src/kernels/dispatch.cpp
    src/kernels/gemm_naive.cu
    src/kernels/gemm_shared.cu
    src/kernels/gemm_register.cu
    src/kernels/gemm_cublas.cu
    src/kernels/gemm_cutlass.cu
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gemm_lib PUBLIC CUDA::cublas)

# Tests
add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm PRIVATE gemm_lib)

# Benchmarks
add_executable(bench_gemm benchmarks/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE gemm_lib)
