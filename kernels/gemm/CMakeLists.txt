cmake_minimum_required(VERSION 3.18)

if(EXISTS "/usr/bin/gcc-12")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-12")
endif()

project(CUDA_GEMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Include directories
include_directories(
    include 
    src 
    ../../common/include
    ../../third-party/cutlass/include
)

# Library
add_library(gemm_lib
    src/gemm_api.cpp
    src/dispatch.cpp
    src/gemm_naive.cu
    src/gemm_shared.cu
    src/gemm_register.cu
    src/gemm_vectorized.cu
    src/gemm_double_buffered.cu
    src/gemm_async.cu
    src/gemm_ultimate.cu
    src/gemm_zhihu.cu
    src/gemm_cublas.cu
    src/gemm_cutlass.cu
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gemm_lib PUBLIC CUDA::cublas)

# Tests
add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm PRIVATE gemm_lib)

# Benchmarks
add_executable(bench_gemm tests/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE gemm_lib)

# Profiling
add_executable(profile_gemm tests/profile_gemm.cu)
target_link_libraries(profile_gemm PRIVATE gemm_lib)
