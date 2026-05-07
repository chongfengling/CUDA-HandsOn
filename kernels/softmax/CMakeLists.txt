cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(EXISTS "/usr/bin/gcc-12")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-12")
endif()

project(CUDA_SOFTMAX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Include directories
include_directories(
    include
    src
    ../../common/include
)

# Library
add_library(softmax_lib
    src/softmax_api.cpp
    src/dispatch.cpp
    src/softmax_naive.cu
    src/softmax_warp.cu
    src/softmax_online.cu
)

# Tests
add_executable(test_softmax tests/test_softmax.cu)
target_link_libraries(test_softmax PRIVATE softmax_lib)

# Benchmarks
add_executable(bench_softmax tests/bench_softmax.cu)
target_link_libraries(bench_softmax PRIVATE softmax_lib)
