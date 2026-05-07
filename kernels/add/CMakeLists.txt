cmake_minimum_required(VERSION 3.18)
project(VectorAdd LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

# Include paths
include_directories(
    include
    ../../common/include
)

# Source files
set(ADD_SOURCES
    src/add_naive.cu
    src/add_single_block.cu
    src/add_multi_block.cu
    src/dispatch.cpp
)

# Test executable
add_executable(test_add tests/test_add.cu ${ADD_SOURCES})
