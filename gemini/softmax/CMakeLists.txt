cmake_minimum_required(VERSION 3.18)
project(SoftmaxBenchmark CUDA CXX)

# 1. 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 指定显卡架构 (native 表示适配当前机器) replace -arch=sm_80
set(CMAKE_CUDA_ARCHITECTURES native)

# 2. 包含头文件目录
include_directories(include)

# 3. 收集源文件
# 这里把所有 .cu 文件打包成一个库 (Library)
add_library(softmax_lib 
    src/softmax_naive.cu 
    src/softmax_warp.cu
    src/softmax_online.cu
    src/dispatch.cpp 
)

# 4. 创建可执行文件
add_executable(benchmark benchmarks/main.cpp)

# 5. 链接
target_link_libraries(benchmark PRIVATE softmax_lib)