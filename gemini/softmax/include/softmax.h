#pragma once

// 定义枚举，方便切换不同的实现
enum class SoftmaxType {
    Naive,
    Warp
};

// 统一的 C++ 入口
// 内部会根据 type 分发到不同的 kernel
void launch_softmax(float* input, float* output, int M, int N, SoftmaxType type);

// 如果你想单独暴露也可以，但通常会封装起来
void launch_softmax_naive(float* input, float* output, int M, int N);
void launch_softmax_warp(float* input, float* output, int M, int N);