// src/dispatch.cpp
#include "softmax.h"
#include <stdexcept>

void launch_softmax(float* input, float* output, int M, int N, SoftmaxType type) {
    switch (type) {
        case SoftmaxType::Naive:
            launch_softmax_naive(input, output, M, N);
            break;
        case SoftmaxType::Warp:
            launch_softmax_warp(input, output, M, N);
            break;
        
        default:
            throw std::runtime_error("Unknown softmax type");
    }
}