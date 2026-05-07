#pragma once

#ifndef SOFTMAX_H
#define SOFTMAX_H

// Softmax algorithms
enum class SoftmaxAlgo {
    NAIVE,
    WARP,
    ONLINE
};

// Row-wise softmax over an M x N matrix.
// input and output are assumed to be device pointers.
void softmax(int M, int N, const float* input, float* output, SoftmaxAlgo algo = SoftmaxAlgo::WARP);

// Backward-compatible entry point for older call sites.
void launch_softmax(const float* input, float* output, int M, int N, SoftmaxAlgo algo);

#endif // SOFTMAX_H
