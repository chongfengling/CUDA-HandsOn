#include "dispatch.h"
#include "kernels.h"
#include <stdexcept>

void dispatch_softmax(int M, int N, const float* input, float* output, SoftmaxAlgo algo) {
    switch (algo) {
        case SoftmaxAlgo::NAIVE:
            launch_softmax_naive(input, output, M, N);
            break;
        case SoftmaxAlgo::WARP:
            launch_softmax_warp(input, output, M, N);
            break;
        case SoftmaxAlgo::ONLINE:
            launch_softmax_online(input, output, M, N);
            break;
        default:
            throw std::invalid_argument("Unsupported Softmax algorithm");
    }
}
