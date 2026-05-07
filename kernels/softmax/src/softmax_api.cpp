#include "softmax.h"
#include "dispatch.h"

void softmax(int M, int N, const float* input, float* output, SoftmaxAlgo algo) {
    dispatch_softmax(M, N, input, output, algo);
}

void launch_softmax(const float* input, float* output, int M, int N, SoftmaxAlgo algo) {
    dispatch_softmax(M, N, input, output, algo);
}
