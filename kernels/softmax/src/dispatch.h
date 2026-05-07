#pragma once
#include "softmax.h"

void dispatch_softmax(int M, int N, const float* input, float* output, SoftmaxAlgo algo);
