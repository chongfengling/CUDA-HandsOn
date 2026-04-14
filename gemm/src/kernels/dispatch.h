#pragma once
#include "gemm.h"

void dispatch_gemm(int M, int N, int K, const float* A, const float* B, float* C, GemmAlgo algo);
