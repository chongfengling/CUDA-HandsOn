#pragma once

#ifndef GEMM_H
#define GEMM_H

// GEMM Algorithms
enum class GemmAlgo {
    NAIVE,
    SHARED_MEMORY,
    REGISTER,
    VECTORIZED,
    DOUBLE_BUFFERED,
    TENSOR_CORE,
    ASYNC,
    ULTIMATE,
    ZHIHU,
    CUBLAS,
    CUTLASS
};

// C = A * B
// A is M x K
// B is K x N
// C is M x N
// A, B, C are assumed to be device pointers
void sgemm(int M, int N, int K, const float* A, const float* B, float* C, GemmAlgo algo = GemmAlgo::SHARED_MEMORY);

#endif // GEMM_H
