#include "gemm.h"
#include "dispatch.h"

void sgemm(int M, int N, int K, const float* A, const float* B, float* C, GemmAlgo algo) {
    dispatch_gemm(M, N, K, A, B, C, algo);
}
