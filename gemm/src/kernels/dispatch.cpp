#include "dispatch.h"
#include "kernels.h"
#include <stdexcept>

void dispatch_gemm(int M, int N, int K, const float* A, const float* B, float* C, GemmAlgo algo) {
    if (algo == GemmAlgo::NAIVE) {
        launch_gemm_naive(A, B, C, M, N, K);
    } else if (algo == GemmAlgo::SHARED_MEMORY) {
        launch_gemm_shared(A, B, C, M, N, K);
    } else if (algo == GemmAlgo::REGISTER) {
        launch_gemm_register(A, B, C, M, N, K);
    } else if (algo == GemmAlgo::CUBLAS) {
        launch_gemm_cublas(A, B, C, M, N, K);
    } else if (algo == GemmAlgo::CUTLASS) {
        launch_gemm_cutlass(A, B, C, M, N, K);
    } else {
        throw std::invalid_argument("Unsupported GEMM algorithm");
    }
}
