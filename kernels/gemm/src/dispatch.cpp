#include "dispatch.h"
#include "kernels.h"
#include <stdexcept>

void dispatch_gemm(int M, int N, int K, const float* A, const float* B, float* C, GemmAlgo algo) {
    switch (algo) {
        case GemmAlgo::NAIVE:
            launch_gemm_naive(A, B, C, M, N, K);
            break;
        case GemmAlgo::SHARED_MEMORY:
            launch_gemm_shared(A, B, C, M, N, K);
            break;
        case GemmAlgo::REGISTER:
            launch_gemm_register(A, B, C, M, N, K);
            break;
        case GemmAlgo::VECTORIZED:
            launch_gemm_vectorized(A, B, C, M, N, K);
            break;
        case GemmAlgo::DOUBLE_BUFFERED:
            launch_gemm_double_buffered(A, B, C, M, N, K);
            break;
        case GemmAlgo::TENSOR_CORE:
            launch_gemm_tensor_core(A, B, C, M, N, K);
            break;
        case GemmAlgo::ASYNC:
            launch_gemm_async(A, B, C, M, N, K);
            break;
        case GemmAlgo::ULTIMATE:
            launch_gemm_ultimate(A, B, C, M, N, K);
            break;
        case GemmAlgo::ZHIHU:
            launch_gemm_zhihu(A, B, C, M, N, K);
            break;
        case GemmAlgo::CUBLAS:
            launch_gemm_cublas(A, B, C, M, N, K);
            break;
        case GemmAlgo::CUTLASS:
            launch_gemm_cutlass(A, B, C, M, N, K);
            break;
        default:
            throw std::invalid_argument("Unsupported GEMM algorithm");
    }
}
