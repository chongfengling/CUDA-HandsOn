#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "kernels.h"

void launch_gemm_cublas(const float* A, const float* B, float* C, int M, int N, int K) {
    // 使用 thread_local 保证每个线程只有一个 Handle，避免每次启动算子都 Create/Destroy 带来的巨大开销
    thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS handle creation failed");
        }
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS 默认列主序 (Column-Major)，而这里的实现是行主序 (Row-Major)
    // C = A * B (Row-Major) 等价于 C^T = B^T * A^T (Column-Major)
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                    &alpha, B, N, A, K, &beta, C, N) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS SGEMM failed");
    }
}
