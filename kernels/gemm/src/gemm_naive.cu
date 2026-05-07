#include <cuda_runtime.h>
#include "kernels.h"
#include "cuda_utils.h"


/**
 * @file gemm_naive.cu
 * @brief Naive GEMM implementation: C = A * B
 *
 * @algorithm
 * - Each thread computes exactly one element of the output matrix C.
 * - Reads a full row of A and a full column of B from Global Memory.
 *
 * @bottleneck
 * - **Memory Bound**: Extremely low compute-to-memory ratio.
 * - **Uncoalesced Access**: Threads in a warp access elements of matrix B in a strided manner (column-major reading of a row-major matrix), resulting in massive cache line misses.
 */

template <typename T>
__global__ void gemm_naive_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = T(0);
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
    gemm_naive_kernel<float><<<grid, block>>>(A, B, C, M, N, K);
    LAST_KERNEL_CHECK();
}
