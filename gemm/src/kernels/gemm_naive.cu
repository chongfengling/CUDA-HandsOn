#include <cuda_runtime.h>
#include "kernels.h"

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * ty + ty;
    int col = bx * tx + tx;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
