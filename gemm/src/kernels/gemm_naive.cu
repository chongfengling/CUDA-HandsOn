#include <cuda_runtime.h>
#include "kernels.h"

/**
 * Naive GEMM kernel: C = A * B
 * 
 * Strategy:
 * Each thread is responsible for computing one single element of the output matrix C.
 * This is the simplest implementation but suffers from:
 * 1. Low global memory coalescing efficiency (for matrix A).
 * 2. High redundant global memory accesses (each element of A and B is loaded N and M times respectively).
 */
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Map 2D grid/block indices to matrix row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to prevent out-of-bounds access for non-multiple-of-block-size dimensions
    if (row < M && col < N) {
        float sum = 0.0f;
        // Dot product of row 'row' from A and column 'col' from B
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        // Write the result to global memory
        C[row * N + col] = sum;
    }
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // Standard 16x16 block size
    dim3 block(16, 16);
    // Ceiling division to cover all elements in M and N dimensions
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
