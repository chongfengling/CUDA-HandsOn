#include <cuda_runtime.h>
#include "kernels.h"

#define TILEWIDTH 16

__global__ void gemm_shared_kernel(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    __shared__ float As[TILEWIDTH][TILEWIDTH];
    __shared__ float Bs[TILEWIDTH][TILEWIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // identify the row and column of the C elemnet
    // where blockDim.x = blockDim.y = TILE_WIDTH
    int row = by * TILEWIDTH + ty;
    int col = bx * TILEWIDTH + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILEWIDTH - 1) / TILEWIDTH; t++) {
        int tileA = t * TILEWIDTH + tx;
        int tileB = t * TILEWIDTH + ty;
        
        // location of an element in A matrix: A[row][col] = row * width + col
        // tile A from left to right (k direction)
        if (row < M && tileA < K)
            As[ty][tx] = A[row * K + tileA];
        else
            As[ty][tx] = 0.0f;

        // tile B from top to bottom (k direction)
        if (col < N && tileB < K)
            Bs[ty][tx] = B[tileB * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILEWIDTH; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void launch_gemm_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILEWIDTH, TILEWIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_shared_kernel<<<grid, block>>>(M, N, K, A, B, C);
}
