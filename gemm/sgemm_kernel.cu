#include "sgemm.cuh"

// We use a Tiled Matrix Multiplication approach using shared memory 
// to significantly reduce global memory accesses and improve performance.
#define TILE 16 //same shape of blocks

__global__ void sgemm_shared_kernel(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    // shared memory tile
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y; // TILE=blockDIm(x,y)  和构造相关 one tile one block
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // 遍历 K 维度的 tile
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // row major, from left to right
        int tiledA = t * TILE + threadIdx.x;
        // from top to bottom
        int tiledB = t * TILE + threadIdx.y;

        // load A tile
        if (row < M && tiledA < K)
            //row base, As[ty][tx] to avoid bank conflict
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledA]; // t * Tile + row * K + threadIdx.x
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // load B tile
        if (col < N && tiledB < K)
            Bs[threadIdx.y][threadIdx.x] = B[tiledB * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // compute partial result
        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}


__global__ void sgemm_naive_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        float sum = 0.0f;
        for (int i = 0; i< K; ++i){
            sum += A[row * K + i] * B[N * i + col];
        }
        C[row * N + col] = sum;
    }

}
// The required interface function
// Assumes that A, B, and C pointers already reside on device memory.
void solve(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    sgemm_tiled_kernel<<<gridDim, blockDim>>>(M, N, K, A, B, C);
    // sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, A, B, C);
    CHECK_CUDA(cudaGetLastError());
}