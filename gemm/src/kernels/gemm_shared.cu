#include <cuda_runtime.h>
#include "kernels.h"

// TILEWIDTH defines the size of the square tile stored in shared memory.
#define TILEWIDTH 16

/**
 * Shared Memory GEMM: C = A * B
 * 
 * Strategy:
 * 1. Divide the output matrix C into tiles of TILEWIDTH x TILEWIDTH.
 * 2. Each thread block computes one tile of C.
 * 3. Before computing, threads cooperatively load tiles of A and B into shared memory.
 * 4. This reduces redundant global memory reads by a factor of TILEWIDTH.
 */
__global__ void gemm_shared_kernel(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    // Declare shared memory for tiles of A and B.
    // __shared__ memory is visible to all threads within the same block.
    __shared__ float As[TILEWIDTH][TILEWIDTH];
    __shared__ float Bs[TILEWIDTH][TILEWIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the element in C this thread computes.
    int row = by * TILEWIDTH + ty;
    int col = bx * TILEWIDTH + tx;

    // Use a register for local accumulation of the dot product result.
    float sum = 0.0f;

    // Iterate through all tiles across the K (inner) dimension.
    for (int t = 0; t < (K + TILEWIDTH - 1) / TILEWIDTH; t++) {
        
        // Load tiles of A and B into shared memory.
        // Threads in a block cooperatively fetch data from global memory.
        int tileA_col = t * TILEWIDTH + tx;
        int tileB_row = t * TILEWIDTH + ty;
        
        // Boundary checks for matrix A (M x K)
        if (row < M && tileA_col < K)
            As[ty][tx] = A[row * K + tileA_col];
        else
            As[ty][tx] = 0.0f; // Padding for out-of-bounds tiles

        // Boundary checks for matrix B (K x N)
        if (col < N && tileB_row < K)
            Bs[ty][tx] = B[tileB_row * N + col];
        else
            Bs[ty][tx] = 0.0f; // Padding for out-of-bounds tiles

        // Synchronization 1: Ensure the entire tile is loaded before any thread starts computing.
        __syncthreads();

        // Perform partial dot product for the current tile.
        #pragma unroll
        for (int i = 0; i < TILEWIDTH; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        // Synchronization 2: Ensure computation is finished before the next tile is loaded.
        __syncthreads();
    }

    // Write the final accumulated sum back to global memory C.
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void launch_gemm_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    // block size matches the tile size
    dim3 block(TILEWIDTH, TILEWIDTH);
    // Grid covers all elements of C
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_shared_kernel<<<grid, block>>>(M, N, K, A, B, C);
}
