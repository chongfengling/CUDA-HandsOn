#include <cuda_runtime.h>
#include "kernels.h"
#include "cuda_utils.h"


/**
 * @file gemm_shared.cu
 * @brief Shared Memory Tiled GEMM
 *
 * @algorithm
 * - Divides the computation into 2D blocks (Tiles) of size TILE_SIZE x TILE_SIZE.
 * - Threads cooperatively load a tile of A and a tile of B from Global Memory to Shared Memory.
 * - **Memory Coalescing**: By reading contiguous chunks into Shared Memory, we fix the uncoalesced access pattern of matrix B.
 * - **Data Reuse**: Each element loaded into Shared Memory is used TILE_SIZE times, reducing Global Memory bandwidth requirements by a factor of TILE_SIZE.
 *
 * @bottleneck
 * - **Shared Memory Latency**: Inner loop still issues too many Shared Memory reads relative to FMA (Fused Multiply-Add) instructions.
 */

template <typename T, int TILEWIDTH>
__global__ void gemm_shared_kernel(
    int M, int N, int K,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C)
{
    // Declare shared memory for tiles of A and B.
    // __shared__ memory is visible to all threads within the same block.
    __shared__ T As[TILEWIDTH][TILEWIDTH];
    __shared__ T Bs[TILEWIDTH][TILEWIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the element in C this thread computes.
    int row = by * TILEWIDTH + ty;
    int col = bx * TILEWIDTH + tx;

    // Use a register for local accumulation of the dot product result.
    T sum = T(0);

    // Iterate through all tiles across the K (inner) dimension.
    for (int t = 0; t < ceil_div(K, TILEWIDTH); t++) {
        
        // Load tiles of A and B into shared memory.
        // Threads in a block cooperatively fetch data from global memory.
        int tileA_col = t * TILEWIDTH + tx;
        int tileB_row = t * TILEWIDTH + ty;
        
        // Boundary checks for matrix A (M x K)
        if (row < M && tileA_col < K)
            As[ty][tx] = A[row * K + tileA_col];
        else
            As[ty][tx] = T(0); // Padding for out-of-bounds tiles

        // Boundary checks for matrix B (K x N)
        if (col < N && tileB_row < K)
            Bs[ty][tx] = B[tileB_row * N + col];
        else
            Bs[ty][tx] = T(0); // Padding for out-of-bounds tiles

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
    const int TILE_SIZE = 16;
    // block size matches the tile size
    dim3 block(TILE_SIZE, TILE_SIZE);
    // Grid covers all elements of C
    dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
    gemm_shared_kernel<float, TILE_SIZE><<<grid, block>>>(M, N, K, A, B, C);
    LAST_KERNEL_CHECK();
}
