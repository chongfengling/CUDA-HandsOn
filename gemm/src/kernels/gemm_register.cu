#include <cuda_runtime.h>
#include "kernels.h"

// Tiling parameters
// BM: M dimension covered by each thread block
// BN: N dimension covered by each thread block
// BK: K dimension covered by each thread block (tile depth)

// Thread-level tiling parameters
// TM: Rows of C processed by each thread
// TN: Columns of C processed by each thread

/**
 * GEMM kernel optimized with Register Tiling and Shared Memory.
 * 
 * Strategy:
 * 1. Each block of 16x16 threads computes a BM x BN (128x128) tile of C.
 * 2. Each thread computes a TM x TN (8x8) sub-tile of C using its private registers.
 * 3. Shared memory tiles As and Bs are used to cache Global Memory data for reuse.
 * 4. Data is moved from Shared Memory to Registers to minimize SMEM bank conflict 
 *    overhead and latency during the innermost compute loop.
 */
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_register_kernel(
    int M, int N, int K,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C)
{
    // Thread and block indices
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int bx = blockIdx.x; 
    int by = blockIdx.y;

    // Shared memory for tiling A and B
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    // Accumulators in registers (8x8 sub-tile per thread)
    T r_c[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            r_c[m][n] = T(0);
        }
    }
    
    // Register fragments for A and B to support outer-product computation
    T fragA[TM]; 
    T fragB[TN];  

    // Top-left corner of the C sub-tile this thread is responsible for
    int row = by * BM + ty * TM;
    int col = bx * BN + tx * TN;
    
    // Linear thread ID in a 16x16 block (0-255)
    int tid = ty * blockDim.x + tx; 

    // Iterate over tiles in the K dimension
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        
        // Load tiles from Global Memory to Shared Memory.
        // Each block has 256 threads. As and Bs both have 1024 elements (128*8).
        // Each thread needs to load 4 elements for As and 4 for Bs per K-tile.
        for (int i = 0; i < 4; i++) {
            int load_id = i * 256 + tid;

            // Load As: Matrix A is M x K, tile is BM x BK
            int a_tile_row = load_id / BK;
            int a_tile_col = load_id % BK;
            int a_global_row = by * BM + a_tile_row;
            int a_global_col = k_tile + a_tile_col;

            if (a_global_row < M && a_global_col < K) {
                As[a_tile_row][a_tile_col] = A[a_global_row * K + a_global_col];
            } else {
                As[a_tile_row][a_tile_col] = T(0);
            }

            // Load Bs: Matrix B is K x N, tile is BK x BN
            int b_tile_row = load_id / BN;
            int b_tile_col = load_id % BN;
            int b_global_row = k_tile + b_tile_row;
            int b_global_col = bx * BN + b_tile_col;

            if (b_global_row < K && b_global_col < N) {
                Bs[b_tile_row][b_tile_col] = B[b_global_row * N + b_global_col];
            } else {
                Bs[b_tile_row][b_tile_col] = T(0);
            }
        }

        // Wait for all threads to finish loading to SMEM
        __syncthreads();

        // Compute the outer product for the current tile.
        // We iterate through the BK dimension and compute an outer product of TMx1 and 1xTN.
        // Note: #pragma unroll here might increase register pressure significantly.
        for (int k = 0; k < BK; k++) {
            for (int m = 0; m < TM; m++) {
                fragA[m] = As[ty * TM + m][k];
            }
            for (int n = 0; n < TN; n++) {
                fragB[n] = Bs[k][tx * TN + n];
            }

            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += fragA[m] * fragB[n];
                }
            }
        }
        
        // Synchronize before the next K-tile to ensure SMEM is ready for new data.
        __syncthreads();
    }

    // Write the final accumulated results from registers back to Global Memory C.
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int c_global_row = row + m;
            int c_global_col = col + n;

            if (c_global_row < M && c_global_col < N) {
                C[c_global_row * N + c_global_col] = r_c[m][n];
            }
        }
    }
}

void launch_gemm_register(const float* A, const float* B, float* C, int M, int N, int K) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    // 16x16 threads = 256 threads per block.
    dim3 block(16, 16);
    // Each block handles 128x128 output elements.
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_register_kernel<float, BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, A, B, C);
}
