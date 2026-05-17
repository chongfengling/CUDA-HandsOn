#include <cuda_runtime.h>
#include <cstdint>
#include "kernels.h"
#include "cuda_utils.h"


/**
 * @file gemm_vectorized.cu
 * @brief Vectorized Memory Access GEMM
 *
 * @algorithm
 * - Uses `float4` data types to perform 128-bit vectorized memory transactions.
 * - **Maximizing Bandwidth**: A single 128-bit instruction (`LDG.E.128`) is significantly more efficient than four separate 32-bit instructions (`LDG.E`), fully utilizing the L2 cache and memory controllers.
 * - Keeps the vectorized shared-memory layout while matching the code structure
 *   and 2D thread-tile mapping of `gemm_register.cu`.
 */

template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_vectorized_kernel(
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

    // Shared memory for tiling A and B. As is transposed for vectorized loads.
    __shared__ T As[BK][BM]; // Transposed for bank-free access
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

    int a_load_row = tid / (BK / 4); 
    int a_load_col = (tid % (BK / 4)) * 4;
    int b_load_row = tid / (BN / 4); 
    int b_load_col = (tid % (BN / 4)) * 4;

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Vectorized load from Global to Shared Memory
        int a_global_row = by * BM + a_load_row;
        int a_global_col = k_tile + a_load_col;
        if (a_global_row < M && a_global_col < K) {
            float4 tmp = reinterpret_cast<const float4*>(&A[a_global_row * K + a_global_col])[0];
            As[a_load_col + 0][a_load_row] = tmp.x;
            As[a_load_col + 1][a_load_row] = tmp.y;
            As[a_load_col + 2][a_load_row] = tmp.z;
            As[a_load_col + 3][a_load_row] = tmp.w;
        } else {
            for (int i = 0; i < 4; i++) As[a_load_col + i][a_load_row] = T(0);
        }

        int b_global_row = k_tile + b_load_row;
        int b_global_col = bx * BN + b_load_col;
        if (b_global_row < K && b_global_col < N) {
            float4 tmp = reinterpret_cast<const float4*>(&B[b_global_row * N + b_global_col])[0];
            reinterpret_cast<float4*>(&Bs[b_load_row][b_load_col])[0] = tmp;
        } else {
            reinterpret_cast<float4*>(&Bs[b_load_row][b_load_col])[0] = make_float4(0,0,0,0);
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) fragA[m] = As[k][ty * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; n++) fragB[n] = Bs[k][tx * TN + n];

            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += fragA[m] * fragB[n];
                }
            }
        }
        __syncthreads();
    }

    // Write the final accumulated results from registers back to Global Memory C.
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int c_global_row = row + m;
            int c_global_col = col + n;

            if (c_global_row < M && c_global_col < N) {
                C[c_global_row * N + c_global_col] = r_c[m][n];
            }
        }
    }
}

void launch_gemm_vectorized(const float* A, const float* B, float* C, int M, int N, int K) {
    bool is_aligned = (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
                      (K % 4 == 0) && (N % 4 == 0);

    if (!is_aligned) {
        launch_gemm_register(A, B, C, M, N, K);
        return;
    }

    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    gemm_vectorized_kernel<float, BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, A, B, C);
    LAST_KERNEL_CHECK();
}
