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
 * - Shared memory arrays (`As`) are often transposed to eliminate bank conflicts during the float4 read/write phase.
 */

template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_vectorized_kernel(
    int M, int N, int K,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C)
{
    __shared__ T As[BK][BM]; // Transposed for bank-free access
    __shared__ T Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    T r_c[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            r_c[i][j] = T(0);
        }
    }

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
            for(int i=0; i<4; i++) As[a_load_col+i][a_load_row] = T(0);
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
            T fragA[TM];
            T fragB[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) fragA[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int i = 0; i < TN; i++) fragB[i] = Bs[k][tx * TN + i];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    r_c[i][j] += fragA[i] * fragB[j];
                }
            }
        }
        __syncthreads();
    }

    // Write back
    int row = by * BM + ty * TM;
    int col = bx * BN + tx * TN;
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + (col + j)] = r_c[i][j];
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
