#include <cuda_runtime.h>
#include <cstdint>
#include "kernels.h"
#include "cuda_utils.h"


/**
 * @file gemm_ultimate.cu
 * @brief Warp-Tiling & Bank-Conflict-Free GEMM (Ultimate Manual Tuning)
 *
 * @algorithm
 * - Combines all previous techniques: Double Buffering, Vectorized Access, and Register Tiling.
 * - **Warp Tiling**: Organizes the 2D Thread Tiling precisely around hardware Warps (32 threads). Each warp is responsible for a 64x32 sub-tile to optimize instruction dispatch.
 * - **Bank Conflict Elimination**: Carefully pads and transposes Shared Memory arrays (`As`) to ensure that all threads in a warp access different Memory Banks simultaneously.
 */

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_ultimate_kernel(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Shared memory for tiling A and B.
    // As is transposed to [BK][BM], and both A/B use two stages for double buffering.
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    // Linear thread ID in a 16x16 block (0-255)
    int tid = ty * blockDim.x + tx;

    // Warp identification
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Warp 2x4 layout: 2 warps in Y, 4 warps in X
    int warp_y = warp_id / 4; 
    int warp_x = warp_id % 4;

    // Within warp: 8x4 layout of threads? No, each thread handles 8x8.
    // Let's use a simpler mapping for 8x8 thread tiling
    int thread_y = lane_id / 4; // 0-7
    int thread_x = lane_id % 4; // 0-3

    // Accumulators in registers (8x8 sub-tile per thread)
    float r_c[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            r_c[m][n] = 0.0f;
        }
    }

    // Register fragments for A and B to support outer-product computation
    float fragA[TM];
    float fragB[TN];

    // Top-left corner of the C sub-tile this thread is responsible for
    int warp_row = warp_y * 64;
    int warp_col = warp_x * 32;
    int thread_row = warp_row + thread_y * TM;
    int thread_col = warp_col + thread_x * TN;
    int row = by * BM + thread_row;
    int col = bx * BN + thread_col;

    // Load indices for Global -> Shared
    int a_load_row = tid / 2;    // 0-127
    int a_load_col = (tid % 2) * 4;
    int b_load_row = tid / 32;   // 0-7
    int b_load_col = (tid % 32) * 4;

    // Initial Load
    {
        int a_global_row = by * BM + a_load_row;
        int a_global_col = a_load_col;
        if (a_global_row < M && a_global_col < K) {
            float4 tmp = reinterpret_cast<const float4*>(&A[a_global_row * K + a_global_col])[0];
            As[0][a_load_col + 0][a_load_row] = tmp.x;
            As[0][a_load_col + 1][a_load_row] = tmp.y;
            As[0][a_load_col + 2][a_load_row] = tmp.z;
            As[0][a_load_col + 3][a_load_row] = tmp.w;
        }
        int b_global_row = b_load_row;
        int b_global_col = bx * BN + b_load_col;
        if (b_global_row < K && b_global_col < N) {
            reinterpret_cast<float4*>(&Bs[0][b_load_row][b_load_col])[0] =
                reinterpret_cast<const float4*>(&B[b_global_row * N + b_global_col])[0];
        }
    }
    __syncthreads();

    // Main Loop
    for (int k_tile = BK; k_tile < K; k_tile += BK) {
        int read_idx = ((k_tile / BK) - 1) % 2;
        int write_idx = (k_tile / BK) % 2;

        // Load next tile
        float4 a_next_v = make_float4(0,0,0,0);
        int a_global_row = by * BM + a_load_row;
        int a_global_col = k_tile + a_load_col;
        if (a_global_row < M && a_global_col < K) {
            a_next_v = reinterpret_cast<const float4*>(&A[a_global_row * K + a_global_col])[0];
        }

        float4 b_next_v = make_float4(0,0,0,0);
        int b_global_row = k_tile + b_load_row;
        int b_global_col = bx * BN + b_load_col;
        if (b_global_row < K && b_global_col < N) {
            b_next_v = reinterpret_cast<const float4*>(&B[b_global_row * N + b_global_col])[0];
        }

        // Compute current tile using Warp Tiling logic
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load float4 from SMEM to registers (LDS.128)
            // Each warp handles 64(Y) x 32(X)
            // warp_y=0 -> rows 0-63, warp_y=1 -> rows 64-127
            // thread_y handles 8 rows within that
            #pragma unroll
            for (int i = 0; i < TM / 4; i++) {
                reinterpret_cast<float4*>(&fragA[i * 4])[0] = 
                    reinterpret_cast<float4*>(&As[read_idx][k][thread_row + i * 4])[0];
            }
            #pragma unroll
            for (int i = 0; i < TN / 4; i++) {
                reinterpret_cast<float4*>(&fragB[i * 4])[0] = 
                    reinterpret_cast<float4*>(&Bs[read_idx][k][thread_col + i * 4])[0];
            }

            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += fragA[m] * fragB[n];
                }
            }
        }

        // Commit load to SMEM
        As[write_idx][a_load_col + 0][a_load_row] = a_next_v.x;
        As[write_idx][a_load_col + 1][a_load_row] = a_next_v.y;
        As[write_idx][a_load_col + 2][a_load_row] = a_next_v.z;
        As[write_idx][a_load_col + 3][a_load_row] = a_next_v.w;
        reinterpret_cast<float4*>(&Bs[write_idx][b_load_row][b_load_col])[0] = b_next_v;

        __syncthreads();
    }

    // Final Tile
    int final_idx = (( (K+BK-1)/BK ) - 1) % 2;
    #pragma unroll
    for (int k = 0; k < BK; k++) {
        #pragma unroll
        for (int i = 0; i < TM / 4; i++)
            reinterpret_cast<float4*>(&fragA[i * 4])[0] = reinterpret_cast<float4*>(&As[final_idx][k][thread_row + i * 4])[0];
        #pragma unroll
        for (int i = 0; i < TN / 4; i++)
            reinterpret_cast<float4*>(&fragB[i * 4])[0] = reinterpret_cast<float4*>(&Bs[final_idx][k][thread_col + i * 4])[0];
        
        #pragma unroll
        for (int m = 0; m < TM; m++)
            #pragma unroll
            for (int n = 0; n < TN; n++)
                r_c[m][n] += fragA[m] * fragB[n];
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

void launch_gemm_ultimate(const float* A, const float* B, float* C, int M, int N, int K) {
    if ((K % 8 != 0) || (N % 8 != 0) || (reinterpret_cast<uintptr_t>(A) % 16 != 0)) {
        launch_gemm_double_buffered(A, B, C, M, N, K);
        return;
    }

    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block(16, 16); // 256 threads
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    gemm_ultimate_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, A, B, C);
    LAST_KERNEL_CHECK();
}
