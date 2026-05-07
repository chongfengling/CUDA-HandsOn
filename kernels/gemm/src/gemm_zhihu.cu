#include <cuda_runtime.h>
#include <cstdint>
#include "kernels.h"
#include "cuda_utils.h"

#define WARP_SIZE 32
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// double buffering kernel from "懒蚂蚁呀不嘿" (Zhihu)
// Ref: https://zhuanlan.zhihu.com/p/1910636263666610461
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int A_BLOCK_Y = 32, int B_BLOCK_X = 32, int C_BLOCK_X = 16, int C_BLOCK_Y = 16,
         int C_WARP_X = 8, int C_WARP_Y = 4, int C_WARP_DIM_X = 2, int Tm = 8, int Tn = 8>
__global__ void doublebufferingGEMM(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                   const int M, const int N, const int K) {
  __shared__ float As[2][Bk][Bm];  // Store transposed tileA
  __shared__ float Bs[2][Bk][Bn];  // Store tileB

  // Calculate block's tileC top-left coordinates
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  int tid = threadIdx.x;

  /*------ tileA ------*/
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid & (A_BLOCK_X - 1);

  /*------ tileB ------*/
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid & (B_BLOCK_X - 1);

  // Warp info
  int warpId = tid / WARP_SIZE;
  int laneId = tid & (WARP_SIZE - 1);

  int warpX = warpId & (C_WARP_DIM_X - 1);
  int warpY = warpId / C_WARP_DIM_X;

  // Z-order mapping for lanes within warp
  int laneY = (laneId & 1) + ((laneId >> 4) << 1);
  int laneX = (laneId & 15) >> 1;

  int C_THREAD_Y = warpY * C_WARP_Y + laneY;
  int C_THREAD_X = warpX * C_WARP_X + laneX;

  // Accumulators in registers
  float Ct[Tm][Tn];
  #pragma unroll
  for(int i=0; i<Tm; ++i) 
    for(int j=0; j<Tn; ++j) Ct[i][j] = 0.0f;

  // Fragments for A and B
  float regA[2][Tm];
  float regB[2][Tn];

  int buffer_id = 0;

  // Initial load k = 0
#pragma unroll
  for (int i = 0; i < Bm; i += A_BLOCK_Y) {
    int r = r0 + i + A_THREAD_Y;
    As[0][A_THREAD_X][(i + A_THREAD_Y) ^ (A_THREAD_X << 2)] =
        (r < M && A_THREAD_X < K) ? A[r * K + A_THREAD_X] : 0.0f;
  }

#pragma unroll
  for (int j = 0; j < Bn; j += B_BLOCK_X) { // Note: Bn instead of Bm (corrected from snippet)
    int c = c0 + j + B_THREAD_X;
    Bs[0][B_THREAD_Y][j + B_THREAD_X] = (B_THREAD_Y < K && c < N) ? B[B_THREAD_Y * N + c] : 0.0f;
  }

  __syncthreads();

  // K-Loop
  for (int k_outer = Bk; k_outer < K + Bk; k_outer += Bk) {
#pragma unroll
    for (int p = 0; p < Bk + 1; ++p) {
      if (p > 0) {
#pragma unroll
        for (int i = 0; i < Tm; ++i) {
#pragma unroll
          for (int j = 0; j < Tn; ++j) { 
              Ct[i][j] += regA[(p - 1) & 1][i] * regB[(p - 1) & 1][j]; 
          }
        }
      }

      if (p < Bk) {
#pragma unroll
        for (int i = 0; i < (Tm >> 2); ++i) {
          int r_idx = (C_THREAD_Y + i * C_BLOCK_Y) << 2;
          FLOAT4(regA[p & 1][i << 2]) = FLOAT4(As[buffer_id][p][r_idx ^ (p << 2)]);
        }

#pragma unroll
        for (int j = 0; j < (Tn >> 2); ++j) {
          int c_idx = (C_THREAD_X + j * C_BLOCK_X) << 2;
          FLOAT4(regB[p & 1][j << 2]) = FLOAT4(Bs[buffer_id][p][c_idx]);
        }
      }
    }

    if (k_outer < K) {
      int c_load = k_outer + A_THREAD_X;
#pragma unroll
      for (int i = 0; i < Bm; i += A_BLOCK_Y) {
        int r = r0 + i + A_THREAD_Y;
        As[buffer_id ^ 1][A_THREAD_X][(i + A_THREAD_Y) ^ (A_THREAD_X << 2)] =
            (r < M && c_load < K) ? A[r * K + c_load] : 0.f;
      }

      int r_load = k_outer + B_THREAD_Y;
#pragma unroll
      for (int j = 0; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j + B_THREAD_X;
        Bs[buffer_id ^ 1][B_THREAD_Y][j + B_THREAD_X] = (r_load < K && c < N) ? B[r_load * N + c] : 0.f;
      }

      __syncthreads();
    }
    buffer_id ^= 1;
  }

  // Final Write out
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + (C_THREAD_Y << 2) + ((i >> 2) << 2) * C_BLOCK_Y + (i & 3);
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + (C_THREAD_X << 2) + ((j >> 2) << 2) * C_BLOCK_X + (j & 3);
      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}

void launch_gemm_zhihu(const float* A, const float* B, float* C, int M, int N, int K) {
    const int BM = 128;
    const int BN = 128;
    
    dim3 block(256); // 1D block as per kernel design
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

    doublebufferingGEMM<<<grid, block>>>(A, B, C, M, N, K);
    LAST_KERNEL_CHECK();
}
