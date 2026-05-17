#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include "kernels.h"
#include "cuda_utils.h"


/**
 * @file gemm_tensor_core.cu
 * @brief Tensor Core GEMM using shared-memory tiling, register fragments,
 *        vectorized loads, and double buffering.
 *
 * @algorithm
 * - CTA tile: 128x128 output elements.
 * - Warp tile: each of 16 warps computes one 32x32 output sub-tile using 2x2
 *   WMMA accumulator fragments.
 * - K tile: 8 elements, matching WMMA TF32 m16n16k8.
 * - Shared memory: A and B tiles are staged once per CTA and reused by warps.
 * - Registers: WMMA accumulator fragments keep each warp's C tile in registers.
 * - Vectorized loads: global-to-shared copies use float4 when the edge permits.
 * - Double buffer: two shared-memory stages ping-pong across K tiles.
 */

using namespace nvcuda;

template <
    int BM, int BN, int BK,
    int WMMA_M, int WMMA_N, int WMMA_K,
    int WARPS_M, int WARPS_N,
    int WARP_TILES_M, int WARP_TILES_N,
    int AS_STRIDE, int BS_STRIDE>
__global__ void gemm_tensor_core_kernel(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    // Thread and block indices
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Double-buffered shared memory for CTA-level A and B tiles. The padded
    // leading dimensions reduce regular shared-memory bank conflicts while
    // preserving WMMA-compatible row-major fragments.
    __shared__ float As[2][BM][AS_STRIDE];
    __shared__ float Bs[2][BK][BS_STRIDE];

    // Accumulator fragments in registers for one 32x32 warp tile.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int wm = 0; wm < WARP_TILES_M; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; wn++) {
            wmma::fill_fragment(c_frag[wm][wn], 0.0f);
        }
    }

    int block_row = by * BM;
    int block_col = bx * BN;
    int warp_row = warp_m * WARP_TILES_M * WMMA_M;
    int warp_col = warp_n * WARP_TILES_N * WMMA_N;

    auto load_stage = [&](int stage, int k_tile) {
        constexpr int A_VEC_COLS = BK / 4;
        constexpr int B_VEC_COLS = BN / 4;
        constexpr int A_VEC_COUNT = BM * A_VEC_COLS;
        constexpr int B_VEC_COUNT = BK * B_VEC_COLS;

        for (int vec = tid; vec < A_VEC_COUNT; vec += blockDim.x) {
            int row = vec / A_VEC_COLS;
            int col = (vec % A_VEC_COLS) * 4;
            int global_row = block_row + row;
            int global_col = k_tile + col;

            if ((K % 4 == 0) && global_row < M && global_col + 3 < K) {
                float4 tmp = reinterpret_cast<const float4*>(&A[global_row * K + global_col])[0];
                As[stage][row][col + 0] = tmp.x;
                As[stage][row][col + 1] = tmp.y;
                As[stage][row][col + 2] = tmp.z;
                As[stage][row][col + 3] = tmp.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int k_col = global_col + i;
                    As[stage][row][col + i] =
                        (global_row < M && k_col < K) ? A[global_row * K + k_col] : 0.0f;
                }
            }
        }

        for (int vec = tid; vec < B_VEC_COUNT; vec += blockDim.x) {
            int row = vec / B_VEC_COLS;
            int col = (vec % B_VEC_COLS) * 4;
            int global_row = k_tile + row;
            int global_col = block_col + col;

            if ((N % 4 == 0) && global_row < K && global_col + 3 < N) {
                float4 tmp = reinterpret_cast<const float4*>(&B[global_row * N + global_col])[0];
                Bs[stage][row][col + 0] = tmp.x;
                Bs[stage][row][col + 1] = tmp.y;
                Bs[stage][row][col + 2] = tmp.z;
                Bs[stage][row][col + 3] = tmp.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int n_col = global_col + i;
                    Bs[stage][row][col + i] =
                        (global_row < K && n_col < N) ? B[global_row * N + n_col] : 0.0f;
                }
            }
        }
    };

    // Prologue: stage the first K tile.
    load_stage(0, 0);
    __syncthreads();

    int read_stage = 0;
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        int next_k_tile = k_tile + BK;
        int write_stage = read_stage ^ 1;

        if (next_k_tile < K) {
            load_stage(write_stage, next_k_tile);
        }

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> a_frag[WARP_TILES_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> b_frag[WARP_TILES_N];

        #pragma unroll
        for (int wm = 0; wm < WARP_TILES_M; wm++) {
            wmma::load_matrix_sync(a_frag[wm],
                                   &As[read_stage][warp_row + wm * WMMA_M][0],
                                   AS_STRIDE);
        }
        #pragma unroll
        for (int wn = 0; wn < WARP_TILES_N; wn++) {
            wmma::load_matrix_sync(b_frag[wn],
                                   &Bs[read_stage][0][warp_col + wn * WMMA_N],
                                   BS_STRIDE);
        }
        #pragma unroll
        for (int wm = 0; wm < WARP_TILES_M; wm++) {
            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; wn++) {
                wmma::mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
            }
        }

        __syncthreads();
        read_stage = write_stage;
    }

    bool full_tile = (N % 4 == 0) && (block_row + BM <= M) && (block_col + BN <= N);
    if (full_tile) {
        #pragma unroll
        for (int wm = 0; wm < WARP_TILES_M; wm++) {
            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; wn++) {
                int c_row = block_row + warp_row + wm * WMMA_M;
                int c_col = block_col + warp_col + wn * WMMA_N;
                wmma::store_matrix_sync(&C[c_row * N + c_col],
                                        c_frag[wm][wn], N, wmma::mem_row_major);
            }
        }
    } else {
        __shared__ float C_tile[WMMA_M][WMMA_N];

        for (int owner_warp = 0; owner_warp < WARPS_M * WARPS_N; owner_warp++) {
            int owner_warp_m = owner_warp / WARPS_N;
            int owner_warp_n = owner_warp % WARPS_N;
            int owner_row = owner_warp_m * WARP_TILES_M * WMMA_M;
            int owner_col = owner_warp_n * WARP_TILES_N * WMMA_N;

            #pragma unroll
            for (int wm = 0; wm < WARP_TILES_M; wm++) {
                #pragma unroll
                for (int wn = 0; wn < WARP_TILES_N; wn++) {
                    if (warp_id == owner_warp) {
                        wmma::store_matrix_sync(&C_tile[0][0],
                                                c_frag[wm][wn], WMMA_N, wmma::mem_row_major);
                    }
                    __syncthreads();

                    int tile_global_row = block_row + owner_row + wm * WMMA_M;
                    int tile_global_col = block_col + owner_col + wn * WMMA_N;
                    for (int idx = tid; idx < WMMA_M * WMMA_N; idx += blockDim.x) {
                        int row = idx / WMMA_N;
                        int col = idx % WMMA_N;
                        int global_row = tile_global_row + row;
                        int global_col = tile_global_col + col;

                        if (global_row < M && global_col < N) {
                            C[global_row * N + global_col] = C_tile[row][col];
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}

void launch_gemm_tensor_core(const float* A, const float* B, float* C, int M, int N, int K) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 8;
    const int WARP_TILES_M = 2;
    const int WARP_TILES_N = 2;
    const int WARPS_M = BM / (WARP_TILES_M * WMMA_M);
    const int WARPS_N = BN / (WARP_TILES_N * WMMA_N);
    const int THREADS = WARPS_M * WARPS_N * 32;
    const int AS_STRIDE = BK + 8;
    const int BS_STRIDE = BN + 8;

    dim3 block(THREADS);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    gemm_tensor_core_kernel<
        BM, BN, BK, WMMA_M, WMMA_N, WMMA_K,
        WARPS_M, WARPS_N, WARP_TILES_M, WARP_TILES_N,
        AS_STRIDE, BS_STRIDE>
        <<<grid, block>>>(M, N, K, A, B, C);
    LAST_KERNEL_CHECK();
}
