#include <cuda_runtime.h>
#include <math.h>
#include "kernels.h"
#include "cuda_utils.h"

#define BLOCK_SIZE 256

// Block-wide tree reduction over shared memory.
// Preconditions:
// - sdata[threadIdx.x] already contains each thread's local maximum.
// - blockDim.x is a power of two. The launcher uses BLOCK_SIZE = 256.
// Result:
// - sdata[0] contains the maximum for the whole block.
__device__ void block_reduce_max(float* sdata) {
    unsigned int tid = threadIdx.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        // The next reduction round reads values written by the current round.
        __syncthreads();
    }
}

// Same shared-memory tree pattern as block_reduce_max, but for sums.
// sdata[0] contains the block-wide sum when the function returns.
__device__ void block_reduce_sum(float* sdata) {
    unsigned int tid = threadIdx.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}

// Baseline row-wise safe softmax.
//
// Grid mapping:
// - One CUDA block owns one matrix row.
// - Threads in the block scan columns with a block-stride loop, so N may be
//   larger than BLOCK_SIZE.
//
// Algorithm:
// 1. Reduce row max for numerical stability.
// 2. Compute exp(x - row_max), cache it in output, and reduce row sum.
// 3. Normalize the cached exponentials in-place.
static __global__ void softmax_naive_kernel(const float* input, float* output, int N) {
    int row = blockIdx.x;
    const float* input_row = input + row * N;
    float* output_row = output + row * N;

    // Reused for max and sum reductions. The explicit sync after reading
    // row_max prevents the sum phase from overwriting data too early.
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    // Pass 1: each thread scans its columns, then the block reduces the row max.
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input_row[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    block_reduce_max(sdata);
    float row_max = sdata[0];
    __syncthreads();

    // Pass 2: compute exponentials and reduce their sum.
    // output is used as temporary storage so pass 3 does not recompute expf.
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(input_row[i] - row_max);
        local_sum += val;
        output_row[i] = val; 
    }
    sdata[tid] = local_sum;
    __syncthreads();

    block_reduce_sum(sdata);
    float row_sum = sdata[0];

    // Pass 3: normalize cached exponentials by the row sum.
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] /= row_sum;
    }
}

void launch_softmax_naive(const float* d_input, float* d_output, int M, int N) {
    // One block per row. This is simple and easy to reason about, but exposes
    // limited parallelism when M is small.
    dim3 grid(M);
    dim3 block(BLOCK_SIZE); 
    softmax_naive_kernel<<<grid, block>>>(d_input, d_output, N);
    LAST_KERNEL_CHECK();
}
