#include <cuda_runtime.h>
#include <math.h>
#include "kernels.h"
#include "cuda_utils.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-local tree reduction using shuffle instructions.
// Only lane 0 receives the final reduction result. Callers that need a
// block-wide result write one value per warp to shared memory and reduce again
// in warp 0.
static __inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

static __inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Row-wise safe softmax with warp-level reductions.
//
// Compared with softmax_naive_kernel, this keeps the first reduction stage in
// registers with __shfl_down_sync. Shared memory only stores one partial value
// per warp, which reduces synchronization and shared-memory traffic.
static __global__ void softmax_warp_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int laneId = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    
    int row = blockIdx.x;
    const float* input_row = input + row * N;
    float* output_row = output + row * N;

    // One slot per warp: BLOCK_SIZE / WARP_SIZE = 8 for the current launch.
    __shared__ float sdata[BLOCK_SIZE / WARP_SIZE]; 

    // Pass 1: compute the row max.
    // Each thread scans a strided subset of columns, then reductions proceed
    // in two levels: intra-warp first, then across warp leaders in warp 0.
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, input_row[i]);
    }

    local_max = warpReduceMax(local_max);

    if (laneId == 0) {
        sdata[warpId] = local_max;
    }
    __syncthreads();

    float row_max = -INFINITY;
    if (warpId == 0) {
        // Lanes beyond the number of warps contribute the identity value.
        float val = (tid < (BLOCK_SIZE / WARP_SIZE)) ? sdata[tid] : -INFINITY;
        row_max = warpReduceMax(val);
    }

    // Broadcast the final row max to the whole block through shared memory.
    if (tid == 0) sdata[0] = row_max;
    __syncthreads();
    row_max = sdata[0]; 

    // Pass 2: compute exp(x - row_max), cache it, and reduce the row sum.
    float local_sum = 0.0f; 
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = expf(input_row[i] - row_max);
        local_sum += val;
        output_row[i] = val; // Cache exp result
    }

    local_sum = warpReduceSum(local_sum);

    if (laneId == 0) sdata[warpId] = local_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (warpId == 0) {
        float val = (tid < (BLOCK_SIZE / WARP_SIZE)) ? sdata[tid] : 0.0f;
        row_sum = warpReduceSum(val);
    }

    // Broadcast the row sum before normalizing the cached exponentials.
    if (tid == 0) sdata[0] = row_sum;
    __syncthreads();
    row_sum = sdata[0];

    // Pass 3: normalize cached exponentials in-place.
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        output_row[i] /= row_sum;
    }
}

void launch_softmax_warp(const float* input, float* output, int M, int N) {
    // Same one-block-per-row mapping as the naive kernel, with a faster
    // block-wide reduction path.
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    softmax_warp_kernel<<<grid, block>>>(input, output, N);
    LAST_KERNEL_CHECK();
}
