#include <cuda_runtime.h>
#include <math.h>
#include "kernels.h"
#include "cuda_utils.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 32

struct __align__(8) OnlineStat {
    float max_val;
    float sum;
};

// Running state for online softmax:
// - max_val is the maximum observed so far.
// - sum is sum(exp(x - max_val)) under that current max.
//
// Merging two states rescales both partial sums to the new shared max:
//   new_max = max(a.max_val, b.max_val)
//   new_sum = a.sum * exp(a.max_val - new_max)
//           + b.sum * exp(b.max_val - new_max)
__inline__ __device__ OnlineStat combine_stat(OnlineStat a, OnlineStat b) {
    OnlineStat res;
    res.max_val = fmaxf(a.max_val, b.max_val);
    res.sum = a.sum * expf(a.max_val - res.max_val) + b.sum * expf(b.max_val - res.max_val);
    return res;
}

// Warp-level reduction for OnlineStat. Shuffle each float field separately
// because __shfl_down_sync operates on scalar 32-bit values.
__inline__ __device__ OnlineStat warpReduceStat(OnlineStat val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        OnlineStat other;
        other.max_val = __shfl_down_sync(0xffffffff, val.max_val, offset);
        other.sum = __shfl_down_sync(0xffffffff, val.sum, offset);
        val = combine_stat(val, other);
    }
    return val;
}

// Online row-wise safe softmax.
//
// Unlike the naive and warp kernels, this combines max and sum discovery into
// one streaming reduction pass. A second pass is still required to write the
// final normalized values because every output element needs the final row max
// and row sum.
__global__ void softmax_online_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int laneId = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    
    int row = blockIdx.x;
    const float* input_row = input + row * N;
    float* output_row = output + row * N;

    // One OnlineStat per warp for the inter-warp reduction stage.
    __shared__ OnlineStat sdata[BLOCK_SIZE / WARP_SIZE];

    // Pass 1: each thread builds an online state over its strided columns.
    OnlineStat local_state = {-INFINITY, 0.0f};

    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float x = input_row[i];
        
        // Fast path for updating a single scalar into the running state.
        // When x becomes the new max, the old sum is rescaled into x's frame.
        if (x > local_state.max_val) {
            local_state.sum = local_state.sum * expf(local_state.max_val - x) + 1.0f;
            local_state.max_val = x;
        } else {
            local_state.sum += expf(x - local_state.max_val);
        }
    }

    // Reduce within each warp, then reduce the warp leaders in warp 0.
    local_state = warpReduceStat(local_state);

    if (laneId == 0) {
        sdata[warpId] = local_state;
    }
    __syncthreads();

    OnlineStat row_state = {-INFINITY, 0.0f};
    if (warpId == 0) {
        // Lanes beyond the number of warps contribute the identity state.
        OnlineStat val = (tid < (BLOCK_SIZE / WARP_SIZE)) ? sdata[tid] : (OnlineStat){-INFINITY, 0.0f};
        row_state = warpReduceStat(val);
    }

    // Broadcast the final row state to all threads by reusing sdata[0].
    if (tid == 0) sdata[0] = row_state;
    __syncthreads();
    
    float global_max = sdata[0].max_val;
    float global_sum = sdata[0].sum;

    // Pass 2: write the final normalized values.
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        output_row[i] = expf(input_row[i] - global_max) / global_sum;
    }
}

void launch_softmax_online(const float* input, float* output, int M, int N) {
    // One block per row. Online reduction reduces global-memory traffic by not
    // storing intermediate exponentials.
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    softmax_online_kernel<<<grid, block>>>(input, output, N);
    LAST_KERNEL_CHECK();
}
