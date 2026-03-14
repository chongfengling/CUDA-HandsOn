#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "softmax.h"

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ==========================================
// Part 1: Warp Level Primitives (寄存器级通信)
// ==========================================

static __inline__ __device__ float warpReduceMax(float val) {
    // 32 -> 16 -> 8 -> 4 -> 2 -> 1
    // 使用 XOR (蝴蝶归约) 还是 Down (树形归约)？
    // 这里我们用 Down，因为我们最后只需要 Lane 0 拿到结果
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        // __shfl_down_sync: 去拿 "当前ID + offset" 的值
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

// ==========================================
// Part 2: Softmax Kernel (Warp Optimized)
// ==========================================

static __global__ void softmax_warp_kernel(float* input, float* output, int N) {
    // 1. Setup
    int tid = threadIdx.x;
    int laneId = tid % WARP_SIZE; // 0~31
    int warpId = tid / WARP_SIZE; // 0~7
    
    // 定位当前行
    int row = blockIdx.x;
    float* input_row = input + row * N;
    float* output_row = output + row * N;

    // Shared Memory 只需要存每个 Warp 的结果
    // 256 threads / 32 = 8 warps -> 需要 8 个 float
    __shared__ float sdata[BLOCK_SIZE / WARP_SIZE]; 

    // ============================================
    // Pass 1: Find Row Max (为了数值稳定)
    // ============================================
    
    float local_max = -INFINITY;
    // Grid Stride Loop (处理 N > 256 的情况) each block deals with N elements
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, input_row[i]);
    }

    // A. Warp 内归约
    local_max = warpReduceMax(local_max);

    // B. Warp 间通信：每个 Warp 的 Leader (Lane 0) 把结果写入 Shared Mem
    if (laneId == 0) {
        sdata[warpId] = local_max;
    }
    __syncthreads(); // 等待所有 Warp 写完

    // C. Block 级最终归约 (由 Warp 0 完成)
    float row_max = -INFINITY;
    if (warpId == 0) {
        // 读取 Warp 结果。如果 tid >= 8，读取 -INF (Padding)
        float val = (tid < (BLOCK_SIZE / WARP_SIZE)) ? sdata[tid] : -INFINITY;
        row_max = warpReduceMax(val);
    }

    // D. 广播 (Broadcast)
    // 只有 tid 0 拿着最终结果，把它写回 sdata[0] 让大家读
    if (tid == 0) sdata[0] = row_max;
    __syncthreads();
    row_max = sdata[0]; // 所有人拿到 row_max

    // ============================================
    // Pass 2: Compute Exp & Sum
    // ============================================
    
    float local_sum = 0.0f; // 重置累加器
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = expf(input_row[i] - row_max);
        local_sum += val;
        output_row[i] = val; // 暂存 exp 结果
    }

    // A. Warp 内归约
    local_sum = warpReduceSum(local_sum);

    // B. Warp 间通信
    if (laneId == 0) sdata[warpId] = local_sum;
    __syncthreads();

    // C. Block 级最终归约
    float row_sum = 0.0f;
    if (warpId == 0) {
        float val = (tid < (BLOCK_SIZE / WARP_SIZE)) ? sdata[tid] : 0.0f; // Padding 0
        row_sum = warpReduceSum(val);
    }

    // D. 广播
    if (tid == 0) sdata[0] = row_sum;
    __syncthreads();
    row_sum = sdata[0];

    // ============================================
    // Pass 3: Normalize
    // ============================================
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        output_row[i] /= row_sum;
    }
}

void launch_softmax_warp(float* input, float* output, int M, int N) {
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    // Warp 版本可能需要不同的 Shared Memory 配置
    softmax_warp_kernel<<<grid, block>>>(input, output, N);
}