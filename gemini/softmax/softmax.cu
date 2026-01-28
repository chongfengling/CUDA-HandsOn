#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // for std::max_element
#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 256

// --- CPU 版 Safe Softmax ---
// 假设输入是 [N] 的向量 (也就是 Batch=1, Heads=1, SeqLen=N 的简化情况)
void softmax_cpu(float* input, float* output, int N) {
    // 1. Find Max (Row Max)
    // 初始化为一个极小值
    float m = -INFINITY; 
    for (int i = 0; i < N; i++) {
        if (input[i] > m) {
            m = input[i];
        }
    }

    // 2. Compute Sum of Exponentials (Row Sum)
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        // 先减去 max 再求 exp，保证数值稳定
        float val = expf(input[i] - m);
        sum += val;
        output[i] = val; // 暂时把中间结果 exp 存在 output 里
    }

    // 3. Normalize (Division)
    for (int i = 0; i < N; i++) {
        output[i] = output[i] / sum;
    }
}

// --- Helper: 树形归约找最大值 ---
// 这是一个经典的模式：折半归约
__device__ void block_reduce_max(float* sdata) {
    unsigned int tid = threadIdx.x;
    // 假设 blockDim 是 2 的幂次，进行折半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 只有前一半线程工作：比较自己和对应后半部分的值
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        // 必须同步！否则可能读到还没更新的数据
        __syncthreads();
    }
}

// --- Helper: 树形归约求和 ---
__device__ void block_reduce_sum(float* sdata) {
    unsigned int tid = threadIdx.x;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}

// --- Kernel: Naive Softmax ---
__global__ void softmax_naive_kernel(float* input, float* output, int N) {
    // 1. 确定当前 Block 负责哪一行
    int row = blockIdx.x;
    // 指向当前行的起始位置
    float* input_row = input + row * N;
    float* output_row = output + row * N;

    // 2. 声明共享内存 (大小为 BlockSize)
    // 这里的 sdata 复用于 max 和 sum 计算
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;

    // ============================================
    // 第一步：寻找 Max (为了数值稳定)
    // ============================================
    
    // 初始化局部最大值 (处理 N > BlockSize 的情况)
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input_row[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // 全局归约 (结果存在 sdata[0])
    block_reduce_max(sdata);

    // 广播 Max 给所有线程
    float row_max = sdata[0];
    // 这里不需要 __syncthreads() 因为下面大家都不改 sdata[0] 了吗？
    // 为了安全起见，或者为了复用 sdata，最好同步一下，
    // 但因为下面我们重写 sdata[tid]，如果不读 sdata[0] 就没事。
    // 为了代码逻辑清晰，通常大家会在读完 sdata[0] 后不立即写 sdata。
    
    // ============================================
    // 第二步：计算 Exp 并求 Sum
    // ============================================
    
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        // Safe Softmax: exp(x - max)
        float val = expf(input_row[i] - row_max);
        local_sum += val;
        // 顺便把 exp 结果存入 output 暂存，避免等下重复计算 exp
        output_row[i] = val; 
    }
    
    // 必须同步，确保大家读完了 row_max 且 output 写完了
    __syncthreads(); 
    
    // 写入 Shared Memory 准备归约
    sdata[tid] = local_sum;
    __syncthreads();

    // 全局归约 (结果存在 sdata[0])
    block_reduce_sum(sdata);

    float row_sum = sdata[0];

    // ============================================
    // 第三步：归一化 (Normalize)
    // ============================================
    
    // 这里不需要同步，因为 block_reduce_sum 内部最后有同步
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] /= row_sum;
    }
}

// Host 调用代码示例
void launch_softmax_naive(float* d_input, float* d_output, int M, int N) {
    // 假设 N <= 256，直接用 1 个 Block 处理 1 行
    // 如果 N > 256，上面的循环逻辑也能处理
    dim3 grid(M); // M 行
    dim3 block(BLOCK_SIZE); 
    
    // 动态共享内存大小 (如果需要动态指定)
    // 这里我们用了静态数组 __shared__ float sdata[BLOCK_SIZE]
    softmax_naive_kernel<<<grid, block>>>(d_input, d_output, N);
}

int main() {
    int N = 1024; // Sequence Length
    size_t bytes = N * sizeof(float);

    float *h_input, *h_output;
    cudaMallocManaged(&h_input, bytes);
    cudaMallocManaged(&h_output, bytes);

    // --- 测试用例 1: 普通数据 ---
    // 初始化：让数据有些波动
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 10); 
    }

    // --- 测试用例 2: 压力测试 (大数值) ---
    // 这种情况下，如果不减 max，exp(1000) 会直接溢出
    h_input[0] = 1000.0f; 
    h_input[1] = 1002.0f; 

    // 执行 CPU 计算
    softmax_cpu(h_input, h_output, N);

    dim3 grid(N); // M 行
    dim3 block(BLOCK_SIZE); 
    softmax_naive_kernel<<<grid, block>>>(h_input, h_output, N);
    cudaDeviceSynchronize();

    // 验证结果
    // 理论上 h_input[1] 比 h_input[0] 大 1.0
    // 所以 h_output[1] 应该是 h_output[0] 的 e^1 (约 2.718) 倍
    // 且其他很小的数对应的概率应该接近 0
    printf("Verification (Top 2 elements):\n");
    printf("Input: %.1f -> Output: %.6f\n", h_input[0], h_output[0]);
    printf("Input: %.1f -> Output: %.6f\n", h_input[1], h_output[1]);
    
    // 简单的概率和检查
    float total_prob = 0.0f;
    for(int i=0; i<N; i++) total_prob += h_output[i];
    printf("Sum of probabilities: %.6f (Expect 1.000000)\n", total_prob);

    cudaFree(h_input);
    cudaFree(h_output);
    return 0;
}