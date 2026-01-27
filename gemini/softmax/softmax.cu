#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // for std::max_element
#include <cuda_runtime.h>

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
    h_input[1] = 1001.0f; 

    // 执行 CPU 计算
    softmax_cpu(h_input, h_output, N);

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