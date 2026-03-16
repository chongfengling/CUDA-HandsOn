#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#include "softmax.h" // 引用接口头文件

// --- 配置参数 ---
const int N = 1024;        // 序列长度 (Sequence Length)
const int M = 8192;        // Batch Size (加大行数以填满 GPU)
const int REPEAT_TIMES = 100; // 计时的循环次数
const int WARMUP_TIMES = 10;  // 预热次数

// 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while(0)

// --- 验证函数 (简单版) ---
// 验证每一行的概率和是否为 1.0
bool verify_result(float* d_output, int M, int N) {
    std::vector<float> h_output(N * M);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    // 随机抽查 5 行进行验证，避免 CPU 耗时太久
    for (int i = 0; i < 5; i++) {
        int row_idx = rand() % M;
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += h_output[row_idx * N + j];
        }
        if (fabs(sum - 1.0f) > 1e-4) {
            printf("❌ Verification Failed at row %d: Sum = %f (Expect 1.0)\n", row_idx, sum);
            return false;
        }
    }
    printf("✅ Verification Passed (Random Sampling)\n");
    return true;
}

// --- 核心 Benchmark 函数 ---
void run_benchmark(const char* name, SoftmaxType type, float* d_input, float* d_output, int M, int N) {
    printf("----------------------------------------------------------------\n");
    printf("Benchmarking: %s\n", name);

    // 1. Warmup (预热)
    // 目的：唤醒 GPU，填充指令 Cache，让 GPU 进入高频状态
    for (int i = 0; i < WARMUP_TIMES; i++) {
        launch_softmax(d_input, d_output, M, N, type);
    }
    CHECK_CUDA(cudaDeviceSynchronize()); // 必须同步，确保预热结束

    // 2. Record (计时)
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < REPEAT_TIMES; i++) {
        // 调用封装好的接口
        launch_softmax(d_input, d_output, M, N, type);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // 3. 计算指标
    float avg_latency = milliseconds / REPEAT_TIMES; // ms
    
    // 有效带宽 (Effective Bandwidth)
    // 理论读取量: M * N * 4 bytes
    // 理论写入量: M * N * 4 bytes
    size_t total_bytes = (size_t)M * N * sizeof(float) * 2;
    float gb_per_sec = (total_bytes / 1e9) / (avg_latency / 1000.0f);

    printf("  Latency   : %.4f ms\n", avg_latency);
    printf("  Bandwidth : %.2f GB/s\n", gb_per_sec);

    // 4. 验证正确性 (只验证一次，避免影响性能测试)
    verify_result(d_output, M, N);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    printf("Softmax Performance Benchmark\n");
    printf("Shape: [M=%d, N=%d], Total Elements: %lu\n", M, N, (size_t)M * N);
    
    // 1. 内存分配
    size_t bytes = (size_t)M * N * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    // 2. 数据初始化
    srand(2024);
    for (size_t i = 0; i < M * N; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f; // 0.0 ~ 10.0
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // 3. 运行测试
    // Test Case 1: Naive (Shared Memory)
    run_benchmark("Naive Softmax (Shared Mem)", SoftmaxType::Naive, d_input, d_output, M, N);

    // Test Case 2: Warp Shuffle
    run_benchmark("Warp Softmax (Shuffle)",     SoftmaxType::Warp,  d_input, d_output, M, N);

    // Test Case 3: Online Softmax
    run_benchmark("Online Softmax",             SoftmaxType::Online, d_input, d_output, M, N);

    // 4. 清理
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);

    return 0;
}