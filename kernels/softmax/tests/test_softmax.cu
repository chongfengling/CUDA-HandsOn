#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#include "softmax.h"
#include "timer.h"
#include "cuda_utils.h"

// --- Configuration ---
const int N = 1024;        
const int M = 8192;        
const int REPEAT_TIMES = 100; 
const int WARMUP_TIMES = 10;  

// --- Verification ---
bool verify_result(float* d_output, int M, int N) {
    std::vector<float> h_output(N * M);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));

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

// --- Benchmark ---
void run_benchmark(const char* name, SoftmaxAlgo algo, float* d_input, float* d_output, int M, int N) {
    printf("----------------------------------------------------------------\n");
    printf("Benchmarking: %s\n", name);

    // 1. Warmup
    for (int i = 0; i < WARMUP_TIMES; i++) {
        softmax(M, N, d_input, d_output, algo);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. Performance Measurement
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < REPEAT_TIMES; i++) {
        softmax(M, N, d_input, d_output, algo);
    }
    timer.stop();
    
    float avg_latency = timer.elapsed_msecs() / REPEAT_TIMES;
    
    // Effective Bandwidth
    size_t total_bytes = (size_t)M * N * sizeof(float) * 2;
    float gb_per_sec = (total_bytes / 1e9) / (avg_latency / 1000.0f);

    printf("  Latency   : %.4f ms\n", avg_latency);
    printf("  Bandwidth : %.2f GB/s\n", gb_per_sec);

    // 3. Verification
    verify_result(d_output, M, N);
}

int main() {
    printf("Softmax Performance Benchmark\n");
    printf("Shape: [M=%d, N=%d], Total Elements: %lu\n", M, N, (size_t)M * N);
    
    size_t bytes = (size_t)M * N * sizeof(float);
    std::vector<float> h_input(M * N);
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    srand(2024);
    for (size_t i = 0; i < M * N; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    run_benchmark("Naive Softmax", SoftmaxAlgo::NAIVE, d_input, d_output, M, N);
    run_benchmark("Warp Softmax",  SoftmaxAlgo::WARP,  d_input, d_output, M, N);
    run_benchmark("Online Softmax", SoftmaxAlgo::ONLINE, d_input, d_output, M, N);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
