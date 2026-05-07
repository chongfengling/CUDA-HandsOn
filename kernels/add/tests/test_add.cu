#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include "add.h"
#include "cuda_utils.h"
#include "timer.h"

// For comparison, use a slightly smaller N if we want to run Naive in reasonable time
// 10^7 is large enough to show difference but won't hang the GPU for Naive
#define N 10000000
#define MAX_ERR 1e-6

void verify(float *out, const float *a, const float *b, int n) {
    for(int i = 0; i < n; i++){
        if(std::fabs(out[i] - (a[i] + b[i])) > MAX_ERR){
            printf("Mismatch at %d: %f != %f\n", i, out[i], a[i]+b[i]);
            exit(1);
        }
    }
}

void run_benchmark(const float* d_a, const float* d_b, float* d_out, float* h_out, 
                  const float* h_a, const float* h_b, int n, 
                  AddAlgo algo, std::string name, int iterations) {
    CudaTimer timer;
    
    // Warmup
    launch_vector_add(d_a, d_b, d_out, n, algo);
    
    // Benchmark
    timer.start();
    for(int i = 0; i < iterations; i++) {
        launch_vector_add(d_a, d_b, d_out, n, algo);
    }
    timer.stop();

    // Verify
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float)*n, cudaMemcpyDeviceToHost));
    verify(h_out, h_a, h_b, n);

    float avg_time = timer.elapsed_msecs() / iterations;
    printf("[%s] Average time: %f ms\n", name.c_str(), avg_time);
    
    // if (algo == AddAlgo::MULTI_BLOCK_MULTI) {
        float gb = (float)n * sizeof(float) * 3 / (1024 * 1024 * 1024);
        float throughput = gb / (avg_time / 1000.0f);
        printf("  Effective Bandwidth: %.2f GB/s\n", throughput);
    // }
}

int main() {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    a = (float*)malloc(sizeof(float)*N);
    b = (float*)malloc(sizeof(float)*N);
    out = (float*)malloc(sizeof(float)*N);

    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)*N));

    CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice));

    printf("Vector Size: %d elements\n", N);
    printf("-------------------------------------------------\n");

    // Naive: Only 1 iteration because it's slow
    run_benchmark(d_a, d_b, d_out, out, a, b, N, AddAlgo::NAIVE_SINGLE_THREAD, "Naive (Single Thread)", 1);
    
    // Single Block: 10 iterations
    run_benchmark(d_a, d_b, d_out, out, a, b, N, AddAlgo::SINGLE_BLOCK_MULTI, "Single Block Multi-thread", 10);
    
    // Multi Block: 100 iterations (Fastest)
    run_benchmark(d_a, d_b, d_out, out, a, b, N, AddAlgo::MULTI_BLOCK_MULTI, "Multi Block (Optimized)", 100);

    CUDA_CHECK(cudaFree(d_a)); 
    CUDA_CHECK(cudaFree(d_b)); 
    CUDA_CHECK(cudaFree(d_out));
    free(a); free(b); free(out);

    return 0;
}
