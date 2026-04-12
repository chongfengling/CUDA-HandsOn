#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "sgemm.cuh"

// =============================================================
// CPU Reference: A(M×K) * B(K×N) = C(M×N)
// =============================================================
void cpu_sgemm(int M, int N, int K,
               const float* A, const float* B, float* C) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// =============================================================
// Utils
// =============================================================
void init_matrix(std::vector<float>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : mat) x = dist(gen);
}

bool verify(int M, int N,
            const std::vector<float>& C_cpu,
            const std::vector<float>& C_gpu) {

    const float eps = 1e-3f;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(C_cpu[i] - C_gpu[i]) > eps) {
            std::cerr << "Mismatch at " << i
                      << " CPU=" << C_cpu[i]
                      << " GPU=" << C_gpu[i] << std::endl;
            return false;
        }
    }
    std::cout << "Verification SUCCESS\n";
    return true;
}

// =============================================================
// 通用测试函数
// =============================================================
void run_test(int M, int N, int K,
              const std::vector<float>& A,
              const std::vector<float>& B) {

    std::vector<float> C_cpu(M * N, 0);
    std::vector<float> C_gpu(M * N, 0);

    cpu_sgemm(M, N, K, A.data(), B.data(), C_cpu.data());

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    solve(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    verify(M, N, C_cpu, C_gpu);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// =============================================================
// Example Tests（重点加强）
// =============================================================
void run_example_tests() {

    std::cout << "==== Example 1: 2x2 ====\n";
    run_test(2, 2, 2,
        {1,2,3,4},
        {5,6,7,8}
    );

    std::cout << "==== Example 2: 1x3 * 3x1 ====\n";
    run_test(1, 1, 3,
        {1,2,3},
        {4,5,6}
    );

    std::cout << "==== Example 3: 非方阵 ====\n";
    run_test(2, 2, 3,
        {
            1,2,3,
            4,5,6
        },
        {
            7,8,
            9,10,
            11,12
        }
    );

    std::cout << "==== Example 4: 含负数 ====\n";
    run_test(2, 2, 2,
        {
            -1,2,
            3,-4
        },
        {
            5,-6,
            7,8
        }
    );

    std::cout << "==== Example 5: 单位矩阵 ====\n";
    run_test(3, 3, 3,
        {
            1,0,0,
            0,1,0,
            0,0,1
        },
        {
            1,2,3,
            4,5,6,
            7,8,9
        }
    );

    std::cout << "==== Example 6: 全1矩阵 ====\n";
    run_test(2, 2, 4,
        std::vector<float>(2*4, 1.0f),
        std::vector<float>(4*2, 1.0f)
    );
}

// =============================================================
// Correctness Test
// =============================================================
void run_correctness_test() {
    int M = 128, N = 127, K = 129;

    std::vector<float> A(M*K);
    std::vector<float> B(K*N);

    init_matrix(A);
    init_matrix(B);

    run_test(M, N, K, A, B);
}

// =============================================================
// Benchmark
// =============================================================
void run_benchmark(int M, int N, int K, int iters=10) {

    std::cout << "\n==== Benchmark M=" << M
              << " N=" << N << " K=" << K << " ====\n";

    std::vector<float> A(M*K);
    std::vector<float> B(K*N);

    init_matrix(A);
    init_matrix(B);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M*N*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));

    // warmup
    solve(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        solve(M, N, K, d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float avg = ms / iters;
    double gflops = (2.0 * M * N * K) / (avg * 1e6);

    std::cout << "Time: " << avg << " ms\n";
    std::cout << "GFLOPS: " << gflops << "\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// =============================================================
// Main
// =============================================================
int main() {

    run_example_tests();

    run_correctness_test();

    run_benchmark(8192, 4096, 6144);

    return 0;
}