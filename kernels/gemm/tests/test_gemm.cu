#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include "gemm.h"
#include "cuda_utils.h"

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
            const std::vector<float>& C_gpu,
            float eps = 1e-3f) {

    for (int i = 0; i < M * N; ++i) {
        if (std::abs(C_cpu[i] - C_gpu[i]) > eps) {
            std::cerr << "Mismatch at " << i
                      << " CPU=" << C_cpu[i]
                      << " GPU=" << C_gpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

// =============================================================
// Test Runner
// =============================================================
void run_test(int M, int N, int K,
              const std::vector<float>& A,
              const std::vector<float>& B,
              GemmAlgo algo, const std::string& algo_name) {

    std::cout << "Testing " << algo_name << " M=" << M << " N=" << N << " K=" << K << " ... ";
    std::vector<float> C_cpu(M * N, 0);
    std::vector<float> C_gpu(M * N, 0);

    cpu_sgemm(M, N, K, A.data(), B.data(), C_cpu.data());

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    sgemm(M, N, K, d_A, d_B, d_C, algo);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float eps = (algo == GemmAlgo::TENSOR_CORE) ? 5e-2f : 1e-3f;
    if (!verify(M, N, C_cpu, C_gpu, eps)) {
        std::cerr << "FAILED!" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "SUCCESS\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void run_algo_tests(GemmAlgo algo, const std::string& algo_name) {
    std::cout << "\n==== Running Tests for " << algo_name << " ====\n";
    
    run_test(2, 2, 2, {1,2,3,4}, {5,6,7,8}, algo, algo_name);
    run_test(1, 1, 3, {1,2,3}, {4,5,6}, algo, algo_name);
    run_test(2, 2, 3, {1,2,3,4,5,6}, {7,8,9,10,11,12}, algo, algo_name);
    run_test(2, 2, 2, {-1,2,3,-4}, {5,-6,7,8}, algo, algo_name);
    run_test(3, 3, 3, {1,0,0,0,1,0,0,0,1}, {1,2,3,4,5,6,7,8,9}, algo, algo_name);
    run_test(2, 2, 4, std::vector<float>(2*4, 1.0f), std::vector<float>(4*2, 1.0f), algo, algo_name);

    int M = 127, N = 129, K = 128;
    std::vector<float> A(M*K);
    std::vector<float> B(K*N);
    init_matrix(A);
    init_matrix(B);
    run_test(M, N, K, A, B, algo, algo_name);
}

int main() {
    run_algo_tests(GemmAlgo::NAIVE, "NAIVE");
    run_algo_tests(GemmAlgo::SHARED_MEMORY, "SHARED_MEMORY");
    run_algo_tests(GemmAlgo::REGISTER, "REGISTER");
    run_algo_tests(GemmAlgo::VECTORIZED, "VECTORIZED");
    run_algo_tests(GemmAlgo::DOUBLE_BUFFERED, "DOUBLE_BUFFERED");
    run_algo_tests(GemmAlgo::TENSOR_CORE, "TENSOR_CORE");
    run_algo_tests(GemmAlgo::ASYNC, "ASYNC");
    run_algo_tests(GemmAlgo::ULTIMATE, "ULTIMATE");
    run_algo_tests(GemmAlgo::ZHIHU, "ZHIHU");
    run_algo_tests(GemmAlgo::CUBLAS, "CUBLAS");
    run_algo_tests(GemmAlgo::CUTLASS, "CUTLASS");

    std::cout << "\nAll tests passed successfully!\n";
    return 0;
}
