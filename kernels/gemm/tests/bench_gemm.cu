#include <iostream>
#include <vector>
#include <random>
#include <tuple>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>
#include "gemm.h"
#include "cuda_utils.h"
#include "timer.h"

// 存储单次测试结果的结构体
struct BenchmarkResult {
    std::string name;
    float avg_ms;
    double gflops;
};

void init_matrix(std::vector<float>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : mat) x = dist(gen);
}

// 统一的测试函数
BenchmarkResult run_benchmark(int M, int N, int K, GemmAlgo algo, const std::string& algo_name, int iters=10) {
    std::vector<float> A(M*K), B(K*N);
    init_matrix(A); init_matrix(B);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));

    sgemm(M, N, K, d_A, d_B, d_C, algo); // warmup
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        sgemm(M, N, K, d_A, d_B, d_C, algo);
    }
    timer.stop();

    float avg = timer.elapsed_msecs() / iters;
    double gflops = (2.0 * M * N * K) / (avg * 1e6);

    CUDA_CHECK(cudaFree(d_A)); 
    CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_C));

    return {algo_name, avg, gflops};
}

int main() {
    std::vector<std::tuple<int, int, int>> shapes = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 4096, 6144}
    };

    std::string separator = "+-------+-------+-------+-----------------+------------+------------+------------+";
    std::cout << "\n" << separator << "\n";
    std::cout << "|   M   |   N   |   K   |    Algorithm    | Time (ms)  |   GFLOPS   | vs cuBLAS  |\n";
    std::cout << separator << "\n";

    for (const auto& shape : shapes) {
        int M, N, K;
        std::tie(M, N, K) = shape;

        std::vector<BenchmarkResult> results;
        results.push_back(run_benchmark(M, N, K, GemmAlgo::NAIVE, "NAIVE"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::SHARED_MEMORY, "SHARED_MEM"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::REGISTER, "REGISTER"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::VECTORIZED, "VECTORIZED"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::DOUBLE_BUFFERED, "DOUBLE_BUF"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::ASYNC, "ASYNC"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::ULTIMATE, "ULTIMATE"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::ZHIHU, "ZHIHU"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::CUBLAS, "CUBLAS"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::CUTLASS, "CUTLASS"));

        double cublas_gflops = 1.0;
        for (const auto& r : results) {
            if (r.name == "CUBLAS") cublas_gflops = r.gflops;
        }

        for (const auto& r : results) {
            double ratio = (r.gflops / cublas_gflops) * 100.0;
            std::cout << "| " << std::setw(5) << M 
                      << " | " << std::setw(5) << N 
                      << " | " << std::setw(5) << K 
                      << " | " << std::setw(15) << r.name 
                      << " | " << std::setw(10) << std::fixed << std::setprecision(4) << r.avg_ms 
                      << " | " << std::setw(10) << std::fixed << std::setprecision(2) << r.gflops 
                      << " | " << std::setw(9) << std::fixed << std::setprecision(2) << ratio << "% |" << std::endl;
        }
        std::cout << separator << "\n";
    }

    return 0;
}
