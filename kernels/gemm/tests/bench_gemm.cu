#include <iostream>
#include <vector>
#include <random>
#include <tuple>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdlib>
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

bool parse_algo(std::string name, GemmAlgo& algo, std::string& algo_name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    if (name == "NAIVE") {
        algo = GemmAlgo::NAIVE;
        algo_name = "NAIVE";
    } else if (name == "SHARED_MEM" || name == "SHARED_MEMORY" || name == "SHARED") {
        algo = GemmAlgo::SHARED_MEMORY;
        algo_name = "SHARED_MEM";
    } else if (name == "REGISTER") {
        algo = GemmAlgo::REGISTER;
        algo_name = "REGISTER";
    } else if (name == "VECTORIZED" || name == "VECTOR") {
        algo = GemmAlgo::VECTORIZED;
        algo_name = "VECTORIZED";
    } else if (name == "DOUBLE_BUF" || name == "DOUBLE_BUFFERED" || name == "BUFFER") {
        algo = GemmAlgo::DOUBLE_BUFFERED;
        algo_name = "DOUBLE_BUF";
    } else if (name == "TENSOR_CORE" || name == "TENSOR") {
        algo = GemmAlgo::TENSOR_CORE;
        algo_name = "TENSOR_CORE";
    } else if (name == "ASYNC") {
        algo = GemmAlgo::ASYNC;
        algo_name = "ASYNC";
    } else if (name == "ULTIMATE") {
        algo = GemmAlgo::ULTIMATE;
        algo_name = "ULTIMATE";
    } else if (name == "ZHIHU") {
        algo = GemmAlgo::ZHIHU;
        algo_name = "ZHIHU";
    } else if (name == "CUBLAS") {
        algo = GemmAlgo::CUBLAS;
        algo_name = "CUBLAS";
    } else if (name == "CUTLASS") {
        algo = GemmAlgo::CUTLASS;
        algo_name = "CUTLASS";
    } else {
        return false;
    }
    return true;
}

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << "\n"
              << "  " << prog << " M N K ALGO [iters]\n\n"
              << "ALGO: NAIVE, SHARED_MEM, REGISTER, VECTORIZED, DOUBLE_BUF, "
              << "TENSOR_CORE, ASYNC, ULTIMATE, ZHIHU, CUBLAS, CUTLASS\n";
}

int main(int argc, char** argv) {
    if (argc != 1 && argc != 5 && argc != 6) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 5 || argc == 6) {
        int M = std::stoi(argv[1]);
        int N = std::stoi(argv[2]);
        int K = std::stoi(argv[3]);
        int iters = (argc == 6) ? std::stoi(argv[5]) : 10;

        GemmAlgo algo;
        std::string algo_name;
        if (!parse_algo(argv[4], algo, algo_name)) {
            std::cerr << "Unsupported algorithm: " << argv[4] << "\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        BenchmarkResult result = run_benchmark(M, N, K, algo, algo_name, iters);
        std::cout << "M,N,K,Algorithm,Time_ms,GFLOPS\n"
                  << M << "," << N << "," << K << ","
                  << result.name << ","
                  << std::fixed << std::setprecision(4) << result.avg_ms << ","
                  << std::fixed << std::setprecision(2) << result.gflops << "\n";
        return EXIT_SUCCESS;
    }

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
        results.push_back(run_benchmark(M, N, K, GemmAlgo::TENSOR_CORE, "TENSOR_CORE"));
        // results.push_back(run_benchmark(M, N, K, GemmAlgo::ASYNC, "ASYNC"));
        results.push_back(run_benchmark(M, N, K, GemmAlgo::ULTIMATE, "ULTIMATE"));
        // results.push_back(run_benchmark(M, N, K, GemmAlgo::ZHIHU, "ZHIHU"));
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
