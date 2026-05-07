#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "softmax.h"
#include "cuda_utils.h"
#include "timer.h"

struct BenchmarkResult {
    std::string algo_name;
    float avg_ms;
    double logical_gb_s;
    double elements_per_s;
    bool verified;
};

struct Algorithm {
    SoftmaxAlgo algo;
    const char* name;
};

void init_input(std::vector<float>& input) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    for (float& x : input) {
        x = dist(gen);
    }
}

int repeats_for_shape(int M, int N) {
    const long long elements = static_cast<long long>(M) * N;
    if (elements <= 1 << 20) return 300;
    if (elements <= 8 << 20) return 150;
    return 80;
}

bool verify_rows(const float* d_output, int M, int N) {
    std::vector<float> output(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const std::vector<int> rows = {
        0,
        M / 2,
        std::max(0, M - 1)
    };

    for (int row : rows) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            sum += output[static_cast<size_t>(row) * N + col];
        }
        if (std::fabs(sum - 1.0f) > 1e-3f) {
            std::cerr << "Verification failed at row " << row
                      << ": sum=" << sum << "\n";
            return false;
        }
    }
    return true;
}

BenchmarkResult run_benchmark(int M, int N,
                              const float* d_input,
                              float* d_output,
                              SoftmaxAlgo algo,
                              const std::string& algo_name) {
    const int warmups = 10;
    const int repeats = repeats_for_shape(M, N);

    for (int i = 0; i < warmups; ++i) {
        softmax(M, N, d_input, d_output, algo);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < repeats; ++i) {
        softmax(M, N, d_input, d_output, algo);
    }
    timer.stop();

    const float avg_ms = timer.elapsed_msecs() / repeats;
    const double elements = static_cast<double>(M) * N;
    const double logical_bytes = elements * sizeof(float) * 2.0;
    const double logical_gb_s = (logical_bytes / 1e9) / (avg_ms / 1000.0);
    const double elements_per_s = elements / (avg_ms / 1000.0);
    const bool verified = verify_rows(d_output, M, N);

    return {algo_name, avg_ms, logical_gb_s, elements_per_s, verified};
}

std::vector<std::pair<int, int>> parse_shapes(int argc, char** argv) {
    if (argc == 1) {
        return {
            {32768, 128},
            {8192, 512},
            {8192, 1024},
            {4096, 4096},
            {1024, 8192},
            {64, 16384}
        };
    }

    if ((argc - 1) % 2 != 0) {
        std::cerr << "Usage: " << argv[0] << " [M N]...\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::pair<int, int>> shapes;
    for (int i = 1; i < argc; i += 2) {
        int M = std::atoi(argv[i]);
        int N = std::atoi(argv[i + 1]);
        if (M <= 0 || N <= 0) {
            std::cerr << "M and N must be positive integers.\n";
            std::exit(EXIT_FAILURE);
        }
        shapes.push_back({M, N});
    }
    return shapes;
}

int main(int argc, char** argv) {
    const std::vector<Algorithm> algorithms = {
        {SoftmaxAlgo::NAIVE, "NAIVE"},
        {SoftmaxAlgo::WARP, "WARP"},
        {SoftmaxAlgo::ONLINE, "ONLINE"}
    };
    const std::vector<std::pair<int, int>> shapes = parse_shapes(argc, argv);

    const std::string separator =
        "+--------+--------+----------+------------+------------+--------------+----------+";

    std::cout << "\nSoftmax Benchmark\n";
    std::cout << "Pass custom shapes as: ./bench_softmax M N [M N]...\n\n";
    std::cout << separator << "\n";
    std::cout << "|      M |      N | Algorithm| Time (ms)  | Logical GB/s | Elements/s   | Verified |\n";
    std::cout << separator << "\n";

    for (const auto& shape : shapes) {
        const int M = shape.first;
        const int N = shape.second;
        const size_t elements = static_cast<size_t>(M) * N;
        const size_t bytes = elements * sizeof(float);

        std::vector<float> h_input(elements);
        init_input(h_input);

        float* d_input = nullptr;
        float* d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

        for (const Algorithm& algorithm : algorithms) {
            BenchmarkResult result = run_benchmark(M, N, d_input, d_output,
                                                   algorithm.algo, algorithm.name);
            std::cout << "| " << std::setw(6) << M
                      << " | " << std::setw(6) << N
                      << " | " << std::setw(8) << result.algo_name
                      << " | " << std::setw(10) << std::fixed << std::setprecision(4) << result.avg_ms
                      << " | " << std::setw(12) << std::fixed << std::setprecision(2) << result.logical_gb_s
                      << " | " << std::setw(12) << std::scientific << std::setprecision(3) << result.elements_per_s
                      << " | " << std::setw(8) << (result.verified ? "yes" : "no")
                      << " |\n";
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        std::cout << separator << "\n";
    }

    return 0;
}
