# CUDA GEMM Implementation

A comprehensive, modular implementation of Single Precision General Matrix Multiplication (SGEMM) using CUDA. This project serves as an educational and practical example of how to optimize matrix multiplication on NVIDIA GPUs.

## Overview

This repository demonstrates the step-by-step optimization of matrix multiplication ( $C = A \times B$ ), providing a clean API and separating the interface from the underlying CUDA kernel implementations.

### Supported Algorithms

*   **NAIVE:** A straightforward, unoptimized GPU implementation where each thread computes one element of the output matrix $C$. It serves as a baseline for correctness and performance.
*   **SHARED_MEMORY (Tiled):** An optimized implementation that utilizes CUDA shared memory to reduce global memory bandwidth bottleneck. By loading "tiles" of matrices $A$ and $B$ into fast, on-chip shared memory, threads within a block can reuse data, significantly improving performance. The tile size used in this implementation is $16 \times 16$.

## Project Structure

The project follows a standard C++ engineering layout, separating public interfaces from private implementations.

```text
├── CMakeLists.txt        # CMake build configuration
├── README.md             # This file
├── include/
│   └── gemm.h            # Public API header (contains the sgemm function)
├── src/
│   ├── gemm_api.cpp      # Public API implementation (calls the dispatcher)
│   └── kernels/
│       ├── dispatch.h    # Internal header for kernel dispatching
│       ├── dispatch.cpp  # Routes the call to the selected algorithm
│       ├── kernels.h     # Internal header declaring kernel launch functions
│       ├── gemm_naive.cu # Implementation of the naive kernel
│       └── gemm_shared.cu# Implementation of the shared memory tiled kernel
├── tests/
│   └── test_gemm.cu      # Comprehensive unit tests (correctness verification)
└── benchmarks/
    └── bench_gemm.cu     # Performance benchmarking script
```

**Design Philosophy:** The headers in `src/kernels/` (`dispatch.h`, `kernels.h`) are intentionally kept internal. Users of the library only need to include `include/gemm.h` and link against the library. This encapsulates the CUDA specific launch details and kernel declarations, preventing namespace pollution and reducing recompilation dependencies for external code.

## Requirements

*   **CUDA Toolkit:** (Tested with 12.x, but should work with older versions supporting C++14)
*   **C++ Compiler:** Supporting C++14 (e.g., GCC, Clang, MSVC)
*   **CMake:** >= 3.18 (Optional, but recommended for building)

## Building the Project

### Using CMake (Recommended)

```bash
mkdir build
cd build
cmake ..
make bench_gemm
make test_gemm
```

This will build the core library (`libgemm_lib.a`), the test executable (`test_gemm`), and the benchmark executable (`bench_gemm`).

### Manual Compilation (using nvcc)

If you encounter CMake configuration issues (e.g., with newer GCC versions conflicting with CUDA), you can compile the executables directly using `nvcc`:

**Compile Tests:**
```bash
nvcc -std=c++14 -Iinclude -Isrc src/gemm_api.cpp src/kernels/dispatch.cpp src/kernels/gemm_naive.cu src/kernels/gemm_shared.cu tests/test_gemm.cu -o test_gemm
```

**Compile Benchmarks:**
```bash
nvcc -std=c++14 -O3 -I ./include -I ./src -I ../third-party/cutlass/include src/gemm_api.cpp src/kernels/dispatch.cpp src/kernels/gemm_naive.cu src/kernels/gemm_shared.cu benchmarks/bench_gemm.cu -o bench_gemm
```

nvcc -std=c++17 -arch=sm_80 -I./include -I./src -I../third-party/cutlass/include src/gemm_api.cpp src/kernels/dispatch.cpp src/kernels/gemm_naive.cu src/kernels/gemm_shared.cu

## Running

### Tests

The test suite verifies the GPU results against a standard CPU implementation across various shapes and edge cases, including non-square matrices, matrices with negative numbers, and dimensions that are not perfectly divisible by the tile size.

```bash
./test_gemm
```

### Benchmarks

The benchmark suite measures the execution time and calculates the GFLOPS (Giga Floating-Point Operations Per Second) for both algorithms across different matrix sizes.

```bash
./bench_gemm
```

## Performance Comparison

Here is an example benchmark run on an NVIDIA GPU, demonstrating the performance advantage of the Shared Memory (Tiled) approach over the Naive implementation, especially for larger matrices.

| Matrix Size (M=N=K) | Algorithm     | Avg Time (ms) | Performance (GFLOPS) |
| :------------------ | :------------ | :------------ | :------------------- |
| 1024                | NAIVE         | 0.612         | ~ 3505               |
| 1024                | SHARED_MEMORY | 0.522         | ~ 4111               |
| 2048                | NAIVE         | 5.290         | ~ 3247               |
| 2048                | SHARED_MEMORY | 3.961         | ~ 4337               |
| 4096                | NAIVE         | 42.160        | ~ 3259               |
| 4096                | SHARED_MEMORY | 29.991        | ~ 4582               |
| 8192 x 4096 x 6144  | NAIVE         | 132.492       | ~ 3112               |
| 8192 x 4096 x 6144  | SHARED_MEMORY | 96.943        | ~ 4253               |

*Note: Performance will vary based on the specific GPU hardware used.*
