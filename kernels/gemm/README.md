# CUDA GEMM (General Matrix Multiplication)

This directory contains a suite of SGEMM (Single-precision General Matrix Multiplication) implementations, ranging from a basic naive version to highly optimized kernels utilizing shared memory tiling, register tiling, and double buffering.

## Mathematical Definition

The operation is defined as:
$$C = \alpha (A \times B) + \beta C$$
In this implementation, we focus on the core $C = A \times B$ case, where:
- $A$ is an $M \times K$ matrix.
- $B$ is a $K \times N$ matrix.
- $C$ is an $M \times N$ matrix.

## Optimization Evolution

We follow a progressive optimization path to reach high performance:

1.  **NAIVE**: The baseline implementation. Each thread computes one element of $C$. Heavily memory-bound and suffers from uncoalesced memory access for matrix B.
2.  **SHARED_MEMORY**: Uses 2D tiling in shared memory to reduce global memory traffic. Improves memory coalescing.
3.  **REGISTER**: Introduces 2D thread-level tiling. Data is moved from shared memory to registers to reduce shared memory bandwidth pressure.
4.  **VECTORIZED**: Uses `float4` vectorized loads to maximize memory throughput.
5.  **DOUBLE_BUFFERED**: Implements software pipelining to overlap global-to-shared memory transfers with computation.
6.  **ULTIMATE**: A hand-tuned kernel combining warp-tiling, bank-conflict-free shared memory access, and extreme register reuse.
7.  **CUBLAS/CUTLASS**: Industry-standard implementations used as performance gold standards.

## Performance Comparison (GFLOPS)

Benchmarks performed on local GPU (Architecture: SM 80/Blackwell).
Currenrtly, we only have results for M=N=K=1024 and don't benchmark across different shapes. The results are as follows:

| M | N | K | Algorithm | Time (ms) | GFLOPS | vs cuBLAS |
| :---: | :---: | :---: | :--- | ---: | ---: | ---: |
| 1024 | 1024 | 1024 | NAIVE | 0.6752 | 3180.54 | 11.78% |
| 1024 | 1024 | 1024 | SHARED_MEM | 0.4383 | 4899.96 | 18.15% |
| 1024 | 1024 | 1024 | REGISTER | 0.1598 | 13439.24 | 49.77% |
| 1024 | 1024 | 1024 | VECTORIZED | 0.1341 | 16017.58 | 59.32% |
| 1024 | 1024 | 1024 | DOUBLE_BUF | 0.1197 | 17945.47 | 66.46% |
| 1024 | 1024 | 1024 | TENSOR_CORE | 0.1627 | 13196.64 | 48.87% |
| 1024 | 1024 | 1024 | ULTIMATE | 0.0900 | 23861.78 | 88.37% |
| 1024 | 1024 | 1024 | CUBLAS | 0.0795 | 27002.32 | 100.00% |
| 1024 | 1024 | 1024 | CUTLASS | 0.0764 | 28126.10 | 104.16% |


## Directory Structure

- `include/`: Public interface (`gemm.h`).
- `src/`: Kernel implementations and dispatcher.
- `tests/`: Correctness tests and performance benchmarks.
- `scripts/`: Profiling and build scripts.

## Building and Running

### Build
```bash
mkdir build && cd build
cmake ..
make
```

### Run Tests
```bash
./test_gemm
```

### Run Benchmarks
```bash
./bench_gemm
```

### Profiling
```bash
./scripts/profile.sh <ALGO_ID>
```
