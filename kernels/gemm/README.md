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
6.  **ASYNC**: Utilizes NVIDIA Ampere+ `cp.async` instructions for asynchronous data movement from global to shared memory.
7.  **ULTIMATE**: A hand-tuned kernel combining warp-tiling, bank-conflict-free shared memory access, and extreme register reuse.
8.  **ZHIHU**: A high-performance implementation based on community best practices (Ref: [Zhihu](https://zhuanlan.zhihu.com/p/1910636263666610461)).
9.  **CUBLAS/CUTLASS**: Industry-standard implementations used as performance gold standards.

## Performance Comparison (GFLOPS)

Benchmarks performed on local GPU (Architecture: SM 80/Blackwell).

| Matrix Size ($M=N=K$) | NAIVE | SHARED | REGISTER | VECTORIZED | ULTIMATE | CUBLAS | PyTorch |
| :------------------- | :---- | :----- | :------- | :--------- | :------- | :----- | :------ |
| 1024                 | 3568  | 4291   | 13432    | 15836      | 24364    | 26867  | 26826   |
| 2048                 | 3191  | 4352   | 16687    | 23733      | 20902    | 32413  | 33375   |
| 4096                 | 3288  | 4518   | 19065    | 25646      | 25503    | 37975  | 35040   |

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
