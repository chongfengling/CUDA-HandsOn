# Softmax Operator Optimization

## Mathematical Definition

(**softmax**) Given a vector $\mathbf{x} = [x_1, x_2, \dots, x_n] \in \mathbb{R}^n$, the standard Softmax output is defined as:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^N e^{x_j}}
$$

When the input $\mathbf{x} \in \mathbb{R}^{m \times n}$ is a matrix, the softmax is applied to the last dimension (columns) of the matrix usually. 


(**safe softmax**) Given a vector $\mathbf{x} = [x_1, x_2, \dots, x_n] \in \mathbb{R}^n$, the safe Softmax output is defined as:

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_{j=1}^N e^{x_j - \max(\mathbf{x})}}
$$

Subtracting the maximum value ensures numerical stability by preventing overflow in the exponential function.

## Implementations

### 1. NAIVE_SHARED_MEM
A basic implementation using tree reduction in shared memory to find the row maximum and sum of exponentials. It requires multiple synchronization points and multiple passes over the data.

### 2. WARP_OPTIMIZED
Uses warp-level primitives (`__shfl_down_sync`) for faster reduction within warps. This reduces shared memory usage and synchronization overhead between threads in the same warp.

### 3. ONLINE_SOFTMAX
Implements the Online Softmax algorithm, which computes the maximum and the sum of exponentials in a single pass over the data. This significantly reduces memory traffic and improves performance.

## Performance Comparison

| Version | Latency (ms) | Bandwidth (GB/s) |
| :--- | :--- | :--- |
| Naive | 0.0542 | 1238.49 |
| Warp Optimized | 0.0474 | 1414.63 |
| Online Softmax | 0.0426 | 1573.65 |

*Note: Benchmarks performed on [M=8192, N=1024] input.*

## How to Build and Run

```bash
mkdir build && cd build
cmake ..
make
./test_softmax
```

### Benchmark

```bash
./bench_softmax
./bench_softmax 8192 1024 64 16384
```
