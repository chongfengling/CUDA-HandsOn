# Vector Add Operator Optimization

This directory contains various implementations of the Vector Addition operator ($C = A + B$), demonstrating the transition from naive single-threaded execution to optimized grid-stride loops.

## Mathematical Definition

Given two vectors $\mathbf{a}$ and $\mathbf{b}$ of length $N$:

$$
c_i = a_i + b_i \quad \text{for } i = 0, 1, \dots, N-1
$$

## Implementations

### 1. NAIVE_SINGLE_THREAD
A single-threaded kernel where one thread iterates through the entire array. Extremely slow and fails to utilize GPU parallelism.

### 2. SINGLE_BLOCK_MULTI
Uses multiple threads within a single block. Limited by the maximum threads per block (usually 1024), but uses a stride loop to handle larger $N$.

### 3. MULTI_BLOCK_MULTI
The standard optimized implementation using a grid-stride loop. Distributes work across multiple blocks and threads, maximizing hardware utilization.

## Performance Comparison

| Version | Average Time (ms) | Bandwidth (GB/s) |
| :--- | :--- | :--- |
| Naive | 658.7781 | 0.17 |
| Single Block | 15.3652 | 7.27 |
| Multi Block | 0.1300 | 859.39 |

*Note: Benchmarks performed on $N = 10^7$ elements.*

## How to Build and Run

```bash
mkdir build && cd build
cmake ..
make
./test_add
```

## References

- [CUDA Tutorial](https://cuda-tutorial.readthedocs.io/en/latest/)