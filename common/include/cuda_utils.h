#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

/**
 * @brief Macro for checking CUDA API errors.
 */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            throw std::runtime_error("CUDA Error");                         \
        }                                                                   \
    } while (0)

/**
 * @brief Macro for checking the last CUDA error.
 * Useful after kernel launches which are asynchronous.
 */
#define LAST_KERNEL_CHECK()                                                 \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA kernel error at %s %d: %s\n", __FILE__,   \
                    __LINE__, cudaGetErrorString(err));                     \
            throw std::runtime_error("CUDA Kernel Error");                  \
        }                                                                   \
    } while (0)

/**
 * @brief Ceiling division for grid/block calculation.
 */
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
