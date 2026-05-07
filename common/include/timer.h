#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"

/**
 * @brief A simple timer using CUDA events for precise GPU timing.
 */
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }

    void start(cudaStream_t stream = 0) {
        stream_ = stream;
        CUDA_CHECK(cudaEventRecord(start_, stream_));
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    /**
     * @brief Returns the elapsed time in milliseconds.
     */
    float elapsed_msecs() {
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

private:
    cudaEvent_t start_, stop_;
    cudaStream_t stream_ = 0;
};
