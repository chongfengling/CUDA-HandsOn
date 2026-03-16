#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 100000000
#define MAX_ERR 1e-6

// Uncomment ONE of the following lines to select the kernel to run:
// #define KERNEL_SINGLE
// #define KERNEL_MULTITHREAD
#define KERNEL_MULTIBLOCK

// ------------------- Kernels -------------------

// 1️⃣ Single-thread kernel
__global__ void vector_add_kernel(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

// 2️⃣ Single-block multi-thread kernel
__global__ void vector_add_kernel_multithreads(float *out, float *a, float *b, int n) {
    int idx = threadIdx.x;
    int stride = blockDim.x;
    for(int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

// 3️⃣ Multi-block multi-thread kernel
__global__ void vector_add_kernel_multiblocks(float *out, float *a, float *b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

// ------------------- Utility Functions -------------------
void verify(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        if(fabs(out[i] - (a[i] + b[i])) > MAX_ERR){
            printf("Mismatch at %d: %f != %f\n", i, out[i], a[i]+b[i]);
            exit(1);
        }
    }
}

// Print GPU memory usage
void printGPUMemUsage(const char *msg) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    printf("%s - GPU memory used: %.2f MB / %.2f MB\n", 
        msg, (total_bytes - free_bytes)/1024.0/1024.0, total_bytes/1024.0/1024.0);
}

// ------------------- Main -------------------
int main() {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate host memory
    a = (float*)malloc(sizeof(float)*N);
    b = (float*)malloc(sizeof(float)*N);
    out = (float*)malloc(sizeof(float)*N);

    // Initialize input vectors on host
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Print initial GPU memory usage
    printGPUMemUsage("Before device allocation");

    // Allocate device memory
    cudaMalloc(&d_a, sizeof(float)*N);
    cudaMalloc(&d_b, sizeof(float)*N);
    cudaMalloc(&d_out, sizeof(float)*N);

    // Print GPU memory usage after allocation
    printGPUMemUsage("After device allocation");
    
    // Copy input vectors from host to device
    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);

    // CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

#ifdef KERNEL_SINGLE
    cudaEventRecord(start);
    vector_add_kernel<<<1,1>>>(d_out, d_a, d_b, N);
    cudaEventRecord(stop);
#elif defined(KERNEL_MULTITHREAD)
    int blockSize = 256;
    cudaEventRecord(start);
    vector_add_kernel_multithreads<<<1, blockSize>>>(d_out, d_a, d_b, N);
    cudaEventRecord(stop);
#elif defined(KERNEL_MULTIBLOCK)
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1)/blockSize;
    cudaEventRecord(start);
    vector_add_kernel_multiblocks<<<numBlocks, blockSize>>>(d_out, d_a, d_b, N);
    cudaEventRecord(stop);
#else
    #error "Please define one kernel macro: KERNEL_SINGLE, KERNEL_MULTITHREAD, KERNEL_MULTIBLOCK"
#endif

    // Copy result back to host and verify
    cudaEventSynchronize(stop);
    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);
    verify(out, a, b, N);

#ifdef KERNEL_SINGLE
    printf("Single-thread kernel PASSED\n");
#elif defined(KERNEL_MULTITHREAD)
    printf("Single-block multi-thread kernel PASSED\n");
#elif defined(KERNEL_MULTIBLOCK)
    printf("Multi-block multi-thread kernel PASSED\n");
#endif

    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f ms\n", ms);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(a); free(b); free(out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
