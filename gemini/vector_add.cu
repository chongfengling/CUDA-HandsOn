#include <stdio.h>
#include <cuda_runtime.h>

// 补充定义 Kernel，以保证代码完整
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int n = 1 << 20; // 1M 元素
    size_t bytes = n * sizeof(float);

    // 1. 定义指针
    float *a, *b, *c;

    // 2. 直接分配 Unified Memory (代替 malloc)
    // 这块内存 CPU 和 GPU 都能访问
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // 3. 在 CPU 端初始化数据 (直接操作统一内存)
    // 此时数据会被放在 CPU 页缓存或者通过缺页中断处理
    for(int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // 设置并发配置
    int blockSize = 256; 
    int gridSize = (n + blockSize - 1) / blockSize; 

    // --- 准备计时器 ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Launching kernel with Grid Size: %d, Block Size: %d\n", gridSize, blockSize);

    // 4. 记录开始
    cudaEventRecord(start);

    // 5. 启动核函数
    // GPU 驱动会处理数据迁移，将数据从 CPU 侧搬运到 GPU 侧 (Page Faults)
    vector_add<<<gridSize, blockSize>>>(a, b, c, n);

    // 6. 记录结束
    cudaEventRecord(stop);

    // 7. 同步并计算时间
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel Execution Time: %.3f ms\n", milliseconds);
    // 带宽公式推导: (Bytes / 10^9) / (ms / 1000) = (Bytes / ms) / 10^6
    printf("Effective Bandwidth: %.3f GB/s\n", (3.0 * bytes) / milliseconds / 1e6); 

    // 8. 验证结果
    // 此时 CPU 再次访问 c[i]，驱动会自动将数据从 GPU 搬回 CPU (如果还在 GPU 的话)
    // 这里的同步其实由上面的 cudaEventSynchronize 保证了，
    // 如果没有 Event，必须调用 cudaDeviceSynchronize() 才能在 CPU 读数据
    printf("Result verification (expecting 3.0):\n");
    for(int i = 0; i < 5; i++) {
        printf("Index %d: %.1f + %.1f = %.1f\n", i, a[i], b[i], c[i]);
    }

    // 9. 释放内存 (必须使用 cudaFree)
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    
    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}