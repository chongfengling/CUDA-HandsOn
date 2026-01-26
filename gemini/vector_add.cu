#include <cuda_runtime.h>
#include <stdio.h>

// === TODO 1: 编写核函数 ===
__global__ void vector_add(float *a, float *b, float *c, int n) {
    // 挑战：如何通过 blockIdx, blockDim, threadIdx 计算出当前线程负责数组中的哪一个索引？
    // 提示：就像之前的公式一样，算出全局唯一的 ID
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    // 边界检查：防止计算出的索引超过数组长度 (n)
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1 << 20; // 2的20次方，大约 100 万 (1,048,576) 个元素
    size_t bytes = n * sizeof(float);

    // 1. 分配 Host (CPU) 内存
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // 初始化数据 (A全为1.0, B全为2.0)
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 2. 分配 Device (GPU) 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 3. 将数据从 Host 拷贝到 Device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // === TODO 2: 设置并发配置 ===
    // 提示：每个 Block 线程数最好是 128, 256 或 512
    int blockSize = 256; 
    
    // 挑战：你需要多少个 Block 才能覆盖 n 个元素？
    // 提示：公式 (Total + Block - 1) / Block
    int gridSize = (n + blockSize - 1) / blockSize; 

    printf("Launching kernel with Grid Size: %d, Block Size: %d\n", gridSize, blockSize);

    // ... (前面的内存分配和拷贝代码保持不变) ...

    // --- 新增：准备计时器 ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Launching kernel with Grid Size: %d, Block Size: %d\n", gridSize, blockSize);

    // 1. 记录开始时间
    cudaEventRecord(start);

    // 2. 启动核函数
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // 3. 记录结束时间
    cudaEventRecord(stop);

    // 4. 等待事件完成 (这一步非常重要，必须等待 GPU 也就是 stop 事件真正发生)
    cudaEventSynchronize(stop);

    // 5. 计算时间差
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel Execution Time: %.3f ms\n", milliseconds);
    printf("Effective Bandwidth: %.3f GB/s\n", 
           (3.0 * bytes) / milliseconds / 1e6); // 读取A, 读取B, 写入C = 3倍内存

    // ... (后面的内存拷贝回 Host 和验证代码保持不变) ...

    // 4. 将结果从 Device 拷贝回 Host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 验证结果 (只检查前5个)
    printf("Result verification (expecting 3.0):\n");
    for(int i = 0; i < 5; i++) {
        printf("Index %d: %f + %f = %f\n", i, h_a[i], h_b[i], h_c[i]);
    }

    // 释放内存
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}