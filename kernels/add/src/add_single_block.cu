#include "add.h"
#include "cuda_utils.h"

// 2️⃣ Single Block: Multiple threads, but limited to 1024 (or block size)
__global__ void vector_add_kernel_single_block(const float *a, const float *b, float *out, int n) {
    int idx = threadIdx.x;
    int stride = blockDim.x;
    for(int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

void launch_vector_add_single_block(const float* a, const float* b, float* c, int n) {
    int blockSize = 256;
    vector_add_kernel_single_block<<<1, blockSize>>>(a, b, c, n);
    LAST_KERNEL_CHECK();
}
