#include "add.h"
#include "cuda_utils.h"

// 3️⃣ Multi Block: Standard grid-stride loop (Optimized)
__global__ void vector_add_kernel_multi_block(const float *a, const float *b, float *out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

void launch_vector_add_multi_block(const float* a, const float* b, float* c, int n) {
    int blockSize = 256;
    int numBlocks = ceil_div(n, blockSize);
    vector_add_kernel_multi_block<<<numBlocks, blockSize>>>(a, b, c, n);
    LAST_KERNEL_CHECK();
}
