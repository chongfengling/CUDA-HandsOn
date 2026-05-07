#include "add.h"
#include "cuda_utils.h"

// 1️⃣ Naive: Single thread does everything (Very slow!)
__global__ void vector_add_kernel_naive(const float *a, const float *b, float *out, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void launch_vector_add_naive(const float* a, const float* b, float* c, int n) {
    vector_add_kernel_naive<<<1, 1>>>(a, b, c, n);
    LAST_KERNEL_CHECK();
}
