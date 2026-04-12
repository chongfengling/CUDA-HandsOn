

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        float sum = 0.0f;
        for (int i = 0; i< K; ++i){
            sum += A[row * K + i] * B[N * i + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K){
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x,
              (K + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
}