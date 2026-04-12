#define TILE 16 //same shape of blocks

__global__ void gemm_shared_kernel(
    int M, int N, int K,
    const float* A,
    const float* B,
    float* C)
{
    // shared memory tile
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y; // TILE=blockDIm(x,y)  和构造相关 one tile one block
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // 遍历 K 维度的 tile
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // row major, from left to right
        int tiledA = t * TILE + threadIdx.x;
        // from top to bottom
        int tiledB = t * TILE + threadIdx.y;

        // load A tile
        if (row < M && tiledA < K)
            //row base, As[ty][tx] to avoid bank conflict
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledA]; // t * Tile + row * K + threadIdx.x
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // load B tile
        if (col < N && tiledB < K)
            Bs[threadIdx.y][threadIdx.x] = B[tiledB * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // compute partial result
        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void launch_gemm_shared(const float* A, const float* B, float* C, int M, int N, int K){
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x,
              (K + block.y - 1) / block.y);
    gemm_shared_kernel<<<grid, block>>>(A, B, C, M, N, K);
}