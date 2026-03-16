#include <stdio.h>

__global__ void kernel_hello(){
    printf("Hello World from GPU!\n");
}

int main(){
    // Launch kernel with a single thread
    kernel_hello<<<1,1>>>();
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    return 0;
}
// To compile this code, use the following command:
// nvcc kernel_hello.cu -o hello
// To run the compiled program, use:
// ./hello