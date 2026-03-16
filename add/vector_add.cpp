#include <stdio.h>
#include <stdlib.h>

#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    vector_add(out, a, b, N);

    // Print first 5 results to verify
    for (int i = 0; i < 5; i++) {
        printf("out[%d] = %.2f\n", i, out[i]);
    }

    // Free memory
    free(a);
    free(b);
    free(out);
}

// To compile this code, use the following command:
// gcc vector_add.cpp -o cpu_vector_add