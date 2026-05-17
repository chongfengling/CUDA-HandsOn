#pragma once

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_shared(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_register(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_vectorized(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_double_buffered(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_tensor_core(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_async(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_ultimate(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_zhihu(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_cublas(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_cutlass(const float* A, const float* B, float* C, int M, int N, int K);
