#pragma once

void launch_gemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_shared(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_cublas(const float* A, const float* B, float* C, int M, int N, int K);
void launch_gemm_cutlass(const float* A, const float* B, float* C, int M, int N, int K);
