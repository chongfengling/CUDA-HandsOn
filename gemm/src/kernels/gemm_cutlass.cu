#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <stdexcept>
#include "kernels.h"

void launch_gemm_cutlass(const float* A, const float* B, float* C, int M, int N, int K) {
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor
    >;

    Gemm gemm_op;
    float alpha = 1.0f;
    float beta = 0.0f;
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // CUTLASS API requires non-const pointers for arguments structure layout even for inputs,
    // hence the const_cast. The inputs will not be mutated.
    typename Gemm::Arguments arguments{
        problem_size,
        {const_cast<float*>(A), K},
        {const_cast<float*>(B), N},
        {C, N},
        {C, N},
        {alpha, beta}
    };

    cutlass::Status status = gemm_op(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS SGEMM failed");
    }
}
