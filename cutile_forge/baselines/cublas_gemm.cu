#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

torch::Tensor cublas_gemm_fp16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16, "Must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    int M = static_cast<int>(A_c.size(0));
    int K = static_cast<int>(A_c.size(1));
    int N = static_cast<int>(B_c.size(1));

    auto C = torch::empty({M, N}, A.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    // Row-major C[M, N] = A[M, K] * B[K, N] -> col-major C^T[N, M] = B^T[N, K] * A^T[K, M]
    cublasStatus_t status = cublasHgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        reinterpret_cast<const __half*>(B_c.data_ptr<at::Half>()),
        N,
        reinterpret_cast<const __half*>(A_c.data_ptr<at::Half>()),
        K,
        &beta,
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        N
    );
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasHgemm failed");
    return C;
}
