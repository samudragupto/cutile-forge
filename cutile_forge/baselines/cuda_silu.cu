#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void cuda_silu_kernel(
    const half* __restrict__ X,
    half* __restrict__ Y,
    int64_t numel
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float v = __half2float(X[idx]);
    float s = v / (1.0f + expf(-v));
    Y[idx] = __float2half(s);
}

}  // namespace

torch::Tensor cuda_silu(torch::Tensor X) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    auto X_c = X.contiguous();
    auto Y = torch::empty_like(X_c);
    int64_t numel = X_c.numel();
    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    cuda_silu_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half*>(X_c.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        numel
    );
    return Y;
}
