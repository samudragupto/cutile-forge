#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__global__ void cuda_layernorm_kernel(
    const half* __restrict__ X,
    const half* __restrict__ W,
    const half* __restrict__ B,
    half* __restrict__ Y,
    int D,
    float eps
) {
    __shared__ float shared[32];
    int row = blockIdx.x;
    const half* x_row = X + static_cast<size_t>(row) * D;
    half* y_row = Y + static_cast<size_t>(row) * D;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        sum += __half2float(x_row[c]);
    }
    float mean = block_reduce_sum(sum, shared) / static_cast<float>(D);

    float var_sum = 0.0f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float diff = __half2float(x_row[c]) - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrtf(block_reduce_sum(var_sum, shared) / static_cast<float>(D) + eps);

    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float val = (__half2float(x_row[c]) - mean) * inv_std;
        float w = __half2float(W[c]);
        float b = (B != nullptr) ? __half2float(B[c]) : 0.0f;
        y_row[c] = __float2half(val * w + b);
    }
}

__global__ void cuda_rmsnorm_kernel(
    const half* __restrict__ X,
    const half* __restrict__ W,
    half* __restrict__ Y,
    int D,
    float eps
) {
    __shared__ float shared[32];
    int row = blockIdx.x;
    const half* x_row = X + static_cast<size_t>(row) * D;
    half* y_row = Y + static_cast<size_t>(row) * D;

    float sq_sum = 0.0f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float v = __half2float(x_row[c]);
        sq_sum += v * v;
    }
    float inv_rms = rsqrtf(block_reduce_sum(sq_sum, shared) / static_cast<float>(D) + eps);

    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float val = __half2float(x_row[c]) * inv_rms * __half2float(W[c]);
        y_row[c] = __float2half(val);
    }
}

}  // namespace

torch::Tensor cuda_layernorm(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor B,
    double eps
) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    auto X_c = X.contiguous();
    auto W_c = W.contiguous();
    auto B_c = B.contiguous();
    int D = static_cast<int>(X_c.size(-1));
    int rows = static_cast<int>(X_c.numel() / D);
    auto Y = torch::empty_like(X_c);

    int threads = 256;
    cuda_layernorm_kernel<<<rows, threads>>>(
        reinterpret_cast<const half*>(X_c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W_c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_c.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        D,
        static_cast<float>(eps)
    );
    return Y;
}

torch::Tensor cuda_rmsnorm(
    torch::Tensor X,
    torch::Tensor W,
    double eps
) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    auto X_c = X.contiguous();
    auto W_c = W.contiguous();
    int D = static_cast<int>(X_c.size(-1));
    int rows = static_cast<int>(X_c.numel() / D);
    auto Y = torch::empty_like(X_c);

    int threads = 256;
    cuda_rmsnorm_kernel<<<rows, threads>>>(
        reinterpret_cast<const half*>(X_c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W_c.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        D,
        static_cast<float>(eps)
    );
    return Y;
}
