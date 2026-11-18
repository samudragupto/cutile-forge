#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Forward declaration from cuda_gemm.cu
torch::Tensor cuda_gemm_fp16(torch::Tensor A, torch::Tensor B);

namespace {

__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ qw,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    half* __restrict__ w_out,
    int K_half,
    int N,
    int group_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K_half * N;
    if (idx >= total) return;

    int kh = idx / N;
    int col = idx % N;
    uint8_t packed = qw[idx];

    float lo = static_cast<float>(packed & 0x0F);
    float hi = static_cast<float>((packed >> 4) & 0x0F);

    int k0 = kh * 2;
    int k1 = k0 + 1;
    int g0 = k0 / group_size;
    int g1 = k1 / group_size;

    float s0 = __half2float(scales[g0 * N + col]);
    float z0 = __half2float(zeros[g0 * N + col]);
    float s1 = __half2float(scales[g1 * N + col]);
    float z1 = __half2float(zeros[g1 * N + col]);

    w_out[k0 * N + col] = __float2half((lo - z0) * s0);
    w_out[k1 * N + col] = __float2half((hi - z1) * s1);
}

}  // namespace

torch::Tensor cuda_gemm_dequant_separate(
    torch::Tensor A,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size
) {
    TORCH_CHECK(A.is_cuda() && qweight.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(qweight.dtype() == torch::kUInt8, "qweight must be uint8 (K//2, N)");
    auto qw_c = qweight.contiguous();
    auto s_c = scales.contiguous();
    auto z_c = zeros.contiguous();

    int K_half = static_cast<int>(qw_c.size(0));
    int N = static_cast<int>(qw_c.size(1));
    int K = K_half * 2;

    auto w_fp16 = torch::empty({K, N}, A.options());
    int total = K_half * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    dequant_int4_kernel<<<blocks, threads>>>(
        qw_c.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(s_c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(z_c.data_ptr<at::Half>()),
        reinterpret_cast<half*>(w_fp16.data_ptr<at::Half>()),
        K_half,
        N,
        group_size
    );
    return cuda_gemm_fp16(A, w_fp16);
}
