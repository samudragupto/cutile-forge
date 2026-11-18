#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>

namespace {

constexpr int HEAD_DIM_MAX = 128;
constexpr int BK_TILE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Hand-written FlashAttention-2 style online-softmax GQA kernel (1 warp per query row)
__global__ void cuda_flash_attention_gqa_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float sm_scale,
    int S_q,
    int S_kv,
    int kv_offset,
    int D,
    int gqa_ratio,
    int is_causal
) {
    const int bh_q = blockIdx.y;
    const int q_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (q_idx >= S_q) return;

    const int lane = threadIdx.x;  // 0..31
    const int bh_kv = bh_q / gqa_ratio;
    const int q_pos = q_idx + kv_offset;

    const half* q_ptr = Q + (static_cast<size_t>(bh_q) * S_q + q_idx) * D;
    const half* k_base = K + static_cast<size_t>(bh_kv) * S_kv * D;
    const half* v_base = V + static_cast<size_t>(bh_kv) * S_kv * D;
    half* o_ptr = O + (static_cast<size_t>(bh_q) * S_q + q_idx) * D;

    // Each lane holds up to 4 elements of Q and O (for D <= 128)
    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        int d = lane + i * 32;
        if (d < D) {
            q_reg[i] = __half2float(q_ptr[d]);
        }
    }

    float m_i = -1e9f;
    float l_i = 0.0f;

    int max_k = is_causal ? min(S_kv, q_pos + 1) : S_kv;
    for (int k_idx = 0; k_idx < max_k; ++k_idx) {
        const half* k_row = k_base + static_cast<size_t>(k_idx) * D;
        const half* v_row = v_base + static_cast<size_t>(k_idx) * D;

        float dot = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int d = lane + i * 32;
            if (d < D) {
                dot += q_reg[i] * __half2float(k_row[d]);
            }
        }
        dot = warp_reduce_sum(dot) * sm_scale;

        float m_new = fmaxf(m_i, dot);
        float alpha = expf(m_i - m_new);
        float p = expf(dot - m_new);
        l_i = l_i * alpha + p;
        m_i = m_new;

        for (int i = 0; i < 4; ++i) {
            int d = lane + i * 32;
            if (d < D) {
                acc_reg[i] = acc_reg[i] * alpha + p * __half2float(v_row[d]);
            }
        }
    }

    float inv_l = 1.0f / fmaxf(l_i, 1e-6f);
    for (int i = 0; i < 4; ++i) {
        int d = lane + i * 32;
        if (d < D) {
            o_ptr[d] = __float2half(acc_reg[i] * inv_l);
        }
    }
}

}  // namespace

torch::Tensor cuda_attention_gqa(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Must be 4D");

    auto Q_c = Q.contiguous();
    auto K_c = K.contiguous();
    auto V_c = V.contiguous();

    int B = static_cast<int>(Q_c.size(0));
    int H_q = static_cast<int>(Q_c.size(1));
    int S_q = static_cast<int>(Q_c.size(2));
    int D = static_cast<int>(Q_c.size(3));
    int H_kv = static_cast<int>(K_c.size(1));
    int S_kv = static_cast<int>(K_c.size(2));
    TORCH_CHECK(D <= HEAD_DIM_MAX, "Head dim > 128 not supported");
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv");

    int gqa_ratio = H_q / H_kv;
    int kv_offset = S_kv - S_q;
    float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto O = torch::empty_like(Q_c);
    dim3 block(32, 4);  // 4 warps per block -> 4 query rows per block
    dim3 grid((S_q + 3) / 4, B * H_q);

    cuda_flash_attention_gqa_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(Q_c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K_c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V_c.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        sm_scale,
        S_q,
        S_kv,
        kv_offset,
        D,
        gqa_ratio,
        causal ? 1 : 0
    );
    return O;
}
