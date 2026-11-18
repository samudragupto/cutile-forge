#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int STAGES = 3;
constexpr int THREADS = 128;  // 4 warps (2x2 warp grid of 32x32 output each)

__device__ __forceinline__ unsigned int smem_u32addr(const void* ptr) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    unsigned int addr = smem_u32addr(smem_ptr);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(addr), "l"(gmem_ptr), "r"(src_bytes)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// 3-stage cp.async + WMMA tiled FP16 GEMM: C[M, N] = A[M, K] @ B[K, N]
__global__ void cuda_gemm_3stage_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M,
    int N,
    int K
) {
    // Pad stride by 8 halfs to avoid shared memory bank conflicts
    constexpr int SA_STRIDE = BK + 8;
    constexpr int SB_STRIDE = BN + 8;

    __shared__ half smem_A[STAGES][BM * SA_STRIDE];
    __shared__ half smem_B[STAGES][BK * SB_STRIDE];
    __shared__ float smem_C[BM * BN];

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = (warp_id / 2) * 32;  // 0 or 32
    const int warp_n = (warp_id % 2) * 32;  // 0 or 32

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const int num_k_tiles = (K + BK - 1) / BK;

    auto load_tile = [&](int k_tile, int stage) {
        const int k_base = k_tile * BK;
        // Load A tile (BM x BK = 64 x 32 = 2048 halfs = 256 x 8-half vectors)
        #pragma unroll
        for (int idx = tid; idx < (BM * BK) / 8; idx += THREADS) {
            int r = idx / (BK / 8);
            int c = (idx % (BK / 8)) * 8;
            int gr = block_m + r;
            int gc = k_base + c;
            int bytes = (gr < M && gc + 7 < K) ? 16 : 0;
            const half* src = (bytes > 0) ? (A + gr * K + gc) : A;
            cp_async_16(&smem_A[stage][r * SA_STRIDE + c], src, bytes);
            if (bytes == 0 && gr < M && gc < K) {
                // Tail handling if K not multiple of 8
                for (int t = 0; t < 8; ++t) {
                    smem_A[stage][r * SA_STRIDE + c + t] = (gc + t < K) ? A[gr * K + gc + t] : __float2half(0.0f);
                }
            }
        }
        // Load B tile (BK x BN = 32 x 64 = 2048 halfs = 256 x 8-half vectors)
        #pragma unroll
        for (int idx = tid; idx < (BK * BN) / 8; idx += THREADS) {
            int r = idx / (BN / 8);
            int c = (idx % (BN / 8)) * 8;
            int gr = k_base + r;
            int gc = block_n + c;
            int bytes = (gr < K && gc + 7 < N) ? 16 : 0;
            const half* src = (bytes > 0) ? (B + gr * N + gc) : B;
            cp_async_16(&smem_B[stage][r * SB_STRIDE + c], src, bytes);
            if (bytes == 0 && gr < K && gc < N) {
                for (int t = 0; t < 8; ++t) {
                    smem_B[stage][r * SB_STRIDE + c + t] = (gc + t < N) ? B[gr * N + gc + t] : __float2half(0.0f);
                }
            }
        }
        cp_async_commit();
    };

    // Prime pipeline stages 0 and 1
    int preload = (num_k_tiles < STAGES - 1) ? num_k_tiles : (STAGES - 1);
    for (int s = 0; s < preload; ++s) {
        load_tile(s, s);
    }

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int next_tile = k_tile + (STAGES - 1);
        if (next_tile < num_k_tiles) {
            load_tile(next_tile, next_tile % STAGES);
        }

        // Wait until at most (STAGES - 2) or fewer groups remain in flight
        if (k_tile + STAGES - 1 < num_k_tiles) {
            cp_async_wait<STAGES - 2>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        int stage = k_tile % STAGES;
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(
                    frag_a[i],
                    &smem_A[stage][(warp_m + i * 16) * SA_STRIDE + kk],
                    SA_STRIDE
                );
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(
                    frag_b[j],
                    &smem_B[stage][kk * SB_STRIDE + (warp_n + j * 16)],
                    SB_STRIDE
                );
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(acc[i][j], frag_a[i], frag_b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Store warp accumulators to shared memory staging buffer, then bounds-checked global store
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(
                &smem_C[(warp_m + i * 16) * BN + (warp_n + j * 16)],
                acc[i][j],
                BN,
                wmma::mem_row_major
            );
        }
    }
    __syncthreads();

    for (int idx = tid; idx < BM * BN; idx += THREADS) {
        int r = idx / BN;
        int c = idx % BN;
        int gr = block_m + r;
        int gc = block_n + c;
        if (gr < M && gc < N) {
            C[gr * N + gc] = __float2half(smem_C[idx]);
        }
    }
}

}  // namespace

torch::Tensor cuda_gemm_fp16(torch::Tensor A, torch::Tensor B) {
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
    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cuda_gemm_3stage_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A_c.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_c.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );
    return C;
}
