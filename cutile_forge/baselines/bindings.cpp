#include <torch/extension.h>

torch::Tensor cuda_gemm_fp16(torch::Tensor A, torch::Tensor B);
torch::Tensor cuda_gemm_dequant_separate(
    torch::Tensor A,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size
);
torch::Tensor cublas_gemm_fp16(torch::Tensor A, torch::Tensor B);
torch::Tensor cuda_attention_gqa(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
);
torch::Tensor cuda_layernorm(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor B,
    double eps
);
torch::Tensor cuda_rmsnorm(
    torch::Tensor X,
    torch::Tensor W,
    double eps
);
torch::Tensor cuda_silu(torch::Tensor X);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CuTileForge hand-written CUDA C++ and cuBLAS baseline kernels";
    m.def("cuda_gemm_fp16", &cuda_gemm_fp16, "Hand-written 3-stage CUDA tiled GEMM (fp16)");
    m.def(
        "cuda_gemm_dequant_separate",
        &cuda_gemm_dequant_separate,
        "Separate int4 dequant + CUDA GEMM baseline",
        py::arg("A"),
        py::arg("qweight"),
        py::arg("scales"),
        py::arg("zeros"),
        py::arg("group_size") = 128
    );
    m.def("cublas_gemm_fp16", &cublas_gemm_fp16, "cuBLAS HGEMM wrapper (fp16)");
    m.def(
        "cuda_attention_gqa",
        &cuda_attention_gqa,
        "Hand-written CUDA FlashAttention-2 style GQA kernel",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("causal") = true
    );
    m.def(
        "cuda_layernorm",
        &cuda_layernorm,
        "Hand-written CUDA LayerNorm kernel",
        py::arg("X"),
        py::arg("W"),
        py::arg("B"),
        py::arg("eps") = 1e-5
    );
    m.def(
        "cuda_rmsnorm",
        &cuda_rmsnorm,
        "Hand-written CUDA RMSNorm kernel",
        py::arg("X"),
        py::arg("W"),
        py::arg("eps") = 1e-5
    );
    m.def("cuda_silu", &cuda_silu, "Hand-written CUDA SiLU activation kernel");
}
