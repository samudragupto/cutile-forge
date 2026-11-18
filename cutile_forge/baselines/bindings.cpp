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
}
