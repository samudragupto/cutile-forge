cmake_minimum_required(VERSION 3.24)
project(cutile_forge_baselines LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(CUDAToolkit REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
find_package(Torch REQUIRED)

pybind11_add_module(_cutile_baselines
  cutile_forge/baselines/bindings.cpp
  cutile_forge/baselines/cuda_gemm.cu
  cutile_forge/baselines/cuda_gemm_dequant.cu
  cutile_forge/baselines/cublas_gemm.cu
  cutile_forge/baselines/cuda_attention.cu
  cutile_forge/baselines/cuda_layernorm.cu
  cutile_forge/baselines/cuda_silu.cu
)

target_link_libraries(_cutile_baselines PRIVATE ${TORCH_LIBRARIES} CUDA::cublas)

