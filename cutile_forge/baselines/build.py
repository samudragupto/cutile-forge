"""Build script for compiling CUDA C++ baseline kernels via pybind11 + nvcc."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _find_vcvars64() -> Path | None:
    """Locate Visual Studio vcvars64.bat on Windows."""
    if platform.system() != "Windows":
        return None
    if shutil.which("cl.exe"):
        return None  # Already on PATH

    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.exists():
        try:
            out = subprocess.check_output(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                text=True,
            ).strip()
            if out:
                cand = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if cand.exists():
                    return cand
        except subprocess.SubprocessError:
            pass

    for base in [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        Path(r"C:\Program Files\Microsoft Visual Studio"),
    ]:
        if base.exists():
            matches = sorted(base.rglob("vcvars64.bat"))
            if matches:
                return matches[-1]
    return None


def _inject_msvc_env() -> None:
    """Load MSVC environment variables into os.environ when cl.exe is missing on Windows."""
    vcvars = _find_vcvars64()
    if vcvars is None:
        return
    cmd = f'"{vcvars}" >nul 2>&1 && set'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val


def build_baselines(verbose: bool = True) -> Path:
    """Compile all baseline CUDA sources into a Python extension inside baselines/."""
    _inject_msvc_env()

    from torch.utils.cpp_extension import load

    baselines_dir = Path(__file__).resolve().parent
    build_dir = baselines_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    sources: List[str] = [str(baselines_dir / "bindings.cpp")]
    for cu_name in [
        "cuda_gemm.cu",
        "cuda_gemm_dequant.cu",
        "cublas_gemm.cu",
        "cuda_attention.cu",
        "cuda_layernorm.cu",
        "cuda_silu.cu",
    ]:
        cu_path = baselines_dir / cu_name
        if cu_path.exists():
            sources.append(str(cu_path))

    # Target sm_89 (RTX 4060 Laptop) cleanly; allow override via TORCH_CUDA_ARCH_LIST
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    if platform.system() == "Windows":
        extra_cflags = ["/O2", "/std:c++20", "/Zc:preprocessor"]
        extra_cuda_cflags.extend(["-Xcompiler=/std:c++20", "-Xcompiler=/Zc:preprocessor"])
    else:
        extra_cflags = ["-O3", "-std=c++20"]

    module = load(
        name="_cutile_baselines",
        sources=sources,
        build_directory=str(build_dir),
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=["cublas.lib"] if platform.system() == "Windows" else ["-lcublas"],
        verbose=verbose,
    )
    compiled_path = Path(module.__file__)
    target_path = baselines_dir / compiled_path.name
    if compiled_path.resolve() != target_path.resolve():
        shutil.copy2(compiled_path, target_path)
    if verbose:
        print(f"Built CUDA baselines extension: {target_path}")
    return target_path


if __name__ == "__main__":
    try:
        build_baselines(verbose=True)
    except Exception as exc:
        print(f"Failed to build CUDA baselines: {exc}", file=sys.stderr)
        sys.exit(1)
