"""Standalone CuTileForge SDK example — zero dependency on `cutile_forge.engine`."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cutile_forge.sdk import Device, DType, Graph, compile


def main() -> None:
    device_info = Device.auto("auto")
    torch_dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Detected device: {device_info.name} ({device_info.arch}, {device_info.num_sms} SMs)")

    # Build a 4-op graph: x -> matmul(w_up) -> silu -> matmul(w_down) -> rmsnorm
    g = Graph("ffn_rmsnorm_demo")
    x = g.input("x", (64, 128), DType.FP16)
    w_up = g.input("w_up", (128, 256), DType.FP16)
    w_down = g.input("w_down", (256, 128), DType.FP16)
    norm_w = g.input("norm_w", (128,), DType.FP16)

    h = g.matmul(x, w_up)
    a = g.silu(h)
    y = g.matmul(a, w_down)
    out = g.rmsnorm(y, norm_w, eps=1e-5)
    g.output(out)

    compiled = compile(g, target="auto")
    print(compiled.summary())

    t_x = torch.randn((64, 128), device=torch_dev, dtype=torch.float16) * 0.1
    t_wu = torch.randn((128, 256), device=torch_dev, dtype=torch.float16) * 0.1
    t_wd = torch.randn((256, 128), device=torch_dev, dtype=torch.float16) * 0.1
    t_nw = torch.ones((128,), device=torch_dev, dtype=torch.float16)

    result = compiled.run(x=t_x, w_up=t_wu, w_down=t_wd, norm_w=t_nw)
    assert isinstance(result, torch.Tensor)

    # PyTorch reference
    ref_y = torch.matmul(F.silu(torch.matmul(t_x, t_wu)), t_wd).float()
    ref = (ref_y * torch.rsqrt(ref_y.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * t_nw.float()).to(
        torch.float16
    )
    max_err = (result - ref).abs().max().item()
    print(f"Max numerical error vs PyTorch: {max_err:.6f}")
    assert max_err < 1e-2


if __name__ == "__main__":
    main()
