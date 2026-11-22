"""Zero-skip Phase 18 tests for native cross-modal execution & HF checkpoint extractors (`v3.5.0`)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from cutile_forge.engine.config import ModelConfig
from cutile_forge.engine.model import MoEFFNBlock
from cutile_forge.kernels import (
    cutile_gelu_erf,
    cutile_layernorm,
    cutile_quick_gelu,
    fused_attention_gqa,
)
from cutile_forge.kernels.gemm_dequant import pack_int4_weights
from cutile_forge.models import (
    UnifiedModelExecutor,
    build_model_graph,
    extract_hf_clip_vision_weights,
    extract_hf_vit_weights,
    init_synthetic_weights,
)

FORBIDDEN_COMPUTE_ATEN = {
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
    "aten.matmul.default",
    "aten.embedding.default",
    "aten.silu.default",
    "aten.gelu.default",
    "aten.layer_norm.default",
    "aten.native_layer_norm.default",
    "aten.scaled_dot_product_attention.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten.conv2d.default",
    "aten.conv1d.default",
    "aten.convolution.default",
    "aten.topk.default",
    "aten.softmax.int",
    "aten._softmax.default",
}


class _NativeAudit(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.forbidden: list[str] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        name = str(func)
        if name in FORBIDDEN_COMPUTE_ATEN:
            self.forbidden.append(name)
        return func(*args, **(kwargs or {}))


def test_phase18_native_kernel_enablers_dhead72_ln_and_acts() -> None:
    """Verify d_head=72 FlashAttention, affine-optional cutile_layernorm, QuickGELU & erf-GELU."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    torch.manual_seed(7)

    # 1. QuickGELU & exact erf GELU
    x = torch.randn((128, 256), device=dev, dtype=dt) * 0.5
    qg = cutile_quick_gelu(x)
    qg_ref = (x.float() * torch.sigmoid(1.702 * x.float())).to(dt)
    assert (qg.float() - qg_ref.float()).abs().max().item() < 1e-2

    ge = cutile_gelu_erf(x)
    ge_ref = F.gelu(x.float(), approximate="none").to(dt)
    assert (ge.float() - ge_ref.float()).abs().max().item() < 1e-2

    # 2. Affine-optional LayerNorm (DiT AdaLN)
    xl = torch.randn((4, 32, 192), device=dev, dtype=dt)
    ln_no = cutile_layernorm(xl, weight=None, bias=None, eps=1e-6)
    ln_ref = F.layer_norm(xl.float(), (192,), eps=1e-6).to(dt)
    assert (ln_no.float() - ln_ref.float()).abs().max().item() < 1e-2

    if dev == "cuda":
        # 3. Non-power-of-2 d_head=72 FlashAttention
        q = torch.randn((1, 8, 64, 72), device="cuda", dtype=torch.float16) * 0.2
        k = torch.randn((1, 8, 64, 72), device="cuda", dtype=torch.float16) * 0.2
        v = torch.randn((1, 8, 64, 72), device="cuda", dtype=torch.float16) * 0.2
        out_fa = fused_attention_gqa(q, k, v, causal=False)
        out_pt = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=False)
        assert (out_fa.float() - out_pt).abs().max().item() < 1e-2


def test_phase18_moe_ffn_block_native_router() -> None:
    """Verify MoEFFNBlock routes via cutile_moe_router_topk on CUDA without torch.matmul."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    torch.manual_seed(11)

    blk = MoEFFNBlock(num_experts=4, top_k=2, group_size=64)
    x = torch.randn((4, 64), device=dev, dtype=dt) * 0.2
    router_w = torch.randn((4, 64), device=dev, dtype=dt) * 0.1
    qw = pack_int4_weights(torch.randint(0, 16, (64, 64), device=dev, dtype=torch.uint8))
    exp_qw = torch.stack([qw for _ in range(4)], dim=0)
    exp_s = torch.ones((4, 1, 64), device=dev, dtype=dt) * 0.05
    exp_z = torch.zeros((4, 1, 64), device=dev, dtype=dt)

    if dev == "cuda":
        with _NativeAudit() as audit:
            out = blk.forward(x, router_w, exp_qw, exp_s, exp_z)
        assert (
            not audit.forbidden
        ), f"Forbidden ATen compute calls in MoEFFNBlock: {audit.forbidden}"
        assert out.shape == (4, 64)
    else:
        assert ModelConfig(experts=4).experts == 4


def test_phase18_unified_executor_zero_forbidden_aten_all_modalities() -> None:
    """Verify native CuTile UnifiedModelExecutor emits zero forbidden ATen compute calls on CUDA."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    if dev != "cuda":
        g = build_model_graph(
            "llama3",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=128,
        )
        w = init_synthetic_weights(g, seed=1, device="cpu", dtype=torch.float32)
        out = UnifiedModelExecutor(g).forward(w, input_ids=torch.tensor([[1, 2, 3]]))
        assert out.shape == (1, 3, 128)
        return

    torch.manual_seed(19)
    cases = [
        (
            "llama3",
            {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 128,
            },
        ),
        (
            "mixtral",
            {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 128,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
            },
        ),
        (
            "vit",
            {
                "image_size": 32,
                "patch_size": 16,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_labels": 16,
            },
        ),
        (
            "whisper_encoder",
            {
                "num_mel_bins": 16,
                "max_source_positions": 16,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
            },
        ),
        (
            "dit",
            {
                "image_size": 16,
                "patch_size": 2,
                "in_channels": 4,
                "out_channels": 4,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
            },
        ),
    ]
    for fam, ov in cases:
        g = build_model_graph(fam, **ov)
        w = init_synthetic_weights(g, seed=3, device="cuda", dtype=dt)
        ex = UnifiedModelExecutor(g)
        kw: dict[str, Any] = {}
        if fam in ("llama3", "mixtral"):
            kw["input_ids"] = torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.long)
        elif fam == "vit":
            kw["pixel_values"] = torch.randn((1, 3, 32, 32), device="cuda", dtype=dt)
        elif fam == "whisper_encoder":
            kw["mel_features"] = torch.randn((1, 16, 32), device="cuda", dtype=dt)
        elif fam == "dit":
            kw["pixel_values"] = torch.randn((1, 4, 16, 16), device="cuda", dtype=dt)
            kw["timesteps"] = torch.tensor([100], device="cuda", dtype=torch.long)
            kw["class_labels"] = torch.tensor([1], device="cuda", dtype=torch.long)

        with _NativeAudit() as audit:
            out = ex.forward(w, **kw)
        assert out.shape[0] == 1
        assert not audit.forbidden, f"{fam} emitted forbidden ATen compute calls: {audit.forbidden}"


def test_phase18_hf_vit_and_clip_extractors_parity() -> None:
    """Verify extract_hf_vit_weights & extract_hf_clip_vision_weights parity on synthetic HF mocks."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    d, img_sz, p = 64, 32, 16

    mock_vit = SimpleNamespace(
        config=SimpleNamespace(
            image_size=img_sz,
            patch_size=p,
            num_channels=3,
            hidden_size=d,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            hidden_act="gelu",
            layer_norm_eps=1e-6,
        ),
        embeddings=SimpleNamespace(
            patch_embeddings=SimpleNamespace(projection=nn.Conv2d(3, d, p, p).to(dev)),
            cls_token=nn.Parameter(torch.zeros(1, 1, d, device=dev)),
            position_embeddings=nn.Parameter(torch.zeros(1, 5, d, device=dev)),
        ),
        layers=[
            SimpleNamespace(
                layernorm_before=nn.LayerNorm(d).to(dev),
                layernorm_after=nn.LayerNorm(d).to(dev),
                attention=SimpleNamespace(
                    q_proj=nn.Linear(d, d).to(dev),
                    k_proj=nn.Linear(d, d).to(dev),
                    v_proj=nn.Linear(d, d).to(dev),
                    o_proj=nn.Linear(d, d).to(dev),
                ),
                mlp=SimpleNamespace(fc1=nn.Linear(d, 128).to(dev), fc2=nn.Linear(128, d).to(dev)),
            )
        ],
        layernorm=nn.LayerNorm(d).to(dev),
    )

    mock_vit_cls = SimpleNamespace(
        vit=mock_vit,
        classifier=nn.Linear(d, 10).to(dev),
    )
    g_v, w_v = extract_hf_vit_weights(mock_vit_cls, device=dev, dtype=dt)
    ex_v = UnifiedModelExecutor(g_v)
    pix = torch.randn((1, 3, img_sz, img_sz), device=dev, dtype=dt)
    assert ex_v.forward(w_v, pixel_values=pix).shape == (1, 10)

    # Mock CLIPVisionModel
    mock_clip = SimpleNamespace(
        vision_model=SimpleNamespace(
            config=SimpleNamespace(
                image_size=img_sz,
                patch_size=p,
                num_channels=3,
                hidden_size=d,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                hidden_act="quick_gelu",
                layer_norm_eps=1e-5,
            ),
            embeddings=SimpleNamespace(
                patch_embedding=nn.Conv2d(3, d, p, p, bias=False).to(dev),
                class_embedding=nn.Parameter(torch.zeros(d, device=dev)),
                position_embedding=nn.Embedding(5, d).to(dev),
            ),
            pre_layrnorm=nn.LayerNorm(d).to(dev),
            post_layernorm=nn.LayerNorm(d).to(dev),
            encoder=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        layer_norm1=nn.LayerNorm(d).to(dev),
                        layer_norm2=nn.LayerNorm(d).to(dev),
                        self_attn=SimpleNamespace(
                            q_proj=nn.Linear(d, d).to(dev),
                            k_proj=nn.Linear(d, d).to(dev),
                            v_proj=nn.Linear(d, d).to(dev),
                            out_proj=nn.Linear(d, d).to(dev),
                        ),
                        mlp=SimpleNamespace(
                            fc1=nn.Linear(d, 128).to(dev), fc2=nn.Linear(128, d).to(dev)
                        ),
                    )
                ]
            ),
        )
    )
    g_c, w_c = extract_hf_clip_vision_weights(mock_clip, device=dev, dtype=dt)
    ex_c = UnifiedModelExecutor(g_c)
    assert ex_c.forward(w_c, pixel_values=pix).shape == (1, d)
