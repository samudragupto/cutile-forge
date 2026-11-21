"""CuTile Python DSL kernels for LLM inference."""

from cutile_forge.cuda_tile_init import ct  # noqa: F401
from cutile_forge.kernels.activations import (
    cutile_gelu,
    cutile_gelu_backward,
    cutile_layernorm_backward,
    cutile_rmsnorm_backward,
    cutile_silu,
    cutile_silu_backward,
    cutile_softmax_backward,
)
from cutile_forge.kernels.attention_gqa import flash_decoding_split_kv, fused_attention_gqa
from cutile_forge.kernels.bmm import cutile_bmm
from cutile_forge.kernels.ffn_fused import (
    fused_ffn_dequant_up_silu_down,
    fused_ffn_swiglu,
    fused_ffn_up_silu_down,
)
from cutile_forge.kernels.gemm_dequant import (
    cutile_gemm_fp8,
    cutile_gemm_fp16,
    fused_gemm_dequant,
    fused_gemm_nvfp4_dequant,
    gemm_fp8_kernel,
    pack_int4_weights,
    unpack_int4_weights_torch,
)
from cutile_forge.kernels.gemm_dequant_silu import fused_gemm_dequant_silu, fused_gemm_fp16_silu
from cutile_forge.kernels.indexing import cutile_conv2d, cutile_gather
from cutile_forge.kernels.layernorm import cutile_layernorm
from cutile_forge.kernels.layernorm_residual import fused_layernorm_residual, fused_rmsnorm_residual
from cutile_forge.kernels.moe_fused import fused_moe_int4_gemm, fused_topk_gating
from cutile_forge.kernels.reductions import cutile_reduce_max, cutile_reduce_sum
from cutile_forge.kernels.rmsnorm import cutile_rmsnorm
from cutile_forge.kernels.rope import apply_rope, precompute_rope_freqs
from cutile_forge.kernels.sampling import sample_tokens
from cutile_forge.kernels.scan import cutile_cumsum
from cutile_forge.kernels.softmax import online_softmax
from cutile_forge.kernels.training import (
    CuTileGELUFn,
    CuTileLayerNormFn,
    CuTileLinearFn,
    CuTileRMSNormFn,
    CuTileSiLUFn,
    cutile_adamw_step,
    cutile_bmm_backward,
    cutile_gelu_autograd,
    cutile_grad_clip_norm,
    cutile_layernorm_autograd,
    cutile_linear_autograd,
    cutile_linear_backward,
    cutile_multi_tensor_adamw,
    cutile_rmsnorm_autograd,
    cutile_silu_autograd,
)
from cutile_forge.kernels.transpose import cutile_transpose
from cutile_forge.kernels.vision import (
    CuTilePatchEmbed,
    cutile_avg_pool2d,
    cutile_conv2d_backward,
    cutile_depthwise_conv2d,
    cutile_max_pool2d,
)

__all__ = [
    "CuTileGELUFn",
    "CuTileLayerNormFn",
    "CuTileLinearFn",
    "CuTilePatchEmbed",
    "CuTileRMSNormFn",
    "CuTileSiLUFn",
    "apply_rope",
    "cutile_adamw_step",
    "cutile_avg_pool2d",
    "cutile_bmm",
    "cutile_bmm_backward",
    "cutile_conv2d",
    "cutile_conv2d_backward",
    "cutile_cumsum",
    "cutile_depthwise_conv2d",
    "cutile_gather",
    "cutile_gelu",
    "cutile_gelu_autograd",
    "cutile_gelu_backward",
    "cutile_gemm_fp8",
    "cutile_gemm_fp16",
    "cutile_grad_clip_norm",
    "cutile_layernorm",
    "cutile_layernorm_autograd",
    "cutile_layernorm_backward",
    "cutile_linear_autograd",
    "cutile_linear_backward",
    "cutile_max_pool2d",
    "cutile_multi_tensor_adamw",
    "cutile_reduce_max",
    "cutile_reduce_sum",
    "cutile_rmsnorm",
    "cutile_rmsnorm_autograd",
    "cutile_rmsnorm_backward",
    "cutile_silu",
    "cutile_silu_autograd",
    "cutile_silu_backward",
    "cutile_softmax_backward",
    "cutile_transpose",
    "flash_decoding_split_kv",
    "fused_attention_gqa",
    "fused_ffn_dequant_up_silu_down",
    "fused_ffn_swiglu",
    "fused_ffn_up_silu_down",
    "fused_gemm_dequant",
    "fused_gemm_dequant_silu",
    "fused_gemm_fp16_silu",
    "fused_gemm_nvfp4_dequant",
    "fused_layernorm_residual",
    "fused_moe_int4_gemm",
    "fused_rmsnorm_residual",
    "fused_topk_gating",
    "gemm_fp8_kernel",
    "online_softmax",
    "pack_int4_weights",
    "precompute_rope_freqs",
    "sample_tokens",
    "unpack_int4_weights_torch",
]
