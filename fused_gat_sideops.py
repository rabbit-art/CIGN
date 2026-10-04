from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None
_EXT_ERROR = None


def _extension_name() -> str:
    return "graph_clifford_fused_gat_sideops_a3v3_v1"


def ensure_fused_gat_sideops_loaded(verbose: bool = False):
    """Build/load A3-v3 side-op CUDA kernels once, outside epoch timing."""
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError("A3-v3 fused GAT side-op extension previously failed to load") from _EXT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("A3-v3 fused GAT side ops require CUDA")

    root = Path(__file__).resolve().parent
    src_dir = root / "fused_gat_sideops_ext"
    build_dir = root / ".fused_gat_sideops_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    try:
        _EXT = load(
            name=_extension_name(),
            sources=[
                str(src_dir / "fused_gat_sideops.cpp"),
                str(src_dir / "fused_gat_sideops_cuda.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir),
            verbose=verbose,
        )
        return _EXT
    except Exception as exc:
        _EXT_ERROR = exc
        raise


class _AttentionLogitsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, att_src, att_dst):
        ext = ensure_fused_gat_sideops_loaded(False)
        x = x.contiguous()
        att_src = att_src.contiguous()
        att_dst = att_dst.contiguous()
        alpha_src, alpha_dst = ext.attention_forward(x, att_src, att_dst)
        ctx.save_for_backward(x, att_src, att_dst)
        return alpha_src, alpha_dst

    @staticmethod
    def backward(ctx, grad_src, grad_dst):
        ext = ensure_fused_gat_sideops_loaded(False)
        x, att_src, att_dst = ctx.saved_tensors
        grad_src = grad_src.contiguous()
        grad_dst = grad_dst.contiguous()
        grad_x, grad_att_src, grad_att_dst = ext.attention_backward(
            grad_src, grad_dst, x, att_src, att_dst
        )
        return grad_x, grad_att_src, grad_att_dst


class _HeadMeanBiasSiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, bias):
        ext = ensure_fused_gat_sideops_loaded(False)
        out = out.contiguous()
        bias = bias.contiguous()
        y, pre = ext.head_silu_forward(out, bias)
        ctx.save_for_backward(pre)
        ctx.heads = int(out.size(1))
        return y

    @staticmethod
    def backward(ctx, grad_y):
        ext = ensure_fused_gat_sideops_loaded(False)
        (pre,) = ctx.saved_tensors
        grad_out, grad_bias = ext.head_silu_backward(
            grad_y.contiguous(), pre, ctx.heads
        )
        return grad_out, grad_bias


def fused_attention_logits(x: torch.Tensor, att_src: torch.Tensor, att_dst: torch.Tensor):
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise TypeError("A3-v3 fused attention logits require CUDA float32 tensors")
    return _AttentionLogitsFn.apply(x, att_src, att_dst)


def fused_head_mean_bias_silu(out: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if bias is None:
        raise ValueError("A3-v3 fused head reduction currently requires GAT bias=True")
    if out.dtype != torch.float32 or out.device.type != "cuda":
        raise TypeError("A3-v3 fused head reduction requires CUDA float32 tensors")
    return _HeadMeanBiasSiLUFn.apply(out, bias)
