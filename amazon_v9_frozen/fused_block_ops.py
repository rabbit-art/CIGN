from __future__ import annotations

import itertools
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None
_EXT_ERROR = None
_DROPOUT_COUNTER = itertools.count()


def _extension_name() -> str:
    return "graph_clifford_fused_block_ops_a3v8_v1"


def ensure_fused_block_ops_loaded(verbose: bool = False):
    """Build/load the local A3-v8 block-fusion CUDA extension once."""
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError("A3-v8 fused block-ops extension previously failed to load") from _EXT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("A3-v8 fused block ops require CUDA")

    root = Path(__file__).resolve().parent
    src_dir = root / "fused_block_ops_ext"
    build_dir = root / ".fused_block_ops_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

    try:
        _EXT = load(
            name=_extension_name(),
            sources=[
                str(src_dir / "fused_block_ops.cpp"),
                str(src_dir / "fused_block_ops_cuda.cu"),
            ],
            extra_cflags=["-O3"],
            # No --use_fast_math: keep this path as close as practical to FP32.
            extra_cuda_cflags=["-O3", "-std=c++17"],
            build_directory=str(build_dir),
            verbose=verbose,
        )
        return _EXT
    except Exception as exc:
        _EXT_ERROR = exc
        raise


def _next_seed(tag: int, dropout_p: float) -> int:
    if float(dropout_p) <= 0.0:
        return 0
    base_seed = int(torch.initial_seed()) & ((1 << 63) - 1)
    call_id = next(_DROPOUT_COUNTER)
    # Different tags decorrelate the H-input and residual-path dropout streams.
    return int(
        (
            base_seed
            + 0x9E3779B97F4A7C15 * (call_id + 1)
            + 0xD1B54A32D192ED03 * int(tag)
        )
        & ((1 << 63) - 1)
    )


class _DropoutLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, dropout_p: float, eps: float, seed: int):
        ext = ensure_fused_block_ops_loaded(False)
        y, mean, rstd, mask = ext.dropout_layernorm_forward(
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            float(dropout_p),
            float(eps),
            int(seed),
        )
        ctx.save_for_backward(x, weight, mean, rstd, mask)
        ctx.dropout_p = float(dropout_p)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        ext = ensure_fused_block_ops_loaded(False)
        x, weight, mean, rstd, mask = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = ext.dropout_layernorm_backward(
            grad_y.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            mean.contiguous(),
            rstd.contiguous(),
            mask.contiguous(),
            float(ctx.dropout_p),
        )
        return grad_x, grad_weight, grad_bias, None, None, None


class _AddLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, weight, bias, eps: float):
        ext = ensure_fused_block_ops_loaded(False)
        y, mean, rstd = ext.add_layernorm_forward(
            a.contiguous(),
            b.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            float(eps),
        )
        ctx.save_for_backward(a, b, weight, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        ext = ensure_fused_block_ops_loaded(False)
        a, b, weight, mean, rstd = ctx.saved_tensors
        grad_z, grad_weight, grad_bias = ext.add_layernorm_backward(
            grad_y.contiguous(),
            a.contiguous(),
            b.contiguous(),
            weight.contiguous(),
            mean.contiguous(),
            rstd.contiguous(),
        )
        # z = a + b, so both branches receive the same gradient tensor.
        return grad_z, grad_z, grad_weight, grad_bias, None


class _DropoutGammaResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, g, gamma, dropout_p: float, seed: int):
        ext = ensure_fused_block_ops_loaded(False)
        y, mask = ext.dropout_gamma_residual_forward(
            h.contiguous(),
            g.contiguous(),
            gamma.contiguous(),
            float(dropout_p),
            int(seed),
        )
        ctx.save_for_backward(g, gamma, mask)
        ctx.dropout_p = float(dropout_p)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        ext = ensure_fused_block_ops_loaded(False)
        g, gamma, mask = ctx.saved_tensors
        grad_g, grad_gamma = ext.dropout_gamma_residual_backward(
            grad_y.contiguous(),
            g.contiguous(),
            gamma.contiguous(),
            mask.contiguous(),
            float(ctx.dropout_p),
        )
        # y = h + ..., therefore dy/dh = 1 exactly. Returning grad_y directly
        # avoids another [N,D] copy/kernel.
        return grad_y, grad_g, grad_gamma, None, None


def fused_dropout_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dropout_p: float,
    eps: float,
) -> torch.Tensor:
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise TypeError("A3-v8 fused_dropout_layernorm requires CUDA float32")
    if x.dim() != 2 or x.size(1) != 256:
        raise ValueError(f"A3-v8 v1 is specialized for [N,256], got {tuple(x.shape)}")
    seed = _next_seed(tag=1, dropout_p=dropout_p)
    return _DropoutLayerNormFn.apply(x, weight, bias, float(dropout_p), float(eps), int(seed))


def fused_add_layernorm(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if a.device.type != "cuda" or a.dtype != torch.float32:
        raise TypeError("A3-v8 fused_add_layernorm requires CUDA float32")
    if a.shape != b.shape or a.dim() != 2 or a.size(1) != 256:
        raise ValueError(f"A3-v8 v1 expects matching [N,256] tensors, got {tuple(a.shape)}, {tuple(b.shape)}")
    return _AddLayerNormFn.apply(a, b, weight, bias, float(eps))


def fused_dropout_gamma_residual(
    h: torch.Tensor,
    g: torch.Tensor,
    gamma: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    if h.device.type != "cuda" or h.dtype != torch.float32:
        raise TypeError("A3-v8 fused_dropout_gamma_residual requires CUDA float32")
    if h.shape != g.shape or h.dim() != 2 or h.size(1) != 256:
        raise ValueError(f"A3-v8 v1 expects matching [N,256] tensors, got {tuple(h.shape)}, {tuple(g.shape)}")
    if gamma.dim() != 1 or gamma.numel() != 256:
        raise ValueError("A3-v8 v1 currently supports vector gamma with shape [256]")
    seed = _next_seed(tag=2, dropout_p=dropout_p)
    return _DropoutGammaResidualFn.apply(h, g, gamma, float(dropout_p), int(seed))
