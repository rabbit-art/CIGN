from __future__ import annotations

import itertools
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None
_EXT_ERROR = None
_CALL_COUNTER = itertools.count()


def _extension_name() -> str:
    return "graph_clifford_collapsed_gat_a3v6_v1"


def ensure_collapsed_gat_loaded(verbose: bool = False):
    """Build/load A3-v6 collapsed-head GAT CUDA kernels once.

    The build is intentionally local to this experiment directory and happens
    before model creation / epoch timing. No site-packages are modified.
    """
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError("A3-v6 collapsed GAT extension previously failed to load") from _EXT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("A3-v6 collapsed GAT requires CUDA")

    root = Path(__file__).resolve().parent
    src_dir = root / "collapsed_gat_ext"
    build_dir = root / ".collapsed_gat_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

    try:
        _EXT = load(
            name=_extension_name(),
            sources=[
                str(src_dir / "collapsed_gat.cpp"),
                str(src_dir / "collapsed_gat_cuda.cu"),
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


class _CollapsedGATFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        attn_row,
        attn_col,
        rowptr,
        col,
        colptr,
        row,
        perm,
        negative_slope,
        in_feat,
        bias,
        attn_drop,
        seed,
    ):
        ext = ensure_collapsed_gat_loaded(False)
        attn_row = attn_row.contiguous()
        attn_col = attn_col.contiguous()
        rowptr = rowptr.contiguous()
        col = col.contiguous()
        colptr = colptr.contiguous()
        row = row.contiguous()
        perm = perm.contiguous()
        in_feat = in_feat.contiguous()
        bias = bias.contiguous()

        y, pre, edge_max, edge_sum, edge_mask = ext.forward(
            attn_row,
            attn_col,
            rowptr,
            col,
            float(negative_slope),
            in_feat,
            bias,
            float(attn_drop),
            int(seed),
        )
        ctx.save_for_backward(
            rowptr,
            col,
            colptr,
            row,
            perm,
            edge_max,
            edge_sum,
            edge_mask,
            in_feat,
            attn_row,
            attn_col,
            pre,
        )
        ctx.negative_slope = float(negative_slope)
        ctx.attn_drop = float(attn_drop)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        ext = ensure_collapsed_gat_loaded(False)
        (
            rowptr,
            col,
            colptr,
            row,
            perm,
            edge_max,
            edge_sum,
            edge_mask,
            in_feat,
            attn_row,
            attn_col,
            pre,
        ) = ctx.saved_tensors
        grad_feat, grad_attn_row, grad_attn_col, grad_bias = ext.backward(
            float(ctx.negative_slope),
            float(ctx.attn_drop),
            rowptr,
            col,
            colptr,
            row,
            perm,
            edge_max,
            edge_sum,
            edge_mask,
            in_feat,
            attn_row,
            attn_col,
            pre,
            grad_y.contiguous(),
        )
        return (
            grad_attn_row,
            grad_attn_col,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_feat,
            grad_bias,
            None,
            None,
        )


def collapsed_gat_mean_bias_silu(
    attn_row: torch.Tensor,
    attn_col: torch.Tensor,
    csr,
    csc,
    perm: torch.Tensor,
    negative_slope: float,
    in_feat: torch.Tensor,
    bias: torch.Tensor,
    attn_drop: float,
) -> torch.Tensor:
    """A3-v6 GAT core specialized for concat=False (safe for 1<=H<=32, 1<=C<=256).

    It is algebraically the same as:
        dgNN GATConvFuse(...).mean(dim=1) + bias -> SiLU

    but never materializes dgNN's [N, heads, channels] output nor the matching
    expanded grad-output tensor in backward. Attention logits remain the A3-v3
    fused side-op so its already-fast parameter-gradient path is preserved.
    """
    if in_feat.dtype != torch.float32 or in_feat.device.type != "cuda":
        raise TypeError("A3-v6 collapsed GAT requires CUDA float32 in_feat")
    if bias is None:
        raise ValueError("A3-v6 collapsed GAT currently requires bias=True")
    if in_feat.dim() != 3:
        raise ValueError(f"in_feat must be [N,H,C], got {tuple(in_feat.shape)}")
    C = int(in_feat.size(2))
    H = int(in_feat.size(1))
    # Kernel safety range: feature tile is 256 threads; attention backward uses
    # blockDim=(32,H), so H must not exceed 32. Supporting smaller C/H is
    # important for clean11 lightweight hyperparameter sweeps.
    if not (1 <= C <= 256):
        raise ValueError(f"A3-v6 collapsed GAT requires 1<=C<=256; got C={C}")
    if not (1 <= H <= 32):
        raise ValueError(f"A3-v6 collapsed GAT requires 1<=H<=32; got H={H}")

    (rowptr, col), (row, colptr) = csr, csc
    # Tie the custom dropout stream to torch.manual_seed while giving every GAT
    # call a different deterministic counter. This is more reproducible than
    # dgNN's clock()-seeded CURAND path while preserving Bernoulli dropout.
    base_seed = int(torch.initial_seed()) & ((1 << 63) - 1)
    # Evaluation has attn_drop=0 and should not advance the training dropout
    # stream. This makes repeated validation forwards independent of the next
    # training mask while retaining one distinct deterministic mask per train call.
    call_id = next(_CALL_COUNTER) if float(attn_drop) > 0.0 else -1
    seed = 0 if call_id < 0 else (
        base_seed + 0x9E3779B97F4A7C15 * (call_id + 1)
    ) & ((1 << 63) - 1)
    return _CollapsedGATFn.apply(
        attn_row,
        attn_col,
        rowptr,
        col,
        colptr,
        row,
        perm,
        float(negative_slope),
        in_feat,
        bias,
        float(attn_drop),
        int(seed),
    )
