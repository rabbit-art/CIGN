from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.cpp_extension import load

_EXT = None
_EXT_ERROR = None


def _extension_name() -> str:
    # Keep a unique name so it does not collide with older experiments.
    return "graph_clifford_fused_pack_a3_v1"


def ensure_fused_clifford_pack_loaded(verbose: bool = False):
    """Build/load the A3 CUDA extension once, outside epoch timing."""
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError("A3 fused Clifford extension previously failed to load") from _EXT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("A3 fused Clifford pack requires CUDA.")

    root = Path(__file__).resolve().parent
    src_dir = root / "fused_clifford_pack_ext"
    build_dir = root / ".fused_clifford_pack_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # The server uses RTX 3090 (sm_86). Respect an explicit user override.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

    try:
        _EXT = load(
            name=_extension_name(),
            sources=[
                str(src_dir / "fused_clifford_pack.cpp"),
                str(src_dir / "fused_clifford_pack_cuda.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir),
            verbose=verbose,
        )
        return _EXT
    except Exception as exc:  # pragma: no cover - executed on target CUDA machine
        _EXT_ERROR = exc
        raise


class FusedCliffordPackFunction(torch.autograd.Function):
    """Fuse Clifford feature construction + hop gating into one CUDA op.

    The output layout is exactly the legacy layout:
        [hop1_W, hop1_D, hop2_W, hop2_D, hop3_W, hop3_D]
    so the existing nn.Linear(1536 -> 256) projection and all learned parameters
    are unchanged.

    This A3-v1 kernel deliberately keeps the highly optimized cuBLAS projection
    as a single large GEMM. It removes the large Python/PyTorch chain that used
    to materialize H_mul_TC, D_s, W_s, F_s, stack(...), gate multiplication and
    reshape before that GEMM.
    """

    @staticmethod
    def forward(ctx, H, C, P0, P1, P2, alpha):
        ext = ensure_fused_clifford_pack_loaded(verbose=False)
        tensors = (H, C, P0, P1, P2, alpha)
        names = ("H", "C", "P0", "P1", "P2", "alpha")
        for name, t in zip(names, tensors):
            if not t.is_cuda:
                raise RuntimeError(f"{name} must be CUDA tensor")
            if t.dtype != torch.float32:
                raise TypeError(f"A3 fused pack currently requires float32; {name} has {t.dtype}")
            if not t.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous; got stride={t.stride()}")

        if H.dim() != 2 or C.shape != H.shape:
            raise ValueError(f"H and C must both be [N,D], got {tuple(H.shape)} and {tuple(C.shape)}")
        N, D = H.shape
        for name, P in (("P0", P0), ("P1", P1), ("P2", P2)):
            if tuple(P.shape) != (N, 2 * D):
                raise ValueError(f"{name} must have shape {(N, 2*D)}, got {tuple(P.shape)}")
        if tuple(alpha.shape) != (N, 3):
            raise ValueError(f"alpha must have shape {(N,3)}, got {tuple(alpha.shape)}")

        out = ext.forward(H, C, P0, P1, P2, alpha)
        ctx.save_for_backward(H, C, P0, P1, P2, alpha)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        ext = ensure_fused_clifford_pack_loaded(verbose=False)
        H, C, P0, P1, P2, alpha = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grads = ext.backward(grad_output, H, C, P0, P1, P2, alpha)
        return tuple(grads)


def fused_clifford_pack(
    H_norm: torch.Tensor,
    C_norm: torch.Tensor,
    propagated: Sequence[torch.Tensor],
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Return the exact legacy G_raw tensor using one fused CUDA feature kernel.

    A3-v1 is intentionally specialized for the current Amazon configuration
    with exactly three requested hops. Other configurations can keep the legacy
    path by setting --use_fused_clifford_pack False.
    """
    if len(propagated) != 3:
        raise ValueError(
            "A3 fused Clifford pack v1 is specialized for exactly three hops. "
            f"Got {len(propagated)} propagated tensors. Disable it for other hop counts."
        )
    if alpha.dim() == 3 and alpha.size(-1) == 1:
        alpha = alpha.squeeze(-1)
    if alpha.dim() != 2 or alpha.size(1) != 3:
        raise ValueError(f"Expected alpha [N,3], got {tuple(alpha.shape)}")
    # Node-wise softmax is already contiguous in the target configuration.
    # Calling contiguous() is a no-op there, while also making global-gate use safe.
    alpha = alpha.contiguous()
    return FusedCliffordPackFunction.apply(
        H_norm.contiguous(),
        C_norm.contiguous(),
        propagated[0].contiguous(),
        propagated[1].contiguous(),
        propagated[2].contiguous(),
        alpha,
    )
