from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.cpp_extension import load

_EXT = None
_EXT_ERROR = None


def _extension_name() -> str:
    return "graph_clifford_cign_pack_k3_v1"


def ensure_fused_clifford_switch_pack_loaded(verbose: bool = False):
    """Build/load the CIGN switch-aware K=3 CUDA extension once.

    This extension is used only by the experimental six-switch model and never
    changes the project's original fused_clifford_pack implementation.
    """
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError(
            "CIGN switch fused Clifford extension previously failed to load"
        ) from _EXT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("CIGN switch fused Clifford pack requires CUDA.")

    root = Path(__file__).resolve().parent
    src_dir = root / "fused_clifford_cign_pack_ext"
    build_dir = root / ".fused_clifford_cign_pack_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

    try:
        _EXT = load(
            name=_extension_name(),
            sources=[
                str(src_dir / "fused_clifford_switch_pack.cpp"),
                str(src_dir / "fused_clifford_switch_pack_cuda.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir),
            verbose=verbose,
        )
        return _EXT
    except Exception as exc:  # target CUDA machine
        _EXT_ERROR = exc
        raise


def _mode_to_int(name: str, kind: str) -> int:
    value = str(name).strip().lower()
    if value == "old":
        return 0
    if value == "new":
        return 1
    raise ValueError(f"{kind} must be old/new, got {name!r}")


def _normalize_alpha(alpha, N: int, device) -> torch.Tensor:
    if alpha is None:
        return torch.ones((N, 3), device=device, dtype=torch.float32)
    if alpha.dim() == 3 and alpha.size(-1) == 1:
        alpha = alpha.squeeze(-1)
    if tuple(alpha.shape) != (N, 3):
        raise ValueError(f"Expected alpha {(N,3)}, got {tuple(alpha.shape)}")
    return alpha.to(dtype=torch.float32).contiguous()


class FusedCliffordSwitchPackFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        H,
        C,
        P0,
        P1,
        P2,
        alpha,
        interaction_mode_int: int,
        weighting_mode_int: int,
    ):
        ext = ensure_fused_clifford_switch_pack_loaded(False)

        out = ext.forward(
            H, C, P0, P1, P2, alpha,
            int(interaction_mode_int),
            int(weighting_mode_int),
        )

        ctx.interaction_mode_int = int(interaction_mode_int)
        ctx.weighting_mode_int = int(weighting_mode_int)
        ctx.save_for_backward(H, C, P0, P1, P2, alpha)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        ext = ensure_fused_clifford_switch_pack_loaded(False)
        H, C, P0, P1, P2, alpha = ctx.saved_tensors

        grad_output = grad_output.to(
            dtype=torch.float32
        ).contiguous()

        grads = ext.backward(
            grad_output,
            H, C, P0, P1, P2, alpha,
            int(ctx.interaction_mode_int),
            int(ctx.weighting_mode_int),
        )

        # Two integer mode arguments are non-differentiable.
        return (*grads, None, None)


def fused_clifford_switch_pack(
    H_norm: torch.Tensor,
    C_norm: torch.Tensor,
    propagated: Sequence[torch.Tensor],
    alpha: torch.Tensor | None,
    interaction_mode: str,
    alpha_weighting_mode: str,
) -> torch.Tensor:
    """Fused K=3 Clifford interaction + alpha weighting.

    interaction_mode="old":
        B_HR = H*T_C
        B_RH = C*T_H
        D = SiLU(B_HR)
        W = B_HR - B_RH

    interaction_mode="new":
        D = SiLU(0.5*(B_HR+B_RH))
        W = 0.5*(B_HR-B_RH)

    alpha_weighting_mode="old":
        output = Cat_k(alpha_k * [W_k,D_k]) -> [N,6D]

    alpha_weighting_mode="new":
        D_agg = sum_k alpha_k D_k
        W_agg = sum_k alpha_k W_k
        output = Cat(D_agg,W_agg) -> [N,2D]
    """
    if len(propagated) != 3:
        raise ValueError(
            "CIGN switch fused pack v1 is specialized for exactly 3 hops; "
            f"got {len(propagated)}."
        )
    if H_norm.dim() != 2 or C_norm.shape != H_norm.shape:
        raise ValueError("H_norm/C_norm must both be [N,D]")

    N, D = H_norm.shape
    H = H_norm.to(dtype=torch.float32).contiguous()
    C = C_norm.to(dtype=torch.float32).contiguous()
    props = [
        p.to(dtype=torch.float32).contiguous()
        for p in propagated
    ]
    for p in props:
        if tuple(p.shape) != (N, 2 * D):
            raise ValueError(
                f"Every propagated tensor must be {(N, 2*D)}, "
                f"got {tuple(p.shape)}"
            )

    A = _normalize_alpha(alpha, int(N), H.device)
    interaction_int = _mode_to_int(interaction_mode, "interaction_mode")
    weighting_int = _mode_to_int(
        alpha_weighting_mode, "alpha_weighting_mode"
    )

    return FusedCliffordSwitchPackFunction.apply(
        H, C, props[0], props[1], props[2], A,
        interaction_int, weighting_int,
    )
