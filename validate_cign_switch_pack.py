#!/usr/bin/env python3
"""Numerically validate the CIGN switch-aware fused Clifford CUDA kernel.

Run once before full training:
PYTHONNOUSERSITE=1 python \
    validate_cign_switch_pack.py
"""

import torch

from generic_engine.fused_clifford_switch_pack import (
    fused_clifford_switch_pack,
    ensure_fused_clifford_switch_pack_loaded,
)


def reference(H, C, props, alpha, interaction, weighting):
    Ds = []
    Ws = []

    for P in props:
        TH, TC = P.split(H.size(1), dim=-1)
        bhr = H * TC
        brh = C * TH

        if interaction == "old":
            D = torch.nn.functional.silu(bhr)
            W = bhr - brh
        else:
            D = torch.nn.functional.silu(
                0.5 * (bhr + brh)
            )
            W = torch.tanh(
                0.5 * (bhr - brh)
            )

        Ds.append(D)
        Ws.append(W)

    D = torch.stack(Ds, dim=1)
    W = torch.stack(Ws, dim=1)
    A = alpha.unsqueeze(-1)

    if weighting == "old":
        F = torch.cat([W, D], dim=-1) * A
        return F.reshape(H.size(0), -1)

    Dagg = (D * A).sum(dim=1)
    Wagg = (W * A).sum(dim=1)
    return torch.cat([Dagg, Wagg], dim=-1)


def relerr(a, b):
    denom = b.norm().clamp_min(1e-12)
    return float((a - b).norm() / denom)


def one_case(interaction, weighting):
    torch.manual_seed(42)
    device = "cuda"

    N = 257
    D = 64

    raw = [
        torch.randn(N, D, device=device, dtype=torch.float32)
        for _ in range(2)
    ]
    prop_raw = [
        torch.randn(N, 2 * D, device=device, dtype=torch.float32)
        for _ in range(3)
    ]
    alpha_raw = torch.randn(
        N, 3, device=device, dtype=torch.float32
    )

    def clone_inputs():
        H = raw[0].detach().clone().requires_grad_(True)
        C = raw[1].detach().clone().requires_grad_(True)
        P = [
            x.detach().clone().requires_grad_(True)
            for x in prop_raw
        ]
        logits = alpha_raw.detach().clone().requires_grad_(True)
        A = torch.softmax(logits, dim=-1)
        return H, C, P, logits, A

    H1, C1, P1, logits1, A1 = clone_inputs()
    out_ref = reference(
        H1, C1, P1, A1, interaction, weighting
    )
    grad_seed = torch.randn_like(out_ref)
    loss_ref = (out_ref * grad_seed).sum()
    loss_ref.backward()

    H2, C2, P2, logits2, A2 = clone_inputs()
    out_fast = fused_clifford_switch_pack(
        H2, C2, P2, A2,
        interaction_mode=interaction,
        alpha_weighting_mode=weighting,
    )
    loss_fast = (out_fast * grad_seed).sum()
    loss_fast.backward()

    rows = {
        "out": relerr(out_fast, out_ref),
        "grad_H": relerr(H2.grad, H1.grad),
        "grad_C": relerr(C2.grad, C1.grad),
        "grad_P0": relerr(P2[0].grad, P1[0].grad),
        "grad_P1": relerr(P2[1].grad, P1[1].grad),
        "grad_P2": relerr(P2[2].grad, P1[2].grad),
        "grad_alpha_logits": relerr(logits2.grad, logits1.grad),
    }

    print(
        f"{interaction=}, {weighting=}: "
        + ", ".join(f"{k}={v:.3e}" for k, v in rows.items())
    )

    worst = max(rows.values())
    if not torch.isfinite(out_fast).all():
        raise RuntimeError("non-finite fused output")
    if worst > 5e-5:
        raise RuntimeError(
            f"validation failed: worst relative error {worst}"
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("Compiling/loading CIGN switch CUDA kernel...")
    ensure_fused_clifford_switch_pack_loaded(False)
    print("Kernel loaded.")

    for interaction in ["old", "new"]:
        for weighting in ["old", "new"]:
            one_case(interaction, weighting)

    print("VALIDATION PASS")


if __name__ == "__main__":
    main()
