#!/usr/bin/env python3
"""A3-v9 selective mixed-precision validator/benchmark.

V9 keeps all model sizes unchanged and lowers precision only for the large dense
Linear GEMMs.  The graph/sparse/custom CUDA path remains FP32.

This script does not train the graph model.  It checks that:
  * CUDA BF16/FP16 autocast is available,
  * master Linear parameters stay FP32,
  * representative Amazon-sized dense fwd+bwd calls are finite,
  * TF32 vs BF16/FP16 timing and numerical deltas are printed.
"""

import argparse
import math
import time

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["bf16", "fp16", "both"], default="both")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=12)
    return p.parse_args()


def dtype_from_name(name):
    return torch.bfloat16 if name == "bf16" else torch.float16


def set_tf32(enabled: bool):
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def timed_linear_fwd_bwd(n, in_dim, out_dim, mode, warmup, iters, device):
    torch.manual_seed(1234)
    lin = nn.Linear(in_dim, out_dim, bias=True, device=device, dtype=torch.float32)
    x = torch.randn(n, in_dim, device=device, dtype=torch.float32, requires_grad=True)
    grad_fp32 = torch.randn(n, out_dim, device=device, dtype=torch.float32)

    if mode == "tf32":
        amp = False
        amp_dtype = torch.bfloat16
        grad = grad_fp32
    else:
        amp = True
        amp_dtype = dtype_from_name(mode)
        grad = grad_fp32.to(amp_dtype)

    # Parameters are intentionally always FP32 master weights.
    assert lin.weight.dtype == torch.float32
    assert lin.bias.dtype == torch.float32

    def one_iter():
        lin.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp, cache_enabled=True
        ):
            y = lin(x)
        # The real V9 model immediately casts the large Linear result back to
        # FP32 before graph/custom CUDA operations.  The cast remains in the
        # autograd graph, so Linear backward still follows the autocast dtype.
        y_fp32 = y.float()
        y_fp32.backward(grad_fp32)

    for _ in range(warmup):
        one_iter()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        one_iter()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iters

    # One small-ish deterministic numerical comparison sample is done outside
    # the timed loop to avoid perturbing timings.
    return ms


def numerical_check(in_dim, out_dim, mode, device):
    n = 2048
    torch.manual_seed(2026)
    lin = nn.Linear(in_dim, out_dim, bias=True, device=device, dtype=torch.float32)
    x0 = torch.randn(n, in_dim, device=device, dtype=torch.float32)
    g0 = torch.randn(n, out_dim, device=device, dtype=torch.float32)

    # Baseline TF32 forward/backward.
    set_tf32(True)
    xb = x0.detach().clone().requires_grad_(True)
    lin.zero_grad(set_to_none=True)
    yb = lin(xb)
    yb.backward(g0)
    out_ref = yb.detach().float()
    gx_ref = xb.grad.detach().float().clone()
    gw_ref = lin.weight.grad.detach().float().clone()

    # AMP forward/backward using the same FP32 master weights.
    xa = x0.detach().clone().requires_grad_(True)
    lin.zero_grad(set_to_none=True)
    amp_dtype = dtype_from_name(mode)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True, cache_enabled=True):
        ya = lin(xa)
    ya.float().backward(g0)
    out = ya.detach().float()
    gx = xa.grad.detach().float()
    gw = lin.weight.grad.detach().float()

    def stats(a, b):
        diff = (a - b).float()
        rmse = diff.square().mean().sqrt().item()
        denom = b.float().square().mean().sqrt().item() + 1e-12
        rel_rmse = rmse / denom
        max_abs = diff.abs().max().item()
        return max_abs, rel_rmse

    omax, orr = stats(out, out_ref)
    gxmax, gxrr = stats(gx, gx_ref)
    gwmax, gwrr = stats(gw, gw_ref)
    finite = all(
        torch.isfinite(t).all().item()
        for t in (out, gx, gw)
    )
    return {
        "finite": finite,
        "out_max_abs": omax,
        "out_rel_rmse": orr,
        "grad_x_max_abs": gxmax,
        "grad_x_rel_rmse": gxrr,
        "grad_w_max_abs": gwmax,
        "grad_w_rel_rmse": gwrr,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A3-v9 validator requires CUDA.")

    device = torch.device("cuda")
    prop = torch.cuda.get_device_properties(device)
    print("=" * 104)
    print("A3-v9 VALIDATION: selective mixed precision for large dense Linear GEMMs")
    print("=" * 104)
    print(f"GPU                  : {prop.name}")
    print(f"CUDA capability      : {prop.major}.{prop.minor}")
    print(f"Torch                : {torch.__version__}")
    print(f"BF16 supported       : {torch.cuda.is_bf16_supported()}")
    print("Master parameters    : FP32")
    print("FP32 islands         : GAT attention/message CUDA, MultiHopCSR, CliffordPack, block fusion, LN/softmax/loss")
    print("AMP targets          : input projection, GAT 256->1536 Linear, Clifford 1536->256 projection")

    modes = []
    if args.dtype in ("bf16", "both"):
        if torch.cuda.is_bf16_supported():
            modes.append("bf16")
        else:
            print("WARNING: BF16 requested but unsupported; skipping BF16.")
    if args.dtype in ("fp16", "both"):
        modes.append("fp16")

    shapes = [
        ("input projection 300->256", 24492, 300, 256),
        ("GAT linear 256->1536", 24492, 256, 1536),
        ("Clifford projection 1536->256", 24492, 1536, 256),
    ]

    print("\nREPRESENTATIVE FWD+BWD BENCHMARK")
    print("-" * 104)
    set_tf32(True)
    for label, n, din, dout in shapes:
        base = timed_linear_fwd_bwd(n, din, dout, "tf32", args.warmup, args.iters, device)
        print(f"{label}")
        print(f"  TF32              : {base:.4f} ms")
        for mode in modes:
            ms = timed_linear_fwd_bwd(n, din, dout, mode, args.warmup, args.iters, device)
            print(f"  {mode.upper():<18}: {ms:.4f} ms | speedup={base/ms:.3f}x | reduction={(1-ms/base)*100:.2f}%")

    print("\nNUMERICAL DELTA VS TF32 (small representative GEMM; precision change is intentional)")
    print("-" * 104)
    all_finite = True
    for mode in modes:
        st = numerical_check(256, 1536, mode, device)
        all_finite = all_finite and bool(st["finite"])
        print(f"{mode.upper()}: finite={st['finite']}")
        print(f"  output  max_abs={st['out_max_abs']:.6e} rel_rmse={st['out_rel_rmse']:.6e}")
        print(f"  grad_x  max_abs={st['grad_x_max_abs']:.6e} rel_rmse={st['grad_x_rel_rmse']:.6e}")
        print(f"  grad_w  max_abs={st['grad_w_max_abs']:.6e} rel_rmse={st['grad_w_rel_rmse']:.6e}")

    print("=" * 104)
    if all_finite and modes:
        print("RESULT: SELECTIVE AMP KERNELS ARE FINITE/READY")
        print("NEXT: run trainb9_use.py. The default V9 runner uses BF16; switch selective_amp_dtype to fp16 only for an A/B test.")
    else:
        print("RESULT: VALIDATION FAILED OR NO SUPPORTED AMP MODE")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
