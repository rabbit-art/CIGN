#!/usr/bin/env python3
"""Validate A3-v5 MultiHopCSR3 numerics and benchmark grouped sparse recurrence."""

import time
import torch

from model import prepare_fixed_csr_operator, FixedCSRSpMMFunction
from multihop_csr_pipeline import prepare_multihop_csr_operator, multihop_csr3


def make_regular_operator(n: int, offsets):
    device = torch.device("cuda")
    offsets = torch.tensor(offsets, device=device, dtype=torch.long)
    row = torch.arange(n, device=device, dtype=torch.long).repeat_interleave(offsets.numel())
    col = (torch.arange(n, device=device, dtype=torch.long).unsqueeze(1) + offsets.view(1, -1)) % n
    col = col.reshape(-1)
    val = torch.full((row.numel(),), 1.0 / float(offsets.numel()), device=device, dtype=torch.float32)
    coo = torch.sparse_coo_tensor(torch.stack([row, col]), val, (n, n), device=device).coalesce()
    return coo


def baseline3(base, x):
    y1 = FixedCSRSpMMFunction.apply(x, base.csr, base.csr_t)
    y2 = FixedCSRSpMMFunction.apply(y1, base.csr, base.csr_t)
    y3 = FixedCSRSpMMFunction.apply(y2, base.csr, base.csr_t)
    return y1, y2, y3


def report(name, a, b):
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-8)
    print(f"  {name:<18} max_abs={diff.max().item():.8e} mean_abs={diff.mean().item():.8e} max_rel={(diff/denom).max().item():.8e}")
    return diff.max().item()


def numerical_test():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    n, d = 4096, 128
    op = make_regular_operator(n, [0, 1, -1, 7, -7, 31, -31, 127, -127])
    base = prepare_fixed_csr_operator(op)
    pipe = prepare_multihop_csr_operator(base, dense_cols=d, algorithm=2)

    x0 = torch.randn(n, d, device="cuda", dtype=torch.float32)
    g1 = torch.randn(n, d, device="cuda", dtype=torch.float32)
    g2 = torch.randn(n, d, device="cuda", dtype=torch.float32)
    g3 = torch.randn(n, d, device="cuda", dtype=torch.float32)

    xa = x0.clone().requires_grad_(True)
    ya = baseline3(base, xa)
    la = (ya[0]*g1).sum() + (ya[1]*g2).sum() + (ya[2]*g3).sum()
    la.backward()
    ga = xa.grad.detach().clone()

    xb = x0.clone().requires_grad_(True)
    yb = multihop_csr3(pipe, xb)
    lb = (yb[0]*g1).sum() + (yb[1]*g2).sum() + (yb[2]*g3).sum()
    lb.backward()
    gb = xb.grad.detach().clone()

    print("NUMERICAL ALIGNMENT (N=4096,D=128,HOPS=1/2/3)")
    errs = [
        report("P^1 X", yb[0], ya[0]),
        report("P^2 X", yb[1], ya[1]),
        report("P^3 X", yb[2], ya[2]),
        report("grad_X", gb, ga),
    ]
    print("  native_info       :", pipe.info)
    ok = max(errs) < 5e-4
    print("  numerical_status  :", "PASS" if ok else "FAIL")
    if not ok:
        raise RuntimeError("A3-v5 numerical alignment failed.")


def bench_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def benchmark():
    n, d = 24492, 512
    op = make_regular_operator(n, [0, 1, -1, 7, -7, 31, -31, 127, -127])
    base = prepare_fixed_csr_operator(op)
    pipe = prepare_multihop_csr_operator(base, dense_cols=d, algorithm=2)
    x = torch.randn(n, d, device="cuda", dtype=torch.float32)
    g1 = torch.randn_like(x)
    g2 = torch.randn_like(x)
    g3 = torch.randn_like(x)

    def legacy_native_sequence():
        y1 = torch.sparse.mm(base.csr, x)
        y2 = torch.sparse.mm(base.csr, y1)
        y3 = torch.sparse.mm(base.csr, y2)
        t2 = torch.sparse.mm(base.csr_t, g3) + g2
        t1 = torch.sparse.mm(base.csr_t, t2) + g1
        gx = torch.sparse.mm(base.csr_t, t1)
        return y3, gx

    def a3v5_sequence():
        ys = pipe.plan.forward3(x)
        gx = pipe.plan.backward3(g1, g2, g3)
        return ys[2], gx

    legacy = bench_ms(legacy_native_sequence)
    v5 = bench_ms(a3v5_sequence)
    print("A3-v5 MULTIHOP FWD3+BWD3 SYNTHETIC BENCHMARK")
    print(f"  shape             : N={n}, D={d}, nnz={base.nnz}")
    print(f"  fixedcsr_sequence : {legacy:.4f} ms")
    print(f"  a3v5_pipeline     : {v5:.4f} ms")
    print(f"  speedup           : {legacy / v5:.3f}x")
    print("  note              : synthetic regular sparsity; final truth is trainb9_use.py on Amazon-ratings")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    print("=" * 100)
    print("A3-v5 VALIDATION: grouped three-hop fixed-CSR forward/backward pipeline")
    print("=" * 100)
    t0 = time.perf_counter()
    numerical_test()
    benchmark()
    print(f"TOTAL_VALIDATION_WALL: {time.perf_counter()-t0:.3f}s (outside training timing)")
    print("=" * 100)
    print("RESULT: NUMERICAL ALIGNMENT LOOKS GOOD")
    print("NEXT: run trainb9_use.py and compare against A3-v4-TF32 = 0.091367 s/epoch.")


if __name__ == "__main__":
    main()
