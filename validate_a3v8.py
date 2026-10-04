from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from fused_block_ops import (
    ensure_fused_block_ops_loaded,
    fused_dropout_layernorm,
    fused_add_layernorm,
    fused_dropout_gamma_residual,
)


def max_abs(a, b):
    return float((a - b).abs().max().item())


def max_rel(a, b, eps=1e-12):
    return float(((a - b).abs() / (b.abs() + eps)).max().item())


def clone_leaf(x):
    return x.detach().clone().requires_grad_(True)


def legacy_block_ops(H, Cgat, G, hw, hb, cw, cb, gamma, p, eps):
    Hn = F.layer_norm(F.dropout(H, p=p, training=True), (H.size(1),), hw, hb, eps)
    Cn = F.layer_norm(Cgat + Hn, (H.size(1),), cw, cb, eps)
    Y = H + gamma.unsqueeze(0) * F.dropout(G, p=p, training=True)
    return Hn, Cn, Y


def fused_block_ops(H, Cgat, G, hw, hb, cw, cb, gamma, p, eps):
    Hn = fused_dropout_layernorm(H, hw, hb, p, eps)
    Cn = fused_add_layernorm(Cgat, Hn, cw, cb, eps)
    Y = fused_dropout_gamma_residual(H, G, gamma, p)
    return Hn, Cn, Y


def numerical_test():
    torch.manual_seed(1234)
    device = 'cuda'
    N, D = 4096, 256
    eps = 1e-5
    p = 0.0  # strict algebraic comparison; dropout RNG streams intentionally differ in training

    base = {
        'H': torch.randn(N, D, device=device, dtype=torch.float32),
        'Cgat': torch.randn(N, D, device=device, dtype=torch.float32),
        'G': torch.randn(N, D, device=device, dtype=torch.float32),
        'hw': torch.randn(D, device=device, dtype=torch.float32) * 0.2 + 1.0,
        'hb': torch.randn(D, device=device, dtype=torch.float32) * 0.1,
        'cw': torch.randn(D, device=device, dtype=torch.float32) * 0.2 + 1.0,
        'cb': torch.randn(D, device=device, dtype=torch.float32) * 0.1,
        'gamma': torch.randn(D, device=device, dtype=torch.float32) * 0.05 + 0.29,
    }
    probe_h = torch.randn(N, D, device=device)
    probe_c = torch.randn(N, D, device=device)
    probe_y = torch.randn(N, D, device=device)

    A = {k: clone_leaf(v) for k, v in base.items()}
    B = {k: clone_leaf(v) for k, v in base.items()}

    out_a = legacy_block_ops(**A, p=p, eps=eps)
    loss_a = (out_a[0] * probe_h).sum() + (out_a[1] * probe_c).sum() + (out_a[2] * probe_y).sum()
    loss_a.backward()

    out_b = fused_block_ops(**B, p=p, eps=eps)
    loss_b = (out_b[0] * probe_h).sum() + (out_b[1] * probe_c).sum() + (out_b[2] * probe_y).sum()
    loss_b.backward()

    rows = []
    for name, aa, bb in [
        ('H_norm', out_a[0], out_b[0]),
        ('C_norm', out_a[1], out_b[1]),
        ('residual_out', out_a[2], out_b[2]),
    ]:
        rows.append((name, max_abs(aa, bb), max_rel(aa, bb)))
    for k in ['H', 'Cgat', 'G', 'hw', 'hb', 'cw', 'cb', 'gamma']:
        rows.append((f'grad_{k}', max_abs(A[k].grad, B[k].grad), max_rel(A[k].grad, B[k].grad)))

    print('NUMERICAL ALIGNMENT (dropout=0)')
    for name, ma, mr in rows:
        print(f'  {name:<18s} max_abs={ma:.8e} max_rel={mr:.8e}')

    # LayerNorm reductions use a different reduction order than ATen/Welford.
    # Output/input gradients should be close; affine parameter reductions can
    # accumulate somewhat larger absolute differences over thousands of rows.
    forward_ok = all(ma <= 3e-4 for name, ma, _ in rows if not name.startswith('grad_'))
    grad_ok = all(ma <= (2e-2 if name in {'grad_hw','grad_hb','grad_cw','grad_cb','grad_gamma'} else 2e-3)
                  for name, ma, _ in rows if name.startswith('grad_'))
    return forward_ok and grad_ok


def bench_one(label, fn, tensors, warmup=5, iters=20):
    # tensors is a dict of leaf tensors; fn accepts them and returns 3 tensors.
    for _ in range(warmup):
        for t in tensors.values():
            if t.grad is not None:
                t.grad = None
        outs = fn(**tensors)
        loss = outs[0].square().mean() + outs[1].square().mean() + outs[2].square().mean()
        loss.backward()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        for t in tensors.values():
            if t.grad is not None:
                t.grad = None
        outs = fn(**tensors)
        loss = outs[0].square().mean() + outs[1].square().mean() + outs[2].square().mean()
        loss.backward()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f'  {label:<28s} = {ms:.4f} ms/call')
    return ms


def benchmark():
    torch.manual_seed(42)
    N, D = 24492, 256
    p, eps = 0.3, 1e-5
    device = 'cuda'

    def make():
        return {
            'H': torch.randn(N, D, device=device, requires_grad=True),
            'Cgat': torch.randn(N, D, device=device, requires_grad=True),
            'G': torch.randn(N, D, device=device, requires_grad=True),
            'hw': (torch.ones(D, device=device) + 0.01 * torch.randn(D, device=device)).requires_grad_(),
            'hb': torch.zeros(D, device=device, requires_grad=True),
            'cw': (torch.ones(D, device=device) + 0.01 * torch.randn(D, device=device)).requires_grad_(),
            'cb': torch.zeros(D, device=device, requires_grad=True),
            'gamma': torch.full((D,), 0.29, device=device, requires_grad=True),
        }

    legacy_t = make()
    fused_t = {k: v.detach().clone().requires_grad_(True) for k, v in legacy_t.items()}

    print('A3-v8 BLOCK-SIDEOPS FWD+BWD BENCHMARK (N=24492,D=256,p=0.3)')
    legacy_ms = bench_one(
        'V6 legacy block sideops',
        lambda **kw: legacy_block_ops(**kw, p=p, eps=eps),
        legacy_t,
    )
    fused_ms = bench_one(
        'V8 fused block sideops',
        lambda **kw: fused_block_ops(**kw, p=p, eps=eps),
        fused_t,
    )
    print(f'  speedup                      = {legacy_ms / fused_ms:.3f}x')
    print(f'  time_reduction               = {(1.0 - fused_ms / legacy_ms) * 100.0:.2f}%')

    # Peak allocated memory diagnostic for one complete forward+backward call.
    def peak(fn, ts):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for t in ts.values():
            t.grad = None
        outs = fn(**ts)
        (outs[0].square().mean() + outs[1].square().mean() + outs[2].square().mean()).backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    legacy_peak = peak(lambda **kw: legacy_block_ops(**kw, p=p, eps=eps), legacy_t)
    fused_peak = peak(lambda **kw: fused_block_ops(**kw, p=p, eps=eps), fused_t)
    print(f'  legacy_peak_alloc_mb         = {legacy_peak:.2f}')
    print(f'  fused_peak_alloc_mb          = {fused_peak:.2f}')
    print(f'  peak_mem_reduction           = {(1.0 - fused_peak / legacy_peak) * 100.0:.2f}%')
    return legacy_ms, fused_ms


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    print('=' * 110)
    print('A3-v8 VALIDATION: block-side elementwise/norm fusion on top of the retained V6 baseline')
    print('=' * 110)
    t0 = time.perf_counter()
    ensure_fused_block_ops_loaded(verbose=True)
    torch.cuda.synchronize()
    print(f'EXTENSION_BUILD_OR_LOAD: {time.perf_counter() - t0:.3f}s (outside training timing)')

    ok = numerical_test()
    legacy_ms, fused_ms = benchmark()
    print('=' * 110)
    print(f'numerical_status : {"PASS" if ok else "FAIL"}')
    if ok:
        print('RESULT: NUMERICAL ALIGNMENT LOOKS GOOD')
    else:
        print('RESULT: NUMERICAL ALIGNMENT FAILED -- DO NOT RUN FULL TRAINING')
    if ok and fused_ms < legacy_ms:
        print('NEXT: run trainb9_use.py and compare against retained V6 = 0.076903 s/epoch.')
    elif ok:
        print('NEXT: fused sideops are not faster synthetically; do not spend time on full training yet.')


if __name__ == '__main__':
    main()
