from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from fused_gat_sideops import (
    ensure_fused_gat_sideops_loaded,
    fused_attention_logits,
    fused_head_mean_bias_silu,
)
from model import CompatibleFusedGATConv, CompatibleFusedGATConvSideOps


def max_abs(a, b):
    return float((a - b).abs().max().item())


def clone_leaf(x):
    return x.detach().clone().requires_grad_(True)


def check_sideops(device):
    torch.manual_seed(123)
    N, H, C = 4096, 6, 256
    x0 = torch.randn(N, H, C, device=device, dtype=torch.float32)
    as0 = torch.randn(1, H, C, device=device, dtype=torch.float32) * 0.1
    ad0 = torch.randn(1, H, C, device=device, dtype=torch.float32) * 0.1
    gs = torch.randn(N, H, device=device)
    gd = torch.randn(N, H, device=device)

    x = clone_leaf(x0); ats = clone_leaf(as0); atd = clone_leaf(ad0)
    ls = (x * ats).sum(-1); ld = (x * atd).sum(-1)
    loss = (ls * gs + ld * gd).sum(); loss.backward()
    legacy = (ls.detach(), ld.detach(), x.grad.detach(), ats.grad.detach(), atd.grad.detach())

    x = clone_leaf(x0); ats = clone_leaf(as0); atd = clone_leaf(ad0)
    fs, fd = fused_attention_logits(x, ats, atd)
    loss = (fs * gs + fd * gd).sum(); loss.backward()
    fused = (fs.detach(), fd.detach(), x.grad.detach(), ats.grad.detach(), atd.grad.detach())

    print("ATTENTION SIDE-OP NUMERICS")
    labels = ["alpha_src", "alpha_dst", "grad_x", "grad_att_src", "grad_att_dst"]
    att_errs = []
    for name, a, b in zip(labels, legacy, fused):
        e = max_abs(a, b); att_errs.append(e); print(f"  {name:16s} max_abs={e:.8e}")

    out0 = torch.randn(N, H, C, device=device, dtype=torch.float32)
    bias0 = torch.randn(C, device=device, dtype=torch.float32) * 0.1
    gy = torch.randn(N, C, device=device)

    out = clone_leaf(out0); bias = clone_leaf(bias0)
    y = F.silu(out.mean(dim=1) + bias)
    (y * gy).sum().backward()
    legacy_h = (y.detach(), out.grad.detach(), bias.grad.detach())

    out = clone_leaf(out0); bias = clone_leaf(bias0)
    y2 = fused_head_mean_bias_silu(out, bias)
    (y2 * gy).sum().backward()
    fused_h = (y2.detach(), out.grad.detach(), bias.grad.detach())

    print("HEAD MEAN+BIAS+SILU NUMERICS")
    labels = ["output", "grad_out", "grad_bias"]
    head_errs = []
    for name, a, b in zip(labels, legacy_h, fused_h):
        e = max_abs(a, b); head_errs.append(e); print(f"  {name:16s} max_abs={e:.8e}")

    # Reduction order differs, especially for parameter gradients. These bounds
    # are intentionally absolute and strict enough to catch semantic mistakes.
    ok = (
        att_errs[0] < 2e-5 and att_errs[1] < 2e-5 and
        att_errs[2] < 2e-5 and att_errs[3] < 5e-3 and att_errs[4] < 5e-3 and
        head_errs[0] < 2e-5 and head_errs[1] < 2e-5 and head_errs[2] < 5e-3
    )
    return ok


def prepare_graph(conv, edge_index, n):
    ei = edge_index.flip(0).contiguous()
    csr, csc, perm = conv.to_graph_format(ei, size=(n, n))
    rowptr, col = csr; row, colptr = csc
    rowptr = rowptr.clone().contiguous(); col = col.clone().contiguous()
    row = row.clone().contiguous(); colptr = colptr.clone().contiguous(); perm = perm.clone().contiguous()
    return ((rowptr, col), (row, colptr), perm)


def check_full_fusedgat(device):
    torch.manual_seed(7)
    N, D, H = 1000, 256, 6
    E = 5000
    src = torch.randint(0, N, (E,), device=device)
    dst = torch.randint(0, N, (E,), device=device)
    loops = torch.arange(N, device=device)
    edge_index = torch.stack([torch.cat([src, loops]), torch.cat([dst, loops])], dim=0)

    base = CompatibleFusedGATConv(D, D, heads=H, concat=False, dropout=0.0, add_self_loops=False, bias=True).to(device)
    fast = CompatibleFusedGATConvSideOps(D, D, heads=H, concat=False, dropout=0.0, add_self_loops=False, bias=True).to(device)
    fast.load_state_dict(base.state_dict(), strict=True)
    base.eval(); fast.eval()
    graph = prepare_graph(base, edge_index, N)

    x0 = torch.randn(N, D, device=device)
    probe = torch.randn(N, D, device=device)

    x1 = clone_leaf(x0)
    y1 = F.silu(base(x1, *graph))
    (y1 * probe).sum().backward()
    g1 = x1.grad.detach().clone()
    p1 = {n: p.grad.detach().clone() for n, p in base.named_parameters() if p.grad is not None}

    x2 = clone_leaf(x0)
    y2 = fast(x2, *graph)
    (y2 * probe).sum().backward()
    g2 = x2.grad.detach().clone()
    p2 = {n: p.grad.detach().clone() for n, p in fast.named_parameters() if p.grad is not None}

    print("FULL dgNN FUSEDGAT INTEGRATION (dropout=0)")
    print(f"  output           max_abs={max_abs(y1,y2):.8e}")
    print(f"  input_grad       max_abs={max_abs(g1,g2):.8e}")
    max_param = 0.0
    for name in p1:
        e = max_abs(p1[name], p2[name]); max_param=max(max_param,e)
        print(f"  {name:16s} max_abs={e:.8e}")
    # Parameter reductions can differ slightly in summation order.
    return max_abs(y1,y2) < 5e-5 and max_abs(g1,g2) < 1e-4 and max_param < 2e-2


def benchmark_sideops(device):
    torch.manual_seed(9)
    N,H,C=24492,6,256
    x=torch.randn(N,H,C,device=device,requires_grad=True)
    ats=torch.randn(1,H,C,device=device,requires_grad=True)*0.1
    atd=torch.randn(1,H,C,device=device,requires_grad=True)*0.1
    out=torch.randn(N,H,C,device=device,requires_grad=True)
    bias=torch.randn(C,device=device,requires_grad=True)

    def legacy():
        a=(x*ats).sum(-1); b=(x*atd).sum(-1)
        y=F.silu(out.mean(1)+bias)
        return a.mean()+b.mean()+y.mean()
    def fused():
        a,b=fused_attention_logits(x,ats,atd)
        y=fused_head_mean_bias_silu(out,bias)
        return a.mean()+b.mean()+y.mean()

    def timeit(fn, iters=20):
        for _ in range(5):
            loss=fn(); torch.autograd.grad(loss,(x,ats,atd,out,bias),retain_graph=False)
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(iters):
            loss=fn(); torch.autograd.grad(loss,(x,ats,atd,out,bias),retain_graph=False)
        torch.cuda.synchronize(); return (time.perf_counter()-t0)*1000/iters

    lt=timeit(legacy); ft=timeit(fused)
    print("SYNTHETIC SIDE-OP FWD+BWD BENCHMARK (N=24492,H=6,C=256)")
    print(f"  legacy_ms = {lt:.4f}")
    print(f"  fused_ms  = {ft:.4f}")
    print(f"  speedup   = {lt/ft:.3f}x")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device=torch.device('cuda')
    print('='*100)
    print('A3-v2 VALIDATION: fused attention logits + head mean/bias/SiLU around dgNN')
    print('='*100)
    t0=time.perf_counter(); ensure_fused_gat_sideops_loaded(verbose=True); torch.cuda.synchronize()
    print(f'EXTENSION_BUILD_OR_LOAD: {time.perf_counter()-t0:.3f}s (outside training timing)')
    ok1=check_sideops(device)
    ok2=check_full_fusedgat(device)
    benchmark_sideops(device)
    print('='*100)
    if ok1 and ok2:
        print('RESULT: NUMERICAL ALIGNMENT LOOKS GOOD')
    else:
        raise RuntimeError('A3-v2 numerical validation failed; do not run formal training yet.')

if __name__ == '__main__':
    main()
