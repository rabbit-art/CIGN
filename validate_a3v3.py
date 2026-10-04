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


def max_rel(a, b, floor=1e-4):
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(floor)
    return float(((a - b).abs() / denom).max().item())


def clone_leaf(x):
    return x.detach().clone().requires_grad_(True)


def check_attention_numerics(device, N=4096):
    torch.manual_seed(123)
    H, C = 6, 256
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

    print(f"ATTENTION NUMERICS (N={N}, H=6, C=256)")
    labels = ["alpha_src", "alpha_dst", "grad_x", "grad_att_src", "grad_att_dst"]
    errs = []
    for name, a, b in zip(labels, legacy, fused):
        ae = max_abs(a, b); re = max_rel(a, b)
        errs.append((ae, re))
        print(f"  {name:16s} max_abs={ae:.8e} max_rel={re:.8e}")

    # Parameter gradients use a different reduction tree. Absolute tolerance is
    # therefore intentionally looser than pointwise x-gradient tolerance.
    return (
        errs[0][0] < 2e-5 and errs[1][0] < 2e-5 and
        errs[2][0] < 2e-5 and
        errs[3][0] < 1.5e-2 and errs[4][0] < 1.5e-2
    )


def check_head_sideop(device):
    torch.manual_seed(321)
    N,H,C = 4096,6,256
    out0 = torch.randn(N,H,C,device=device,dtype=torch.float32)
    bias0 = torch.randn(C,device=device,dtype=torch.float32)*0.1
    gy = torch.randn(N,C,device=device)

    out = clone_leaf(out0); bias = clone_leaf(bias0)
    y = F.silu(out.mean(dim=1)+bias); (y*gy).sum().backward()
    legacy=(y.detach(),out.grad.detach(),bias.grad.detach())

    out = clone_leaf(out0); bias = clone_leaf(bias0)
    y2=fused_head_mean_bias_silu(out,bias); (y2*gy).sum().backward()
    fused=(y2.detach(),out.grad.detach(),bias.grad.detach())
    print("HEAD MEAN+BIAS+SILU NUMERICS")
    labels=["output","grad_out","grad_bias"]
    errs=[]
    for name,a,b in zip(labels,legacy,fused):
        e=max_abs(a,b); errs.append(e); print(f"  {name:16s} max_abs={e:.8e}")
    return errs[0] < 2e-5 and errs[1] < 2e-5 and errs[2] < 5e-3


def prepare_graph(conv, edge_index, n):
    ei=edge_index.flip(0).contiguous()
    csr,csc,perm=conv.to_graph_format(ei,size=(n,n))
    rowptr,col=csr; row,colptr=csc
    return ((rowptr.clone().contiguous(),col.clone().contiguous()),
            (row.clone().contiguous(),colptr.clone().contiguous()),
            perm.clone().contiguous())


def check_full_fusedgat(device):
    torch.manual_seed(7)
    N,D,H=1000,256,6; E=5000
    src=torch.randint(0,N,(E,),device=device); dst=torch.randint(0,N,(E,),device=device)
    loops=torch.arange(N,device=device)
    edge_index=torch.stack([torch.cat([src,loops]),torch.cat([dst,loops])],0)
    base=CompatibleFusedGATConv(D,D,heads=H,concat=False,dropout=0.0,add_self_loops=False,bias=True).to(device)
    fast=CompatibleFusedGATConvSideOps(D,D,heads=H,concat=False,dropout=0.0,add_self_loops=False,bias=True).to(device)
    fast.load_state_dict(base.state_dict(),strict=True); base.eval(); fast.eval()
    graph=prepare_graph(base,edge_index,N)
    x0=torch.randn(N,D,device=device); probe=torch.randn(N,D,device=device)

    x1=clone_leaf(x0); y1=F.silu(base(x1,*graph)); (y1*probe).sum().backward()
    g1=x1.grad.detach().clone(); p1={n:p.grad.detach().clone() for n,p in base.named_parameters() if p.grad is not None}
    x2=clone_leaf(x0); y2=fast(x2,*graph); (y2*probe).sum().backward()
    g2=x2.grad.detach().clone(); p2={n:p.grad.detach().clone() for n,p in fast.named_parameters() if p.grad is not None}

    print("FULL dgNN FUSEDGAT INTEGRATION (dropout=0)")
    oe=max_abs(y1,y2); ge=max_abs(g1,g2)
    print(f"  output           max_abs={oe:.8e}")
    print(f"  input_grad       max_abs={ge:.8e}")
    mp=0.0
    for name in p1:
        e=max_abs(p1[name],p2[name]); mp=max(mp,e); print(f"  {name:16s} max_abs={e:.8e}")
    return oe < 5e-5 and ge < 1e-4 and mp < 3e-2


def benchmark_attention(device, iters=30):
    torch.manual_seed(9)
    N,H,C=24492,6,256
    x=torch.randn(N,H,C,device=device,requires_grad=True)
    ats=(torch.randn(1,H,C,device=device)*0.1).requires_grad_(True)
    atd=(torch.randn(1,H,C,device=device)*0.1).requires_grad_(True)
    gs=torch.randn(N,H,device=device); gd=torch.randn(N,H,device=device)

    def legacy():
        a=(x*ats).sum(-1); b=(x*atd).sum(-1)
        return (a*gs+b*gd).sum()
    def fused():
        a,b=fused_attention_logits(x,ats,atd)
        return (a*gs+b*gd).sum()

    def timeit(fn):
        for _ in range(5):
            loss=fn(); torch.autograd.grad(loss,(x,ats,atd),retain_graph=False)
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(iters):
            loss=fn(); torch.autograd.grad(loss,(x,ats,atd),retain_graph=False)
        torch.cuda.synchronize(); return (time.perf_counter()-t0)*1000/iters

    lt=timeit(legacy); ft=timeit(fused)
    print("ATTENTION LOGITS FWD+BWD BENCHMARK (N=24492,H=6,C=256)")
    print(f"  pytorch_legacy_ms = {lt:.4f}")
    print(f"  a3v3_fused_ms     = {ft:.4f}")
    print(f"  speedup            = {lt/ft:.3f}x")


def benchmark_native_backward(device, iters=50):
    # Isolate the extension backward. This is the exact kernel family changed
    # in A3-v3 and should be much faster than the A3-v2 profiler's ~8.7 ms/call
    # (43.55 ms / 5 steps for one call per block layer aggregate).
    ext=ensure_fused_gat_sideops_loaded(False)
    torch.manual_seed(11)
    N,H,C=24492,6,256
    x=torch.randn(N,H,C,device=device)
    ats=torch.randn(1,H,C,device=device)*0.1
    atd=torch.randn(1,H,C,device=device)*0.1
    gs=torch.randn(N,H,device=device); gd=torch.randn(N,H,device=device)
    for _ in range(10):
        ext.attention_backward(gs,gd,x,ats,atd)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters):
        ext.attention_backward(gs,gd,x,ats,atd)
    torch.cuda.synchronize(); ms=(time.perf_counter()-t0)*1000/iters
    print("A3-v3 NATIVE ATTENTION BACKWARD ONLY")
    print(f"  mean_ms_per_call = {ms:.4f}")
    print("  target           = clearly below the old A3-v2 custom backward path")


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device=torch.device('cuda')
    print('='*100)
    print('A3-v3 VALIDATION: two-stage coalesced attention-parameter backward reduction')
    print('='*100)
    t0=time.perf_counter(); ensure_fused_gat_sideops_loaded(verbose=True); torch.cuda.synchronize()
    print(f'EXTENSION_BUILD_OR_LOAD: {time.perf_counter()-t0:.3f}s (outside training timing)')
    ok1=check_attention_numerics(device,4096)
    ok2=check_head_sideop(device)
    ok3=check_full_fusedgat(device)
    benchmark_attention(device)
    benchmark_native_backward(device)
    print('='*100)
    if ok1 and ok2 and ok3:
        print('RESULT: NUMERICAL ALIGNMENT LOOKS GOOD')
        print('NEXT: run trainb9_use.py and compare against A3-v2 = 0.105699 s/epoch.')
    else:
        raise RuntimeError('A3-v3 numerical validation failed; do not run formal training yet.')

if __name__=='__main__':
    main()
