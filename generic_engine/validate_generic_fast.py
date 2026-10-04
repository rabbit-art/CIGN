#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from trainb9 import (
    load_dataset, build_graph_operator, prepare_gat_edge_index,
    prepare_fixed_csr_operator,
)
from model import CompatibleFusedGATConvSideOps
from chunked_dgnn_gat import prepare_chunked_dgnn_graph
from multihop_csr_pipeline import prepare_multihop_csr_operator, generic_multihop_csr
from fused_clifford_pack import fused_clifford_pack, ensure_fused_clifford_pack_loaded
from fused_gat_sideops import ensure_fused_gat_sideops_loaded


def str2bool(v):
    if isinstance(v, bool): return v
    return str(v).lower() in {"1","true","yes","y"}


def lin_of(m):
    if getattr(m, "lin", None) is not None: return m.lin
    if getattr(m, "lin_src", None) is not None: return m.lin_src
    raise RuntimeError("No homogeneous GAT linear")


def copy_gat(src, dst):
    with torch.no_grad():
        lin_of(dst).weight.copy_(lin_of(src).weight)
        dst.att_src.copy_(src.att_src)
        dst.att_dst.copy_(src.att_dst)
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)


def err(a, b):
    if a is None or b is None:
        return {"finite": False, "max_abs": None, "rel_l2": None}
    a=a.detach().float(); b=b.detach().float()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "max_abs": None, "rel_l2": None}
    d=a-b
    return {
        "finite": True,
        "max_abs": float(d.abs().max().item()) if d.numel() else 0.0,
        "rel_l2": float(d.norm().item()/(b.norm().item()+1e-12)),
    }


def aligned(stats, rel=1e-4, abs_=1e-4):
    return all(v["finite"] and v["rel_l2"] <= rel and v["max_abs"] <= abs_ for v in stats.values())


def bench_cuda(fn, params, x, warmup=3, iters=10):
    for _ in range(warmup):
        for p in params:
            p.grad = None
        x.grad = None
        y=fn(x); loss=y.square().mean(); loss.backward()
    torch.cuda.synchronize()
    vals=[]
    for _ in range(iters):
        for p in params:
            p.grad = None
        x.grad = None
        st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True)
        st.record(); y=fn(x); loss=y.square().mean(); loss.backward(); en.record(); en.synchronize()
        vals.append(st.elapsed_time(en))
    vals=sorted(vals)
    return {"median_ms": float(vals[len(vals)//2]), "mean_ms": float(sum(vals)/len(vals)), "all_ms": vals}


def validate_gat(edge_index, n, hidden, heads, device, chunk_max, chunk_targets, dropout, warmup, iters):
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    ensure_fused_gat_sideops_loaded(False)
    ref=GATConv(hidden,hidden,heads=heads,concat=False,dropout=0.0,add_self_loops=False,bias=True).to(device)
    cand=CompatibleFusedGATConvSideOps(hidden,hidden,heads=heads,concat=False,dropout=0.0,add_self_loops=False,bias=True).to(device)
    copy_gat(ref,cand)
    graph=prepare_chunked_dgnn_graph(cand,edge_index,n,chunk_max,chunk_targets)
    x0=torch.randn(n,hidden,device=device)*0.1
    xr=x0.detach().clone().requires_grad_(True); xc=x0.detach().clone().requires_grad_(True)
    yr=F.silu(ref(xr,edge_index)); yc=cand(xc,graph,None,None)
    probe=torch.randn_like(yr)
    (yr*probe).sum().backward(); (yc*probe).sum().backward()
    stats={
        "out":err(yc,yr), "xgrad":err(xc.grad,xr.grad),
        "wgrad":err(lin_of(cand).weight.grad,lin_of(ref).weight.grad),
        "asgrad":err(cand.att_src.grad,ref.att_src.grad),
        "adgrad":err(cand.att_dst.grad,ref.att_dst.grad),
        "bgrad":err(cand.bias.grad,ref.bias.grad),
    }
    ok=aligned(stats)

    refb=GATConv(hidden,hidden,heads=heads,concat=False,dropout=dropout,add_self_loops=False,bias=True).to(device)
    candb=CompatibleFusedGATConvSideOps(hidden,hidden,heads=heads,concat=False,dropout=dropout,add_self_loops=False,bias=True).to(device)
    copy_gat(refb,candb)
    xb=torch.randn(n,hidden,device=device,requires_grad=True)*0.1
    xb=xb.detach().requires_grad_(True)
    ref_b=bench_cuda(lambda z:F.silu(refb(z,edge_index)),list(refb.parameters()),xb,warmup,iters)
    cand_b=bench_cuda(lambda z:candb(z,graph,None,None),list(candb.parameters()),xb,warmup,iters)
    speed=ref_b["median_ms"]/cand_b["median_ms"] if cand_b["median_ms"]>0 else 0.0
    return {"aligned":ok,"errors":stats,"graph_info":graph.info,"pyg":ref_b,"chunked_dgnn":cand_b,"speedup_vs_pyg":speed}


def validate_multihop(fixed, n, hidden, powers, device):
    gen=prepare_multihop_csr_operator(fixed,dense_cols=2*hidden,powers=powers,prefer_native_three_hop=False)
    x0=torch.randn(n,2*hidden,device=device)*0.1
    xr=x0.detach().clone().requires_grad_(True); xc=x0.detach().clone().requires_grad_(True)
    req=sorted(set(int(p) for p in powers if int(p)>0))
    cur=xr; ro={}
    for p in range(1,max(req)+1):
        cur=torch.sparse.mm(fixed.csr,cur)
        if p in req: ro[p]=cur
    co=generic_multihop_csr(gen,xc)
    probes={p:torch.randn_like(ro[p]) for p in req}
    sum((ro[p]*probes[p]).sum() for p in req).backward()
    sum((co[p]*probes[p]).sum() for p in req).backward()
    stats={f"out_p{p}":err(co[p],ro[p]) for p in req}; stats["xgrad"]=err(xc.grad,xr.grad)
    return {"aligned":aligned(stats),"errors":stats,"info":gen.info}


def legacy_pack(H,C,props,alpha):
    feats=[]
    for p in props:
        D=H.size(1); TH,TC=p.split(D,dim=-1); u=H*TC
        feats.append(torch.cat([u-TH*C,F.silu(u)],dim=-1))
    x=torch.stack(feats,dim=1)
    if alpha is not None: x=x*alpha.unsqueeze(-1)
    return x.reshape(H.size(0),-1)


def validate_pack(hidden, powers, device):
    N=2048; K=len(powers)
    ensure_fused_clifford_pack_loaded(False,hop_count=K)
    H0=torch.randn(N,hidden,device=device)*0.1; C0=torch.randn(N,hidden,device=device)*0.1
    P0=[torch.randn(N,2*hidden,device=device)*0.1 for _ in range(K)]
    A0=torch.softmax(torch.randn(N,K,device=device),dim=-1)
    Hr=H0.detach().clone().requires_grad_(True); Hc=H0.detach().clone().requires_grad_(True)
    Cr=C0.detach().clone().requires_grad_(True); Cc=C0.detach().clone().requires_grad_(True)
    Pr=[x.detach().clone().requires_grad_(True) for x in P0]; Pc=[x.detach().clone().requires_grad_(True) for x in P0]
    Ar=A0.detach().clone().requires_grad_(True); Ac=A0.detach().clone().requires_grad_(True)
    yr=legacy_pack(Hr,Cr,Pr,Ar); yc=fused_clifford_pack(Hc,Cc,Pc,Ac)
    probe=torch.randn_like(yr); (yr*probe).sum().backward(); (yc*probe).sum().backward()
    stats={"out":err(yc,yr),"Hgrad":err(Hc.grad,Hr.grad),"Cgrad":err(Cc.grad,Cr.grad),"Agrad":err(Ac.grad,Ar.grad)}
    for i,(a,b) in enumerate(zip(Pc,Pr)): stats[f"P{i}grad"]=err(a.grad,b.grad)
    return {"aligned":aligned(stats,rel=3e-4,abs_=3e-4),"errors":stats,"K":K,"D":hidden}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_name",default='Amazon-ratings')
    ap.add_argument("--data_root",default="./data")
    ap.add_argument("--use_undirected",type=str2bool,default=True)
    ap.add_argument("--hidden_dim",type=int,default=256)
    ap.add_argument("--gat_heads",type=int,default=6)
    ap.add_argument("--gat_dropout",type=float,default=0.32)
    ap.add_argument("--hop_scales",type=int,nargs="+",default=[1, 2, 3])
    ap.add_argument("--operator_mode",default="adjacency",choices=["adjacency","laplacian","hybrid"])
    ap.add_argument("--hybrid_alpha",type=float,default=0.5)
    ap.add_argument("--chunk_max_nodes",type=int,default=45000)
    ap.add_argument("--chunk_target_nodes",type=int,default=8192)
    ap.add_argument("--warmup",type=int,default=2)
    ap.add_argument("--bench_iters",type=int,default=5)
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--output",default="generic_fast_validation.json")
    args=ap.parse_args()
    device=torch.device(args.device); torch.cuda.set_device(device.index or 0)
    ds,data=load_dataset(args.dataset_name,args.data_root)
    ei,op=build_graph_operator(data.edge_index,data.num_nodes,args.use_undirected,args.operator_mode,args.hybrid_alpha)
    ei=prepare_gat_edge_index(ei,data.num_nodes).to(device); op=op.to(device)
    fixed=prepare_fixed_csr_operator(op)
    result={
        "dataset":args.dataset_name,"nodes":int(data.num_nodes),"edges":int(ei.size(1)),
        "shape":{"hidden":args.hidden_dim,"heads":args.gat_heads,"hops":args.hop_scales},
    }
    print("[1/3] validating chunked dgNN GAT ...",flush=True)
    result["gat"]=validate_gat(ei,int(data.num_nodes),args.hidden_dim,args.gat_heads,device,args.chunk_max_nodes,args.chunk_target_nodes,args.gat_dropout,args.warmup,args.bench_iters)
    print("      aligned=",result["gat"]["aligned"],"speedup_vs_pyg=",f"{result['gat']['speedup_vs_pyg']:.3f}x",flush=True)
    print("[2/3] validating generic MultiHopCSR ...",flush=True)
    result["multihop"]=validate_multihop(fixed,int(data.num_nodes),args.hidden_dim,args.hop_scales,device)
    print("      aligned=",result["multihop"]["aligned"],flush=True)
    print("[3/3] validating generic CliffordPackK ...",flush=True)
    result["clifford_pack"]=validate_pack(args.hidden_dim,args.hop_scales,device)
    print("      aligned=",result["clifford_pack"]["aligned"],flush=True)
    result["all_aligned"]=bool(result["gat"]["aligned"] and result["multihop"]["aligned"] and result["clifford_pack"]["aligned"])
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("="*100)
    print("ALL_ALIGNED:",result["all_aligned"])
    print("GAT_SPEEDUP_VS_PYG:",f"{result['gat']['speedup_vs_pyg']:.3f}x")
    print("JSON:",Path(args.output).resolve())
    print("="*100)

if __name__=="__main__": main()
