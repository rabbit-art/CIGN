from __future__ import annotations

import time
import torch

from collapsed_gat import ensure_collapsed_gat_loaded
from fused_gat_sideops import ensure_fused_gat_sideops_loaded
from model import CompatibleFusedGATConvSideOps, CompatibleCollapsedFusedGATConv


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def max_rel(a: torch.Tensor, b: torch.Tensor, floor: float = 1e-4) -> float:
    denom = torch.maximum(a.abs(), b.abs()).clamp_min(floor)
    return float(((a - b).abs() / denom).max().item())


def clone_leaf(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


def prepare_graph(conv, edge_index: torch.Tensor, n: int):
    # Match the production model's PyG source->target -> dgNN target-row conversion.
    ei = edge_index.flip(0).contiguous()
    csr, csc, perm = conv.to_graph_format(ei, size=(n, n))
    rowptr, col = csr
    row, colptr = csc

    # dgNN graph conversion normally returns int32. Force the exact dtype expected
    # by both the official dgNN op and the V6 custom CUDA wrapper, while also
    # guaranteeing zero-offset independent storage.
    rowptr = rowptr.to(torch.int32).clone().contiguous()
    col = col.to(torch.int32).clone().contiguous()
    row = row.to(torch.int32).clone().contiguous()
    colptr = colptr.to(torch.int32).clone().contiguous()
    perm = perm.to(torch.int32).clone().contiguous()
    return ((rowptr, col), (row, colptr), perm)


def make_graph(device, n: int, extra_edges: int):
    # Add one self edge per node so every destination row is non-empty.
    src = torch.randint(0, n, (extra_edges,), device=device)
    dst = torch.randint(0, n, (extra_edges,), device=device)
    loops = torch.arange(n, device=device)
    return torch.stack(
        [torch.cat([src, loops]), torch.cat([dst, loops])], dim=0
    ).contiguous()


def clear_grads(module):
    for p in module.parameters():
        p.grad = None


def numerical_integration(device):
    torch.manual_seed(7001)
    N, D, H = 2048, 256, 6
    edge_index = make_graph(device, N, extra_edges=N * 8)

    old = CompatibleFusedGATConvSideOps(
        D, D, heads=H, concat=False, dropout=0.0,
        add_self_loops=False, bias=True,
    ).to(device)
    new = CompatibleCollapsedFusedGATConv(
        D, D, heads=H, concat=False, dropout=0.0,
        add_self_loops=False, bias=True,
    ).to(device)
    new.load_state_dict(old.state_dict(), strict=True)
    old.eval(); new.eval()
    graph = prepare_graph(old, edge_index, N)

    x0 = torch.randn(N, D, device=device, dtype=torch.float32)
    probe = torch.randn(N, D, device=device, dtype=torch.float32)

    x1 = clone_leaf(x0)
    clear_grads(old)
    y1 = old(x1, *graph)
    (y1 * probe).sum().backward()
    xg1 = x1.grad.detach().clone()
    pg1 = {name: p.grad.detach().clone() for name, p in old.named_parameters() if p.grad is not None}

    x2 = clone_leaf(x0)
    clear_grads(new)
    y2 = new(x2, *graph)
    (y2 * probe).sum().backward()
    xg2 = x2.grad.detach().clone()
    pg2 = {name: p.grad.detach().clone() for name, p in new.named_parameters() if p.grad is not None}

    print("A3-v6 FULL GAT NUMERICS (dropout=0)")
    oe = max_abs(y1, y2)
    ore = max_rel(y1, y2)
    xge = max_abs(xg1, xg2)
    xgre = max_rel(xg1, xg2)
    print(f"  output           max_abs={oe:.8e} max_rel={ore:.8e}")
    print(f"  input_grad       max_abs={xge:.8e} max_rel={xgre:.8e}")

    max_param = 0.0
    for name in pg1:
        if name not in pg2:
            raise RuntimeError(f"V6 missing gradient for parameter {name}")
        e = max_abs(pg1[name], pg2[name])
        re = max_rel(pg1[name], pg2[name])
        max_param = max(max_param, e)
        print(f"  {name:16s} max_abs={e:.8e} max_rel={re:.8e}")

    # Different reduction/atomic ordering is expected. These are deliberately
    # much tighter than the model-scale training tolerance but not bitwise tests.
    ok = (oe < 5e-4 and xge < 2e-3 and max_param < 5e-2)
    return ok


def _time_module(module, graph, x0, iters: int, warmup: int):
    module.train()
    # Reuse the same leaf allocation; each forward still creates a fresh graph.
    # This keeps the diagnostic focused on GAT forward/backward rather than a
    # 24k x 256 tensor clone/allocation per iteration.
    x = x0.detach().clone().requires_grad_(True)

    def one_iter():
        clear_grads(module)
        x.grad = None
        y = module(x, *graph)
        loss = y.square().mean()
        loss.backward()

    for _ in range(warmup):
        one_iter()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        one_iter()
        ends[i].record()
    torch.cuda.synchronize()
    vals = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    return sum(vals) / len(vals)


def benchmark_full_gat(device):
    torch.manual_seed(7002)
    N, D, H = 24492, 256, 6
    # Aim at the production graph's ~210k edge scale including one loop/node.
    edge_index = make_graph(device, N, extra_edges=186100)

    old = CompatibleFusedGATConvSideOps(
        D, D, heads=H, concat=False, dropout=0.2,
        add_self_loops=False, bias=True,
    ).to(device)
    new = CompatibleCollapsedFusedGATConv(
        D, D, heads=H, concat=False, dropout=0.2,
        add_self_loops=False, bias=True,
    ).to(device)
    new.load_state_dict(old.state_dict(), strict=True)
    graph = prepare_graph(old, edge_index, N)
    x0 = torch.randn(N, D, device=device, dtype=torch.float32)

    # Keep current production TF32 setting for the linear projection GEMM.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    old_ms = _time_module(old, graph, x0, iters=8, warmup=3)
    new_ms = _time_module(new, graph, x0, iters=8, warmup=3)
    speedup = old_ms / new_ms
    reduction = (old_ms - new_ms) / old_ms * 100.0

    print("A3-v5 vs A3-v6 FULL GAT FWD+BWD BENCHMARK")
    print(f"  graph_nodes       = {N}")
    print(f"  graph_edges       = {edge_index.size(1)}")
    print(f"  old_a3v5_ms       = {old_ms:.4f}")
    print(f"  new_a3v6_ms       = {new_ms:.4f}")
    print(f"  speedup            = {speedup:.3f}x")
    print(f"  reduction          = {reduction:.2f}%")
    return speedup


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    print("=" * 108)
    print("A3-v6 VALIDATION: collapsed-head concat=False GAT core")
    print("=" * 108)

    t0 = time.perf_counter()
    ensure_fused_gat_sideops_loaded(verbose=False)
    ensure_collapsed_gat_loaded(verbose=True)
    torch.cuda.synchronize()
    print(f"EXTENSION_BUILD_OR_LOAD: {time.perf_counter() - t0:.3f}s (outside training timing)")

    ok = numerical_integration(device)
    speedup = benchmark_full_gat(device)

    print("=" * 108)
    if not ok:
        raise RuntimeError("A3-v6 numerical validation failed; do NOT run formal training.")

    print("RESULT: NUMERICAL ALIGNMENT LOOKS GOOD")
    if speedup > 1.05:
        print("BENCHMARK: V6 GAT is faster; formal one-split training is worth running.")
    else:
        print("BENCHMARK: V6 GAT speedup is small/non-positive; send this output before formal training.")
    print("REFERENCE FULL-TRAIN BASELINE: A3-v5 = 0.087593 s/epoch")


if __name__ == "__main__":
    main()
