import time
import torch
import torch.nn.functional as F

from fused_clifford_pack import ensure_fused_clifford_pack_loaded, fused_clifford_pack


def legacy_pack(H, C, Ps, alpha):
    D = H.size(1)
    feats = []
    for P in Ps:
        TH, TC = P.split(D, dim=-1)
        z = H * TC
        D_s = F.silu(z)
        W_s = z - TH * C
        feats.append(torch.cat([W_s, D_s], dim=-1))
    x = torch.stack(feats, dim=1)
    x = x * alpha.unsqueeze(-1)
    return x.reshape(H.size(0), -1)


def stat(name, a, b):
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = b.abs().clamp_min(1e-7)
    max_rel = float((diff / denom).max().item())
    print(f"{name:<28s} max_abs={max_abs:.8e} mean_abs={mean_abs:.8e} max_rel={max_rel:.8e}")
    return max_abs, mean_abs


def make_leaf_like(x):
    return x.detach().clone().requires_grad_(True)


def numerical_validation():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = "cuda"
    N, D = 513, 256

    H0 = torch.randn(N, D, device=device, dtype=torch.float32) * 0.3
    C0 = torch.randn(N, D, device=device, dtype=torch.float32) * 0.3
    P0s = [torch.randn(N, 2 * D, device=device, dtype=torch.float32) * 0.25 for _ in range(3)]
    logits0 = torch.randn(N, 3, device=device, dtype=torch.float32) * 0.2
    weight0 = torch.randn(D, 6 * D, device=device, dtype=torch.float32) * 0.02
    bias0 = torch.randn(D, device=device, dtype=torch.float32) * 0.01
    probe = torch.randn(N, D, device=device, dtype=torch.float32)

    # Legacy branch.
    H1, C1 = make_leaf_like(H0), make_leaf_like(C0)
    Ps1 = [make_leaf_like(p) for p in P0s]
    logits1 = make_leaf_like(logits0)
    W1, b1 = make_leaf_like(weight0), make_leaf_like(bias0)
    alpha1 = torch.softmax(logits1, dim=-1)
    raw1 = legacy_pack(H1, C1, Ps1, alpha1)
    out1 = F.linear(raw1, W1, b1)
    loss1 = (out1 * probe).sum() / float(N * D)
    loss1.backward()

    # A3 branch.
    H2, C2 = make_leaf_like(H0), make_leaf_like(C0)
    Ps2 = [make_leaf_like(p) for p in P0s]
    logits2 = make_leaf_like(logits0)
    W2, b2 = make_leaf_like(weight0), make_leaf_like(bias0)
    alpha2 = torch.softmax(logits2, dim=-1)
    raw2 = fused_clifford_pack(H2, C2, Ps2, alpha2)
    out2 = F.linear(raw2, W2, b2)
    loss2 = (out2 * probe).sum() / float(N * D)
    loss2.backward()
    torch.cuda.synchronize()

    print("=" * 100)
    print("A3 NUMERICAL ALIGNMENT: legacy Clifford pack vs fused CUDA pack")
    print("=" * 100)
    results = []
    results.append(stat("G_raw", raw2, raw1)[0])
    results.append(stat("projected output", out2, out1)[0])
    results.append(stat("grad H", H2.grad, H1.grad)[0])
    results.append(stat("grad C", C2.grad, C1.grad)[0])
    for i in range(3):
        results.append(stat(f"grad propagated[{i}]", Ps2[i].grad, Ps1[i].grad)[0])
    results.append(stat("grad gate logits", logits2.grad, logits1.grad)[0])
    results.append(stat("grad projection weight", W2.grad, W1.grad)[0])
    results.append(stat("grad projection bias", b2.grad, b1.grad)[0])
    print(f"loss legacy={loss1.item():.10f} | fused={loss2.item():.10f}")

    # Forward is expected to be almost bitwise-close; alpha reduction in backward
    # can differ slightly because the CUDA kernel uses a parallel reduction.
    ok = all(torch.isfinite(t).all().item() for t in [raw2, out2, H2.grad, C2.grad, logits2.grad, W2.grad, b2.grad])
    ok = ok and max(results) <= 5e-4
    print("RESULT:", "NUMERICAL ALIGNMENT LOOKS GOOD" if ok else "NUMERICAL ALIGNMENT FAILED")
    if not ok:
        raise SystemExit(2)


def _timed_tail(use_fused, N=24492, D=256, warmup=5, iters=12):
    torch.manual_seed(123)
    H = (torch.randn(N, D, device="cuda") * 0.2).requires_grad_(True)
    C = (torch.randn(N, D, device="cuda") * 0.2).requires_grad_(True)
    Ps = [(torch.randn(N, 2 * D, device="cuda") * 0.2).requires_grad_(True) for _ in range(3)]
    gate_logits = (torch.randn(N, 3, device="cuda") * 0.1).requires_grad_(True)
    W = (torch.randn(D, 6 * D, device="cuda") * 0.01).requires_grad_(True)
    b = torch.zeros(D, device="cuda", requires_grad=True)
    probe = torch.randn(N, D, device="cuda")
    leaves = [H, C, *Ps, gate_logits, W, b]

    def one():
        for t in leaves:
            t.grad = None
        alpha = torch.softmax(gate_logits, dim=-1)
        raw = fused_clifford_pack(H, C, Ps, alpha) if use_fused else legacy_pack(H, C, Ps, alpha)
        y = F.linear(raw, W, b)
        loss = (y * probe).mean()
        loss.backward()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        one()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def benchmark():
    print("\n" + "=" * 100)
    print("A3 SYNTHETIC TAIL BENCHMARK (N=24492, D=256, 3 hops; forward+projection+backward)")
    print("This is diagnostic only. Final decision must use trainb9.py full-epoch timing.")
    print("=" * 100)
    legacy_ms = _timed_tail(False)
    fused_ms = _timed_tail(True)
    speedup = legacy_ms / fused_ms if fused_ms > 0 else float("inf")
    reduction = 100.0 * (legacy_ms - fused_ms) / legacy_ms
    print(f"legacy tail : {legacy_ms:.4f} ms")
    print(f"A3 tail     : {fused_ms:.4f} ms")
    print(f"tail speedup: {speedup:.3f}x | reduction={reduction:.2f}%")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print("=" * 100)
    print("A3 FUSED CLIFFORD PACK VALIDATION")
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.get_device_name(0))
    print("Building/loading CUDA extension (outside training timing)...")
    t0 = time.perf_counter()
    ensure_fused_clifford_pack_loaded(verbose=True)
    torch.cuda.synchronize()
    print(f"extension ready in {time.perf_counter()-t0:.3f}s")
    numerical_validation()
    benchmark()
    print("\nNEXT: if validation passed, run trainb9_use.py and compare against 0.125917 s/epoch.")


if __name__ == "__main__":
    main()
