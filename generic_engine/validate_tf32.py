import time
import torch


def bench_matmul(m, k, n, use_tf32, warmup=10, iters=50):
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
    w = torch.randn(k, n, device="cuda", dtype=torch.float32)
    for _ in range(warmup):
        y = x @ w
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        y = x @ w
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters, y


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print("=" * 100)
    print("A3-v4 TF32 MATMUL SMOKE BENCHMARK")
    print("torch:", torch.__version__)
    print("gpu  :", torch.cuda.get_device_name(0))
    shapes = [
        (24492, 256, 1536, "GAT linear 256->1536"),
        (24492, 1536, 256, "Clifford projection 1536->256"),
        (24492, 300, 256, "Input projection 300->256"),
    ]
    for m, k, n, name in shapes:
        fp32_ms, y_fp32 = bench_matmul(m, k, n, False)
        tf32_ms, y_tf32 = bench_matmul(m, k, n, True)
        # Compare outputs generated from different random tensors? Re-run same tensors below for accuracy check.
        torch.manual_seed(1234)
        x = torch.randn(m, k, device='cuda', dtype=torch.float32)
        w = torch.randn(k, n, device='cuda', dtype=torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        ref = x @ w
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        out = x @ w
        max_abs = (ref - out).abs().max().item()
        mean_abs = (ref - out).abs().mean().item()
        print(f"{name:36s} | FP32={fp32_ms:8.4f} ms | TF32={tf32_ms:8.4f} ms | speedup={fp32_ms/tf32_ms:6.3f}x | max_abs={max_abs:.6g} | mean_abs={mean_abs:.6g}")
    print("=" * 100)
    print("NEXT: if TF32 is faster, run trainb9_use.py and compare with A3-v3 = 0.098784 s/epoch.")


if __name__ == "__main__":
    main()
