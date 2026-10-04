import torch
from model import prepare_fixed_csr_operator, FixedCSRSpMMFunction


def stats(name, a, b):
    d = (a.detach() - b.detach()).abs()
    print(f"{name:24s} max_abs={d.max().item():.8e} mean_abs={d.mean().item():.8e}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation.")
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    n = 2000
    e = 16000
    f = 512
    row = torch.randint(0, n, (e,), device=device)
    col = torch.randint(0, n, (e,), device=device)
    val = torch.rand(e, device=device)
    coo = torch.sparse_coo_tensor(torch.stack([row, col]), val, (n, n), device=device).coalesce()

    fixed = prepare_fixed_csr_operator(coo)
    x0 = torch.randn(n, f, device=device)
    x_coo = x0.clone().requires_grad_(True)
    x_csr = x0.clone().requires_grad_(True)

    y_coo = torch.sparse.mm(coo, x_coo)
    y_csr = FixedCSRSpMMFunction.apply(x_csr, fixed.csr, fixed.csr_t)
    stats("forward", y_coo, y_csr)

    g = torch.randn_like(y_coo)
    y_coo.backward(g)
    y_csr.backward(g)
    stats("input gradient", x_coo.grad, x_csr.grad)

    print("Fixed CSR info:", fixed.info)
    fwd = (y_coo - y_csr).abs().max().item()
    bwd = (x_coo.grad - x_csr.grad).abs().max().item()
    if fwd < 1e-4 and bwd < 1e-4:
        print("RESULT: NUMERICAL ALIGNMENT LOOKS GOOD")
    else:
        print("RESULT: DIFFERENCE TOO LARGE; DO NOT BENCHMARK FULL MODEL YET")


if __name__ == "__main__":
    main()
