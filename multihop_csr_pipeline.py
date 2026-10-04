from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent
_EXT_NAME = "graph_clifford_multihop_csr_a3v5_v1"
_EXT = None


def ensure_multihop_csr_loaded():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError("A3-v5 MultiHopCSR requires CUDA.")
    src = _THIS_DIR / "multihop_csr_ext"
    build_dir = _THIS_DIR / ".multihop_csr_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    verbose = os.environ.get("CLIFFORD_MULTIHOP_CSR_VERBOSE", "0").lower() in {"1", "true", "yes"}
    _EXT = load(
        name=_EXT_NAME,
        sources=[str(src / "multihop_csr_pipeline.cpp"), str(src / "multihop_csr_pipeline_cuda.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=["-lcusparse"],
        with_cuda=True,
        build_directory=str(build_dir),
        verbose=verbose,
    )
    return _EXT


def _csr_parts_int32(csr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if csr.layout != torch.sparse_csr:
        raise TypeError(f"Expected sparse CSR tensor, got {csr.layout}.")
    if csr.device.type != "cuda":
        raise RuntimeError("A3-v5 CSR tensors must already be on CUDA.")
    if csr.dtype != torch.float32:
        raise TypeError(f"A3-v5 currently requires float32 sparse values, got {csr.dtype}.")
    crow = csr.crow_indices().to(dtype=torch.int32).clone().contiguous()
    col = csr.col_indices().to(dtype=torch.int32).clone().contiguous()
    val = csr.values().to(dtype=torch.float32).clone().contiguous()
    return crow, col, val


def _csr_mm(csr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=csr.dtype).contiguous()
    if x.is_cuda:
        with torch.autocast(device_type="cuda", enabled=False):
            return torch.sparse.mm(csr, x)
    return torch.sparse.mm(csr, x)


class MultiHopCSR3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, operator: "MultiHopCSR3GraphOperator"):
        ctx.input_dtype = x.dtype
        x_fp32 = x.to(dtype=torch.float32).contiguous()
        ctx.operator = operator
        y1, y2, y3 = operator.plan.forward3(x_fp32)
        return y1, y2, y3

    @staticmethod
    def backward(ctx, grad_y1, grad_y2, grad_y3):
        ref = grad_y1 if grad_y1 is not None else (grad_y2 if grad_y2 is not None else grad_y3)
        if ref is None:
            return None, None
        if grad_y1 is None:
            grad_y1 = torch.zeros_like(ref)
        if grad_y2 is None:
            grad_y2 = torch.zeros_like(ref)
        if grad_y3 is None:
            grad_y3 = torch.zeros_like(ref)
        grad_x = ctx.operator.plan.backward3(
            grad_y1.to(dtype=torch.float32).contiguous(),
            grad_y2.to(dtype=torch.float32).contiguous(),
            grad_y3.to(dtype=torch.float32).contiguous(),
        )
        if grad_x.dtype != ctx.input_dtype:
            grad_x = grad_x.to(ctx.input_dtype)
        return grad_x, None


class MultiHopCSR3GraphOperator:
    __slots__ = (
        "base", "plan", "_storage_p", "_storage_pt", "shape", "nnz", "dtype",
        "device", "dense_cols", "algorithm", "powers", "info",
    )

    def __init__(self, base_fixed_csr, dense_cols: int, algorithm: int = 2):
        if int(dense_cols) <= 0:
            raise ValueError("dense_cols must be positive.")
        if int(algorithm) not in (1, 2, 3):
            raise ValueError("A3-v5 cuSPARSE algorithm must be 1, 2, or 3.")
        if not hasattr(base_fixed_csr, "csr") or not hasattr(base_fixed_csr, "csr_t"):
            raise TypeError("A3-v5 expects an already prepared FixedCSRGraphOperator.")
        self.base = base_fixed_csr
        self.shape = tuple(base_fixed_csr.shape)
        self.nnz = int(base_fixed_csr.nnz)
        self.dtype = base_fixed_csr.dtype
        self.device = base_fixed_csr.device
        self.dense_cols = int(dense_cols)
        self.algorithm = int(algorithm)
        self.powers = (1, 2, 3)
        p = _csr_parts_int32(base_fixed_csr.csr)
        pt = _csr_parts_int32(base_fixed_csr.csr_t)
        self._storage_p = p
        self._storage_pt = pt
        ext = ensure_multihop_csr_loaded()
        self.plan = ext.MultiHopCsrPlan(
            p[0], p[1], p[2], pt[0], pt[1], pt[2],
            int(self.shape[0]), self.dense_cols, self.algorithm,
        )
        dummy = torch.empty((self.shape[0], self.dense_cols), device=self.device, dtype=torch.float32)
        self.plan.prepare(dummy)
        torch.cuda.synchronize(self.device)
        self.info: Dict[str, object] = {
            "backend": "A3v5_MultiHopCSR3_Native",
            "powers": list(self.powers),
            "shape": self.shape,
            "nnz": self.nnz,
            "dense_cols": self.dense_cols,
            "index_dtype": "torch.int32",
            "algorithm": self.algorithm,
            "workspace_bytes": int(self.plan.workspace_bytes()),
            "native_info": self.plan.info(),
        }

    @property
    def csr(self):
        return self.base.csr

    @property
    def csr_t(self):
        return self.base.csr_t


class _TailFixedCSRFunction(torch.autograd.Function):
    """One exact fixed-CSR propagation step after the native 3-hop prefix."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, csr: torch.Tensor, csr_t: torch.Tensor):
        ctx.csr_t = csr_t
        ctx.input_dtype = x.dtype
        return _csr_mm(csr, x)

    @staticmethod
    def backward(ctx, grad_y):
        grad_x = _csr_mm(ctx.csr_t, grad_y)
        if grad_x.dtype != ctx.input_dtype:
            grad_x = grad_x.to(ctx.input_dtype)
        return grad_x, None, None


class HybridNative3TailCSRGraphOperator:
    """Arbitrary-K propagation using native first 3 powers + exact CSR tail."""

    __slots__ = (
        "base", "native3", "shape", "nnz", "dtype", "device", "dense_cols",
        "algorithm", "powers", "positive_powers", "max_power", "info",
    )

    def __init__(self, base_fixed_csr, dense_cols: int, powers, algorithm: int = 2):
        powers = tuple(sorted(set(int(p) for p in powers)))
        if not powers or min(powers) < 0:
            raise ValueError(f"powers must be non-empty non-negative integers, got {powers}")
        positive = tuple(p for p in powers if p > 0)
        if not positive or max(positive) < 3:
            raise ValueError(
                "HybridNative3TailCSR requires at least one positive requested power >= 3; "
                f"got powers={powers}"
            )

        self.base = base_fixed_csr
        self.native3 = MultiHopCSR3GraphOperator(
            base_fixed_csr, dense_cols=dense_cols, algorithm=algorithm
        )
        self.shape = tuple(base_fixed_csr.shape)
        self.nnz = int(base_fixed_csr.nnz)
        self.dtype = base_fixed_csr.dtype
        self.device = base_fixed_csr.device
        self.dense_cols = int(dense_cols)
        self.algorithm = int(algorithm)
        self.powers = powers
        self.positive_powers = positive
        self.max_power = max(positive)
        self.info = {
            "backend": "HybridNative3TailCSR",
            "powers": list(self.powers),
            "shape": self.shape,
            "nnz": self.nnz,
            "dense_cols": self.dense_cols,
            "algorithm": self.algorithm,
            "prefix": "A3v5_MultiHopCSR3_Native",
            "tail_steps": max(0, self.max_power - 3),
            "implementation": "native forward3/backward3 prefix + exact FixedCSR tail",
            "native_info": self.native3.info,
        }

    @property
    def csr(self):
        return self.base.csr

    @property
    def csr_t(self):
        return self.base.csr_t


def hybrid_native3_tail_csr(operator: "HybridNative3TailCSRGraphOperator", x: torch.Tensor):
    y1, y2, y3 = multihop_csr3(operator.native3, x)
    requested = set(operator.positive_powers)
    out = {}
    if 1 in requested:
        out[1] = y1
    if 2 in requested:
        out[2] = y2
    if 3 in requested:
        out[3] = y3

    current = y3
    for power in range(4, operator.max_power + 1):
        current = _TailFixedCSRFunction.apply(current, operator.csr, operator.csr_t)
        if power in requested:
            out[power] = current
    return out


class GenericMultiHopCSRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, operator: "GenericMultiHopCSRGraphOperator"):
        ctx.operator = operator
        ctx.input_dtype = x.dtype
        requested = operator.positive_powers
        current = x.to(dtype=operator.dtype).contiguous()
        outputs = []
        wanted = set(requested)
        for p in range(1, operator.max_power + 1):
            current = _csr_mm(operator.csr, current)
            if p in wanted:
                outputs.append(current)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        op = ctx.operator
        if not grad_outputs:
            return None, None
        ref = next((g for g in grad_outputs if g is not None), None)
        if ref is None:
            return None, None
        grad_by_power = {p: g for p, g in zip(op.positive_powers, grad_outputs) if g is not None}
        acc = torch.zeros_like(ref, dtype=op.dtype)
        for p in range(op.max_power, 0, -1):
            g = grad_by_power.get(p)
            if g is not None:
                acc = acc + g.to(dtype=op.dtype).contiguous()
            acc = _csr_mm(op.csr_t, acc)
        if acc.dtype != ctx.input_dtype:
            acc = acc.to(ctx.input_dtype)
        return acc, None


class GenericMultiHopCSRGraphOperator:
    __slots__ = (
        "base", "shape", "nnz", "dtype", "device", "dense_cols", "algorithm",
        "powers", "positive_powers", "max_power", "info",
    )

    def __init__(self, base_fixed_csr, dense_cols: int, powers, algorithm: int = 2):
        if not hasattr(base_fixed_csr, "csr") or not hasattr(base_fixed_csr, "csr_t"):
            raise TypeError("GenericMultiHopCSR expects an already prepared FixedCSRGraphOperator.")
        powers = tuple(sorted(set(int(p) for p in powers)))
        if not powers or min(powers) < 0:
            raise ValueError(f"powers must be non-empty non-negative integers, got {powers}")
        self.base = base_fixed_csr
        self.shape = tuple(base_fixed_csr.shape)
        self.nnz = int(base_fixed_csr.nnz)
        self.dtype = base_fixed_csr.dtype
        self.device = base_fixed_csr.device
        self.dense_cols = int(dense_cols)
        self.algorithm = int(algorithm)
        self.powers = powers
        self.positive_powers = tuple(p for p in powers if p > 0)
        self.max_power = max(self.positive_powers) if self.positive_powers else 0
        self.info = {
            "backend": "GenericMultiHopCSR_Autograd",
            "powers": list(self.powers),
            "shape": self.shape,
            "nnz": self.nnz,
            "dense_cols": self.dense_cols,
            "value_dtype": str(self.dtype),
            "implementation": "cached CSR recurrence + single custom autograd node",
        }

    @property
    def csr(self):
        return self.base.csr

    @property
    def csr_t(self):
        return self.base.csr_t


def prepare_multihop_csr_operator(
    base_fixed_csr,
    dense_cols: int,
    algorithm: int = 2,
    powers=(1, 2, 3),
    prefer_native_three_hop: bool = True,
    strategy: str = "auto",
):
    pset = tuple(sorted(set(int(p) for p in powers)))
    positive = tuple(p for p in pset if p > 0)
    max_power = max(positive) if positive else 0
    strategy = str(strategy).strip().lower()

    if strategy not in {"auto", "native3_tail", "generic"}:
        raise ValueError(f"Unknown MultiHop strategy={strategy!r}")

    if prefer_native_three_hop and pset == (1, 2, 3):
        return MultiHopCSR3GraphOperator(
            base_fixed_csr, dense_cols=dense_cols, algorithm=algorithm
        )

    if strategy in {"auto", "native3_tail"} and max_power >= 3:
        return HybridNative3TailCSRGraphOperator(
            base_fixed_csr, dense_cols=dense_cols, powers=pset, algorithm=algorithm
        )

    return GenericMultiHopCSRGraphOperator(
        base_fixed_csr, dense_cols=dense_cols, powers=pset, algorithm=algorithm
    )


def multihop_csr3(operator: MultiHopCSR3GraphOperator, x: torch.Tensor):
    return MultiHopCSR3Function.apply(x, operator)


def generic_multihop_csr(operator: GenericMultiHopCSRGraphOperator, x: torch.Tensor):
    if not operator.positive_powers:
        return {}
    vals = GenericMultiHopCSRFunction.apply(x, operator)
    if len(operator.positive_powers) == 1 and torch.is_tensor(vals):
        vals = (vals,)
    return {p: y for p, y in zip(operator.positive_powers, vals)}
