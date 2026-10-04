"""A3-v5 grouped three-hop fixed-CSR propagation.

This module keeps the model equations unchanged for hop_scales=[1,2,3]:
    Y1 = P X
    Y2 = P Y1
    Y3 = P Y2

The optimization groups all three forward SpMMs into one native extension call,
and groups the complete chain-rule backward into one native extension call:
    T2 = G2 + P^T G3
    T1 = G1 + P^T T2
    GX = P^T T1

P/P^T CSR structures, int32 indices, cuSPARSE descriptors, and workspace are
constructed once before timed training.  This is not the earlier A1 single-SpMM
backend: the unit of execution is the entire fixed 3-hop recurrence.
"""

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
        sources=[
            str(src / "multihop_csr_pipeline.cpp"),
            str(src / "multihop_csr_pipeline_cuda.cu"),
        ],
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


class MultiHopCSR3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, operator: "MultiHopCSR3GraphOperator"):
        if x.dtype != torch.float32:
            raise TypeError(f"A3-v5 expects float32 dense features, got {x.dtype}.")
        if not x.is_contiguous():
            x = x.contiguous()
        ctx.operator = operator
        y1, y2, y3 = operator.plan.forward3(x)
        return y1, y2, y3

    @staticmethod
    def backward(ctx, grad_y1, grad_y2, grad_y3):
        # All three outputs are consumed by the 3-hop Clifford block in the
        # target configuration. Keep a defensive zero fallback for validation.
        ref = grad_y1 if grad_y1 is not None else (grad_y2 if grad_y2 is not None else grad_y3)
        if ref is None:
            return None, None
        z = None
        if grad_y1 is None:
            z = torch.zeros_like(ref)
            grad_y1 = z
        if grad_y2 is None:
            grad_y2 = torch.zeros_like(ref)
        if grad_y3 is None:
            grad_y3 = torch.zeros_like(ref)
        if not grad_y1.is_contiguous():
            grad_y1 = grad_y1.contiguous()
        if not grad_y2.is_contiguous():
            grad_y2 = grad_y2.contiguous()
        if not grad_y3.is_contiguous():
            grad_y3 = grad_y3.contiguous()
        grad_x = ctx.operator.plan.backward3(grad_y1, grad_y2, grad_y3)
        return grad_x, None


class MultiHopCSR3GraphOperator:
    __slots__ = (
        "base", "plan", "_storage_p", "_storage_pt", "shape", "nnz", "dtype",
        "device", "dense_cols", "algorithm", "info",
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

        p = _csr_parts_int32(base_fixed_csr.csr)
        pt = _csr_parts_int32(base_fixed_csr.csr_t)
        self._storage_p = p
        self._storage_pt = pt

        ext = ensure_multihop_csr_loaded()
        self.plan = ext.MultiHopCsrPlan(
            p[0], p[1], p[2],
            pt[0], pt[1], pt[2],
            int(self.shape[0]),
            self.dense_cols,
            self.algorithm,
        )
        dummy = torch.empty((self.shape[0], self.dense_cols), device=self.device, dtype=torch.float32)
        self.plan.prepare(dummy)
        torch.cuda.synchronize(self.device)

        self.info: Dict[str, object] = {
            "backend": "A3v5_MultiHopCSR3",
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


def prepare_multihop_csr_operator(base_fixed_csr, dense_cols: int, algorithm: int = 2):
    return MultiHopCSR3GraphOperator(base_fixed_csr, dense_cols=dense_cols, algorithm=algorithm)


def multihop_csr3(operator: MultiHopCSR3GraphOperator, x: torch.Tensor):
    return MultiHopCSR3Function.apply(x, operator)
