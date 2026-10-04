# ============================================================
# model.py
# Graph Clifford model with optional layer-wise Dirichlet recording.
#
# This version keeps the V1-8 model framework and adds an optional dgNN/PyG
# FusedGAT backend plus fixed-CSR SpMM, A3-v1 fused Clifford packing, and A3-v2 fused GAT side operations.  The original V1-8 optimizations remain: recurrent multi-hop sparse propagation and one-time
# GAT self-loop preparation in trainb9.py. Dirichlet recording remains optional.
#
# Environment variables used by this file:
#   CLIFFORD_RECORD_DIRICHLET=1
#   CLIFFORD_DIRICHLET_JSONL=/path/to/save.jsonl
#   CLIFFORD_DIRICHLET_RECORD_ONLY_EVAL=1       # default: 1
#   CLIFFORD_DIRICHLET_RECORD_EVERY=1           # record every N eval forwards
#   CLIFFORD_DIRICHLET_NORMALIZE=1              # default: 1
#
# It writes JSONL rows like:
#   {"eval_call": 1, "num_layers": 8, "energies": [E_H0, E_H1, ...]}
# ============================================================

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, FusedGATConv

from fused_clifford_pack import fused_clifford_pack
from fused_clifford_switch_pack import fused_clifford_switch_pack
from fused_clifford_cign_pack import fused_clifford_switch_pack as cign_pack3
from fused_clifford_cign_generic import fused_clifford_pack as cign_pack_k
from fused_gat_sideops import fused_attention_logits, fused_head_mean_bias_silu
from fused_block_ops import fused_dropout_layernorm, fused_add_layernorm, fused_dropout_gamma_residual
from collapsed_gat import collapsed_gat_mean_bias_silu
from multihop_csr_pipeline import (
    MultiHopCSR3GraphOperator,
    GenericMultiHopCSRGraphOperator,
    HybridNative3TailCSRGraphOperator,
    multihop_csr3,
    generic_multihop_csr,
    hybrid_native3_tail_csr,
)
from chunked_dgnn_gat import ChunkedDgNNGraph, prepare_chunked_dgnn_graph, run_chunked_dgnn_op


def cign_outer_difference(b_hr, b_rh, mode):
    """Only the outer-difference nonlinearity varies; retain the 0.5 factor."""
    raw = 0.5 * (b_hr - b_rh)
    if mode == "tanh":
        return torch.tanh(raw)
    if mode == "none":
        return raw
    raise ValueError(f"Invalid outer nonlinearity: {mode}")


def _resolve_selective_amp_dtype(name: str):
    name = str(name).strip().lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"selective_amp_dtype must be bf16 or fp16, got {name!r}")


def _resolve_old_new_mode(explicit_value, env_name: str, default: str = "old") -> str:
    """Resolve one isolated CIGN ablation axis.

    Priority:
      1) explicit constructor/factory argument
      2) environment variable
      3) default

    Valid values: "old" / "new".
    """
    if explicit_value is None:
        explicit_value = os.environ.get(env_name, default)
    value = str(explicit_value).strip().lower()
    if value not in {"old", "new"}:
        raise ValueError(
            f"{env_name} must be 'old' or 'new', got {explicit_value!r}"
        )
    return value


def _local_amp_linear(linear: nn.Linear, x: torch.Tensor, enabled: bool, amp_dtype: torch.dtype) -> torch.Tensor:
    """Run only one dense Linear under AMP, then return FP32.

    This is deliberately *local* autocast. Sparse SpMM, attention reductions,
    LayerNorm, Clifford CUDA kernels, and all other graph operations remain
    outside autocast.  This prevents BF16/FP16 gradients from leaking into
    cuSPARSE/custom FP32 kernels during backward.
    """
    if bool(enabled) and x.is_cuda:
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=True,
            cache_enabled=True,
        ):
            y = linear(x)
        return y.float()
    return linear(x).float()


# ============================================================
# Fixed graph-operator CSR SpMM utilities
# ============================================================
class FixedCSRSpMMFunction(torch.autograd.Function):
    """SpMM for a fixed sparse graph operator.

    The graph operator is constant during training, so gradients are required
    only for the dense feature matrix X.  The forward uses a pre-built CSR(P),
    and backward directly uses a pre-built CSR(P^T):

        forward : Y      = P   @ X
        backward: grad_X = P^T @ grad_Y

    This deliberately avoids PyTorch COO sparse-autograd bookkeeping and the
    repeated COO coalesce/sort work seen in the profiler.  It supports the
    first-order gradients used by ordinary neural-network training.
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor, operator_csr: torch.Tensor, operator_t_csr: torch.Tensor):
        if operator_csr.layout != torch.sparse_csr:
            raise TypeError(f"operator_csr must be sparse CSR, got {operator_csr.layout}")
        if operator_t_csr.layout != torch.sparse_csr:
            raise TypeError(f"operator_t_csr must be sparse CSR, got {operator_t_csr.layout}")
        if operator_csr.dtype != operator_t_csr.dtype:
            raise TypeError(
                f"CSR(P) and CSR(P^T) must share dtype, got {operator_csr.dtype} and {operator_t_csr.dtype}"
            )
        ctx.operator_t_csr = operator_t_csr
        ctx.input_dtype = X.dtype

        # cuSPARSE SpMM in the current Torch/CUDA stack requires sparse and
        # dense operands to share the same value type. Keep graph SpMM FP32
        # even when neighboring dense GEMMs use BF16/FP16.
        X_spmm = X.to(dtype=operator_csr.dtype).contiguous()
        if X_spmm.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                out = torch.sparse.mm(operator_csr, X_spmm)
        else:
            out = torch.sparse.mm(operator_csr, X_spmm)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_spmm = grad_output.to(dtype=ctx.operator_t_csr.dtype).contiguous()
        if grad_spmm.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                grad_X = torch.sparse.mm(ctx.operator_t_csr, grad_spmm)
        else:
            grad_X = torch.sparse.mm(ctx.operator_t_csr, grad_spmm)
        if grad_X.dtype != ctx.input_dtype:
            grad_X = grad_X.to(ctx.input_dtype)
        return grad_X, None, None


class FixedCSRGraphOperator:
    """Small immutable cache holding CSR(P) and CSR(P^T)."""

    __slots__ = ("csr", "csr_t", "shape", "nnz", "dtype", "device", "info")

    def __init__(self, csr: torch.Tensor, csr_t: torch.Tensor):
        self.csr = csr
        self.csr_t = csr_t
        self.shape = tuple(csr.shape)
        self.nnz = int(csr.values().numel())
        self.dtype = csr.dtype
        self.device = csr.device
        self.info = {
            "layout": "csr",
            "shape": self.shape,
            "nnz": self.nnz,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "csr_crow_contiguous": bool(csr.crow_indices().is_contiguous()),
            "csr_col_contiguous": bool(csr.col_indices().is_contiguous()),
            "csr_values_contiguous": bool(csr.values().is_contiguous()),
            "csr_t_crow_contiguous": bool(csr_t.crow_indices().is_contiguous()),
            "csr_t_col_contiguous": bool(csr_t.col_indices().is_contiguous()),
            "csr_t_values_contiguous": bool(csr_t.values().is_contiguous()),
        }


def _clone_csr_storage(csr: torch.Tensor) -> torch.Tensor:
    """Rebuild a CSR tensor with independent contiguous index/value storage."""
    return torch.sparse_csr_tensor(
        csr.crow_indices().clone().contiguous(),
        csr.col_indices().clone().contiguous(),
        csr.values().clone().contiguous(),
        size=csr.shape,
        dtype=csr.dtype,
        device=csr.device,
    )


@torch.no_grad()
def prepare_fixed_csr_operator(operator: torch.Tensor) -> FixedCSRGraphOperator:
    """Convert a fixed 2-D COO/CSR graph operator into cached CSR(P), CSR(P^T).

    Any coalesce/transpose/conversion happens exactly once before epoch timing.
    The returned object is then reused by every Clifford block and every epoch.
    """
    if not isinstance(operator, torch.Tensor):
        raise TypeError(f"operator must be a torch.Tensor, got {type(operator)!r}")
    if operator.dim() != 2 or operator.size(0) != operator.size(1):
        raise ValueError(f"Expected a square 2-D graph operator, got shape={tuple(operator.shape)}")

    if operator.layout == torch.sparse_coo:
        coo = operator.coalesce()
    elif operator.layout == torch.sparse_csr:
        coo = operator.to_sparse_coo().coalesce()
    else:
        raise TypeError(
            "FixedCSRSpMM expects a sparse COO or sparse CSR graph operator; "
            f"got layout={operator.layout}."
        )

    # Build P in CSR.
    csr = _clone_csr_storage(coo.to_sparse_csr())

    # Build P^T explicitly once.  Using COO here is intentional: all sorting /
    # coalescing is outside training, while each backward thereafter is a plain
    # CSR SpMM with no gradient requested for sparse values/indices.
    idx_t = coo.indices().flip(0).contiguous()
    coo_t = torch.sparse_coo_tensor(
        idx_t,
        coo.values(),
        size=(coo.size(1), coo.size(0)),
        dtype=coo.dtype,
        device=coo.device,
    ).coalesce()
    csr_t = _clone_csr_storage(coo_t.to_sparse_csr())

    return FixedCSRGraphOperator(csr=csr, csr_t=csr_t)


def fixed_sparse_mm(operator, X: torch.Tensor) -> torch.Tensor:
    """Dispatch one P@X multiply while keeping sparse math dtype-safe."""
    if isinstance(operator, FixedCSRGraphOperator):
        return FixedCSRSpMMFunction.apply(X, operator.csr, operator.csr_t)
    if isinstance(operator, torch.Tensor):
        X_spmm = X.to(dtype=operator.dtype).contiguous()
        if X_spmm.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return torch.sparse.mm(operator, X_spmm)
        return torch.sparse.mm(operator, X_spmm)
    raise TypeError(f"Unsupported graph operator type: {type(operator)!r}")


def selected_sparse_powers(
    operator,
    X: torch.Tensor,
    powers,
):
    """
    Compute only the requested powers of a fixed sparse operator by recurrence.

    When ``operator`` is a FixedCSRGraphOperator, every recurrence step uses the
    custom fixed-CSR autograd function. Otherwise this is exactly the V1-8 COO
    sparse.mm path, which keeps the new optimization switchable for A/B tests.
    """
    requested = {int(power) for power in powers}
    if not requested:
        raise ValueError("powers cannot be empty.")
    if min(requested) < 0:
        raise ValueError(f"Graph-operator powers must be non-negative, got {sorted(requested)}.")

    outputs = {}
    if 0 in requested:
        outputs[0] = X

    # Native A3-v5 specialization remains the fastest exact {1,2,3} path.
    if isinstance(operator, MultiHopCSR3GraphOperator) and requested == {1, 2, 3}:
        y1, y2, y3 = multihop_csr3(operator, X)
        outputs[1] = y1
        outputs[2] = y2
        outputs[3] = y3
        return outputs

    # V3 arbitrary-K fast path: native first 3 powers + exact CSR tail.
    if isinstance(operator, HybridNative3TailCSRGraphOperator):
        outputs.update(hybrid_native3_tail_csr(operator, X))
        if 0 in requested:
            outputs[0] = X
        return outputs

    # Generic K fallback.
    if isinstance(operator, GenericMultiHopCSRGraphOperator):
        outputs.update(generic_multihop_csr(operator, X))
        if 0 in requested:
            outputs[0] = X
        return outputs

    current = X
    for power in range(1, max(requested) + 1):
        current = fixed_sparse_mm(operator, current)
        if power in requested:
            outputs[power] = current

    return outputs


def _unique_undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return unique undirected edges with row <= col removed self-loops."""
    if edge_index is None:
        raise ValueError("edge_index is None.")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    row, col = edge_index[0], edge_index[1]
    mask = row != col
    row = row[mask]
    col = col[mask]

    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    flat = u * int(num_nodes) + v
    flat_unique = torch.unique(flat)
    u_unique = torch.div(flat_unique, int(num_nodes), rounding_mode="floor")
    v_unique = flat_unique % int(num_nodes)
    return torch.stack([u_unique, v_unique], dim=0)


@torch.no_grad()
def dirichlet_energy_torch(
    H: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    normalize: bool = True,
) -> float:
    """
    Compute a normalized graph Dirichlet energy over unique undirected edges:
        E(H) = mean_{(i,j) in E} ||h_i - h_j||_2^2

    If normalize=True, additionally divide by mean node squared norm to reduce
    the scale effect across layers:
        E_norm = E / mean_i ||h_i||_2^2
    """
    if H.numel() == 0:
        return 0.0

    H_float = H.detach().float()
    undirected_edges = _unique_undirected_edges(edge_index.detach(), num_nodes=num_nodes)
    if undirected_edges.numel() == 0 or undirected_edges.size(1) == 0:
        return 0.0

    src = undirected_edges[0].to(H_float.device)
    dst = undirected_edges[1].to(H_float.device)

    diff = H_float[src] - H_float[dst]
    energy = (diff * diff).sum(dim=-1).mean()

    if normalize:
        denom = (H_float * H_float).sum(dim=-1).mean().clamp_min(1e-12)
        energy = energy / denom

    return float(energy.detach().cpu().item())


# ============================================================
# Optional Dirichlet recorder
# ============================================================
class DirichletRecorder:
    def __init__(self):
        self.enabled = os.environ.get("CLIFFORD_RECORD_DIRICHLET", "0") == "1"
        self.jsonl_path = os.environ.get("CLIFFORD_DIRICHLET_JSONL", "").strip()
        self.only_eval = os.environ.get("CLIFFORD_DIRICHLET_RECORD_ONLY_EVAL", "1") != "0"
        self.record_every = max(1, int(os.environ.get("CLIFFORD_DIRICHLET_RECORD_EVERY", "1")))
        self.normalize = os.environ.get("CLIFFORD_DIRICHLET_NORMALIZE", "1") != "0"
        self.eval_call = 0
        self.train_call = 0

    def maybe_record(self, states, edge_index, num_layers: int, training: bool):
        if not self.enabled or not self.jsonl_path:
            return
        if self.only_eval and training:
            return

        if training:
            self.train_call += 1
            call_idx = self.train_call
            phase = "train"
        else:
            self.eval_call += 1
            call_idx = self.eval_call
            phase = "eval"

        if call_idx % self.record_every != 0:
            return

        try:
            num_nodes = int(states[0].size(0))
            energies = [
                dirichlet_energy_torch(
                    H=H,
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    normalize=self.normalize,
                )
                for H in states
            ]
            row = {
                "phase": phase,
                "call": int(call_idx),
                "eval_call": int(self.eval_call),
                "train_call": int(self.train_call),
                "num_layers": int(num_layers),
                "num_states": int(len(states)),
                "normalize": bool(self.normalize),
                "energies": energies,
            }
            path = Path(self.jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            # Avoid breaking training because of analysis logging.
            if os.environ.get("CLIFFORD_DIRICHLET_RAISE_ON_ERROR", "0") == "1":
                raise
            print(f"[DirichletRecorder] skipped because of error: {repr(e)}")


# ============================================================
# PyG 2.8.0 + dgNN compatible fused GAT
# ============================================================
class CompatibleFusedGATConv(FusedGATConv):
    """
    Compatibility wrapper validated against PyG GATConv on the same graph and
    parameters (dropout disabled for the numerical comparison).

    PyG 2.8.0 creates ``self.lin`` for homogeneous graphs, while its
    FusedGATConv.forward still calls ``self.lin_src``.  This wrapper selects
    ``self.lin`` when present and otherwise falls back to ``self.lin_src``.

    Graph direction and CUDA-index alignment are handled once by
    GraphCliffordNet.prepare_fused_gat_graph().
    """

    def forward(self, x, csr, csc, perm):
        H, C = self.heads, self.out_channels
        if x.dim() != 2:
            raise ValueError(f"FusedGAT expects a 2-D node feature tensor, got {tuple(x.shape)}")

        amp_enabled = bool(getattr(self, "_selective_amp_enabled", False))
        amp_dtype = getattr(self, "_selective_amp_dtype", torch.bfloat16)
        if getattr(self, "lin", None) is not None:
            x = _local_amp_linear(self.lin, x, amp_enabled, amp_dtype).view(-1, H, C)
        else:
            if getattr(self, "lin_src", None) is None:
                raise RuntimeError("Neither self.lin nor self.lin_src is available in FusedGATConv.")
            x = _local_amp_linear(self.lin_src, x, amp_enabled, amp_dtype).view(-1, H, C)

        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        dropout = self.dropout if self.training else 0.0

        if isinstance(csr, ChunkedDgNNGraph):
            out = run_chunked_dgnn_op(
                self.op, x, alpha_src, alpha_dst, csr,
                self.negative_slope, dropout,
            )
        else:
            (rowptr, col), (row, colptr) = csr, csc
            out = self.op(
                alpha_dst, alpha_src, rowptr, col, colptr, row, perm,
                self.negative_slope, x, dropout,
            )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out


class CompatibleFusedGATConvSideOps(CompatibleFusedGATConv):
    """A3-v2: keep dgNN's fused message-passing op, but fuse its expensive
    PyTorch-side attention-logit reduction and concat=False head reduction.

    The returned tensor already includes the SiLU that GraphCliffordBlock
    applies immediately after GAT in the legacy/A3-v1 path.
    """

    def forward(self, x, csr, csc, perm):
        if self.concat:
            raise RuntimeError("A3-v2 GAT side-op fusion is specialized for concat=False")
        H, C = self.heads, self.out_channels
        if x.dim() != 2:
            raise ValueError(f"FusedGAT expects a 2-D node feature tensor, got {tuple(x.shape)}")

        amp_enabled = bool(getattr(self, "_selective_amp_enabled", False))
        amp_dtype = getattr(self, "_selective_amp_dtype", torch.bfloat16)
        if getattr(self, "lin", None) is not None:
            x = _local_amp_linear(self.lin, x, amp_enabled, amp_dtype).view(-1, H, C)
        else:
            if getattr(self, "lin_src", None) is None:
                raise RuntimeError("Neither self.lin nor self.lin_src is available in FusedGATConv.")
            x = _local_amp_linear(self.lin_src, x, amp_enabled, amp_dtype).view(-1, H, C)

        alpha_src, alpha_dst = fused_attention_logits(x, self.att_src, self.att_dst)
        dropout = self.dropout if self.training else 0.0
        if isinstance(csr, ChunkedDgNNGraph):
            out = run_chunked_dgnn_op(
                self.op, x, alpha_src, alpha_dst, csr,
                self.negative_slope, dropout,
            )
        else:
            (rowptr, col), (row, colptr) = csr, csc
            out = self.op(
                alpha_dst, alpha_src, rowptr, col, colptr, row, perm,
                self.negative_slope, x, dropout,
            )
        # Fuses out.mean(dim=1), bias, and the immediately following SiLU.
        return fused_head_mean_bias_silu(out, self.bias)


class CompatibleCollapsedFusedGATConv(CompatibleFusedGATConv):
    """Generalized V6: collapsed-head dgNN-style GAT for concat=False.

    A3-v3 still computes the attention logits. The message-passing CUDA core
    then performs softmax/dropout/aggregation and directly returns
    mean(heads)+bias+SiLU as [N,C], avoiding the temporary [N,H,C] GAT output
    and the matching expanded grad-output tensor in backward.
    """

    def forward(self, x, csr, csc, perm):
        if self.concat:
            raise RuntimeError("A3-v6 collapsed GAT is specialized for concat=False")
        H, C = self.heads, self.out_channels
        if x.dim() != 2:
            raise ValueError(f"FusedGAT expects a 2-D node feature tensor, got {tuple(x.shape)}")
        # Generalized fast kernel: all clean11 lightweight shapes stay on the
        # collapsed path. attention_backward uses a (32,H) CUDA block, hence H<=32.
        if not (1 <= H <= 32) or not (1 <= C <= 256):
            raise RuntimeError(
                f"Generalized collapsed GAT requires 1<=heads<=32 and 1<=channels<=256; got H={H},C={C}"
            )

        amp_enabled = bool(getattr(self, "_selective_amp_enabled", False))
        amp_dtype = getattr(self, "_selective_amp_dtype", torch.bfloat16)
        if getattr(self, "lin", None) is not None:
            x = _local_amp_linear(self.lin, x, amp_enabled, amp_dtype).view(-1, H, C)
        else:
            if getattr(self, "lin_src", None) is None:
                raise RuntimeError("Neither self.lin nor self.lin_src is available in FusedGATConv.")
            x = _local_amp_linear(self.lin_src, x, amp_enabled, amp_dtype).view(-1, H, C)

        alpha_src, alpha_dst = fused_attention_logits(x, self.att_src, self.att_dst)
        dropout = self.dropout if self.training else 0.0
        return collapsed_gat_mean_bias_silu(
            attn_row=alpha_dst,
            attn_col=alpha_src,
            csr=csr,
            csc=csc,
            perm=perm,
            negative_slope=self.negative_slope,
            in_feat=x,
            bias=self.bias,
            attn_drop=dropout,
        )


# ============================================================
# Graph Clifford Block
# ============================================================
class GraphCliffordBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        order_set=(1, 2, 3),
        dropout: float = 0.28,
        gat_heads: int = 6,
        gat_dropout: float = 0.32,
        use_fused_gat: bool = False,
        use_fused_gat_sideops: bool = False,
        use_chunked_fused_gat: bool = False,
        use_collapsed_fused_gat: bool = False,
        use_fused_clifford_pack: bool = False,
        use_fused_block_ops: bool = False,
        use_selective_amp: bool = False,
        selective_amp_dtype: str = "bf16",
        init_gamma: float = 0.49,
        gamma_mode: str = "vector",
        hop_gate_mode: str = "node",
        alpha_score_mode: str = None,
        interaction_mode: str = None,
        alpha_weighting_mode: str = None,
        use_bias: bool = True,
    ):
        super().__init__()

        if len(order_set) == 0:
            raise ValueError("order_set / hop_scales cannot be empty.")
        if any(int(s) < 0 for s in order_set):
            raise ValueError("All propagation orders must be non-negative integers.")
        if gamma_mode not in {"scalar", "vector"}:
            raise ValueError("gamma_mode must be either 'scalar' or 'vector'.")
        if hop_gate_mode not in {"none", "global", "node"}:
            raise ValueError("hop_gate_mode must be one of {'none', 'global', 'node'}.")

        self.hidden_dim = hidden_dim
        self.order_set = [int(s) for s in order_set]
        self.num_hops = len(self.order_set)
        self.dropout = dropout
        self.gamma_mode = gamma_mode
        self.hop_gate_mode = hop_gate_mode

        # Three independent old/new axes.
        #
        # 1) alpha_score_mode
        #    old: MLP([H_norm || R_norm])
        #    new: MLP([H_norm + R_norm || |H_norm - R_norm|])
        #
        # 2) interaction_mode
        #    old:
        #      D_k = SiLU(H_norm * T_R^k)
        #      W_k = H_norm * T_R^k - T_H^k * R_norm
        #    new:
        #      B_HR = H_norm * T_R^k
        #      B_RH = R_norm * T_H^k
        #      D_k = SiLU(0.5 * (B_HR + B_RH))
        #      W_k = tanh(0.5 * (B_HR - B_RH))
        #
        # 3) alpha_weighting_mode
        #    old: per-hop alpha -> Cat_k([W_k,D_k]) -> projection
        #    new: sum_k alpha_k D_k and sum_k alpha_k W_k separately,
        #         then Cat([D_agg,W_agg]) -> projection.
        self.alpha_score_mode = _resolve_old_new_mode(
            alpha_score_mode, "CIGN_ALPHA_SCORE_MODE", "new"
        )
        self.interaction_mode = _resolve_old_new_mode(
            interaction_mode, "CIGN_INTERACTION_MODE", "new"
        )
        self.alpha_weighting_mode = _resolve_old_new_mode(
            alpha_weighting_mode, "CIGN_ALPHA_WEIGHTING_MODE", "old"
        )

        self.use_fused_gat = bool(use_fused_gat)
        self.use_fused_gat_sideops = bool(use_fused_gat_sideops)
        self.use_chunked_fused_gat = bool(use_chunked_fused_gat)
        self.use_collapsed_fused_gat = bool(use_collapsed_fused_gat)
        if self.use_chunked_fused_gat and not self.use_fused_gat:
            raise ValueError("use_chunked_fused_gat=True requires use_fused_gat=True")
        if self.use_chunked_fused_gat and self.use_collapsed_fused_gat:
            raise ValueError("chunked dgNN and collapsed GAT are mutually exclusive")
        if self.use_fused_gat_sideops and not self.use_fused_gat:
            raise ValueError("use_fused_gat_sideops=True requires use_fused_gat=True")
        if self.use_collapsed_fused_gat and not self.use_fused_gat:
            raise ValueError("use_collapsed_fused_gat=True requires use_fused_gat=True")
        if self.use_collapsed_fused_gat and not self.use_fused_gat_sideops:
            raise ValueError("Generalized V6 requires use_fused_gat_sideops=True for A3-v3 attention logits")
        self.outer_nonlinearity = os.environ.get("CIGN_OUTER_NONLINEARITY", "none").strip().lower()
        if self.outer_nonlinearity not in {"tanh", "none"}:
            raise ValueError("CIGN_OUTER_NONLINEARITY must be tanh or none")
        self.use_fused_clifford_pack = bool(use_fused_clifford_pack)
        self.use_fused_block_ops = bool(use_fused_block_ops)
        self.use_selective_amp = bool(use_selective_amp)
        self.selective_amp_dtype_name = str(selective_amp_dtype).strip().lower()
        self.selective_amp_dtype = _resolve_selective_amp_dtype(self.selective_amp_dtype_name)
        if self.use_fused_block_ops and gamma_mode != "vector":
            raise ValueError("Generalized V8 block fusion currently requires gamma_mode=vector")
        # Generic CliffordPack supports arbitrary hop counts and also hop_gate_mode=none.
        # The original specialized K=3 CUDA kernel is still selected automatically.

        self.h_norm = nn.LayerNorm(hidden_dim)
        self.c_norm = nn.LayerNorm(hidden_dim)

        if self.use_collapsed_fused_gat:
            gat_cls = CompatibleCollapsedFusedGATConv
        elif self.use_fused_gat_sideops:
            gat_cls = CompatibleFusedGATConvSideOps
        else:
            gat_cls = CompatibleFusedGATConv if self.use_fused_gat else GATConv
        self.context_gat = gat_cls(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=False,
            dropout=gat_dropout,
            add_self_loops=False,
            bias=use_bias,
        )
        # Custom attributes consumed by the compatible fused GAT wrappers.
        # Only their large input Linear is autocast; the attention/message
        # kernels themselves always receive FP32 tensors.
        self.context_gat._selective_amp_enabled = self.use_selective_amp
        self.context_gat._selective_amp_dtype = self.selective_amp_dtype

        # The old weighting keeps all K hop responses before projection:
        #   [N, K * 2d] -> [N, d]
        # The new weighting aggregates over K first:
        #   [N, 2d] -> [N, d]
        if self.alpha_weighting_mode == "old":
            proj_in_dim = 2 * self.num_hops * hidden_dim
        else:
            proj_in_dim = 2 * hidden_dim
        self.proj = nn.Linear(proj_in_dim, hidden_dim, bias=use_bias)

        if hop_gate_mode == "global":
            self.hop_gate_logits = nn.Parameter(torch.zeros(self.num_hops))
            self.hop_gate_mlp = None
        elif hop_gate_mode == "node":
            self.hop_gate_logits = None
            self.hop_gate_mlp = nn.Linear(2 * hidden_dim, self.num_hops, bias=True)
        else:
            self.hop_gate_logits = None
            self.hop_gate_mlp = None

        if gamma_mode == "vector":
            self.gamma = nn.Parameter(torch.full((hidden_dim,), float(init_gamma)))
        else:
            self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def compute_hop_gate(self, H_norm: torch.Tensor, C_norm: torch.Tensor) -> torch.Tensor:
        num_nodes = H_norm.size(0)

        if self.hop_gate_mode == "none":
            return None

        if self.hop_gate_mode == "global":
            alpha = F.softmax(self.hop_gate_logits, dim=0)
            alpha = alpha.view(1, self.num_hops, 1)
            alpha = alpha.expand(num_nodes, -1, -1)
            return alpha

        if self.alpha_score_mode == "old":
            # Original implementation.
            gate_input = torch.cat([H_norm, C_norm], dim=-1).float()
        else:
            # Teacher-modified symmetric score input:
            # [H + R || |H - R|].
            # We deliberately reuse the current normalized state/context streams
            # (H_norm, C_norm) so this axis changes ONLY the score object.
            gate_input = torch.cat(
                [H_norm + C_norm, torch.abs(H_norm - C_norm)],
                dim=-1,
            ).float()

        # Hop-gate MLP and softmax intentionally stay FP32.
        logits = self.hop_gate_mlp(gate_input)
        alpha = F.softmax(logits, dim=-1)
        return alpha.unsqueeze(-1)

    def forward(
        self,
        H: torch.Tensor,
        edge_index: torch.Tensor,
        graph_operator: torch.Tensor,
        fused_gat_graph=None,
    ) -> torch.Tensor:
        if self.use_fused_block_ops:
            H_norm = fused_dropout_layernorm(
                H, self.h_norm.weight, self.h_norm.bias,
                self.dropout if self.training else 0.0, self.h_norm.eps,
            )
        else:
            H_drop = F.dropout(H, p=self.dropout, training=self.training)
            H_norm = self.h_norm(H_drop)

        if self.use_fused_gat:
            if fused_gat_graph is None:
                raise RuntimeError(
                    "FusedGAT is enabled but the graph format cache was not prepared. "
                    "Call model.prepare_fused_gat_graph(...) once before training."
                )
            if isinstance(fused_gat_graph, ChunkedDgNNGraph):
                gat_out = self.context_gat(H_norm, fused_gat_graph, None, None)
            else:
                csr, csc, perm = fused_gat_graph
                gat_out = self.context_gat(H_norm, csr, csc, perm)
            C_gat = gat_out if (self.use_fused_gat_sideops or self.use_collapsed_fused_gat) else F.silu(gat_out)
        else:
            C_gat = F.silu(self.context_gat(H_norm, edge_index))
        if self.use_fused_block_ops:
            C_norm = fused_add_layernorm(
                C_gat, H_norm, self.c_norm.weight, self.c_norm.bias, self.c_norm.eps
            )
        else:
            C = C_gat + H_norm
            C_norm = self.c_norm(C)

        # Propagate H and C together and reuse P^k[H, C] for all requested hops.
        # This is mathematically equivalent to independently computing P^s H and
        # P^s C, but removes repeated sparse matrix multiplications.
        HC_norm = torch.cat([H_norm, C_norm], dim=-1)
        propagated = selected_sparse_powers(
            operator=graph_operator,
            X=HC_norm,
            powers=self.order_set,
        )

        hop_gate = self.compute_hop_gate(H_norm, C_norm)

        # --------------------------------------------------------
        # Interaction + alpha weighting.
        #
        # V3 full-acceleration policy:
        #   OLD interaction:
        #       keep the original fused_clifford_pack path.
        #   NEW interaction + K=3:
        #       use the new fused_clifford_switch_pack CUDA kernel.
        #       This fuses B_HR/B_RH, symmetric/antisymmetric D/W,
        #       alpha weighting, and (for NEW weighting) hop reduction.
        #   Other unsupported cases:
        #       fall back only for this local operation.
        #
        # All graph/GAT/CSR/MultiHop/V8/TF32/BF16 acceleration is independent
        # and remains unchanged.
        # --------------------------------------------------------
        propagated_in_order = [propagated[s] for s in self.order_set]

        if (
            self.use_fused_clifford_pack
            and self.interaction_mode == "old"
        ):
            # Existing production fast path for the original interaction.
            G_packed = fused_clifford_pack(
                H_norm=H_norm,
                C_norm=C_norm,
                propagated=propagated_in_order,
                alpha=hop_gate,
            )

            if self.alpha_weighting_mode == "old":
                G_raw = G_packed
            else:
                # Existing pack already applied alpha.
                packed = G_packed.reshape(
                    H.size(0), self.num_hops, 2 * self.hidden_dim
                )
                W_weighted = packed[..., : self.hidden_dim]
                D_weighted = packed[..., self.hidden_dim :]
                D_agg = D_weighted.sum(dim=1)
                W_agg = W_weighted.sum(dim=1)
                G_raw = torch.cat([D_agg, W_agg], dim=-1)

        elif self.use_fused_clifford_pack and self.interaction_mode == "new" and self.outer_nonlinearity == "none":
            if self.num_hops == 3:
                G_raw = cign_pack3(H_norm, C_norm, propagated_in_order, hop_gate, "new", self.alpha_weighting_mode)
            else:
                G_raw = cign_pack_k(H_norm, C_norm, propagated_in_order, hop_gate)
                if self.alpha_weighting_mode == "new":
                    packed = G_raw.reshape(H.size(0), self.num_hops, 2 * self.hidden_dim)
                    G_raw = torch.cat([packed[..., self.hidden_dim:].sum(1), packed[..., :self.hidden_dim].sum(1)], dim=-1)

        elif (
            self.use_fused_clifford_pack
            and self.interaction_mode == "new"
            and self.num_hops == 3
        ):
            # NEW full-acceleration path.
            #
            # NEW interaction:
            #   B_HR = H*T_C
            #   B_RH = C*T_H
            #   D = SiLU(0.5*(B_HR+B_RH))
            #   W = tanh(0.5*(B_HR-B_RH))
            #
            # OLD weighting -> [N,6D]
            # NEW weighting -> directly [N,2D], avoiding all Python-side
            #                  stack/multiply/sum intermediates.
            G_raw = fused_clifford_switch_pack(
                H_norm=H_norm,
                C_norm=C_norm,
                propagated=propagated_in_order,
                alpha=hop_gate,
                interaction_mode="new",
                alpha_weighting_mode=self.alpha_weighting_mode,
            )

        else:
            # Correct fallback for non-K3 or explicit pack-off cases.
            D_list = []
            W_list = []

            for s in self.order_set:
                T_H, T_C = propagated[s].split(
                    self.hidden_dim, dim=-1
                )

                B_HR = H_norm * T_C
                B_RH = C_norm * T_H

                if self.interaction_mode == "old":
                    D_s = F.silu(B_HR)
                    W_s = B_HR - B_RH
                else:
                    D_s = F.silu(
                        0.5 * (B_HR + B_RH)
                    )
                    W_s = cign_outer_difference(
                        B_HR, B_RH, self.outer_nonlinearity
                    )

                D_list.append(D_s)
                W_list.append(W_s)

            D_hops = torch.stack(D_list, dim=1)
            W_hops = torch.stack(W_list, dim=1)

            if self.alpha_weighting_mode == "old":
                hop_features = torch.cat(
                    [W_hops, D_hops], dim=-1
                )
                if hop_gate is not None:
                    hop_features = hop_features * hop_gate
                G_raw = hop_features.reshape(
                    H.size(0), -1
                )
            else:
                if hop_gate is not None:
                    D_hops = D_hops * hop_gate
                    W_hops = W_hops * hop_gate
                D_agg = D_hops.sum(dim=1)
                W_agg = W_hops.sum(dim=1)
                G_raw = torch.cat(
                    [D_agg, W_agg], dim=-1
                )

        G_feat = _local_amp_linear(
            self.proj, G_raw, self.use_selective_amp, self.selective_amp_dtype
        )
        if self.use_fused_block_ops:
            return fused_dropout_gamma_residual(
                H, G_feat, self.gamma, self.dropout if self.training else 0.0
            )
        G_feat = F.dropout(G_feat, p=self.dropout, training=self.training)
        if self.gamma_mode == "vector":
            return H + self.gamma.unsqueeze(0) * G_feat
        return H + self.gamma * G_feat


# ============================================================
# Full model
# ============================================================
class GraphCliffordNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 4,
        order_set=(1, 2, 3),
        dropout: float = 0.28,
        gat_heads: int = 6,
        gat_dropout: float = 0.32,
        use_fused_gat: bool = False,
        use_fused_gat_sideops: bool = False,
        use_chunked_fused_gat: bool = False,
        dgnn_chunk_max_nodes: int = 45000,
        dgnn_chunk_target_nodes: int = 8192,
        use_collapsed_fused_gat: bool = False,
        use_fused_clifford_pack: bool = False,
        use_fused_block_ops: bool = False,
        use_selective_amp: bool = False,
        selective_amp_dtype: str = "bf16",
        init_gamma: float = 0.49,
        gamma_mode: str = "vector",
        hop_gate_mode: str = "node",
        alpha_score_mode: str = None,
        interaction_mode: str = None,
        alpha_weighting_mode: str = None,
        layer_combine: str = "concat",
        use_dual_input_projection: bool = True,
        init_input_eta: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if layer_combine not in {"last", "concat"}:
            raise ValueError("layer_combine must be either 'last' or 'concat'.")
        if hop_gate_mode not in {"none", "global", "node"}:
            raise ValueError("hop_gate_mode must be one of {'none', 'global', 'node'}.")

        self.dropout = dropout
        self.layer_combine = layer_combine
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.hop_gate_mode = hop_gate_mode
        self.alpha_score_mode = _resolve_old_new_mode(
            alpha_score_mode, "CIGN_ALPHA_SCORE_MODE", "new"
        )
        self.interaction_mode = _resolve_old_new_mode(
            interaction_mode, "CIGN_INTERACTION_MODE", "new"
        )
        self.alpha_weighting_mode = _resolve_old_new_mode(
            alpha_weighting_mode, "CIGN_ALPHA_WEIGHTING_MODE", "old"
        )
        self.use_dual_input_projection = use_dual_input_projection
        self.use_fused_gat = bool(use_fused_gat)
        self.use_fused_gat_sideops = bool(use_fused_gat_sideops)
        self.use_chunked_fused_gat = bool(use_chunked_fused_gat)
        self.dgnn_chunk_max_nodes = int(dgnn_chunk_max_nodes)
        self.dgnn_chunk_target_nodes = int(dgnn_chunk_target_nodes)
        self.use_collapsed_fused_gat = bool(use_collapsed_fused_gat)
        if self.use_chunked_fused_gat and not self.use_fused_gat:
            raise ValueError("use_chunked_fused_gat=True requires use_fused_gat=True")
        if self.use_chunked_fused_gat and self.use_collapsed_fused_gat:
            raise ValueError("chunked dgNN and collapsed GAT are mutually exclusive")
        if self.use_fused_gat_sideops and not self.use_fused_gat:
            raise ValueError("use_fused_gat_sideops=True requires use_fused_gat=True")
        if self.use_collapsed_fused_gat and not self.use_fused_gat:
            raise ValueError("use_collapsed_fused_gat=True requires use_fused_gat=True")
        if self.use_collapsed_fused_gat and not self.use_fused_gat_sideops:
            raise ValueError("A3-v6 requires use_fused_gat_sideops=True")
        self.outer_nonlinearity = os.environ.get("CIGN_OUTER_NONLINEARITY", "none").strip().lower()
        if self.outer_nonlinearity not in {"tanh", "none"}:
            raise ValueError("CIGN_OUTER_NONLINEARITY must be tanh or none")
        self.use_fused_clifford_pack = bool(use_fused_clifford_pack)
        self.use_fused_block_ops = bool(use_fused_block_ops)
        self.use_selective_amp = bool(use_selective_amp)
        self.selective_amp_dtype_name = str(selective_amp_dtype).strip().lower()
        self.selective_amp_dtype = _resolve_selective_amp_dtype(self.selective_amp_dtype_name)
        self._fused_gat_graph = None
        self._fused_gat_graph_info = None
        self.dirichlet_recorder = DirichletRecorder()

        self.input_act = nn.SiLU()
        if self.use_dual_input_projection:
            self.input_proj_0 = nn.Linear(in_dim, hidden_dim, bias=use_bias)
            self.input_proj_1 = nn.Linear(in_dim, hidden_dim, bias=use_bias)
            self.input_eta = nn.Parameter(torch.tensor(float(init_input_eta)))
        else:
            self.input_proj = nn.Linear(in_dim, hidden_dim, bias=use_bias)

        self.blocks = nn.ModuleList([
            GraphCliffordBlock(
                hidden_dim=hidden_dim,
                order_set=order_set,
                dropout=dropout,
                gat_heads=gat_heads,
                gat_dropout=gat_dropout,
                use_fused_gat=self.use_fused_gat,
                use_fused_gat_sideops=self.use_fused_gat_sideops,
                use_chunked_fused_gat=self.use_chunked_fused_gat,
                use_collapsed_fused_gat=self.use_collapsed_fused_gat,
                use_fused_clifford_pack=self.use_fused_clifford_pack,
                use_fused_block_ops=self.use_fused_block_ops,
                use_selective_amp=self.use_selective_amp,
                selective_amp_dtype=self.selective_amp_dtype_name,
                init_gamma=init_gamma,
                gamma_mode=gamma_mode,
                hop_gate_mode=hop_gate_mode,
                alpha_score_mode=self.alpha_score_mode,
                interaction_mode=self.interaction_mode,
                alpha_weighting_mode=self.alpha_weighting_mode,
                use_bias=use_bias,
            )
            for _ in range(num_layers)
        ])

        if layer_combine == "concat":
            final_dim = hidden_dim * (num_layers + 1)
        else:
            final_dim = hidden_dim

        self.output_norm = nn.LayerNorm(final_dim)
        self.classifier = nn.Linear(final_dim, out_dim, bias=use_bias)

    @torch.no_grad()
    def prepare_fused_gat_graph(self, edge_index: torch.Tensor, num_nodes: int = None):
        """
        Prepare dgNN's fixed CSR/CSC graph representation exactly once.

        Two compatibility fixes are required and were numerically validated:
          1) PyG edge_index is source->target, while dgNN's CSR row is the
             destination/output node.  Therefore the two rows are flipped only
             for dgNN graph conversion.
          2) dgNN backward uses vectorized int4 index loads.  clone() forces
             independent CUDA allocations with zero storage offsets and proper
             16-byte alignment, avoiding the observed ``misaligned address``.

        The original ``edge_index`` is left untouched and continues to be used
        by the non-fused path and by optional Dirichlet recording.
        """
        if not self.use_fused_gat:
            self._fused_gat_graph = None
            self._fused_gat_graph_info = None
            return None

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
        if edge_index.device.type != "cuda":
            raise RuntimeError("dgNN FusedGAT requires CUDA graph tensors.")
        if len(self.blocks) == 0:
            raise RuntimeError("Cannot prepare FusedGAT graph for a model with no blocks.")

        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1
        num_nodes = int(num_nodes)

        conv = self.blocks[0].context_gat
        if not isinstance(conv, CompatibleFusedGATConv):
            raise RuntimeError("The model is marked use_fused_gat=True but its block does not use CompatibleFusedGATConv.")

        if self.use_chunked_fused_gat:
            graph = prepare_chunked_dgnn_graph(
                conv=conv,
                edge_index=edge_index,
                num_nodes=num_nodes,
                max_local_nodes=self.dgnn_chunk_max_nodes,
                target_chunk_nodes=self.dgnn_chunk_target_nodes,
            )
            self._fused_gat_graph = graph
            self._fused_gat_graph_info = dict(graph.info)
            return dict(self._fused_gat_graph_info)

        # IMPORTANT: source->target (PyG) -> target->source (dgNN CSR convention).
        edge_index_dgnn = edge_index.flip(0).contiguous()
        csr, csc, perm = conv.to_graph_format(
            edge_index_dgnn,
            size=(num_nodes, num_nodes),
        )

        rowptr, col = csr
        row, colptr = csc

        # IMPORTANT: .clone() is intentional; .contiguous() alone may preserve
        # a storage offset for an already-contiguous 1-D view.
        rowptr = rowptr.clone().contiguous()
        col = col.clone().contiguous()
        row = row.clone().contiguous()
        colptr = colptr.clone().contiguous()
        perm = perm.clone().contiguous()

        tensors = {
            "csr_rowptr": rowptr,
            "csr_col": col,
            "csc_row": row,
            "csc_colptr": colptr,
            "perm": perm,
        }
        for name, tensor in tensors.items():
            if tensor.storage_offset() != 0:
                raise RuntimeError(f"{name} storage_offset={tensor.storage_offset()}, expected 0.")
            if tensor.data_ptr() % 16 != 0:
                raise RuntimeError(f"{name} is not 16-byte aligned (ptr_mod_16={tensor.data_ptr() % 16}).")

        self._fused_gat_graph = ((rowptr, col), (row, colptr), perm)
        self._fused_gat_graph_info = {
            "num_nodes": num_nodes,
            "num_edges": int(col.numel()),
            "csr_rowptr_ptr_mod_16": int(rowptr.data_ptr() % 16),
            "csr_col_ptr_mod_16": int(col.data_ptr() % 16),
            "csc_row_ptr_mod_16": int(row.data_ptr() % 16),
            "csc_colptr_ptr_mod_16": int(colptr.data_ptr() % 16),
            "perm_ptr_mod_16": int(perm.data_ptr() % 16),
        }
        return dict(self._fused_gat_graph_info)

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor, graph_operator):
        H_in = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_dual_input_projection:
            # V9 dtype-safe selective AMP: only the two dense projections enter
            # autocast. Everything after them is explicitly back in FP32.
            H_base = self.input_act(_local_amp_linear(
                self.input_proj_0, H_in, self.use_selective_amp, self.selective_amp_dtype
            ))
            H_vel = self.input_act(_local_amp_linear(
                self.input_proj_1, H_in, self.use_selective_amp, self.selective_amp_dtype
            ))
            H = H_base + self.input_eta * H_vel
        else:
            H = self.input_act(_local_amp_linear(
                self.input_proj, H_in, self.use_selective_amp, self.selective_amp_dtype
            ))

        states = [H]
        for block in self.blocks:
            H = block(
                H,
                edge_index,
                graph_operator,
                fused_gat_graph=self._fused_gat_graph,
            )
            states.append(H)

        if self.dirichlet_recorder.enabled:
            self.dirichlet_recorder.maybe_record(
                states=states,
                edge_index=edge_index,
                num_layers=self.num_layers,
                training=self.training,
            )

        if self.layer_combine == "concat":
            H_out = torch.cat(states, dim=-1)
        else:
            H_out = states[-1]

        # Final LayerNorm and classifier always stay in FP32.
        H_out = self.output_norm(H_out.float())
        H_out = F.dropout(H_out, p=self.dropout, training=self.training)
        logits = self.classifier(H_out)
        return logits.float()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, graph_operator: torch.Tensor = None) -> torch.Tensor:
        if graph_operator is None:
            raise ValueError("GraphCliffordNet requires a sparse graph operator as the third input.")
        # IMPORTANT: no model-wide autocast here. V9 AMP is applied only inside
        # _local_amp_linear() at the selected dense GEMMs. This keeps cuSPARSE
        # and all FP32 custom CUDA forward/backward paths dtype-safe.
        return self._forward_impl(x.float(), edge_index, graph_operator)


# ============================================================
# Factory
# ============================================================
def build_model(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 4,
    dropout: float = 0.28,
    hop_scales=(1, 2, 3),
    gat_heads: int = 6,
    gat_dropout: float = 0.32,
    use_fused_gat: bool = False,
    use_fused_gat_sideops: bool = False,
    use_chunked_fused_gat: bool = False,
    dgnn_chunk_max_nodes: int = 45000,
    dgnn_chunk_target_nodes: int = 8192,
    use_collapsed_fused_gat: bool = False,
    use_fused_clifford_pack: bool = False,
    use_fused_block_ops: bool = False,
    use_selective_amp: bool = False,
    selective_amp_dtype: str = "bf16",
    init_gamma: float = 0.49,
    gamma_mode: str = "vector",
    hop_gate_mode: str = "node",
    alpha_score_mode: str = None,
    interaction_mode: str = None,
    alpha_weighting_mode: str = None,
    layer_combine: str = "concat",
    use_dual_input_projection: bool = True,
    init_input_eta: float = 0.1,
    use_bias: bool = True,
    **kwargs,
):
    return GraphCliffordNet(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        order_set=hop_scales,
        dropout=dropout,
        gat_heads=gat_heads,
        gat_dropout=gat_dropout,
        use_fused_gat=use_fused_gat,
        use_fused_gat_sideops=use_fused_gat_sideops,
        use_chunked_fused_gat=use_chunked_fused_gat,
        dgnn_chunk_max_nodes=dgnn_chunk_max_nodes,
        dgnn_chunk_target_nodes=dgnn_chunk_target_nodes,
        use_collapsed_fused_gat=use_collapsed_fused_gat,
        use_fused_clifford_pack=use_fused_clifford_pack,
        use_fused_block_ops=use_fused_block_ops,
        use_selective_amp=use_selective_amp,
        selective_amp_dtype=selective_amp_dtype,
        init_gamma=init_gamma,
        gamma_mode=gamma_mode,
        hop_gate_mode=hop_gate_mode,
        alpha_score_mode=alpha_score_mode,
        interaction_mode=interaction_mode,
        alpha_weighting_mode=alpha_weighting_mode,
        layer_combine=layer_combine,
        use_dual_input_projection=use_dual_input_projection,
        init_input_eta=init_input_eta,
        use_bias=use_bias,
    )
