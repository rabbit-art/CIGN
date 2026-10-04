from __future__ import annotations

"""Chunked dgNN GAT graph execution for large node counts.

The installed dgNN fused GAT binary used by this project has been empirically
validated only up to N=46340 for a single square graph.  GAT normalization is
row/destination-local, so a fixed graph can be partitioned by destination rows
without changing the mathematics: every target chunk keeps *all* incoming
edges for those targets, while source-only local nodes are added only as
feature providers.

To keep dgNN away from the native N^2/int32 boundary, every local square graph
is adaptively kept below ``max_local_nodes`` (default 45000).  Source-only local
nodes receive a private self-loop solely to avoid zero-degree rows inside the
local dgNN call; their outputs are discarded, so those loops cannot affect any
global target output or gradient.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class ChunkedDgNNChunk:
    target_lo: int
    target_hi: int
    local_nodes: torch.Tensor
    target_local: torch.Tensor
    csr: Tuple[torch.Tensor, torch.Tensor]
    csc: Tuple[torch.Tensor, torch.Tensor]
    perm: torch.Tensor
    num_real_edges: int
    num_guard_loops: int


class ChunkedDgNNGraph:
    __slots__ = ("num_nodes", "num_edges", "max_local_nodes", "target_chunk_nodes", "chunks", "info")

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        max_local_nodes: int,
        target_chunk_nodes: int,
        chunks: List[ChunkedDgNNChunk],
    ):
        self.num_nodes = int(num_nodes)
        self.num_edges = int(num_edges)
        self.max_local_nodes = int(max_local_nodes)
        self.target_chunk_nodes = int(target_chunk_nodes)
        self.chunks = list(chunks)
        local_sizes = [int(c.local_nodes.numel()) for c in chunks]
        real_edges = [int(c.num_real_edges) for c in chunks]
        guard_loops = [int(c.num_guard_loops) for c in chunks]
        self.info = {
            "backend": "ChunkedDgNN_FusedGAT",
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_chunks": len(chunks),
            "max_local_nodes_limit": self.max_local_nodes,
            "target_chunk_nodes_requested": self.target_chunk_nodes,
            "max_local_nodes_observed": max(local_sizes) if local_sizes else 0,
            "total_local_node_references": sum(local_sizes),
            "total_real_edges": sum(real_edges),
            "total_guard_self_loops": sum(guard_loops),
            "chunks": [
                {
                    "target_lo": int(c.target_lo),
                    "target_hi": int(c.target_hi),
                    "targets": int(c.target_hi - c.target_lo),
                    "local_nodes": int(c.local_nodes.numel()),
                    "real_edges": int(c.num_real_edges),
                    "guard_loops": int(c.num_guard_loops),
                }
                for c in chunks
            ],
        }


def _clone_graph_format(csr, csc, perm):
    rowptr, col = csr
    row, colptr = csc
    rowptr = rowptr.clone().contiguous()
    col = col.clone().contiguous()
    row = row.clone().contiguous()
    colptr = colptr.clone().contiguous()
    perm = perm.clone().contiguous()
    for name, t in {
        "rowptr": rowptr,
        "col": col,
        "row": row,
        "colptr": colptr,
        "perm": perm,
    }.items():
        if t.storage_offset() != 0:
            raise RuntimeError(f"chunked dgNN {name} has non-zero storage_offset={t.storage_offset()}")
        if t.is_cuda and t.data_ptr() % 16 != 0:
            raise RuntimeError(f"chunked dgNN {name} is not 16-byte aligned")
    return (rowptr, col), (row, colptr), perm


@torch.no_grad()
def prepare_chunked_dgnn_graph(
    conv,
    edge_index: torch.Tensor,
    num_nodes: int,
    max_local_nodes: int = 45000,
    target_chunk_nodes: int = 8192,
) -> ChunkedDgNNGraph:
    """Partition a PyG source->target graph by destination rows.

    ``edge_index`` must already contain exactly the GAT self-loop semantics used
    by the model (trainb9.py prepares those loops once before this function).
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2,E], got {tuple(edge_index.shape)}")
    if edge_index.device.type != "cuda":
        raise RuntimeError("Chunked dgNN graph preparation requires CUDA edge_index")
    num_nodes = int(num_nodes)
    max_local_nodes = int(max_local_nodes)
    target_chunk_nodes = int(target_chunk_nodes)
    if max_local_nodes <= 0 or max_local_nodes >= 46341:
        raise ValueError("max_local_nodes must be in [1,46340] for the validated dgNN build")
    if target_chunk_nodes <= 0:
        raise ValueError("target_chunk_nodes must be positive")

    # One CPU copy outside epoch timing keeps adaptive partition bookkeeping
    # simple and deterministic.  The final graph-format tensors live on CUDA.
    ei_cpu = edge_index.detach().to("cpu", dtype=torch.long).contiguous()
    src_all, dst_all = ei_cpu[0], ei_cpu[1]
    device = edge_index.device
    chunks: List[ChunkedDgNNChunk] = []

    def build_range(lo: int, hi: int):
        mask = (dst_all >= lo) & (dst_all < hi)
        src = src_all[mask]
        dst = dst_all[mask]
        targets = torch.arange(lo, hi, dtype=torch.long)
        local_nodes = torch.unique(torch.cat([src, dst, targets], dim=0), sorted=True)

        if int(local_nodes.numel()) > max_local_nodes:
            if hi - lo <= 1:
                raise RuntimeError(
                    f"A single target row requires {int(local_nodes.numel())} local nodes, "
                    f"exceeding chunk dgNN limit={max_local_nodes}. Use PyG for this graph."
                )
            mid = lo + (hi - lo) // 2
            build_range(lo, mid)
            build_range(mid, hi)
            return

        # Remap global node ids to a compact local square graph.
        src_local = torch.searchsorted(local_nodes, src)
        dst_local = torch.searchsorted(local_nodes, dst)
        local_ei = torch.stack([src_local, dst_local], dim=0)

        # Rows outside the target interval are source-only in this chunk. Give
        # only those rows a guard self-loop; their outputs are never returned.
        source_only = (local_nodes < lo) | (local_nodes >= hi)
        guard_ids = torch.nonzero(source_only, as_tuple=False).view(-1)
        if guard_ids.numel() > 0:
            guard_ei = torch.stack([guard_ids, guard_ids], dim=0)
            local_ei = torch.cat([local_ei, guard_ei], dim=1)

        target_local = torch.searchsorted(local_nodes, targets)
        local_ei_cuda = local_ei.to(device=device, dtype=torch.long, non_blocking=False).contiguous()

        # Verified dgNN convention: PyG source->target -> dgNN target->source.
        csr, csc, perm = conv.to_graph_format(
            local_ei_cuda.flip(0).contiguous(),
            size=(int(local_nodes.numel()), int(local_nodes.numel())),
        )
        csr, csc, perm = _clone_graph_format(csr, csc, perm)

        chunks.append(
            ChunkedDgNNChunk(
                target_lo=int(lo),
                target_hi=int(hi),
                local_nodes=local_nodes.to(device=device, dtype=torch.long).contiguous(),
                target_local=target_local.to(device=device, dtype=torch.long).contiguous(),
                csr=csr,
                csc=csc,
                perm=perm,
                num_real_edges=int(src.numel()),
                num_guard_loops=int(guard_ids.numel()),
            )
        )

    for lo in range(0, num_nodes, target_chunk_nodes):
        build_range(lo, min(num_nodes, lo + target_chunk_nodes))

    chunks.sort(key=lambda c: c.target_lo)
    expected = 0
    for c in chunks:
        if c.target_lo != expected:
            raise RuntimeError(f"Chunk coverage gap/overlap at target {expected}: got {c.target_lo}")
        expected = c.target_hi
    if expected != num_nodes:
        raise RuntimeError(f"Chunk coverage ended at {expected}, expected {num_nodes}")

    return ChunkedDgNNGraph(
        num_nodes=num_nodes,
        num_edges=int(edge_index.size(1)),
        max_local_nodes=max_local_nodes,
        target_chunk_nodes=target_chunk_nodes,
        chunks=chunks,
    )


def run_chunked_dgnn_op(
    op,
    x_heads: torch.Tensor,
    alpha_src: torch.Tensor,
    alpha_dst: torch.Tensor,
    graph: ChunkedDgNNGraph,
    negative_slope: float,
    dropout: float,
) -> torch.Tensor:
    """Run dgNN on destination chunks and stitch target outputs in global order."""
    if not isinstance(graph, ChunkedDgNNGraph):
        raise TypeError(f"Expected ChunkedDgNNGraph, got {type(graph)!r}")
    if x_heads.dim() != 3:
        raise ValueError(f"x_heads must be [N,H,C], got {tuple(x_heads.shape)}")
    if int(x_heads.size(0)) != graph.num_nodes:
        raise ValueError(f"x_heads N={x_heads.size(0)} does not match graph N={graph.num_nodes}")

    parts = []
    for chunk in graph.chunks:
        nodes = chunk.local_nodes
        x_local = x_heads.index_select(0, nodes).contiguous()
        src_local = alpha_src.index_select(0, nodes).contiguous()
        dst_local = alpha_dst.index_select(0, nodes).contiguous()
        (rowptr, col), (row, colptr), perm = chunk.csr, chunk.csc, chunk.perm
        out_local = op(
            dst_local,
            src_local,
            rowptr,
            col,
            colptr,
            row,
            perm,
            float(negative_slope),
            x_local,
            float(dropout),
        )
        parts.append(out_local.index_select(0, chunk.target_local))

    if not parts:
        return x_heads.new_empty((0, x_heads.size(1), x_heads.size(2)))
    return torch.cat(parts, dim=0)
