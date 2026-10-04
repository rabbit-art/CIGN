# ============================================================
# trainb9.py
# Train Graph Clifford model only
# Dataset loading is kept from the original version.
#
# This version supports:
#   1) LayerNorm + learnable gamma model
#   2) Operator choices: laplacian / adjacency / hybrid
#   3) Final JK-style concatenation through --layer_combine concat
#   4) Residual-enhanced context generation inside model.py
#   5) Hop-level gate through --hop_gate_mode
#   6) Split choices:
#        --split_mode original      : use dataset-provided masks
#        --split_mode random_ratio  : randomly split by train/val/test ratios
#   7) Pure training-step timing with per-epoch mean and sample std
#   8) Optional numerically-aligned dgNN/PyG FusedGAT backend with one-time
#      target->source CSR/CSC conversion and 16-byte alignment fix
#   9) Optional fixed-CSR SpMM for the constant Clifford graph operator:
#        forward  = CSR(P)   @ X
#        backward = CSR(P^T) @ grad
#      with no sparse-operator gradients and no per-step COO coalesce.
#  10) A3-v1 fused Clifford feature construction + hop-gate CUDA kernel
#  11) A3-v2 fused GAT attention-logit + head-mean/bias/SiLU CUDA kernels
#  12) Optional A3-v4 TF32 Tensor-Core matmul mode via --use_tf32
#  13) A3-v5 grouped native three-hop fixed-CSR forward/backward pipeline
#  14) A3-v9 selective BF16/FP16 autocast for large dense Linear GEMMs only
#  15) Strictly disabled Dirichlet recorder during runtime benchmarking
#      unless --enable_dirichlet_recording True is explicitly requested
# ============================================================

import os
import sys
import argparse
import random
import time
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.datasets import (
    Planetoid,
    WebKB,
    WikipediaNetwork,
    Actor,
    HeterophilousGraphDataset,
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import (
    to_undirected,
    get_laplacian,
    add_self_loops,
    remove_self_loops,
    degree,
)

from model_cign import build_model, prepare_fixed_csr_operator, FixedCSRGraphOperator
from multihop_csr_pipeline import prepare_multihop_csr_operator, ensure_multihop_csr_loaded
from fused_clifford_pack import ensure_fused_clifford_pack_loaded
from fused_clifford_switch_pack import ensure_fused_clifford_switch_pack_loaded
from fused_gat_sideops import ensure_fused_gat_sideops_loaded
from fused_block_ops import ensure_fused_block_ops_loaded
from collapsed_gat import ensure_collapsed_gat_loaded

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


DEFAULT_SEED = 42

DEFAULT_DATA_ROOT = "./data"
DEFAULT_DATASET_NAME = 'Amazon-ratings'
DEFAULT_USE_UNDIRECTED = True
DEFAULT_SPLIT_INDEX = 0

# ============================================================
# Split defaults
# ============================================================
# original:
#   use dataset-provided train_mask / val_mask / test_mask
# random_ratio:
#   generate random masks by train_ratio / val_ratio / test_ratio
DEFAULT_SPLIT_MODE = 'random_ratio'
DEFAULT_TRAIN_RATIO = 0.5
DEFAULT_VAL_RATIO = 0.25
DEFAULT_TEST_RATIO = 0.25
DEFAULT_SPLIT_SEED = 42

DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT = 0.28
DEFAULT_HOP_SCALES = [1, 2, 3]
DEFAULT_GAT_HEADS = 6
DEFAULT_GAT_DROPOUT = 0.32
DEFAULT_USE_FUSED_GAT = True
DEFAULT_USE_FUSED_GAT_SIDEOPS = True
DEFAULT_USE_FIXED_CSR_SPMM = True
DEFAULT_USE_FUSED_CLIFFORD_PACK = True
DEFAULT_USE_TF32 = True
DEFAULT_USE_MULTIHOP_CSR_PIPELINE = True
DEFAULT_USE_COLLAPSED_FUSED_GAT = True
DEFAULT_USE_FUSED_BLOCK_OPS = True
DEFAULT_USE_SELECTIVE_AMP = True
DEFAULT_SELECTIVE_AMP_DTYPE = "bf16"
DEFAULT_MULTIHOP_CSR_ALGORITHM = 2
DEFAULT_FAST_BACKEND_POLICY = "auto"

DEFAULT_GENERIC_BACKEND_PROFILE = "auto"
GENERAL_FAST_HIDDEN_DIMS = (16, 32, 64, 96, 128, 160, 192, 224, 256)
GENERAL_FAST_HEADS = tuple(range(1, 7))
GENERAL_FAST_LAYERS = tuple(range(1, 7))
DEFAULT_GAT_BACKEND_REGISTRY = "gat_backend_registry.json"
DEFAULT_GAT_BACKEND_MIN_SPEEDUP = 1.02

# Hard correctness guard for the currently installed dgNN fused-GAT build.
# Dedicated forward/backward alignment tests on this environment found the
# exact boundary N=46340 PASS / N=46341 FAIL, consistent with an N^2 int32
# overflow boundary inside the native fused_gatconv extension. Keep this
# conservative guard until that native extension is rebuilt and revalidated.
DGNN_VALIDATED_MAX_SAFE_NUM_NODES = 46340
DEFAULT_DGNN_CHUNK_MAX_NODES = 45000
DEFAULT_DGNN_CHUNK_TARGET_NODES = 8192
DEFAULT_LARGE_GRAPH_GAT_BACKEND = "auto"
DEFAULT_INIT_GAMMA = 0.49
DEFAULT_GAMMA_MODE = "vector"
DEFAULT_HOP_GATE_MODE = "node"
DEFAULT_OPERATOR_MODE = "adjacency"
DEFAULT_HYBRID_ALPHA = 0.5
DEFAULT_LAYER_COMBINE = "concat"

DEFAULT_LR = 0.003
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EPOCHS = 2000
DEFAULT_PATIENCE = 200

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_PRINT_EVERY = 20
DEFAULT_SAVE_LOG = True
DEFAULT_LOG_DIR = "logs"
DEFAULT_SAVE_MIN_BEST_VAL = 0
DEFAULT_SAVE_MIN_BEST_TEST = 0
DEFAULT_ENABLE_DIRICHLET_RECORDING = False

# Numerical tracing/debug mode. Disabled by default and intended only for
# short diagnostic runs; tensor scans and anomaly detection add heavy overhead.
DEFAULT_DEBUG_NUMERICS = False
DEFAULT_DEBUG_NUMERICS_EPOCHS = 3
DEFAULT_DEBUG_NUMERICS_MODULES = True
DEFAULT_DEBUG_NUMERICS_VERBOSE = False
DEFAULT_DEBUG_DETECT_ANOMALY = True

BINARY_AUC_DATASETS = {"Minesweeper", "Tolokers", "Questions"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {v}")


def sanitize_filename(text: str) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", ".", "="}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def uses_auc(dataset_name: str) -> bool:
    return dataset_name.strip() in BINARY_AUC_DATASETS


def get_metric_name(dataset_name: str) -> str:
    return "ROC AUC" if uses_auc(dataset_name) else "Accuracy"



def get_cign_switch_modes():
    def _mode(env_name):
        value = os.environ.get(env_name, "old").strip().lower()
        if value not in {"old", "new"}:
            raise ValueError(f"{env_name} must be old/new, got {value!r}")
        return value

    return {
        "alpha_score": _mode("CIGN_ALPHA_SCORE_MODE"),
        "interaction": _mode("CIGN_INTERACTION_MODE"),
        "alpha_weighting": _mode("CIGN_ALPHA_WEIGHTING_MODE"),
    }


def build_full_run_config(args) -> str:
    """
    Build a full, human-readable configuration string.
    This string is printed into the log, but it is no longer used directly
    as the filename, because long filenames can trigger OSError: [Errno 36].
    """
    hop_str = "-".join(map(str, args.hop_scales))
    parts = [
        "model=graph_clifford_gat_ln_gamma_operator_jk_cres_hopgate_sixswitch",
        f"alpha_score={get_cign_switch_modes()['alpha_score']}",
        f"interaction={get_cign_switch_modes()['interaction']}",
        f"alpha_weighting={get_cign_switch_modes()['alpha_weighting']}",
        f"data={args.dataset_name}",
        f"seed={args.seed}",
        f"splitmode={args.split_mode}",
        f"split={args.split_index}",
        f"splitseed={args.split_seed}",
        f"ratio={args.train_ratio}-{args.val_ratio}-{args.test_ratio}",
        f"hid={args.hidden_dim}",
        f"layers={args.num_layers}",
        f"drop={args.dropout}",
        f"hops={hop_str}",
        f"hopgate={args.hop_gate_mode}",
        f"gatheads={args.gat_heads}",
        f"gatdrop={args.gat_dropout}",
        f"genericprofile={str(getattr(args, 'generic_backend_profile', 'auto'))}",
        f"fastpolicy={args.fast_backend_policy}",
        f"fusedgat={int(args.use_fused_gat)}",
        f"gatside={int(args.use_fused_gat_sideops)}",
        f"chunkedgat={int(getattr(args, 'use_chunked_fused_gat', False))}",
        f"chunkmax={int(getattr(args, 'dgnn_chunk_max_nodes', DEFAULT_DGNN_CHUNK_MAX_NODES))}",
        f"chunktarget={int(getattr(args, 'dgnn_chunk_target_nodes', DEFAULT_DGNN_CHUNK_TARGET_NODES))}",
        f"largegat={str(getattr(args, 'large_graph_gat_backend', DEFAULT_LARGE_GRAPH_GAT_BACKEND))}",
        f"collapsedgat={int(args.use_collapsed_fused_gat)}",
        f"v8block={int(args.use_fused_block_ops)}",
        f"v9amp={int(args.use_selective_amp)}",
        f"v9dtype={args.selective_amp_dtype}",
        f"fixedcsr={int(args.use_fixed_csr_spmm)}",
        f"multihopcsr={int(args.use_multihop_csr_pipeline)}",
        f"multihopbackend={str(getattr(args, 'multihop_backend', 'auto'))}",
        f"a3pack={int(args.use_fused_clifford_pack)}",
        f"tf32={int(args.use_tf32)}",
        f"gamma={args.init_gamma}",
        f"gammamode={args.gamma_mode}",
        f"op={args.operator_mode}",
        f"hybridalpha={args.hybrid_alpha}",
        f"combine={args.layer_combine}",
        f"lr={args.lr}",
        f"wd={args.weight_decay}",
        f"epochs={args.epochs}",
        f"pat={args.patience}",
        f"timingdiscard={int(getattr(args, 'timing_discard_first_epochs', 10))}",
        f"undir={int(args.use_undirected)}",
        f"dirichlet={int(args.enable_dirichlet_recording)}",
        f"debugnum={int(args.debug_numerics)}",
        f"debugepochs={int(args.debug_numerics_epochs)}",
    ]
    return sanitize_filename("__".join(parts))


def build_run_name(args) -> str:
    """
    Build a short log filename.

    The hash is computed from the full configuration, so different parameter
    settings still get different filenames while avoiding filename-too-long errors.
    """
    full_config = build_full_run_config(args)
    config_hash = hashlib.sha1(full_config.encode("utf-8")).hexdigest()[:10]

    dataset = sanitize_filename(args.dataset_name)
    split_mode = sanitize_filename(args.split_mode)
    prefix = f"gcg_{dataset}_{split_mode}_s{args.seed}"
    max_prefix_len = 50
    if len(prefix) > max_prefix_len:
        prefix = prefix[:max_prefix_len]

    return f"{prefix}_{config_hash}"


def make_unique_log_path(log_dir: Path, base_name: str) -> Path:
    path = log_dir / f"{base_name}.txt"
    if not path.exists():
        return path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return log_dir / f"{base_name}__{timestamp}.txt"


class ConsoleBuffer:
    def __init__(self):
        self.buffer = []

    def write(self, data: str):
        # Do not force a terminal flush for every small print fragment.
        sys.__stdout__.write(data)
        self.buffer.append(data)

    def flush(self):
        sys.__stdout__.flush()

    def get_text(self) -> str:
        return "".join(self.buffer)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Graph Clifford model.")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--use_undirected", type=str2bool, default=DEFAULT_USE_UNDIRECTED)
    parser.add_argument("--split_index", type=int, default=DEFAULT_SPLIT_INDEX)

    # ========================================================
    # Split arguments
    # ========================================================
    parser.add_argument(
        "--split_mode",
        type=str,
        default=DEFAULT_SPLIT_MODE,
        choices=["original", "random_ratio"],
        help=(
            "original: use dataset-provided train/val/test masks. "
            "random_ratio: ignore dataset-provided masks and randomly split nodes by ratios."
        ),
    )
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed used only for random_ratio split.",
    )

    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument(
        "--hop_scales",
        type=int,
        nargs="*",
        default=None,
        help="Order set S in Algorithm 8. Example: --hop_scales 1 2 3",
    )
    parser.add_argument("--gat_heads", type=int, default=DEFAULT_GAT_HEADS)
    parser.add_argument("--gat_dropout", type=float, default=DEFAULT_GAT_DROPOUT)
    parser.add_argument(
        "--use_fused_gat",
        type=str2bool,
        default=DEFAULT_USE_FUSED_GAT,
        help=(
            "Use the numerically-aligned PyG FusedGATConv + dgNN CUDA backend. "
            "Requires the separate pyg_embed_fused environment and dgNN installation."
        ),
    )
    parser.add_argument(
        "--use_fused_gat_sideops",
        type=str2bool,
        default=DEFAULT_USE_FUSED_GAT_SIDEOPS,
        help=(
            "A3-v2: fuse dgNN FusedGAT's PyTorch-side attention-logit reductions "
            "plus concat=False head mean + bias + SiLU into custom CUDA kernels."
        ),
    )
    parser.add_argument(
        "--dgnn_chunk_max_nodes",
        type=int,
        default=DEFAULT_DGNN_CHUNK_MAX_NODES,
        help=(
            "Maximum local square-graph node count for the generic chunked dgNN GAT backend. "
            "Must remain <=46340 for the currently validated dgNN build."
        ),
    )
    parser.add_argument(
        "--dgnn_chunk_target_nodes",
        type=int,
        default=DEFAULT_DGNN_CHUNK_TARGET_NODES,
        help="Initial destination-node chunk size for large-graph dgNN GAT; adaptively split if needed.",
    )

    parser.add_argument(
        "--large_graph_gat_backend",
        type=str,
        default=DEFAULT_LARGE_GRAPH_GAT_BACKEND,
        choices=["auto", "pyg", "chunked"],
        help=(
            "GAT backend when num_nodes exceeds the validated single-graph dgNN limit. "
            "auto defaults to PyG because the current mathematically-correct chunked dgNN "
            "path may duplicate source-only rows and can be slower; chunked keeps the "
            "experimental large-graph dgNN path available for explicit benchmarking."
        ),
    )

    parser.add_argument(
        "--use_collapsed_fused_gat",
        type=str2bool,
        default=DEFAULT_USE_COLLAPSED_FUSED_GAT,
        help=(
            "A3-v6: replace the dgNN message-passing output/head-reduction boundary "
            "with a concat=False collapsed-head CUDA core. In the current safe shape-aware policy, "
            "the validated specialized kernel is used only for hidden_dim=256 and gat_heads=6; "
            "other shapes use dgNN FusedGAT + A3-v3 sideops."
        ),
    )
    parser.add_argument(
        "--use_fused_block_ops",
        type=str2bool,
        default=DEFAULT_USE_FUSED_BLOCK_OPS,
        help=(
            "A3-v8: fuse block input Dropout+LayerNorm, context residual Add+LayerNorm, "
            "and projection Dropout+Gamma+Residual into custom CUDA kernels. "
            "Generalized V8 supports 1<=hidden_dim<=256 and requires gamma_mode=vector."
        ),
    )
    parser.add_argument(
        "--use_fixed_csr_spmm",
        type=str2bool,
        default=DEFAULT_USE_FIXED_CSR_SPMM,
        help=(
            "Use cached CSR(P) and CSR(P^T) with a custom fixed-operator autograd "
            "function for Clifford sparse propagation. All CSR construction is "
            "performed once before epoch timing."
        ),
    )
    parser.add_argument(
        "--use_fused_clifford_pack",
        type=str2bool,
        default=DEFAULT_USE_FUSED_CLIFFORD_PACK,
        help=(
            "A3: fuse Clifford W/D construction + hop-gate multiplication into one "
            "custom CUDA kernel while preserving the original single large projection GEMM. "
            "A3-v1 is specialized for exactly three hops."
        ),
    )
    parser.add_argument(
        "--use_multihop_csr_pipeline",
        type=str2bool,
        default=DEFAULT_USE_MULTIHOP_CSR_PIPELINE,
        help=(
            "A3-v5: group the exact [1,2,3] fixed-CSR recurrence into one native "
            "forward call and one native chain-rule backward call. Requires "
            "--use_fixed_csr_spmm True and hop_scales 1 2 3."
        ),
    )
    parser.add_argument(
        "--multihop_csr_algorithm",
        type=int,
        default=DEFAULT_MULTIHOP_CSR_ALGORITHM,
        choices=[1, 2, 3],
        help="A3-v5 cuSPARSE CSR SpMM algorithm; ALG2 is the profiler-informed default.",
    )
    parser.add_argument(
        "--multihop_backend",
        type=str,
        default="auto",
        choices=["auto", "fixedcsr", "native3", "generic", "native3_tail"],
        help=(
            "Performance-aware multi-hop backend. auto uses native A3-v5 only for exact "
            "hop set {1,2,3}; otherwise it keeps the faster repeated FixedCSR recurrence. "
            "generic and native3_tail remain available for explicit A/B experiments."
        ),
    )
    parser.add_argument(
        "--use_tf32",
        type=str2bool,
        default=DEFAULT_USE_TF32,
        help=(
            "A3-v4 experiment: allow TF32 Tensor-Core math for float32 CUDA matmul/Linear. "
            "This changes internal matmul precision but keeps tensor dtypes as float32."
        ),
    )
    parser.add_argument(
        "--use_selective_amp",
        type=str2bool,
        default=DEFAULT_USE_SELECTIVE_AMP,
        help=(
            "A3-v9: use mixed-precision Tensor-Core math only for the large dense Linear "
            "projections. Sparse propagation, custom GAT/Clifford/block CUDA kernels, "
            "LayerNorm, hop softmax, loss and optimizer master parameters stay FP32."
        ),
    )
    parser.add_argument(
        "--selective_amp_dtype",
        type=str,
        default=DEFAULT_SELECTIVE_AMP_DTYPE,
        choices=["bf16", "fp16"],
        help=(
            "A3-v9 dense GEMM dtype. bf16 is the default because it normally does not "
            "need gradient scaling; fp16 automatically enables GradScaler."
        ),
    )
    parser.add_argument(
        "--fast_backend_policy",
        type=str,
        default=DEFAULT_FAST_BACKEND_POLICY,
        choices=["auto", "strict", "off"],
        help=(
            "clean11-fast backend policy. auto keeps every compatible optimization and "
            "automatically falls back for incompatible hidden/head/hop settings; strict "
            "raises on an incompatible requested backend; off disables all custom fast backends."
        ),
    )
    parser.add_argument(
        "--generic_backend_profile",
        type=str,
        default=DEFAULT_GENERIC_BACKEND_PROFILE,
        choices=[
            "auto",
            "baseline",
            "pyg_tf32",
            "pyg_fixedcsr_tf32",
            "pyg_fixedcsr_pack_tf32",
            "safe_no_amp",
            "amazon_dgnn_fp32",
            "amazon_dgnn_block_fp32",
            "amazon_dgnn_bf16",
            "amazon_dgnn_full_bf16",
            "amazon_pyg_full_bf16",
            "amazon_collapsed_fp32",
            "amazon_collapsed_block_fp32",
            "amazon_collapsed_bf16",
            "amazon_collapsed_full_bf16",
        ],
        help=(
            "Full-model implementation profile used by the generalized speed guard. "
            "'auto' keeps the shape-aware production dispatcher; 'baseline' disables "
            "all custom acceleration; the other profiles are mathematically compatible "
            "candidate stacks benchmarked outside formal epoch timing."
        ),
    )
    parser.add_argument(
        "--gat_backend_registry",
        type=str,
        default=DEFAULT_GAT_BACKEND_REGISTRY,
        help=(
            "JSON registry generated by validate_gat_backend_matrix.py. In fast_backend_policy=auto, "
            "unvalidated hidden/head/dropout shapes use PyG GAT by default; only registry-approved "
            "shapes may use dgNN/A3-v3. Relative paths are resolved next to trainb9.py."
        ),
    )
    parser.add_argument(
        "--gat_backend_min_speedup",
        type=float,
        default=DEFAULT_GAT_BACKEND_MIN_SPEEDUP,
        help=(
            "Minimum measured dgNN/PyG speedup required for AUTO to select a registry-approved "
            "dgNN backend. Numerical safety is always required first."
        ),
    )

    parser.add_argument(
        "--init_gamma",
        type=float,
        default=DEFAULT_INIT_GAMMA,
        help="Initial value of the learnable residual gamma in each Clifford block.",
    )
    parser.add_argument(
        "--gamma_mode",
        type=str,
        default=DEFAULT_GAMMA_MODE,
        choices=["scalar", "vector"],
        help="Use one scalar gamma per block or one gamma per hidden channel.",
    )
    parser.add_argument(
        "--hop_gate_mode",
        type=str,
        default=DEFAULT_HOP_GATE_MODE,
        choices=["none", "global", "node"],
        help=(
            "Hop-level gate mode. "
            "'none' keeps the original concat behavior; "
            "'global' learns one shared softmax weight for each hop; "
            "'node' learns node-wise softmax weights for different hops."
        ),
    )
    parser.add_argument(
        "--operator_mode",
        type=str,
        default=DEFAULT_OPERATOR_MODE,
        choices=["laplacian", "adjacency", "hybrid"],
        help=(
            "Choose the operator P used in T_s(X)=P^sX: "
            "laplacian = L, adjacency = A_norm, "
            "hybrid = hybrid_alpha*L + (1-hybrid_alpha)*A_norm."
        ),
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=DEFAULT_HYBRID_ALPHA,
        help="Only used when --operator_mode hybrid. alpha in alpha*L + (1-alpha)*A_norm.",
    )
    parser.add_argument(
        "--layer_combine",
        type=str,
        default=DEFAULT_LAYER_COMBINE,
        choices=["last", "concat"],
        help=(
            "Final layer combination. 'concat' uses JK-style concatenation "
            "Concat(H0,H1,...,HL); 'last' uses only the last layer."
        ),
    )

    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    parser.add_argument("--print_every", type=int, default=DEFAULT_PRINT_EVERY)
    parser.add_argument(
        "--timing_discard_first_epochs",
        type=int,
        default=10,
        help=(
            "Reporting-only warmup exclusion for steady-state timing statistics. "
            "Training itself is unchanged and all epoch TrainStep values are still recorded. "
            "The legacy all-epoch mean/std is also retained for backward-compatible comparisons."
        ),
    )
    parser.add_argument(
        "--speed_probe_steps",
        type=int,
        default=0,
        help=(
            "Training-only autotune probe. When >0, run this many measured pure "
            "training steps after --speed_probe_warmup warmup steps, print one JSON "
            "result, and exit before evaluation. All graph/JIT preparation remains "
            "outside the measured interval."
        ),
    )
    parser.add_argument(
        "--speed_probe_warmup",
        type=int,
        default=3,
        help="Number of unreported training warmup steps before speed-probe measurement.",
    )
    parser.add_argument("--save_log", type=str2bool, default=DEFAULT_SAVE_LOG)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--save_min_best_val", type=float, default=DEFAULT_SAVE_MIN_BEST_VAL)
    parser.add_argument("--save_min_best_test", type=float, default=DEFAULT_SAVE_MIN_BEST_TEST)

    # ========================================================
    # Numerical tracing/debug arguments
    # ========================================================
    parser.add_argument(
        "--debug_numerics",
        type=str2bool,
        default=DEFAULT_DEBUG_NUMERICS,
        help=(
            "Enable staged NaN/Inf tracing. This checks train-forward logits, loss, "
            "parameter gradients, parameters after optimizer.step, evaluation logits, "
            "and (optionally) module outputs. It is intentionally slow and must not be "
            "used for runtime benchmarking or large sweeps."
        ),
    )
    parser.add_argument(
        "--debug_numerics_epochs",
        type=int,
        default=DEFAULT_DEBUG_NUMERICS_EPOCHS,
        help="When debug_numerics=True, stop after this many clean epochs.",
    )
    parser.add_argument(
        "--debug_numerics_modules",
        type=str2bool,
        default=DEFAULT_DEBUG_NUMERICS_MODULES,
        help="Register forward hooks to report the first module that emits NaN/Inf.",
    )
    parser.add_argument(
        "--debug_numerics_verbose",
        type=str2bool,
        default=DEFAULT_DEBUG_NUMERICS_VERBOSE,
        help="Print finite tensor statistics for every traced module; normally keep False.",
    )
    parser.add_argument(
        "--debug_detect_anomaly",
        type=str2bool,
        default=DEFAULT_DEBUG_DETECT_ANOMALY,
        help="Use torch.autograd.detect_anomaly(check_nan=True) around backward in debug mode.",
    )

    parser.add_argument(
        "--enable_dirichlet_recording",
        type=str2bool,
        default=DEFAULT_ENABLE_DIRICHLET_RECORDING,
        help=(
            "Allow the optional Dirichlet recorder configured by environment variables. "
            "Keep False for runtime benchmarking so no energy computation or JSONL I/O can run."
        ),
    )

    args = parser.parse_args()

    if args.hop_scales is None or len(args.hop_scales) == 0:
        args.hop_scales = DEFAULT_HOP_SCALES

    if int(args.debug_numerics_epochs) <= 0:
        raise ValueError("--debug_numerics_epochs must be >= 1")
    if int(args.timing_discard_first_epochs) < 0:
        raise ValueError("--timing_discard_first_epochs must be >= 0")
    if int(args.speed_probe_steps) < 0:
        raise ValueError("--speed_probe_steps must be >= 0")
    if int(args.speed_probe_warmup) < 0:
        raise ValueError("--speed_probe_warmup must be >= 0")
    if int(args.hidden_dim) <= 0 or int(args.hidden_dim) > 256:
        raise ValueError(
            f"The generalized engine supports 1<=hidden_dim<=256; got {args.hidden_dim}. "
            f"Primary certified grid: {list(GENERAL_FAST_HIDDEN_DIMS)}"
        )
    if int(args.gat_heads) <= 0:
        raise ValueError("--gat_heads must be positive")
    if int(args.num_layers) <= 0:
        raise ValueError("--num_layers must be positive")
    if not (1 <= int(args.dgnn_chunk_max_nodes) <= DGNN_VALIDATED_MAX_SAFE_NUM_NODES):
        raise ValueError(
            f"--dgnn_chunk_max_nodes must be in [1,{DGNN_VALIDATED_MAX_SAFE_NUM_NODES}], "
            f"got {args.dgnn_chunk_max_nodes}"
        )
    if int(args.dgnn_chunk_target_nodes) <= 0:
        raise ValueError("--dgnn_chunk_target_nodes must be positive")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if args.split_mode == "random_ratio":
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                f"When split_mode=random_ratio, train_ratio + val_ratio + test_ratio "
                f"must be 1.0, got {ratio_sum}."
            )
        if args.train_ratio <= 0 or args.val_ratio <= 0 or args.test_ratio <= 0:
            raise ValueError("train_ratio, val_ratio, and test_ratio must all be positive.")

    return args


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_tf32(enabled: bool):
    """Configure the A3-v4 TF32 experiment for CUDA float32 matrix multiplication.

    In the Torch 2.1.x environment used by this project, setting matmul precision
    to 'high' permits TF32 internally for eligible float32 CUDA matmuls/Linear
    operations; 'highest' keeps full FP32 matmul precision. Tensor dtypes remain
    float32 in both modes. cuDNN is set consistently even though this model does
    not rely on convolution layers.
    """
    enabled = bool(enabled)
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")



def _normalize_dataset_registry_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def _gat_registry_key(dataset_name: str, use_undirected: bool, hidden_dim: int, gat_heads: int, gat_dropout: float) -> str:
    return (
        f"dataset={_normalize_dataset_registry_name(dataset_name)}|"
        f"undir={int(bool(use_undirected))}|hidden={int(hidden_dim)}|"
        f"heads={int(gat_heads)}|gatdrop={float(gat_dropout):.8g}"
    )


def _resolve_registry_path(path_like: str) -> Path:
    p = Path(str(path_like)).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _current_gat_env_fingerprint():
    gpu_name = "cpu"
    capability = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            capability = list(torch.cuda.get_device_capability(0))
        except Exception:
            pass
    return {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "gpu_name": gpu_name,
        "compute_capability": capability,
    }


def _load_gat_backend_registry(path_like: str):
    path = _resolve_registry_path(path_like)
    if not path.exists():
        return None, path, "registry file not found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, path, f"registry parse failed: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("entries", {}), dict):
        return None, path, "registry has invalid schema"
    return payload, path, None


def _registry_env_matches(payload: dict):
    saved = payload.get("environment") or {}
    now = _current_gat_env_fingerprint()
    # Be strict on Torch/CUDA/GPU family because dgNN numerical behavior is
    # build/device dependent. Missing legacy fields are treated as mismatch.
    for k in ("torch", "torch_cuda", "gpu_name", "compute_capability"):
        if saved.get(k) != now.get(k):
            return False, f"environment mismatch at {k}: registry={saved.get(k)!r}, current={now.get(k)!r}"
    return True, "environment matched"


def _choose_gat_backend_from_registry(args):
    """Return (backend, source_note).

    backend is one of: 'pyg', 'dgnn_a3v3', 'collapsed_v6'. AUTO is fail-safe:
    absent/stale/unsafe registry entries always choose PyG.
    """
    # Built-in trusted checkpoint: this exact Amazon-ratings shape/dropout was
    # the end-to-end validated V9-BF16 fast path used during optimization.
    if (
        _normalize_dataset_registry_name(args.dataset_name) == "amazon-ratings"
        and bool(args.use_undirected)
        and int(args.hidden_dim) == 256
        and int(args.gat_heads) == 6
        and abs(float(args.gat_dropout) - 0.2) <= 1e-12
    ):
        return "collapsed_v6", "built-in validated Amazon-ratings V9 shape (hidden=256, heads=6, gat_dropout=0.2)"

    payload, path, err = _load_gat_backend_registry(args.gat_backend_registry)
    if payload is None:
        return "pyg", f"safe AUTO fallback: {err}; path={path}"
    env_ok, env_note = _registry_env_matches(payload)
    if not env_ok:
        return "pyg", f"safe AUTO fallback: {env_note}; path={path}"

    key = _gat_registry_key(
        args.dataset_name, args.use_undirected, args.hidden_dim, args.gat_heads, args.gat_dropout
    )
    entry = payload.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        return "pyg", f"safe AUTO fallback: no validated registry entry for {key}"
    if not bool(entry.get("safe", False)):
        return "pyg", f"safe AUTO fallback: registry marks shape unsafe ({entry.get('reason', 'unspecified')})"

    backend = str(entry.get("selected_backend", "pyg")).strip().lower()
    speedup = float(entry.get("selected_speedup_vs_pyg", 1.0) or 1.0)
    min_speedup = float(getattr(args, "gat_backend_min_speedup", DEFAULT_GAT_BACKEND_MIN_SPEEDUP))
    if backend not in {"dgnn_a3v3", "collapsed_v6"}:
        return "pyg", f"registry selected PyG for {key}"
    if speedup < min_speedup:
        return "pyg", f"registry backend {backend} is safe but speedup={speedup:.4f} < required {min_speedup:.4f}"
    return backend, f"registry-approved {backend}, speedup={speedup:.4f}x, key={key}"


def apply_dgnn_node_count_safety_gate(args, num_nodes: int):
    """Final correctness/performance dispatch after data.num_nodes is known.

    For N<=46340, the existing registry/shape-aware single-graph dgNN policy is
    preserved. For larger graphs, the currently installed dgNN binary cannot be
    used as one square graph. The chunked dgNN implementation is numerically
    correct, but it computes source-only local rows in every chunk and therefore
    can be slower than PyG on graphs with substantial cross-chunk neighborhoods.

    Production AUTO therefore selects PyG for the large-graph GAT component while
    keeping all independent generic fast paths (FixedCSR, GenericMultiHopCSR,
    GenericCliffordPackK, TF32, etc.). The chunked backend remains available via
    --large_graph_gat_backend chunked for explicit A/B tests.
    """
    n = int(num_nodes)
    limit = int(DGNN_VALIDATED_MAX_SAFE_NUM_NODES)
    args.dgnn_safe_node_limit = limit
    args.dgnn_node_safety_fallback = False
    args.dgnn_node_safety_reason = "not required"
    args.use_chunked_fused_gat = False

    if n <= limit:
        return args

    policy = str(getattr(args, "fast_backend_policy", "auto")).strip().lower()
    large_mode = str(getattr(args, "large_graph_gat_backend", "auto")).strip().lower()
    requested = bool(getattr(args, "requested_use_fused_gat", False))
    cuda_requested = (
        str(getattr(args, "device", "cuda")).lower().startswith("cuda")
        and torch.cuda.is_available()
    )

    # AUTO intentionally means "choose the proven faster safe backend".  The
    # current chunked dgNN path is correct but not assumed faster than PyG.
    if large_mode in {"auto", "pyg"} or policy == "off" or not requested or not cuda_requested:
        args.use_fused_gat = False
        args.use_fused_gat_sideops = False
        args.use_collapsed_fused_gat = False
        args.use_chunked_fused_gat = False
        args.gat_backend_dispatch = "pyg"
        args.dgnn_node_safety_fallback = True

        if large_mode == "auto":
            why = (
                "AUTO large-graph policy uses PyG; chunked dgNN remains correctness-validated "
                "but is performance-experimental"
            )
        elif large_mode == "pyg":
            why = "explicit --large_graph_gat_backend pyg"
        elif policy == "off":
            why = "fast_backend_policy=off"
        elif not requested:
            why = "fused GAT not requested"
        else:
            why = "CUDA unavailable/not requested"

        args.dgnn_node_safety_reason = (
            f"num_nodes={n}>{limit}; {why}"
        )
        previous_note = getattr(args, "gat_backend_dispatch_note", "n/a")
        args.gat_backend_dispatch_note = (
            f"LARGE-GRAPH SAFE/PERF FALLBACK: {args.dgnn_node_safety_reason}; "
            f"previous_note={previous_note}"
        )
        notes = list(getattr(args, "fast_backend_notes", []) or [])
        notes.append(f"GAT_LARGE_GRAPH: PyG ({args.dgnn_node_safety_reason})")
        args.fast_backend_notes = notes
        return args

    if large_mode != "chunked":
        raise ValueError(f"Unknown large_graph_gat_backend={large_mode!r}")

    previous_dispatch = getattr(args, "gat_backend_dispatch", "manual")
    previous_note = getattr(args, "gat_backend_dispatch_note", "n/a")
    args.use_fused_gat = True
    args.use_fused_gat_sideops = True
    args.use_collapsed_fused_gat = False
    args.use_chunked_fused_gat = True
    args.gat_backend_dispatch = "chunked_dgnn_a3v3"
    args.dgnn_node_safety_fallback = False
    args.dgnn_node_safety_reason = (
        f"num_nodes={n}>{limit}; explicitly using chunked dgNN with local-node cap="
        f"{int(args.dgnn_chunk_max_nodes)}"
    )
    args.gat_backend_dispatch_note = (
        f"EXPERIMENTAL LARGE-GRAPH GAT: {args.dgnn_node_safety_reason}; "
        f"previous_dispatch={previous_dispatch}; previous_note={previous_note}"
    )
    notes = list(getattr(args, "fast_backend_notes", []) or [])
    notes.append(f"GAT_LARGE_GRAPH: explicit chunked dgNN+A3v3 ({args.dgnn_node_safety_reason})")
    args.fast_backend_notes = notes
    return args



def apply_generic_backend_profile(args):
    """Apply one whole-model candidate stack before compatibility resolution.

    Generic profiles remain conservative. Amazon-specific profiles are more
    aggressive because Amazon-ratings is the compute-heavy case that benefits
    most from the full clean11 acceleration stack.

    IMPORTANT:
    - exact Amazon hidden=256/heads=6/hops=[1,2,3] is still routed to the
      completely frozen historical V9 package by the top-level dispatcher;
    - the aggressive Amazon profiles below are *candidates* for other shapes;
    - they are promoted only after a complete TrainStep speed probe succeeds.
    """
    profile = str(getattr(args, "generic_backend_profile", "auto")).strip().lower()
    args.generic_backend_profile = profile

    # Internal opt-in flags.  AUTO/conservative profiles keep these False.
    args.allow_generalized_block_ops = False
    args.allow_generalized_selective_amp = False
    args.allow_generalized_collapsed_gat = False
    args.force_generic_gat_backend = "auto"

    if profile == "auto":
        return args

    if profile == "baseline":
        args.fast_backend_policy = "off"
        return args

    # Common safe initialization for explicit candidate profiles.
    args.fast_backend_policy = "auto"
    args.use_fused_gat = False
    args.use_fused_gat_sideops = False
    args.use_collapsed_fused_gat = False
    args.use_fused_block_ops = False
    args.use_selective_amp = False
    args.large_graph_gat_backend = "pyg"

    if profile == "pyg_tf32":
        args.use_fixed_csr_spmm = False
        args.use_multihop_csr_pipeline = False
        args.use_fused_clifford_pack = False
        args.use_tf32 = True
        return args

    if profile == "pyg_fixedcsr_tf32":
        args.use_fixed_csr_spmm = True
        args.use_multihop_csr_pipeline = True
        args.multihop_backend = "auto"
        args.use_fused_clifford_pack = False
        args.use_tf32 = True
        return args

    if profile == "pyg_fixedcsr_pack_tf32":
        args.use_fixed_csr_spmm = True
        args.use_multihop_csr_pipeline = True
        args.multihop_backend = "auto"
        args.use_fused_clifford_pack = True
        args.use_tf32 = True
        return args

    if profile == "safe_no_amp":
        args.use_fixed_csr_spmm = True
        args.use_multihop_csr_pipeline = True
        args.multihop_backend = "auto"
        args.use_fused_clifford_pack = True
        args.use_tf32 = True
        args.use_fused_gat = True
        args.use_fused_gat_sideops = True
        args.use_collapsed_fused_gat = True
        args.use_fused_block_ops = False
        args.use_selective_amp = False
        return args

    # --------------------------------------------------------------
    # Amazon aggressive family.
    # --------------------------------------------------------------
    amazon_profiles = {
        "amazon_dgnn_fp32",
        "amazon_dgnn_block_fp32",
        "amazon_dgnn_bf16",
        "amazon_dgnn_full_bf16",
        "amazon_pyg_full_bf16",
        "amazon_collapsed_fp32",
        "amazon_collapsed_block_fp32",
        "amazon_collapsed_bf16",
        "amazon_collapsed_full_bf16",
    }
    if profile in amazon_profiles:
        dataset_key = str(args.dataset_name).strip().lower().replace("_", "-")
        if dataset_key != "amazon-ratings":
            raise ValueError(
                f"{profile} is Amazon-specific, but dataset_name={args.dataset_name!r}"
            )

        # Amazon gets the full propagation/Clifford/TF32 stack whenever the
        # current hop/gamma configuration is mathematically compatible.
        args.use_fixed_csr_spmm = True
        args.use_multihop_csr_pipeline = True
        args.multihop_backend = "auto"
        args.use_fused_clifford_pack = True
        args.use_tf32 = True

        # Never use the generalized collapsed kernel here.  The historical
        # collapsed specialization remains isolated in frozen 256x6 V9.
        args.use_collapsed_fused_gat = False

        is_collapsed = profile.startswith("amazon_collapsed_")

        if profile == "amazon_pyg_full_bf16":
            args.force_generic_gat_backend = "pyg"
            args.use_fused_gat = False
            args.use_fused_gat_sideops = False
            args.use_collapsed_fused_gat = False
        else:
            args.force_generic_gat_backend = "dgnn_a3v3"
            args.use_fused_gat = True
            args.use_fused_gat_sideops = True
            args.use_collapsed_fused_gat = bool(is_collapsed)
            if is_collapsed:
                args.allow_generalized_collapsed_gat = True

        if profile in {
            "amazon_dgnn_block_fp32",
            "amazon_dgnn_full_bf16",
            "amazon_pyg_full_bf16",
            "amazon_collapsed_block_fp32",
            "amazon_collapsed_full_bf16",
        }:
            args.use_fused_block_ops = True
            args.allow_generalized_block_ops = True

        if profile in {
            "amazon_dgnn_bf16",
            "amazon_dgnn_full_bf16",
            "amazon_pyg_full_bf16",
            "amazon_collapsed_bf16",
            "amazon_collapsed_full_bf16",
        }:
            args.use_selective_amp = True
            args.selective_amp_dtype = "bf16"
            args.allow_generalized_selective_amp = True

        return args

    raise ValueError(f"Unknown generic_backend_profile={profile!r}")


def build_generic_speed_signature(args):
    """Stable implementation-speed signature (split index intentionally omitted).

    Backend cost is determined mainly by graph/dataset, feature width, heads,
    depth, propagation orders and the structural/noise settings. Split index is
    omitted so ten-split experiments reuse one certified backend.
    """
    payload = {
        "engine_revision": "amazon_stage3_collapsed_v1",
        "dataset": str(args.dataset_name).strip().lower().replace("_", "-"),
        "use_undirected": bool(args.use_undirected),
        "split_mode": str(args.split_mode),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "hop_scales": [int(x) for x in args.hop_scales],
        "gat_heads": int(args.gat_heads),
        "gat_dropout": float(args.gat_dropout),
        "gamma_mode": str(args.gamma_mode),
        "hop_gate_mode": str(args.hop_gate_mode),
        "operator_mode": str(args.operator_mode),
        "hybrid_alpha": float(args.hybrid_alpha),
        "layer_combine": str(args.layer_combine),
        "edge_drop_ratio": float(os.environ.get("CLIFFORD_EDGE_DROP_RATIO", "0") or 0.0),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20], payload


def current_speed_environment():
    info = {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }
    return info


def resolve_fast_backend_compatibility(args):
    """Resolve the optimized backend for the current hyperparameter configuration.

    The original V9 package was tuned around Amazon-ratings (hidden=256,
    heads=6, hops=[1,2,3]). clean11 also contains broad hyperparameter sweeps,
    so specialized kernels must not make those sweeps crash. In ``auto`` mode
    we keep every mathematically compatible optimization and gracefully fall
    back only for the incompatible component. ``strict`` keeps the old fail-fast
    behavior. ``off`` restores the plain PyG/COO path.
    """
    policy = str(getattr(args, "fast_backend_policy", "auto")).strip().lower()
    if not hasattr(args, "requested_use_fused_gat"):
        args.requested_use_fused_gat = bool(getattr(args, "use_fused_gat", False))
    args.use_chunked_fused_gat = False
    notes = []

    def disable(attr: str, reason: str):
        if not bool(getattr(args, attr, False)):
            return
        if policy == "strict":
            raise ValueError(f"{attr}=True is incompatible: {reason}")
        setattr(args, attr, False)
        notes.append(f"{attr}: OFF ({reason})")

    if policy == "off":
        for attr in (
            "use_fused_gat",
            "use_fused_gat_sideops",
            "use_collapsed_fused_gat",
            "use_fused_block_ops",
            "use_fixed_csr_spmm",
            "use_multihop_csr_pipeline",
            "use_fused_clifford_pack",
            "use_tf32",
            "use_selective_amp",
        ):
            setattr(args, attr, False)
        args.fast_backend_notes = ["all optimized backends disabled by fast_backend_policy=off"]
        return args

    wants_cuda = str(args.device).lower().startswith("cuda")
    cuda_ok = torch.cuda.is_available() and wants_cuda
    if not cuda_ok:
        for attr in (
            "use_fused_gat",
            "use_fused_gat_sideops",
            "use_collapsed_fused_gat",
            "use_fused_block_ops",
            "use_multihop_csr_pipeline",
            "use_fused_clifford_pack",
            "use_tf32",
            "use_selective_amp",
        ):
            disable(attr, "CUDA device is required")

    # Numerically safe shape-aware GAT dispatch.  AUTO never sends an unknown
    # shape into dgNN.  A backend must either be the built-in validated Amazon
    # V9 specialization or be approved by validate_gat_backend_matrix.py.
    args.gat_backend_dispatch = "manual"
    args.gat_backend_dispatch_note = "fast_backend_policy is not auto"
    if policy == "auto" and cuda_ok and bool(args.use_fused_gat):
        forced_gat = str(getattr(args, "force_generic_gat_backend", "auto")).strip().lower()
        if forced_gat == "dgnn_a3v3":
            explicit_collapsed = bool(getattr(args, "allow_generalized_collapsed_gat", False)) and bool(args.use_collapsed_fused_gat)
            chosen_gat = "collapsed_v6" if explicit_collapsed else "dgnn_a3v3"
            gat_note = (
                f"explicit whole-model candidate forces {chosen_gat} for "
                f"{args.dataset_name}, hidden={args.hidden_dim}, heads={args.gat_heads}; "
                "node-count safety gate is still applied after data loading"
            )
        elif forced_gat == "pyg":
            chosen_gat = "pyg"
            gat_note = "explicit whole-model candidate forces PyG GAT"
        elif forced_gat == "auto":
            chosen_gat, gat_note = _choose_gat_backend_from_registry(args)
        else:
            raise ValueError(f"Unknown force_generic_gat_backend={forced_gat!r}")

        args.gat_backend_dispatch = chosen_gat
        args.gat_backend_dispatch_note = gat_note
        if chosen_gat == "pyg":
            if bool(args.use_collapsed_fused_gat):
                setattr(args, "use_collapsed_fused_gat", False)
            if bool(args.use_fused_gat_sideops):
                setattr(args, "use_fused_gat_sideops", False)
            setattr(args, "use_fused_gat", False)
            notes.append(f"GAT_AUTO: PyG ({gat_note})")
        elif chosen_gat == "dgnn_a3v3":
            setattr(args, "use_fused_gat", True)
            setattr(args, "use_fused_gat_sideops", True)
            setattr(args, "use_collapsed_fused_gat", False)
            notes.append(f"GAT_AUTO: dgNN+A3v3 ({gat_note})")
        elif chosen_gat == "collapsed_v6":
            setattr(args, "use_fused_gat", True)
            setattr(args, "use_fused_gat_sideops", True)
            setattr(args, "use_collapsed_fused_gat", True)
            notes.append(f"GAT_AUTO: V6-collapsed ({gat_note})")

    if bool(args.use_fused_gat_sideops) and not bool(args.use_fused_gat):
        disable("use_fused_gat_sideops", "requires use_fused_gat=True")

    # Shape-aware GAT dispatch.
    # The custom A3-v6 collapsed-head kernel is a proven specialization for
    # hidden_dim=256 and gat_heads=6.  For every other valid hyperparameter
    # shape we intentionally keep dgNN FusedGAT + A3-v3 sideops instead of
    # forcing the slower generalized collapsed kernel.  This preserves the
    # fastest validated backend currently available for each shape.
    if bool(args.use_collapsed_fused_gat):
        allow_generalized_collapsed = bool(
            getattr(args, "allow_generalized_collapsed_gat", False)
        )
        if not bool(args.use_fused_gat) or not bool(args.use_fused_gat_sideops):
            disable("use_collapsed_fused_gat", "requires fused GAT + fused GAT sideops")
        elif not (int(args.hidden_dim) == 256 and int(args.gat_heads) == 6):
            if not allow_generalized_collapsed:
                disable(
                    "use_collapsed_fused_gat",
                    "conservative AUTO reserves CollapsedGAT for hidden=256,heads=6; "
                    f"got hidden={args.hidden_dim}, heads={args.gat_heads}",
                )
            elif not (1 <= int(args.hidden_dim) <= 256 and 1 <= int(args.gat_heads) <= 32):
                disable(
                    "use_collapsed_fused_gat",
                    "generalized Amazon CollapsedGAT requires "
                    f"1<=hidden<=256 and 1<=heads<=32; got "
                    f"hidden={args.hidden_dim}, heads={args.gat_heads}",
                )

    # Stability-aware V8 dispatch.
    # The D=256 V8 kernel is the only path that has been validated in long
    # end-to-end training.  The generalized small-D kernel is kept in the
    # repository for isolated validation/benchmarking, but AUTO mode does not
    # use it during broad hyperparameter sweeps because small hidden widths can
    # otherwise produce unstable gradients/NaNs in full training.
    if bool(args.use_fused_block_ops):
        allow_generalized_block = bool(getattr(args, "allow_generalized_block_ops", False))
        if int(args.hidden_dim) != 256 and not allow_generalized_block:
            disable(
                "use_fused_block_ops",
                "conservative AUTO uses native PyTorch Dropout/LayerNorm/residual "
                f"for hidden_dim={args.hidden_dim}; generalized V8 is enabled only by "
                "an explicit Amazon full-stack candidate",
            )
        elif str(args.gamma_mode) != "vector":
            disable("use_fused_block_ops", f"V8 block fusion requires gamma_mode=vector, got {args.gamma_mode}")
        elif allow_generalized_block and not (1 <= int(args.hidden_dim) <= 256):
            disable(
                "use_fused_block_ops",
                f"generalized V8 kernel supports 1<=hidden_dim<=256, got {args.hidden_dim}",
            )

    if bool(args.use_multihop_csr_pipeline):
        if not bool(args.use_fixed_csr_spmm):
            disable("use_multihop_csr_pipeline", "requires use_fixed_csr_spmm=True")
        elif not args.hop_scales or min(map(int, args.hop_scales)) < 0:
            disable("use_multihop_csr_pipeline", f"MultiHop backend requires non-negative hop_scales, got {args.hop_scales}")
        else:
            hop_set = tuple(sorted(set(map(int, args.hop_scales))))
            mh_backend = str(getattr(args, "multihop_backend", "auto")).strip().lower()

            if mh_backend == "auto":
                if hop_set != (1, 2, 3):
                    disable(
                        "use_multihop_csr_pipeline",
                        f"performance-aware AUTO selects repeated FixedCSR for hops={list(hop_set)}; "
                        "native A3-v5 is reserved for exact [1,2,3]",
                    )
            elif mh_backend == "fixedcsr":
                disable(
                    "use_multihop_csr_pipeline",
                    f"explicit multihop_backend=fixedcsr for hops={list(hop_set)}",
                )
            elif mh_backend == "native3":
                if hop_set != (1, 2, 3):
                    if policy == "strict":
                        raise ValueError(
                            f"multihop_backend=native3 requires exact hops [1,2,3], got {list(hop_set)}"
                        )
                    setattr(args, "multihop_backend", "auto")
                    disable(
                        "use_multihop_csr_pipeline",
                        f"native3 requires exact [1,2,3], got {list(hop_set)}; falling back to FixedCSR",
                    )
            elif mh_backend in {"generic", "native3_tail"}:
                pass
            else:
                raise ValueError(f"Unknown multihop_backend={mh_backend!r}")

    # Generic CliffordPackK supports arbitrary non-empty hop counts and also
    # hop_gate_mode=none; K=3 automatically retains the specialized CUDA kernel.
    if bool(args.use_fused_clifford_pack) and not args.hop_scales:
        disable("use_fused_clifford_pack", "hop_scales cannot be empty")

    # Stability-aware AMP dispatch.  Selective BF16 is retained for the
    # hidden=256 path where it was validated end-to-end.  For smaller hidden
    # widths AUTO mode keeps TF32/FP32 graph training.  Small models are already
    # much cheaper, and this avoids precision-induced divergence contaminating
    # a large hyperparameter sweep.
    if bool(args.use_selective_amp) and cuda_ok:
        allow_generalized_amp = bool(getattr(args, "allow_generalized_selective_amp", False))
        if int(args.hidden_dim) != 256 and not allow_generalized_amp:
            disable(
                "use_selective_amp",
                f"conservative AUTO keeps FP32/TF32 for hidden_dim={args.hidden_dim}; "
                "generalized selective BF16 is enabled only by an explicit Amazon "
                "whole-model candidate",
            )
        elif str(args.selective_amp_dtype) == "bf16" and not torch.cuda.is_bf16_supported():
            if policy == "strict":
                raise RuntimeError("BF16 selective AMP requested but CUDA BF16 is unsupported")
            args.selective_amp_dtype = "fp16"
            notes.append("selective_amp_dtype: bf16 -> fp16 (CUDA BF16 unsupported)")

    args.fast_backend_notes = notes
    return args


def select_mask(mask: torch.Tensor, split_index: int = 0) -> torch.Tensor:
    if mask.dim() == 1:
        return mask
    if split_index < 0 or split_index >= mask.size(1):
        raise ValueError(
            f"split_index={split_index} is out of range for mask with shape {tuple(mask.shape)}"
        )
    return mask[:, split_index]


def generate_random_ratio_split(
    num_nodes: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate node-level random train/val/test masks.

    The split is performed over all nodes by random permutation.
    The test set receives the remaining nodes to avoid losing nodes due to rounding.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}.")

    generator = torch.Generator()
    generator.manual_seed(int(split_seed))

    perm = torch.randperm(num_nodes, generator=generator)

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    train_idx = perm[:num_train]
    val_idx = perm[num_train:num_train + num_val]
    test_idx = perm[num_train + num_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def apply_split(data, args):
    """
    Apply dataset split according to args.split_mode.

    split_mode='original':
        Use the dataset-provided masks.
        If masks are 2D, select the split specified by split_index.

    split_mode='random_ratio':
        Ignore the dataset-provided masks and generate new random masks.
    """
    if args.split_mode == "original":
        if not hasattr(data, "train_mask") or not hasattr(data, "val_mask") or not hasattr(data, "test_mask"):
            raise ValueError(
                "The dataset does not provide train_mask/val_mask/test_mask. "
                "Please use --split_mode random_ratio."
            )

        data.train_mask = select_mask(data.train_mask, args.split_index)
        data.val_mask = select_mask(data.val_mask, args.split_index)
        data.test_mask = select_mask(data.test_mask, args.split_index)

        return data

    if args.split_mode == "random_ratio":
        train_mask, val_mask, test_mask = generate_random_ratio_split(
            num_nodes=data.num_nodes,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_seed=args.split_seed,
        )

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data

    raise ValueError(f"Unknown split_mode={args.split_mode}.")


def get_raw_mask_shape(data):
    if hasattr(data, "train_mask"):
        return tuple(data.train_mask.shape)
    return None


def load_dataset(name: str, root: str):
    name = name.strip()

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=root, name=name, transform=NormalizeFeatures())
        data = dataset[0]
        return dataset, data

    if name in ["Texas", "Wisconsin", "Cornell"]:
        dataset = WebKB(root=root, name=name, transform=NormalizeFeatures())
        data = dataset[0]
        return dataset, data

    if name in ["Chameleon", "Squirrel"]:
        dataset = WikipediaNetwork(
            root=root,
            name=name.lower(),
            geom_gcn_preprocess=True,
        )
        data = dataset[0]
        return dataset, data

    if name == "Actor":
        dataset = Actor(root=root, transform=NormalizeFeatures())
        data = dataset[0]
        return dataset, data

    if name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
        dataset = HeterophilousGraphDataset(
            root=root,
            name=name,
            # transform=NormalizeFeatures()
        )
        data = dataset[0]
        return dataset, data

    raise ValueError(
        f"Unsupported dataset: {name}. "
        f"Supported datasets are: "
        f"Cora, CiteSeer, PubMed, Texas, Wisconsin, Cornell, "
        f"Chameleon, Squirrel, Actor, "
        f"Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions."
    )


def build_sparse_laplacian(
    edge_index: torch.Tensor,
    num_nodes: int,
    use_undirected: bool = True,
    normalization: str = "sym",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a sparse graph Laplacian L.

    By default this uses the symmetric normalized Laplacian:
        L = I - D^{-1/2} A D^{-1/2}.

    Returns
    -------
    edge_index_used : torch.Tensor
        The possibly undirected edge_index used by both GAT and the graph operator.
    laplacian : torch.sparse_coo_tensor
        Sparse Laplacian matrix L.
    """
    if use_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    lap_edge_index, lap_edge_weight = get_laplacian(
        edge_index=edge_index,
        edge_weight=None,
        normalization=normalization,
        num_nodes=num_nodes,
    )

    laplacian = torch.sparse_coo_tensor(
        indices=lap_edge_index,
        values=lap_edge_weight,
        size=(num_nodes, num_nodes),
        dtype=torch.float,
    ).coalesce()

    return edge_index, laplacian


def build_sparse_normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    use_undirected: bool = True,
    add_self_loop: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build symmetric normalized adjacency:
        A_norm = D^{-1/2} (A + I) D^{-1/2}.

    This corresponds to the B version selected by:
        --operator_mode adjacency
    """
    if use_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    if add_self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adjacency = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight,
        size=(num_nodes, num_nodes),
        dtype=torch.float,
    ).coalesce()

    return edge_index, adjacency


def sparse_linear_combination(A: torch.Tensor, B: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Return alpha*A + (1-alpha)*B for two sparse COO matrices with the same shape.
    Concatenating indices and coalescing safely merges duplicate entries.
    """
    A = A.coalesce()
    B = B.coalesce()
    if A.shape != B.shape:
        raise ValueError(f"Sparse matrices must have the same shape, got {A.shape} and {B.shape}.")

    indices = torch.cat([A.indices(), B.indices()], dim=1)
    values = torch.cat([alpha * A.values(), (1.0 - alpha) * B.values()], dim=0)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=A.shape,
        dtype=A.dtype,
        device=A.device,
    ).coalesce()


def build_graph_operator(
    edge_index: torch.Tensor,
    num_nodes: int,
    use_undirected: bool,
    operator_mode: str,
    hybrid_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the operator P used by the Clifford block:
        T_s(X) = P^s X.

    Available choices:
        laplacian : P = L = I - D^{-1/2} A D^{-1/2}
        adjacency : P = A_norm = D^{-1/2} (A + I) D^{-1/2}
        hybrid    : P = alpha L + (1-alpha) A_norm

    The laplacian and adjacency modes construct only the selected operator.
    This keeps each single-operator experiment clean.
    """
    if not (0.0 <= hybrid_alpha <= 1.0):
        raise ValueError("hybrid_alpha must be in [0, 1].")

    if operator_mode == "laplacian":
        return build_sparse_laplacian(
            edge_index=edge_index,
            num_nodes=num_nodes,
            use_undirected=use_undirected,
            normalization="sym",
        )

    if operator_mode == "adjacency":
        return build_sparse_normalized_adjacency(
            edge_index=edge_index,
            num_nodes=num_nodes,
            use_undirected=use_undirected,
            add_self_loop=True,
        )

    if operator_mode == "hybrid":
        edge_index_used, laplacian = build_sparse_laplacian(
            edge_index=edge_index,
            num_nodes=num_nodes,
            use_undirected=use_undirected,
            normalization="sym",
        )
        _, adjacency = build_sparse_normalized_adjacency(
            edge_index=edge_index_used,
            num_nodes=num_nodes,
            use_undirected=False,
            add_self_loop=True,
        )
        graph_operator = sparse_linear_combination(laplacian, adjacency, hybrid_alpha)
        return edge_index_used, graph_operator

    raise ValueError(f"Unknown operator_mode={operator_mode}.")



def apply_structure_noise_from_env(
    edge_index: torch.Tensor,
    num_nodes: int,
    use_undirected: bool,
):
    """Apply clean11 robustness edge deletion before operator/GAT preprocessing.

    Environment protocol used by ``sweep_train_full_edge_drop_parallel_gpu.py``:
      CLIFFORD_STRUCTURE_NOISE_TYPE=none|edge_drop
      CLIFFORD_EDGE_DROP_RATIO=0.0..1.0
      CLIFFORD_EDGE_DROP_SEED=<int>

    For undirected experiments, an unordered node pair is sampled once and both
    directions are kept/dropped together. Self-loops are excluded from the
    deletion pool because GAT/operator self-loops are rebuilt afterwards.
    """
    noise_type = os.environ.get("CLIFFORD_STRUCTURE_NOISE_TYPE", "none").strip().lower()
    ratio = float(os.environ.get("CLIFFORD_EDGE_DROP_RATIO", "0.0"))
    seed = int(os.environ.get("CLIFFORD_EDGE_DROP_SEED", "42"))

    if noise_type in {"", "none", "off"} or ratio <= 0.0:
        return edge_index, {
            "type": "none",
            "ratio": 0.0,
            "seed": seed,
            "edges_before": int(edge_index.size(1)),
            "edges_after": int(edge_index.size(1)),
        }
    if noise_type != "edge_drop":
        raise ValueError(f"Unsupported CLIFFORD_STRUCTURE_NOISE_TYPE={noise_type!r}")
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"CLIFFORD_EDGE_DROP_RATIO must be in [0,1), got {ratio}")

    # Structural self-loops are ignored here; the graph builder/GAT preparation
    # deterministically re-adds the appropriate self-loop set later.
    edge_index, _ = remove_self_loops(edge_index)
    before = int(edge_index.size(1))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    if use_undirected:
        e = to_undirected(edge_index, num_nodes=num_nodes)
        row, col = e.cpu()
        u = torch.minimum(row, col)
        v = torch.maximum(row, col)
        mask = u < v
        u, v = u[mask], v[mask]
        keys = u * int(num_nodes) + v
        keys = torch.unique(keys, sorted=True)
        num_pairs = int(keys.numel())
        num_drop = min(num_pairs, int(round(num_pairs * ratio)))
        if num_drop > 0:
            perm = torch.randperm(num_pairs, generator=gen)
            keep_mask = torch.ones(num_pairs, dtype=torch.bool)
            keep_mask[perm[:num_drop]] = False
            keys = keys[keep_mask]
        u = torch.div(keys, int(num_nodes), rounding_mode="floor")
        v = keys % int(num_nodes)
        kept = torch.cat([
            torch.stack([u, v], dim=0),
            torch.stack([v, u], dim=0),
        ], dim=1).to(edge_index.device)
        dropped_units = num_drop
        total_units = num_pairs
    else:
        num_edges = int(edge_index.size(1))
        num_drop = min(num_edges, int(round(num_edges * ratio)))
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        if num_drop > 0:
            perm = torch.randperm(num_edges, generator=gen)
            keep_mask[perm[:num_drop]] = False
        kept = edge_index[:, keep_mask.to(edge_index.device)]
        dropped_units = num_drop
        total_units = num_edges

    info = {
        "type": "edge_drop",
        "ratio": ratio,
        "seed": seed,
        "sampling_units": int(total_units),
        "dropped_units": int(dropped_units),
        "edges_before": before,
        "edges_after": int(kept.size(1)),
        "undirected_pairwise": bool(use_undirected),
    }
    return kept.contiguous(), info


def prepare_gat_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Reproduce GATConv(add_self_loops=True) once before training.

    GATConv normally removes existing self-loops and adds exactly one self-loop
    per node on every forward call. The graph is fixed here, so doing the same
    transformation once avoids repeated edge preprocessing without changing the
    GAT graph seen by the model.
    """
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index


@torch.inference_mode()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return 0.0
    pred = logits[mask].argmax(dim=-1)
    correct = (pred == y[mask]).sum().item()
    total = mask.sum().item()
    return correct / total


@torch.inference_mode()
def binary_roc_auc(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if roc_auc_score is None:
        raise ImportError("scikit-learn is required for ROC AUC evaluation. Please install scikit-learn.")

    if mask.sum().item() == 0:
        return 0.0

    masked_logits = logits[mask]
    # Do not let sklearn be the first place that notices numerical divergence.
    # Returning NaN lets the training loop stop this configuration cleanly.
    if not torch.isfinite(masked_logits).all():
        return float("nan")

    y_true = y[mask].detach().cpu().numpy()
    if len(np.unique(y_true)) < 2:
        return 0.0

    if masked_logits.dim() == 1:
        y_score = torch.sigmoid(masked_logits).detach().cpu().numpy()
    elif masked_logits.size(-1) == 1:
        y_score = torch.sigmoid(masked_logits.squeeze(-1)).detach().cpu().numpy()
    else:
        y_score = F.softmax(masked_logits, dim=-1)[:, 1].detach().cpu().numpy()

    if not np.isfinite(y_score).all():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


@torch.inference_mode()
def compute_metric(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, dataset_name: str) -> float:
    if uses_auc(dataset_name):
        return binary_roc_auc(logits, y, mask)
    return accuracy(logits, y, mask)


def forward_model(model, data, graph_operator: torch.Tensor):
    return model(data.x, data.edge_index, graph_operator)




def _iter_tensors(value, prefix=""):
    """Yield (path, tensor) pairs from nested module inputs/outputs."""
    if torch.is_tensor(value):
        yield prefix or "tensor", value
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            child = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _iter_tensors(item, child)
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(item, child)


def _tensor_numeric_stats(tensor: torch.Tensor):
    """Return finite/NaN/Inf statistics. Intended only for debug mode."""
    t = tensor.detach()
    if t.layout != torch.strided:
        return {
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "finite": True,
            "nan": 0,
            "posinf": 0,
            "neginf": 0,
            "max_abs": None,
            "note": f"non-strided layout={t.layout}; value scan skipped",
        }
    if not (t.is_floating_point() or t.is_complex()):
        return {
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "finite": True,
            "nan": 0,
            "posinf": 0,
            "neginf": 0,
            "max_abs": None,
        }
    finite_mask = torch.isfinite(t)
    all_finite = bool(finite_mask.all().item())
    nan_count = int(torch.isnan(t).sum().item())
    posinf_count = int(torch.isposinf(t).sum().item()) if not t.is_complex() else 0
    neginf_count = int(torch.isneginf(t).sum().item()) if not t.is_complex() else 0
    if t.numel() == 0:
        max_abs = 0.0
    else:
        finite_values = t[finite_mask]
        max_abs = float(finite_values.abs().max().item()) if finite_values.numel() else float("nan")
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "finite": all_finite,
        "nan": nan_count,
        "posinf": posinf_count,
        "neginf": neginf_count,
        "max_abs": max_abs,
    }


def _format_numeric_stats(stats) -> str:
    base = (
        f"shape={stats['shape']} dtype={stats['dtype']} device={stats['device']} "
        f"finite={stats['finite']} nan={stats['nan']} +inf={stats['posinf']} -inf={stats['neginf']}"
    )
    if stats.get("max_abs") is not None:
        base += f" max_abs={stats['max_abs']:.6e}"
    if stats.get("note"):
        base += f" note={stats['note']}"
    return base


class NumericalTracer:
    """Stage-aware NaN/Inf tracer for short diagnostic runs.

    Forward hooks are diagnostic only. They do not alter tensors. The tracer keeps
    only the first failure message, so the training loop can stop at the earliest
    known corruption stage instead of letting sklearn or a later CUDA kernel fail.
    """

    def __init__(self, model, enabled=True, trace_modules=True, verbose=False):
        self.model = model
        self.enabled = bool(enabled)
        self.trace_modules = bool(trace_modules)
        self.verbose = bool(verbose)
        self.epoch = 0
        self.stage = "init"
        self.first_failure = None
        self.handles = []
        if self.enabled and self.trace_modules:
            self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if not name:
                continue
            # Container-only modules do not need hooks. All computational modules,
            # including custom GraphClifford blocks/GAT wrappers, are retained.
            if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)):
                continue
            self.handles.append(module.register_forward_hook(self._make_forward_hook(name)))

    def _make_forward_hook(self, name):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            for path, tensor in _iter_tensors(output, "output"):
                if not (tensor.is_floating_point() or tensor.is_complex()):
                    continue
                stats = _tensor_numeric_stats(tensor)
                if self.verbose:
                    print(
                        f"[NUMTRACE][E{self.epoch:03d}][{self.stage}] "
                        f"MODULE={name} {path} {_format_numeric_stats(stats)}"
                    )
                if not stats["finite"]:
                    self.fail(
                        f"module output became non-finite: module={name}, {path}, "
                        f"{_format_numeric_stats(stats)}"
                    )
                    return
        return hook

    def set_context(self, epoch: int, stage: str):
        self.epoch = int(epoch)
        self.stage = str(stage)

    def fail(self, reason: str):
        if self.first_failure is None:
            self.first_failure = f"epoch={self.epoch} stage={self.stage}: {reason}"
            print(f"[NUMTRACE][FIRST_NONFINITE] {self.first_failure}")

    def clear_failure(self):
        self.first_failure = None

    def check_tensor(self, label: str, tensor: torch.Tensor, always_print=False):
        if not self.enabled:
            return True
        stats = _tensor_numeric_stats(tensor)
        if always_print or self.verbose or not stats["finite"]:
            print(
                f"[NUMTRACE][E{self.epoch:03d}][{self.stage}] "
                f"{label}: {_format_numeric_stats(stats)}"
            )
        if not stats["finite"]:
            self.fail(f"{label} became non-finite: {_format_numeric_stats(stats)}")
            return False
        return True

    def check_model_parameters(self, label: str, gradients=False, always_print_summary=True):
        if not self.enabled:
            return True
        checked = 0
        max_abs = 0.0
        worst_name = None
        for name, param in self.model.named_parameters():
            tensor = param.grad if gradients else param.data
            if tensor is None:
                continue
            checked += 1
            stats = _tensor_numeric_stats(tensor)
            if stats.get("max_abs") is not None and np.isfinite(stats["max_abs"]):
                if stats["max_abs"] >= max_abs:
                    max_abs = stats["max_abs"]
                    worst_name = name
            if not stats["finite"]:
                kind = "gradient" if gradients else "parameter"
                print(
                    f"[NUMTRACE][E{self.epoch:03d}][{self.stage}] "
                    f"NONFINITE_{kind.upper()} name={name} {_format_numeric_stats(stats)}"
                )
                self.fail(
                    f"non-finite {kind}: name={name}, {_format_numeric_stats(stats)}"
                )
                return False
        if always_print_summary:
            kind = "gradients" if gradients else "parameters"
            print(
                f"[NUMTRACE][E{self.epoch:03d}][{self.stage}] "
                f"{label}: checked={checked} all_finite=True "
                f"largest_max_abs={max_abs:.6e} name={worst_name}"
            )
        return True

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def train_one_epoch_debug(
    model,
    data,
    graph_operator,
    optimizer,
    device: torch.device,
    tracer: NumericalTracer,
    epoch: int,
    grad_scaler=None,
    detect_anomaly=True,
):
    """One diagnostic training epoch with staged finite-value checks.

    IMPORTANT: elapsed time from this function includes debug synchronization and
    scans and is NOT a valid runtime benchmark.
    """
    model.train()
    tracer.clear_failure()
    tracer.set_context(epoch, "pre_forward")
    tracer.check_tensor("data.x", data.x, always_print=True)
    tracer.check_model_parameters("parameters_before_forward", gradients=False)
    if tracer.first_failure:
        return float("nan"), 0.0, tracer.first_failure

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    optimizer.zero_grad()
    tracer.set_context(epoch, "train_forward")
    logits = forward_model(model, data, graph_operator)
    tracer.check_tensor("train_logits", logits, always_print=True)
    if tracer.first_failure:
        if device.type == "cuda":
            torch.cuda.synchronize()
        return float("nan"), time.perf_counter() - start, tracer.first_failure

    tracer.set_context(epoch, "loss")
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    tracer.check_tensor("loss", loss.reshape(1), always_print=True)
    if tracer.first_failure:
        if device.type == "cuda":
            torch.cuda.synchronize()
        return float(loss.detach().item()), time.perf_counter() - start, tracer.first_failure

    tracer.set_context(epoch, "backward")
    try:
        anomaly_ctx = torch.autograd.detect_anomaly(check_nan=True) if detect_anomaly else nullcontext()
        with anomaly_ctx:
            if grad_scaler is not None and grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                # Expose true (unscaled) gradients before checking them.
                grad_scaler.unscale_(optimizer)
            else:
                loss.backward()
    except RuntimeError as exc:
        tracer.fail(f"backward raised RuntimeError under anomaly detection: {exc}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        return float(loss.detach().item()), time.perf_counter() - start, tracer.first_failure

    tracer.set_context(epoch, "post_backward")
    tracer.check_model_parameters("parameter_gradients", gradients=True)
    if tracer.first_failure:
        if device.type == "cuda":
            torch.cuda.synchronize()
        return float(loss.detach().item()), time.perf_counter() - start, tracer.first_failure

    tracer.set_context(epoch, "optimizer_step")
    if grad_scaler is not None and grad_scaler.is_enabled():
        # Gradients were already unscaled above. GradScaler remembers that state.
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tracer.set_context(epoch, "post_optimizer")
    tracer.check_model_parameters("parameters_after_optimizer", gradients=False)
    return float(loss.detach().item()), elapsed, tracer.first_failure


def train_one_epoch(model, data, graph_operator, optimizer, device: torch.device, grad_scaler=None) -> Tuple[float, float]:
    """
    Run one parameter-update step and return:
        loss_value, pure_training_seconds

    The measured interval contains only:
        optimizer.zero_grad -> training forward -> loss -> backward -> optimizer.step

    It excludes evaluation, metric computation, logging, data loading, early
    stopping logic, and all other runner overhead.
    """
    model.train()

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        optimizer.zero_grad()
        logits = forward_model(model, data, graph_operator)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        end_event.record()

        # Synchronization is performed after the measured interval. It is needed
        # only to obtain an accurate elapsed time from the CUDA events.
        end_event.synchronize()
        train_seconds = start_event.elapsed_time(end_event) / 1000.0
    else:
        start_time = time.perf_counter()
        optimizer.zero_grad()
        logits = forward_model(model, data, graph_operator)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_seconds = time.perf_counter() - start_time

    loss_value = float(loss.detach().item())
    return loss_value, train_seconds


@torch.inference_mode()
def evaluate_logits(model, data, graph_operator) -> torch.Tensor:
    """Run the single full-graph evaluation forward pass for an epoch."""
    model.eval()
    return forward_model(model, data, graph_operator)


def mean_sample_std(values) -> Tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    mean_value = float(np.mean(values))
    if len(values) == 1:
        return mean_value, 0.0
    return mean_value, float(np.std(values, ddof=1))

def maybe_save_log(args, run_name: str, log_text: str, best_val: float, best_test_at_best_val: float):
    if not args.save_log:
        return None

    best_val_pct = best_val * 100.0
    best_test_pct = best_test_at_best_val * 100.0

    if not (best_val_pct > args.save_min_best_val and best_test_pct > args.save_min_best_test):
        return None

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = make_unique_log_path(log_dir, run_name)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)

    return log_path


def main():
    args = parse_args()
    args = apply_generic_backend_profile(args)
    args = resolve_fast_backend_compatibility(args)

    cign_switch_modes = get_cign_switch_modes()

    # V3: NEW interaction no longer disables Clifford acceleration.
    # For K=3 it uses fused_clifford_switch_pack; OLD interaction keeps the
    # existing production fused_clifford_pack. Other acceleration flags remain
    # exactly as resolved by the current clean_new backend policy.
    configure_tf32(args.use_tf32)

    # Buffer the console only when the single-run script itself must save a log.
    # The multi-split runner already captures stdout, so buffering it again when
    # save_log=False would only add Python and memory overhead.
    logger = ConsoleBuffer() if args.save_log else None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if logger is not None:
        sys.stdout = logger
        sys.stderr = logger

    exit_code = 0
    tracer = None
    try:
        seed_everything(args.seed)
        device = torch.device(args.device)
        metric_name = get_metric_name(args.dataset_name)

        dataset, data = load_dataset(args.dataset_name, args.data_root)
        raw_train_mask_shape = get_raw_mask_shape(data)
        data = apply_split(data, args)

        # Optional structural robustness perturbation. This is deliberately
        # outside epoch timing and happens before both graph-operator and GAT
        # fixed-graph preprocessing, so every backend sees the same perturbed graph.
        data.edge_index, structure_noise_info = apply_structure_noise_from_env(
            data.edge_index,
            num_nodes=data.num_nodes,
            use_undirected=args.use_undirected,
        )

        # Final, highest-priority GAT correctness gate. Backend compatibility
        # and registry selection happen before dataset loading, so the node-count
        # limit must be enforced here, after data.num_nodes is known and before
        # any dgNN-dependent compilation/model construction/preprocessing.
        args = apply_dgnn_node_count_safety_gate(args, data.num_nodes)

        # Build the run identity from the EFFECTIVE backend configuration after
        # the safety gate, so filenames/logs cannot claim dgNN when PyG was forced.
        run_name = build_run_name(args)
        full_run_config = build_full_run_config(args)

        operator_edge_index, graph_operator = build_graph_operator(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            use_undirected=args.use_undirected,
            operator_mode=args.operator_mode,
            hybrid_alpha=args.hybrid_alpha,
        )

        # Match GATConv's original remove-self-loops + add-self-loops behavior
        # once, then use GATConv(add_self_loops=False) in every model block.
        data.edge_index = prepare_gat_edge_index(
            edge_index=operator_edge_index,
            num_nodes=data.num_nodes,
        )

        data = data.to(device)
        graph_operator = graph_operator.to(device)

        # --------------------------------------------------------
        # Fixed graph-operator CSR preprocessing.
        # The COO->CSR conversion, transpose, coalesce and storage cloning are
        # intentionally done once here, outside every epoch timing interval.
        # --------------------------------------------------------
        fixed_csr_prep_seconds = 0.0
        fixed_csr_info = None
        if args.use_fixed_csr_spmm:
            if device.type == "cuda":
                torch.cuda.synchronize()
            fixed_csr_prep_start = time.perf_counter()
            graph_operator = prepare_fixed_csr_operator(graph_operator)
            if device.type == "cuda":
                torch.cuda.synchronize()
            fixed_csr_prep_seconds = time.perf_counter() - fixed_csr_prep_start
            fixed_csr_info = dict(graph_operator.info)

        # --------------------------------------------------------
        # Generic MultiHopCSR preprocessing (native three-hop specialization when applicable).
        # The extension, int32 CSR storage, descriptors and workspace are all
        # created here, outside every measured training epoch.
        # --------------------------------------------------------
        multihop_csr_prep_seconds = 0.0
        multihop_csr_info = None
        if args.use_multihop_csr_pipeline:
            if not args.use_fixed_csr_spmm:
                raise ValueError("--use_multihop_csr_pipeline True requires --use_fixed_csr_spmm True")
            if device.type != "cuda":
                raise RuntimeError("Generic MultiHopCSR fast path requires CUDA.")
            torch.cuda.synchronize()
            multihop_start = time.perf_counter()
            hop_set = tuple(sorted(set(map(int, args.hop_scales))))
            mh_backend = str(getattr(args, "multihop_backend", "auto")).strip().lower()

            if mh_backend in {"auto", "native3"}:
                ensure_multihop_csr_loaded()
                strategy = "auto"
                prefer_native3 = True
            elif mh_backend == "generic":
                strategy = "generic"
                prefer_native3 = False
            elif mh_backend == "native3_tail":
                if max(hop_set) < 3:
                    raise ValueError(
                        f"multihop_backend=native3_tail requires max hop >=3, got {list(hop_set)}"
                    )
                ensure_multihop_csr_loaded()
                strategy = "native3_tail"
                prefer_native3 = True
            else:
                raise RuntimeError(
                    f"Unexpected active MultiHop backend {mh_backend!r}; "
                    "fixedcsr should have disabled use_multihop_csr_pipeline before preprocessing"
                )

            graph_operator = prepare_multihop_csr_operator(
                graph_operator,
                dense_cols=2 * int(args.hidden_dim),
                algorithm=int(args.multihop_csr_algorithm),
                powers=args.hop_scales,
                prefer_native_three_hop=prefer_native3,
                strategy=strategy,
            )
            torch.cuda.synchronize()
            multihop_csr_prep_seconds = time.perf_counter() - multihop_start
            multihop_csr_info = dict(graph_operator.info)

        # --------------------------------------------------------
        # A3-v2 FusedGAT side-op CUDA JIT compilation/loading.
        # This remains outside every epoch timing interval.
        # --------------------------------------------------------
        a3v2_gat_compile_seconds = 0.0
        if args.use_fused_gat_sideops:
            if not args.use_fused_gat:
                raise ValueError("--use_fused_gat_sideops True requires --use_fused_gat True")
            if device.type != "cuda":
                raise RuntimeError("--use_fused_gat_sideops True requires CUDA.")
            torch.cuda.synchronize()
            side_compile_start = time.perf_counter()
            ensure_fused_gat_sideops_loaded(verbose=False)
            torch.cuda.synchronize()
            a3v2_gat_compile_seconds = time.perf_counter() - side_compile_start

        # --------------------------------------------------------
        # A3-v8 block fusion CUDA JIT compilation/loading.
        # --------------------------------------------------------
        a3v8_block_compile_seconds = 0.0
        if args.use_fused_block_ops:
            if device.type != "cuda":
                raise RuntimeError("--use_fused_block_ops True requires CUDA.")
            if not (1 <= int(args.hidden_dim) <= 256) or str(args.gamma_mode) != "vector":
                raise ValueError(
                    "Generalized V8 requires 1<=hidden_dim<=256 and gamma_mode=vector; "
                    f"got hidden={args.hidden_dim}, gamma_mode={args.gamma_mode}."
                )
            torch.cuda.synchronize()
            v8_compile_start = time.perf_counter()
            ensure_fused_block_ops_loaded(verbose=False)
            torch.cuda.synchronize()
            a3v8_block_compile_seconds = time.perf_counter() - v8_compile_start

        # --------------------------------------------------------
        # A3-v6 specialized collapsed-head GAT CUDA JIT compilation/loading.
        # Shape-aware dispatch enables this only for hidden=256, heads=6.
        # Compilation remains completely outside measured epoch timing.
        # --------------------------------------------------------
        a3v6_gat_compile_seconds = 0.0
        if args.use_collapsed_fused_gat:
            if not args.use_fused_gat:
                raise ValueError("--use_collapsed_fused_gat True requires --use_fused_gat True")
            if not args.use_fused_gat_sideops:
                raise ValueError("--use_collapsed_fused_gat True requires --use_fused_gat_sideops True")
            if device.type != "cuda":
                raise RuntimeError("--use_collapsed_fused_gat True requires CUDA.")
            allow_generalized_collapsed = bool(
                getattr(args, "allow_generalized_collapsed_gat", False)
            )
            if not (int(args.gat_heads) == 6 and int(args.hidden_dim) == 256):
                if not allow_generalized_collapsed:
                    raise ValueError(
                        "CollapsedGAT outside hidden=256,heads=6 requires an explicit "
                        "Amazon generalized-collapsed profile."
                    )
                if not (
                    1 <= int(args.gat_heads) <= 32
                    and 1 <= int(args.hidden_dim) <= 256
                ):
                    raise ValueError(
                        "Generalized Amazon CollapsedGAT requires "
                        f"1<=heads<=32, 1<=hidden<=256; got "
                        f"heads={args.gat_heads}, hidden={args.hidden_dim}."
                    )
            torch.cuda.synchronize()
            collapsed_compile_start = time.perf_counter()
            ensure_collapsed_gat_loaded(verbose=False)
            torch.cuda.synchronize()
            a3v6_gat_compile_seconds = time.perf_counter() - collapsed_compile_start

        # --------------------------------------------------------
        # CliffordPack CUDA JIT compilation/loading (specialized K=3 or generic K).
        # This happens once before optimizer creation and before any epoch timing.
        # --------------------------------------------------------
        a3_compile_seconds = 0.0
        if args.use_fused_clifford_pack:
            if device.type != "cuda":
                raise RuntimeError("--use_fused_clifford_pack True requires CUDA.")

            torch.cuda.synchronize()
            a3_compile_start = time.perf_counter()

            if cign_switch_modes["interaction"] == "new" and os.environ.get("CIGN_OUTER_NONLINEARITY") == "none":
                if len(args.hop_scales) == 3:
                    from fused_clifford_cign_pack import ensure_fused_clifford_switch_pack_loaded as ensure_cign
                else:
                    from fused_clifford_cign_generic import ensure_generic_fused_clifford_pack_loaded as ensure_cign
                ensure_cign(False)
            elif (
                cign_switch_modes["interaction"] == "new"
                and len(args.hop_scales) == 3
            ):
                ensure_fused_clifford_switch_pack_loaded(
                    verbose=False
                )
            else:
                ensure_fused_clifford_pack_loaded(
                    verbose=False,
                    hop_count=len(args.hop_scales),
                )

            torch.cuda.synchronize()
            a3_compile_seconds = (
                time.perf_counter() - a3_compile_start
            )

        model = build_model(
            in_dim=dataset.num_features,
            hidden_dim=args.hidden_dim,
            out_dim=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            hop_scales=args.hop_scales,
            gat_heads=args.gat_heads,
            gat_dropout=args.gat_dropout,
            use_fused_gat=args.use_fused_gat,
            use_fused_gat_sideops=args.use_fused_gat_sideops,
            use_chunked_fused_gat=getattr(args, "use_chunked_fused_gat", False),
            dgnn_chunk_max_nodes=int(args.dgnn_chunk_max_nodes),
            dgnn_chunk_target_nodes=int(args.dgnn_chunk_target_nodes),
            use_collapsed_fused_gat=args.use_collapsed_fused_gat,
            use_fused_clifford_pack=args.use_fused_clifford_pack,
            use_fused_block_ops=args.use_fused_block_ops,
            use_selective_amp=args.use_selective_amp,
            selective_amp_dtype=args.selective_amp_dtype,
            init_gamma=args.init_gamma,
            gamma_mode=args.gamma_mode,
            hop_gate_mode=args.hop_gate_mode,
            alpha_score_mode=cign_switch_modes["alpha_score"],
            interaction_mode=cign_switch_modes["interaction"],
            alpha_weighting_mode=cign_switch_modes["alpha_weighting"],
            layer_combine=args.layer_combine,
        ).to(device)

        # --------------------------------------------------------
        # FusedGAT fixed-graph preprocessing.
        # This is intentionally outside every epoch timing interval.
        # --------------------------------------------------------
        fused_graph_prep_seconds = 0.0
        fused_graph_info = None
        if args.use_fused_gat:
            if device.type != "cuda":
                raise RuntimeError("--use_fused_gat True requires a CUDA device.")
            torch.cuda.synchronize()
            prep_start = time.perf_counter()
            fused_graph_info = model.prepare_fused_gat_graph(
                data.edge_index,
                num_nodes=data.num_nodes,
            )
            torch.cuda.synchronize()
            fused_graph_prep_seconds = time.perf_counter() - prep_start

        # Runtime benchmarking defaults to a strict recorder-off policy.
        # When explicitly enabled, recording is still restricted to evaluation,
        # which remains outside the pure training-step timing interval.
        dirichlet_effective = False
        if hasattr(model, "dirichlet_recorder"):
            if args.enable_dirichlet_recording:
                model.dirichlet_recorder.only_eval = True
                dirichlet_effective = bool(model.dirichlet_recorder.enabled)
            else:
                model.dirichlet_recorder.enabled = False
                dirichlet_effective = False

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # BF16 does not need loss scaling. FP16 keeps FP32 master parameters but
        # uses GradScaler to protect the low-precision dense backward path.
        grad_scaler = torch.cuda.amp.GradScaler(
            enabled=(
                device.type == "cuda"
                and bool(args.use_selective_amp)
                and str(args.selective_amp_dtype) == "fp16"
            )
        )

        tracer = NumericalTracer(
            model=model,
            enabled=bool(args.debug_numerics),
            trace_modules=bool(args.debug_numerics_modules),
            verbose=bool(args.debug_numerics_verbose),
        ) if args.debug_numerics else None

        print("=" * 90)
        print(f"RUN_NAME           : {run_name}")
        print(f"FULL_RUN_CONFIG    : {full_run_config}")
        print(f"EVAL_METRIC        : {metric_name}")
        print(f"LOG_POLICY         : save only if BestVal>{args.save_min_best_val:.2f}% and BestTest@BestVal>{args.save_min_best_test:.2f}%")
        print("MODEL_NAME         : graph_clifford_gat_ln_gamma_operator_jk_cres_hopgate_sixswitch")
        print(f"ALPHA_SCORE_MODE   : {cign_switch_modes['alpha_score']}")
        print(f"INTERACTION_MODE   : {cign_switch_modes['interaction']}")
        print(f"ALPHA_WEIGHT_MODE  : {cign_switch_modes['alpha_weighting']}")
        print(f"DATASET_NAME       : {args.dataset_name}")
        print(f"DEVICE             : {device}")
        print(f"GENERIC_PROFILE    : {args.generic_backend_profile}")
        print(f"AMAZON_BLOCK_OPTIN : {bool(getattr(args, 'allow_generalized_block_ops', False))}")
        print(f"AMAZON_AMP_OPTIN   : {bool(getattr(args, 'allow_generalized_selective_amp', False))}")
        print(f"AMAZON_COLLAPSED   : {bool(getattr(args, 'allow_generalized_collapsed_gat', False))}")
        print(f"FORCED_GAT_CAND    : {str(getattr(args, 'force_generic_gat_backend', 'auto'))}")
        print(f"FAST_POLICY        : {args.fast_backend_policy}")
        speed_sig, speed_sig_payload = build_generic_speed_signature(args)
        print(f"GENERIC_SPEED_KEY  : {speed_sig}")
        in_hidden_grid = int(args.hidden_dim) in GENERAL_FAST_HIDDEN_DIMS
        in_head_grid = int(args.gat_heads) in GENERAL_FAST_HEADS
        in_layer_grid = int(args.num_layers) in GENERAL_FAST_LAYERS
        print(f"GENERIC_TARGET_GRID: hidden={in_hidden_grid}, heads={in_head_grid}, layers={in_layer_grid}")
        print(f"GAT_DISPATCH       : {getattr(args, 'gat_backend_dispatch', 'manual')}")
        print(f"GAT_DISPATCH_NOTE  : {getattr(args, 'gat_backend_dispatch_note', 'n/a')}")
        print(f"DGNN_SAFE_NODE_LIMIT: {getattr(args, 'dgnn_safe_node_limit', DGNN_VALIDATED_MAX_SAFE_NUM_NODES)}")
        print(f"DGNN_CHUNK_MAX_NODES: {int(args.dgnn_chunk_max_nodes)}")
        print(f"DGNN_CHUNK_TARGETS : {int(args.dgnn_chunk_target_nodes)}")
        print(f"LARGE_GRAPH_GAT    : {str(args.large_graph_gat_backend)}")
        print(f"GAT_LARGE_GRAPH_MODE: {getattr(args, 'dgnn_node_safety_reason', 'not required')}")
        print(f"GAT_REGISTRY       : {_resolve_registry_path(args.gat_backend_registry)}")
        if getattr(args, "fast_backend_notes", None):
            print("FAST_FALLBACKS     : " + " | ".join(args.fast_backend_notes))
        else:
            print("FAST_FALLBACKS     : none")
        print(f"NUM_NODES          : {data.num_nodes}")
        print(f"NUM_EDGES          : {data.edge_index.size(1)}")
        print(f"STRUCTURE_NOISE    : {structure_noise_info}")
        print(f"NUM_FEATURES       : {dataset.num_features}")
        print(f"NUM_CLASSES        : {dataset.num_classes}")

        print(f"RAW_TRAIN_MASK     : {raw_train_mask_shape}")
        print(f"SPLIT_MODE         : {args.split_mode}")
        print(f"SPLIT_INDEX        : {args.split_index}")
        print(f"SPLIT_SEED         : {args.split_seed}")
        print(f"SPLIT_RATIO        : {args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}")
        print(f"TRAIN/VAL/TEST     : {int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")

        print(f"SEED               : {args.seed}")
        print(f"USE_UNDIRECTED     : {args.use_undirected}")
        print(f"HIDDEN_DIM         : {args.hidden_dim}")
        print(f"NUM_LAYERS         : {args.num_layers}")
        print(f"DROPOUT            : {args.dropout}")
        print(f"ORDER_SET S        : {args.hop_scales}")
        print(f"HOP_GATE_MODE      : {args.hop_gate_mode}")
        print(f"MULTIHOP_POLICY    : {str(getattr(args, 'multihop_backend', 'auto'))}")
        print(f"GAT_HEADS          : {args.gat_heads}")
        print(f"GAT_DROPOUT        : {args.gat_dropout}")
        if args.use_collapsed_fused_gat:
            print("GAT_BACKEND        : A3v6_SpecializedCollapsedHeadGAT_H6_C256")
            print("GAT_SIDEOPS        : A3v3_AttentionLogits (head mean+bias+SiLU absorbed into V6 core)")
        elif getattr(args, "use_chunked_fused_gat", False):
            print("GAT_BACKEND        : GenericChunked_dgNN_FusedGAT_A3v3")
            print("GAT_SIDEOPS        : A3v3_GlobalAttentionLogits+HeadSiLU")
        else:
            print(f"GAT_BACKEND        : {'dgNN_FusedGATConv_A3v3_ShapeAware' if (args.use_fused_gat and args.use_fused_gat_sideops) else ('dgNN_FusedGATConv' if args.use_fused_gat else 'PyG_GATConv')}")
            print(f"GAT_SIDEOPS        : {'A3v3_TwoStageAttnBackward+HeadSiLU' if args.use_fused_gat_sideops else 'Legacy_PyTorch'}")
        if args.use_fused_gat_sideops:
            print(f"GAT_SIDEOPS_PREP   : {a3v2_gat_compile_seconds:.6f}s (JIT/load outside epoch timing)")
        if args.use_collapsed_fused_gat:
            print(f"COLLAPSED_GAT_PREP : {a3v6_gat_compile_seconds:.6f}s (JIT/load outside epoch timing)")
        if args.use_fused_block_ops:
            print("BLOCK_FUSION       : A3v8_DropoutLN+AddLN+DropoutGammaResidual")
            print(f"BLOCK_FUSION_PREP  : {a3v8_block_compile_seconds:.6f}s (JIT/load outside epoch timing)")
        if args.use_fused_gat:
            print(f"FUSED_GRAPH_PREP   : {fused_graph_prep_seconds:.6f}s (outside epoch timing)")
            print(f"FUSED_GRAPH_INFO   : {fused_graph_info}")
        if args.use_multihop_csr_pipeline:
            print(f"SPMM_BACKEND       : {multihop_csr_info.get('backend', 'GenericMultiHopCSR') if multihop_csr_info else 'GenericMultiHopCSR'}")
        else:
            print(f"SPMM_BACKEND       : {'FixedCSRSpMM' if args.use_fixed_csr_spmm else 'PyTorch_COO_sparse_mm'}")
        if args.use_fixed_csr_spmm:
            print(f"FIXED_CSR_PREP     : {fixed_csr_prep_seconds:.6f}s (outside epoch timing)")
            print(f"FIXED_CSR_INFO     : {fixed_csr_info}")
        if args.use_multihop_csr_pipeline:
            print(f"MULTIHOP_CSR_PREP  : {multihop_csr_prep_seconds:.6f}s (JIT/plan outside epoch timing)")
            print(f"MULTIHOP_CSR_INFO  : {multihop_csr_info}")
        if args.use_fused_clifford_pack:
            if (
                cign_switch_modes["interaction"] == "new"
                and len(args.hop_scales) == 3
            ):
                clifford_backend = (
                    "CIGN_SwitchPackK3_FusedCUDA_"
                    f"interaction={cign_switch_modes['interaction']}_"
                    f"weight={cign_switch_modes['alpha_weighting']}"
                )
            else:
                clifford_backend = (
                    "A3v1_FusedCliffordPack3_Specialized"
                    if len(args.hop_scales) == 3
                    else f"GenericFusedCliffordPackK{len(args.hop_scales)}"
                )
        else:
            clifford_backend = "Legacy_PyTorch"
        if args.use_fused_clifford_pack and os.environ.get("CIGN_INTERACTION_MODE") == "new" and os.environ.get("CIGN_OUTER_NONLINEARITY") == "none":
            clifford_backend = f"CIGN_CIGNPackK{len(args.hop_scales)}_FusedCUDA"
        print(f"CLIFFORD_BACKEND   : {clifford_backend}")
        if args.use_fused_clifford_pack:
            print(f"A3_KERNEL_PREP     : {a3_compile_seconds:.6f}s (JIT/load outside epoch timing)")
        print(f"TF32_MATMUL        : {bool(args.use_tf32)}")
        print(f"MATMUL_PRECISION   : {torch.get_float32_matmul_precision()}")
        print(f"CUDA_ALLOW_TF32    : {bool(torch.backends.cuda.matmul.allow_tf32)}")
        print(f"CUDNN_ALLOW_TF32   : {bool(torch.backends.cudnn.allow_tf32)}")
        print(f"SELECTIVE_AMP      : {bool(args.use_selective_amp)}")
        print(f"SELECTIVE_AMP_DTYPE: {args.selective_amp_dtype if args.use_selective_amp else 'off'}")
        print(f"AMP_SCOPE          : LOCAL autocast only: input_proj + fused-GAT linear + Clifford projection; sparse/custom kernels forced FP32")
        print(f"AMP_GRAD_SCALER    : {bool(grad_scaler.is_enabled())}")
        print(f"INIT_GAMMA         : {args.init_gamma}")
        print(f"GAMMA_MODE         : {args.gamma_mode}")
        print(f"OPERATOR_MODE      : {args.operator_mode}")
        print(f"HYBRID_ALPHA       : {args.hybrid_alpha}")
        print(f"LAYER_COMBINE      : {args.layer_combine}")
        print(f"LR                 : {args.lr}")
        print(f"WEIGHT_DECAY       : {args.weight_decay}")
        print(f"EPOCHS             : {args.epochs}")
        print(f"PATIENCE           : {args.patience}")
        print(f"PRINT_EVERY        : {args.print_every}")
        print(f"TIMING_DISCARD     : first {int(args.timing_discard_first_epochs)} epoch(s) excluded only from steady-state report")
        print(f"DIRICHLET_REQUEST  : {args.enable_dirichlet_recording}")
        print(f"DIRICHLET_EFFECTIVE: {dirichlet_effective}")
        print(f"DEBUG_NUMERICS     : {bool(args.debug_numerics)}")
        if args.debug_numerics:
            print(f"DEBUG_EPOCHS       : {int(args.debug_numerics_epochs)}")
            print(f"DEBUG_MODULE_HOOKS : {bool(args.debug_numerics_modules)}")
            print(f"DEBUG_VERBOSE      : {bool(args.debug_numerics_verbose)}")
            print(f"DEBUG_ANOMALY      : {bool(args.debug_detect_anomaly)}")
            print("DEBUG_TIMING_VALID : False (debug scans/synchronization intentionally add overhead)")
        print("TIME_SCOPE         : zero_grad + train forward + loss + backward + optimizer.step only")
        print("=" * 90)

        # --------------------------------------------------------
        # Full-model training-only speed probe for generic autotuning.
        # This runs after every JIT/graph/model/optimizer preparation above,
        # uses the exact TrainStep function, performs no evaluation, and exits.
        # --------------------------------------------------------
        if int(args.speed_probe_steps) > 0:
            probe_times = []
            probe_losses = []
            total_probe_steps = int(args.speed_probe_warmup) + int(args.speed_probe_steps)
            probe_status = "ok"
            failure_reason = None

            for probe_i in range(total_probe_steps):
                loss_value, train_seconds = train_one_epoch(
                    model=model,
                    data=data,
                    graph_operator=graph_operator,
                    optimizer=optimizer,
                    device=device,
                    grad_scaler=grad_scaler,
                )
                if not np.isfinite(loss_value) or not np.isfinite(train_seconds):
                    probe_status = "failed"
                    failure_reason = (
                        f"non-finite speed probe at step={probe_i + 1}: "
                        f"loss={loss_value}, time={train_seconds}"
                    )
                    break
                if probe_i >= int(args.speed_probe_warmup):
                    probe_times.append(float(train_seconds))
                    probe_losses.append(float(loss_value))

            sig_key, sig_payload = build_generic_speed_signature(args)
            if probe_times:
                probe_mean, probe_std = mean_sample_std(probe_times)
                probe_median = float(np.median(probe_times))
            else:
                probe_mean = probe_std = probe_median = float("nan")

            result = {
                "status": probe_status,
                "reason": failure_reason,
                "profile": str(args.generic_backend_profile),
                "signature": sig_key,
                "signature_payload": sig_payload,
                "environment": current_speed_environment(),
                "dataset": str(args.dataset_name),
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.edge_index.size(1)),
                "measured_steps": len(probe_times),
                "warmup_steps": int(args.speed_probe_warmup),
                "mean_s": probe_mean,
                "std_s": probe_std,
                "median_s": probe_median,
                "times_s": probe_times,
                "last_loss": probe_losses[-1] if probe_losses else None,
                "effective_backends": {
                    "gat_dispatch": getattr(args, "gat_backend_dispatch", "manual"),
                    "fused_gat": bool(args.use_fused_gat),
                    "collapsed_gat": bool(args.use_collapsed_fused_gat),
                    "fixedcsr": bool(args.use_fixed_csr_spmm),
                    "multihop": bool(args.use_multihop_csr_pipeline),
                    "clifford_pack": bool(args.use_fused_clifford_pack),
                    "block_fusion": bool(args.use_fused_block_ops),
                    "tf32": bool(args.use_tf32),
                    "selective_amp": bool(args.use_selective_amp),
                },
            }
            print("SPEED_PROBE_RESULT_JSON: " + json.dumps(result, sort_keys=True))
            return

        best_val = 0.0
        best_test_at_best_val = 0.0
        best_epoch = 0
        patience_counter = 0
        trained_epochs = 0
        epoch_train_times = []
        numerical_diverged = False
        divergence_reason = ""
        debug_completed = False

        for epoch in range(1, args.epochs + 1):
            if args.debug_numerics:
                loss, train_seconds, debug_failure = train_one_epoch_debug(
                    model=model,
                    data=data,
                    graph_operator=graph_operator,
                    optimizer=optimizer,
                    device=device,
                    tracer=tracer,
                    epoch=epoch,
                    grad_scaler=grad_scaler,
                    detect_anomaly=bool(args.debug_detect_anomaly),
                )
                if debug_failure:
                    numerical_diverged = True
                    divergence_reason = debug_failure
                    print(f"NUMERICAL_DIVERGENCE: {divergence_reason}")
                    epoch_train_times.append(train_seconds)
                    trained_epochs = epoch
                    break
            else:
                loss, train_seconds = train_one_epoch(
                    model=model,
                    data=data,
                    graph_operator=graph_operator,
                    optimizer=optimizer,
                    device=device,
                    grad_scaler=grad_scaler,
                )
            epoch_train_times.append(train_seconds)
            trained_epochs = epoch

            if not np.isfinite(loss):
                numerical_diverged = True
                divergence_reason = f"non-finite training loss at epoch {epoch}: {loss}"
                print(f"NUMERICAL_DIVERGENCE: {divergence_reason}")
                break

            # Evaluation is deliberately outside the timed training interval.
            if tracer is not None:
                tracer.set_context(epoch, "eval_forward")
            eval_logits = evaluate_logits(model, data, graph_operator)
            if tracer is not None:
                tracer.check_tensor("eval_logits", eval_logits, always_print=True)
                if tracer.first_failure:
                    numerical_diverged = True
                    divergence_reason = tracer.first_failure
                    print(f"NUMERICAL_DIVERGENCE: {divergence_reason}")
                    break
            elif not torch.isfinite(eval_logits).all():
                numerical_diverged = True
                divergence_reason = f"non-finite evaluation logits at epoch {epoch}"
                print(f"NUMERICAL_DIVERGENCE: {divergence_reason}")
                break

            val_metric = compute_metric(
                eval_logits,
                data.y,
                data.val_mask,
                args.dataset_name,
            )

            if not np.isfinite(val_metric):
                numerical_diverged = True
                divergence_reason = f"non-finite validation metric at epoch {epoch}"
                print(f"NUMERICAL_DIVERGENCE: {divergence_reason}")
                break

            improved = val_metric > best_val
            if improved:
                best_val = val_metric
                best_test_at_best_val = compute_metric(
                    eval_logits,
                    data.y,
                    data.test_mask,
                    args.dataset_name,
                )
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % args.print_every == 0:
                train_metric = compute_metric(
                    eval_logits,
                    data.y,
                    data.train_mask,
                    args.dataset_name,
                )
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss {loss:.4f} | "
                    f"Train {train_metric*100:.2f}% | "
                    f"Val {val_metric*100:.2f}% | "
                    f"BestVal {best_val*100:.2f}% | "
                    f"BestTest@BestVal {best_test_at_best_val*100:.2f}% | "
                    f"TrainStep {train_seconds:.6f}s"
                )

            if args.debug_numerics and epoch >= int(args.debug_numerics_epochs):
                debug_completed = True
                print(
                    f"[NUMTRACE][CLEAN] completed {epoch} diagnostic epoch(s) with no NaN/Inf "
                    "detected at the traced stages."
                )
                break

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        total_pure_train_time = float(sum(epoch_train_times))
        mean_epoch_train_time, std_epoch_train_time = mean_sample_std(epoch_train_times)
        median_epoch_train_time = float(np.median(epoch_train_times)) if epoch_train_times else 0.0

        discard = min(int(args.timing_discard_first_epochs), len(epoch_train_times))
        steady_epoch_train_times = epoch_train_times[discard:]
        steady_total_train_time = float(sum(steady_epoch_train_times))
        steady_mean_train_time, steady_std_train_time = mean_sample_std(steady_epoch_train_times)
        steady_median_train_time = (
            float(np.median(steady_epoch_train_times)) if steady_epoch_train_times else 0.0
        )

        print("=" * 90)
        print(f"Trained epochs             : {trained_epochs}")
        print(f"Pure training time total   : {total_pure_train_time:.6f}s")
        # Keep the original labels unchanged for existing parsers and fair comparison
        # with old runs. These are ALL measured training epochs, including cold start.
        print(f"Mean epoch training time   : {mean_epoch_train_time:.6f}s")
        print(f"Std epoch training time    : {std_epoch_train_time:.6f}s")
        print(f"Median epoch training time : {median_epoch_train_time:.6f}s")
        print(f"Steady timing discard      : {discard} epoch(s)")
        print(f"Steady measured epochs     : {len(steady_epoch_train_times)}")
        print(f"Steady training time total : {steady_total_train_time:.6f}s")
        print(f"Steady mean epoch time     : {steady_mean_train_time:.6f}s")
        print(f"Steady std epoch time      : {steady_std_train_time:.6f}s")
        print(f"Steady median epoch time   : {steady_median_train_time:.6f}s")
        if numerical_diverged:
            run_status = "NUMERICAL_DIVERGENCE"
            exit_code = 2
        elif args.debug_numerics and debug_completed:
            run_status = "DEBUG_NUMERICS_CLEAN"
        else:
            run_status = "OK"
        print(f"RUN_STATUS                 : {run_status}")
        if numerical_diverged:
            print(f"DIVERGENCE_REASON          : {divergence_reason}")
        elif args.debug_numerics and debug_completed:
            print(f"DEBUG_RESULT               : clean through epoch {trained_epochs}")
        print(f"Best epoch                 : {best_epoch}")
        print(f"Best validation {metric_name:<8}: {best_val*100:.8f}%")
        print(f"Test {metric_name} @ best val : {best_test_at_best_val*100:.8f}%")

        saved_log_path = maybe_save_log(
            args=args,
            run_name=run_name,
            log_text=logger.get_text() if logger is not None else "",
            best_val=best_val,
            best_test_at_best_val=best_test_at_best_val,
        )

        if saved_log_path is not None:
            print(f"LOG_SAVED            : {saved_log_path}")
        elif args.save_log:
            print("LOG_SAVED            : skipped (threshold not met)")
        else:
            print("LOG_SAVED            : disabled")

        print("=" * 90)

    finally:
        if tracer is not None:
            tracer.close()
        if logger is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
