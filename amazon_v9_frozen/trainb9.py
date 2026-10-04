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

from model import build_model, prepare_fixed_csr_operator, FixedCSRGraphOperator
from multihop_csr_pipeline import prepare_multihop_csr_operator, ensure_multihop_csr_loaded
from fused_clifford_pack import ensure_fused_clifford_pack_loaded
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


def build_full_run_config(args) -> str:
    """
    Build a full, human-readable configuration string.
    This string is printed into the log, but it is no longer used directly
    as the filename, because long filenames can trigger OSError: [Errno 36].
    """
    hop_str = "-".join(map(str, args.hop_scales))
    parts = [
        "model=graph_clifford_gat_ln_gamma_operator_jk_cres_hopgate",
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
        f"fastpolicy={args.fast_backend_policy}",
        f"fusedgat={int(args.use_fused_gat)}",
        f"gatside={int(args.use_fused_gat_sideops)}",
        f"collapsedgat={int(args.use_collapsed_fused_gat)}",
        f"v8block={int(args.use_fused_block_ops)}",
        f"v9amp={int(args.use_selective_amp)}",
        f"v9dtype={args.selective_amp_dtype}",
        f"fixedcsr={int(args.use_fixed_csr_spmm)}",
        f"multihopcsr={int(args.use_multihop_csr_pipeline)}",
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
        f"undir={int(args.use_undirected)}",
        f"dirichlet={int(args.enable_dirichlet_recording)}",
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
        "--use_collapsed_fused_gat",
        type=str2bool,
        default=DEFAULT_USE_COLLAPSED_FUSED_GAT,
        help=(
            "A3-v6: replace the dgNN message-passing output/head-reduction boundary "
            "with a concat=False collapsed-head CUDA core. The clean11-fast kernel supports "
            "1<=heads<=32 and 1<=hidden_dim<=256; requires --use_fused_gat True and "
            "--use_fused_gat_sideops True."
        ),
    )
    parser.add_argument(
        "--use_fused_block_ops",
        type=str2bool,
        default=DEFAULT_USE_FUSED_BLOCK_OPS,
        help=(
            "A3-v8: fuse block input Dropout+LayerNorm, context residual Add+LayerNorm, "
            "and projection Dropout+Gamma+Residual into custom CUDA kernels. "
            "V1 is specialized for hidden_dim=256 and gamma_mode=vector."
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
    parser.add_argument("--save_log", type=str2bool, default=DEFAULT_SAVE_LOG)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--save_min_best_val", type=float, default=DEFAULT_SAVE_MIN_BEST_VAL)
    parser.add_argument("--save_min_best_test", type=float, default=DEFAULT_SAVE_MIN_BEST_TEST)
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

    if bool(args.use_fused_gat_sideops) and not bool(args.use_fused_gat):
        disable("use_fused_gat_sideops", "requires use_fused_gat=True")

    # V6 collapsed kernel is safe for C<=256 and H<=32. This deliberately
    # includes clean11's lightweight hidden/head search ranges.
    if bool(args.use_collapsed_fused_gat):
        if not bool(args.use_fused_gat) or not bool(args.use_fused_gat_sideops):
            disable("use_collapsed_fused_gat", "requires fused GAT + fused GAT sideops")
        elif not (1 <= int(args.hidden_dim) <= 256):
            disable("use_collapsed_fused_gat", f"requires 1<=hidden_dim<=256, got {args.hidden_dim}")
        elif not (1 <= int(args.gat_heads) <= 32):
            disable("use_collapsed_fused_gat", f"requires 1<=gat_heads<=32, got {args.gat_heads}")

    # V8 v1 remains the one truly shape-specialized component.
    if bool(args.use_fused_block_ops):
        if int(args.hidden_dim) != 256:
            disable("use_fused_block_ops", f"V8 block fusion is specialized for hidden_dim=256, got {args.hidden_dim}")
        elif str(args.gamma_mode) != "vector":
            disable("use_fused_block_ops", f"V8 block fusion requires gamma_mode=vector, got {args.gamma_mode}")

    if bool(args.use_multihop_csr_pipeline):
        if not bool(args.use_fixed_csr_spmm):
            disable("use_multihop_csr_pipeline", "requires use_fixed_csr_spmm=True")
        elif list(map(int, args.hop_scales)) != [1, 2, 3]:
            disable("use_multihop_csr_pipeline", f"V5 pipeline requires hop_scales=[1,2,3], got {args.hop_scales}")

    if bool(args.use_fused_clifford_pack):
        if len(args.hop_scales) != 3:
            disable("use_fused_clifford_pack", f"A3 Clifford pack requires exactly 3 hops, got {args.hop_scales}")
        elif str(args.hop_gate_mode) == "none":
            disable("use_fused_clifford_pack", "A3 Clifford pack requires global/node hop gating")

    if bool(args.use_selective_amp) and cuda_ok:
        if str(args.selective_amp_dtype) == "bf16" and not torch.cuda.is_bf16_supported():
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

    y_true = y[mask].detach().cpu().numpy()
    if len(np.unique(y_true)) < 2:
        return 0.0

    masked_logits = logits[mask]
    if masked_logits.dim() == 1:
        y_score = torch.sigmoid(masked_logits).detach().cpu().numpy()
    elif masked_logits.size(-1) == 1:
        y_score = torch.sigmoid(masked_logits.squeeze(-1)).detach().cpu().numpy()
    else:
        y_score = F.softmax(masked_logits, dim=-1)[:, 1].detach().cpu().numpy()

    return float(roc_auc_score(y_true, y_score))


@torch.inference_mode()
def compute_metric(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, dataset_name: str) -> float:
    if uses_auc(dataset_name):
        return binary_roc_auc(logits, y, mask)
    return accuracy(logits, y, mask)


def forward_model(model, data, graph_operator: torch.Tensor):
    return model(data.x, data.edge_index, graph_operator)


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
    args = resolve_fast_backend_compatibility(args)
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

    try:
        seed_everything(args.seed)
        device = torch.device(args.device)
        run_name = build_run_name(args)
        full_run_config = build_full_run_config(args)
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
        # A3-v5 grouped three-hop CSR pipeline preprocessing/JIT.
        # The extension, int32 CSR storage, descriptors and workspace are all
        # created here, outside every measured training epoch.
        # --------------------------------------------------------
        multihop_csr_prep_seconds = 0.0
        multihop_csr_info = None
        if args.use_multihop_csr_pipeline:
            if not args.use_fixed_csr_spmm:
                raise ValueError("--use_multihop_csr_pipeline True requires --use_fixed_csr_spmm True")
            if list(args.hop_scales) != [1, 2, 3]:
                raise ValueError(
                    "A3-v5 MultiHopCSR3 is specialized for --hop_scales 1 2 3; "
                    f"got {args.hop_scales}."
                )
            if device.type != "cuda":
                raise RuntimeError("A3-v5 MultiHopCSR3 requires CUDA.")
            torch.cuda.synchronize()
            multihop_start = time.perf_counter()
            ensure_multihop_csr_loaded()
            graph_operator = prepare_multihop_csr_operator(
                graph_operator,
                dense_cols=2 * int(args.hidden_dim),
                algorithm=int(args.multihop_csr_algorithm),
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
            if int(args.hidden_dim) != 256 or str(args.gamma_mode) != "vector":
                raise ValueError(
                    "A3-v8 v1 requires hidden_dim=256 and gamma_mode=vector; "
                    f"got hidden={args.hidden_dim}, gamma_mode={args.gamma_mode}."
                )
            torch.cuda.synchronize()
            v8_compile_start = time.perf_counter()
            ensure_fused_block_ops_loaded(verbose=False)
            torch.cuda.synchronize()
            a3v8_block_compile_seconds = time.perf_counter() - v8_compile_start

        # --------------------------------------------------------
        # A3-v6 collapsed-head GAT CUDA JIT compilation/loading.
        # This specialized core is compiled before model creation and therefore
        # remains completely outside measured epoch timing.
        # --------------------------------------------------------
        a3v6_gat_compile_seconds = 0.0
        if args.use_collapsed_fused_gat:
            if not args.use_fused_gat:
                raise ValueError("--use_collapsed_fused_gat True requires --use_fused_gat True")
            if not args.use_fused_gat_sideops:
                raise ValueError("--use_collapsed_fused_gat True requires --use_fused_gat_sideops True")
            if device.type != "cuda":
                raise RuntimeError("--use_collapsed_fused_gat True requires CUDA.")
            if not (1 <= int(args.gat_heads) <= 32) or not (1 <= int(args.hidden_dim) <= 256):
                raise ValueError(
                    "A3-v6 clean11-fast requires 1<=gat_heads<=32 and 1<=hidden_dim<=256; "
                    f"got heads={args.gat_heads}, hidden={args.hidden_dim}."
                )
            torch.cuda.synchronize()
            collapsed_compile_start = time.perf_counter()
            ensure_collapsed_gat_loaded(verbose=False)
            torch.cuda.synchronize()
            a3v6_gat_compile_seconds = time.perf_counter() - collapsed_compile_start

        # --------------------------------------------------------
        # A3 custom CUDA kernel JIT compilation/loading.
        # This happens once before optimizer creation and before any epoch timing.
        # --------------------------------------------------------
        a3_compile_seconds = 0.0
        if args.use_fused_clifford_pack:
            if device.type != "cuda":
                raise RuntimeError("--use_fused_clifford_pack True requires CUDA.")
            if len(args.hop_scales) != 3:
                raise ValueError(
                    "A3 fused Clifford pack v1 requires exactly 3 hop_scales; "
                    f"got {args.hop_scales}."
                )
            torch.cuda.synchronize()
            a3_compile_start = time.perf_counter()
            ensure_fused_clifford_pack_loaded(verbose=False)
            torch.cuda.synchronize()
            a3_compile_seconds = time.perf_counter() - a3_compile_start

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
            use_collapsed_fused_gat=args.use_collapsed_fused_gat,
            use_fused_clifford_pack=args.use_fused_clifford_pack,
            use_fused_block_ops=args.use_fused_block_ops,
            use_selective_amp=args.use_selective_amp,
            selective_amp_dtype=args.selective_amp_dtype,
            init_gamma=args.init_gamma,
            gamma_mode=args.gamma_mode,
            hop_gate_mode=args.hop_gate_mode,
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

        print("=" * 90)
        print(f"RUN_NAME           : {run_name}")
        print(f"FULL_RUN_CONFIG    : {full_run_config}")
        print(f"EVAL_METRIC        : {metric_name}")
        print(f"LOG_POLICY         : save only if BestVal>{args.save_min_best_val:.2f}% and BestTest@BestVal>{args.save_min_best_test:.2f}%")
        print("MODEL_NAME         : graph_clifford_gat_ln_gamma_operator_jk_cres_hopgate")
        print(f"DATASET_NAME       : {args.dataset_name}")
        print(f"DEVICE             : {device}")
        print(f"FAST_POLICY        : {args.fast_backend_policy}")
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
        print(f"GAT_HEADS          : {args.gat_heads}")
        print(f"GAT_DROPOUT        : {args.gat_dropout}")
        if args.use_collapsed_fused_gat:
            print("GAT_BACKEND        : A3v6_CollapsedHeadGAT")
            print("GAT_SIDEOPS        : A3v3_AttentionLogits (head mean+bias+SiLU absorbed into V6 core)")
        else:
            print(f"GAT_BACKEND        : {'dgNN_FusedGATConv' if args.use_fused_gat else 'PyG_GATConv'}")
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
            print("SPMM_BACKEND       : A3v5_MultiHopCSR3")
        else:
            print(f"SPMM_BACKEND       : {'FixedCSRSpMM' if args.use_fixed_csr_spmm else 'PyTorch_COO_sparse_mm'}")
        if args.use_fixed_csr_spmm:
            print(f"FIXED_CSR_PREP     : {fixed_csr_prep_seconds:.6f}s (outside epoch timing)")
            print(f"FIXED_CSR_INFO     : {fixed_csr_info}")
        if args.use_multihop_csr_pipeline:
            print(f"MULTIHOP_CSR_PREP  : {multihop_csr_prep_seconds:.6f}s (JIT/plan outside epoch timing)")
            print(f"MULTIHOP_CSR_INFO  : {multihop_csr_info}")
        print(f"CLIFFORD_BACKEND   : {'A3v1_FusedCliffordPack' if args.use_fused_clifford_pack else 'Legacy_PyTorch'}")
        if args.use_fused_clifford_pack:
            print(f"A3_KERNEL_PREP     : {a3_compile_seconds:.6f}s (JIT/load outside epoch timing)")
        print(f"TF32_MATMUL        : {bool(args.use_tf32)}")
        print(f"MATMUL_PRECISION   : {torch.get_float32_matmul_precision()}")
        print(f"CUDA_ALLOW_TF32    : {bool(torch.backends.cuda.matmul.allow_tf32)}")
        print(f"CUDNN_ALLOW_TF32   : {bool(torch.backends.cudnn.allow_tf32)}")
        print(f"SELECTIVE_AMP      : {bool(args.use_selective_amp)}")
        print(f"SELECTIVE_AMP_DTYPE: {args.selective_amp_dtype if args.use_selective_amp else 'off'}")
        print(f"AMP_SCOPE          : input_proj + collapsed-GAT linear + Clifford projection only")
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
        print(f"DIRICHLET_REQUEST  : {args.enable_dirichlet_recording}")
        print(f"DIRICHLET_EFFECTIVE: {dirichlet_effective}")
        print("TIME_SCOPE         : zero_grad + train forward + loss + backward + optimizer.step only")
        print("=" * 90)

        best_val = 0.0
        best_test_at_best_val = 0.0
        best_epoch = 0
        patience_counter = 0
        trained_epochs = 0
        epoch_train_times = []

        for epoch in range(1, args.epochs + 1):
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

            # Evaluation is deliberately outside the timed training interval.
            eval_logits = evaluate_logits(model, data, graph_operator)
            val_metric = compute_metric(
                eval_logits,
                data.y,
                data.val_mask,
                args.dataset_name,
            )

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

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        total_pure_train_time = float(sum(epoch_train_times))
        mean_epoch_train_time, std_epoch_train_time = mean_sample_std(epoch_train_times)

        print("=" * 90)
        print(f"Trained epochs             : {trained_epochs}")
        print(f"Pure training time total   : {total_pure_train_time:.6f}s")
        print(f"Mean epoch training time   : {mean_epoch_train_time:.6f}s")
        print(f"Std epoch training time    : {std_epoch_train_time:.6f}s")
        print(f"Best epoch                 : {best_epoch}")
        print(f"Best validation {metric_name:<8}: {best_val*100:.2f}%")
        print(f"Test {metric_name} @ best val : {best_test_at_best_val*100:.2f}%")

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
        if logger is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
