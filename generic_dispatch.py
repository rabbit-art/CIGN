from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


DEFAULTS = {
    "dataset_name": "Amazon-ratings",
    "use_undirected": True,
    "split_mode": 'random_ratio',
    "train_ratio": 0.50,
    "val_ratio": 0.25,
    "test_ratio": 0.25,
    "hidden_dim": 256,
    "num_layers": 4,
    "dropout": 0.28,
    "hop_scales": [1, 2, 3],
    "gat_heads": 6,
    "gat_dropout": 0.32,
    "gamma_mode": "vector",
    "hop_gate_mode": "node",
    "operator_mode": "adjacency",
    "hybrid_alpha": 0.5,
    "layer_combine": "concat",
    "fast_backend_policy": "auto",
    "generic_backend_profile": "auto",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _one(argv, name, default, cast=str):
    flag = "--" + name
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return cast(argv[i + 1])
        if tok.startswith(flag + "="):
            return cast(tok.split("=", 1)[1])
    return default


def _list_int(argv, name, default):
    flag = "--" + name
    for i, tok in enumerate(argv):
        if tok.startswith(flag + "="):
            raw = tok.split("=", 1)[1]
            return [int(x) for x in re.split(r"[, ]+", raw.strip()) if x]
        if tok == flag:
            vals = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                vals.append(int(argv[j]))
                j += 1
            return vals if vals else list(default)
    return list(default)


def canonical_config(argv):
    cfg = {
        "engine_revision": "amazon_stage3_collapsed_v1",
        "dataset": str(_one(argv, "dataset_name", DEFAULTS["dataset_name"])).strip().lower().replace("_", "-"),
        "use_undirected": bool(_one(argv, "use_undirected", DEFAULTS["use_undirected"], str2bool)),
        "split_mode": str(_one(argv, "split_mode", DEFAULTS["split_mode"])),
        "train_ratio": float(_one(argv, "train_ratio", DEFAULTS["train_ratio"], float)),
        "val_ratio": float(_one(argv, "val_ratio", DEFAULTS["val_ratio"], float)),
        "test_ratio": float(_one(argv, "test_ratio", DEFAULTS["test_ratio"], float)),
        "hidden_dim": int(_one(argv, "hidden_dim", DEFAULTS["hidden_dim"], int)),
        "num_layers": int(_one(argv, "num_layers", DEFAULTS["num_layers"], int)),
        "dropout": float(_one(argv, "dropout", DEFAULTS["dropout"], float)),
        "hop_scales": _list_int(argv, "hop_scales", DEFAULTS["hop_scales"]),
        "gat_heads": int(_one(argv, "gat_heads", DEFAULTS["gat_heads"], int)),
        "gat_dropout": float(_one(argv, "gat_dropout", DEFAULTS["gat_dropout"], float)),
        "gamma_mode": str(_one(argv, "gamma_mode", DEFAULTS["gamma_mode"])),
        "hop_gate_mode": str(_one(argv, "hop_gate_mode", DEFAULTS["hop_gate_mode"])),
        "operator_mode": str(_one(argv, "operator_mode", DEFAULTS["operator_mode"])),
        "hybrid_alpha": float(_one(argv, "hybrid_alpha", DEFAULTS["hybrid_alpha"], float)),
        "layer_combine": str(_one(argv, "layer_combine", DEFAULTS["layer_combine"])),
        "edge_drop_ratio": float(os.environ.get("CLIFFORD_EDGE_DROP_RATIO", "0") or 0.0),
    }
    raw = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    return key, cfg


def is_frozen_amazon_v9(argv):
    _, cfg = canonical_config(argv)
    policy = str(_one(argv, "fast_backend_policy", DEFAULTS["fast_backend_policy"])).lower()
    profile = str(_one(argv, "generic_backend_profile", DEFAULTS["generic_backend_profile"])).lower()
    # Any depth 1..6 uses the same frozen V9 per-block implementation. The
    # historical fastest record is specifically layer=4.
    return (
        cfg["dataset"] == "amazon-ratings"
        and cfg["hidden_dim"] == 256
        and cfg["gat_heads"] == 6
        and cfg["hop_scales"] == [1, 2, 3]
        and 1 <= cfg["num_layers"] <= 6
        and policy == "auto"
        and profile == "auto"
    )


def strip_options(argv, names):
    """Remove --name VALUE and --name=VALUE options from argv."""
    names = set(names)
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        hit = None
        for name in names:
            flag = "--" + name
            if tok == flag or tok.startswith(flag + "="):
                hit = name
                break
        if hit is None:
            out.append(tok)
            i += 1
            continue
        if tok == "--" + hit and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            i += 2
        else:
            i += 1
    return out


def load_registry(path):
    p = Path(path)
    if not p.exists():
        return {"schema_version": 1, "entries": {}}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": 1, "entries": {}}
    if not isinstance(payload, dict):
        return {"schema_version": 1, "entries": {}}
    payload.setdefault("schema_version", 1)
    payload.setdefault("entries", {})
    return payload


def save_registry(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def current_environment_fingerprint():
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "torch": str(torch.__version__),
                "torch_cuda": str(torch.version.cuda),
                "device": str(torch.cuda.get_device_name(0)),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        return {
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "device": "cpu",
            "compute_capability": None,
        }
    except Exception:
        return None


def registry_entry_environment_matches(entry):
    saved = entry.get("environment") if isinstance(entry, dict) else None
    now = current_environment_fingerprint()
    if not isinstance(saved, dict) or not isinstance(now, dict):
        return False
    for key in ("torch", "torch_cuda", "device", "compute_capability"):
        if saved.get(key) != now.get(key):
            return False
    return True
