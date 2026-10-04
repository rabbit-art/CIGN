from __future__ import annotations

import hashlib
import json
import os

from generic_dispatch import canonical_config


CIGN_SWITCH_ENGINE_REVISION = "cign_fullaccel_v5_collapsed_dispatch"


def _mode(env_name: str, default: str = "old") -> str:
    value = os.environ.get(env_name, default).strip().lower()
    if value not in {"old", "new"}:
        raise ValueError(f"{env_name} must be old/new, got {value!r}")
    return value


def current_switch_modes():
    return {
        "alpha_score": _mode("CIGN_ALPHA_SCORE_MODE"),
        "interaction": _mode("CIGN_INTERACTION_MODE"),
        "alpha_weighting": _mode("CIGN_ALPHA_WEIGHTING_MODE"),
    }


def all_old():
    modes = current_switch_modes()
    return all(v == "old" for v in modes.values())


def cign_speed_signature(argv):
    """Current clean_new speed key + the three old/new switches.

    The original generic registry cannot be reused here because the NEW
    interaction and NEW weighting change the timed computation and projection
    width. Each of the 2^3 combinations therefore receives its own certificate.
    """
    _, base_cfg = canonical_config(argv)
    cfg = dict(base_cfg)
    cfg["outer_nonlinearity"] = os.environ.get("CIGN_OUTER_NONLINEARITY", "tanh")
    cfg["cign_switch_engine_revision"] = CIGN_SWITCH_ENGINE_REVISION
    cfg.update({
        "cign_alpha_score_mode": current_switch_modes()["alpha_score"],
        "cign_interaction_mode": current_switch_modes()["interaction"],
        "cign_alpha_weighting_mode": current_switch_modes()["alpha_weighting"],
    })
    raw = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    return key, cfg
