#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from generic_dispatch import canonical_config, load_registry, save_registry, strip_options


DEFAULT_PROFILES = [
    "baseline",
    "pyg_tf32",
    "pyg_fixedcsr_tf32",
    "pyg_fixedcsr_pack_tf32",
    "safe_no_amp",
    "auto",
]

AMAZON_PROFILES = [
    "baseline",
    "amazon_dgnn_fp32",
    "amazon_dgnn_full_bf16",
    "amazon_collapsed_fp32",
    "amazon_collapsed_block_fp32",
    "amazon_collapsed_bf16",
    "amazon_collapsed_full_bf16",
    "amazon_pyg_full_bf16",
    "auto",
]


def parse_probe(stdout: str):
    prefix = "SPEED_PROBE_RESULT_JSON:"
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):].strip())
    return None


def current_env_fingerprint():
    # Environment matching is finalized from the child probe, which has Torch/CUDA.
    return None


def run_profile(engine_script: Path, train_args, profile, warmup, steps):
    stripped = strip_options(
        train_args,
        {
            "generic_backend_profile",
            "speed_probe_steps",
            "speed_probe_warmup",
            "save_log",
            "print_every",
            "fast_backend_policy",
            "generic_speed_guard",
            "generic_require_speedup",
            "generic_speed_registry",
            "force_generic_engine",
        },
    )
    cmd = [
        sys.executable,
        str(engine_script),
        *stripped,
        "--generic_backend_profile", profile,
        "--speed_probe_warmup", str(int(warmup)),
        "--speed_probe_steps", str(int(steps)),
        "--save_log", "False",
        "--print_every", "1000000",
    ]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    probe = parse_probe(proc.stdout)
    return {
        "profile": profile,
        "returncode": proc.returncode,
        "probe": probe,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-50:]),
    }


def tune(train_args, registry_path, min_speedup=1.02, warmup=3, steps=10, profiles=None):
    root = Path(__file__).resolve().parent
    engine_script = root / "generic_engine" / "trainb9.py"
    key, cfg = canonical_config(train_args)
    if profiles is None:
        profiles = AMAZON_PROFILES if cfg.get("dataset") == "amazon-ratings" else DEFAULT_PROFILES
    profiles = list(profiles)

    results = {}

    print("=" * 110)
    print("GENERALIZED FULL-MODEL AUTOTUNE")
    print("=" * 110)
    print("signature :", key)
    print("config    :", json.dumps(cfg, sort_keys=True))
    print("profiles  :", profiles)

    for profile in profiles:
        print(f"[AUTOTUNE] profile={profile} ...", flush=True)
        item = run_profile(engine_script, train_args, profile, warmup, steps)
        results[profile] = item
        probe = item.get("probe")
        if item["returncode"] == 0 and probe and probe.get("status") == "ok":
            print(
                f"           median={float(probe['median_s']):.6f}s "
                f"mean={float(probe['mean_s']):.6f}s "
                f"backends={probe.get('effective_backends')}"
            )
        else:
            print("           FAILED")
            print(item["stdout_tail"])

    base = results.get("baseline", {}).get("probe")
    if not base or base.get("status") != "ok":
        raise RuntimeError("Baseline speed probe failed; refusing to certify any optimized profile.")

    base_median = float(base["median_s"])
    candidates = []
    for profile, item in results.items():
        probe = item.get("probe")
        if profile == "baseline" or not probe or probe.get("status") != "ok":
            continue
        med = float(probe["median_s"])
        if med > 0:
            candidates.append((base_median / med, med, profile, probe))

    candidates.sort(reverse=True, key=lambda x: x[0])
    if candidates:
        best_speedup, best_median, best_profile, best_probe = candidates[0]
    else:
        best_speedup, best_median, best_profile, best_probe = 1.0, base_median, "baseline", base

    certified = bool(best_speedup >= float(min_speedup))
    selected_profile = best_profile if certified else "baseline"

    entry = {
        "signature": key,
        "config": cfg,
        "environment": base.get("environment"),
        "num_nodes": base.get("num_nodes"),
        "num_edges": base.get("num_edges"),
        "baseline_profile": "baseline",
        "baseline_median_s": base_median,
        "best_candidate_profile": best_profile,
        "best_candidate_median_s": best_median,
        "best_candidate_speedup_vs_baseline": best_speedup,
        "min_required_speedup": float(min_speedup),
        "certified_speedup": certified,
        "selected_profile": selected_profile,
        "all_profiles": {
            p: {
                "returncode": item["returncode"],
                "probe": item.get("probe"),
            }
            for p, item in results.items()
        },
    }

    registry = load_registry(registry_path)
    registry["entries"][key] = entry
    registry["last_environment"] = base.get("environment")
    save_registry(registry_path, registry)

    print("-" * 110)
    print("baseline median       :", f"{base_median:.6f}s")
    print("best candidate        :", best_profile)
    print("best candidate median :", f"{best_median:.6f}s")
    print("speedup vs baseline   :", f"{best_speedup:.4f}x")
    print("certified             :", certified)
    print("selected profile      :", selected_profile)
    print("registry              :", registry_path)
    print("=" * 110)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry_path", default="generic_speed_registry.json")
    ap.add_argument("--min_speedup", type=float, default=1.02)
    ap.add_argument("--probe_warmup", type=int, default=3)
    ap.add_argument("--probe_steps", type=int, default=10)
    ap.add_argument("--profiles", nargs="+", default=None)
    args, train_args = ap.parse_known_args()

    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    tune(
        train_args=train_args,
        registry_path=args.registry_path,
        min_speedup=args.min_speedup,
        warmup=args.probe_warmup,
        steps=args.probe_steps,
        profiles=args.profiles,
    )


if __name__ == "__main__":
    main()
