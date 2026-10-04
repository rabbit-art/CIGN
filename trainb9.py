#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from generic_dispatch import (
    canonical_config,
    is_frozen_amazon_v9,
    load_registry,
    registry_entry_environment_matches,
    strip_options,
)

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "amazon_v9_frozen" / "trainb9.py"
GENERIC = ROOT / "generic_engine" / "trainb9.py"
AUTOTUNE = ROOT / "backend_autotune.py"
REGISTRY_DEFAULT = ROOT / "generic_speed_registry.json"


def one(argv, name, default=None):
    flag = "--" + name
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return default


def as_bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def exec_python(script: Path, argv):
    os.execv(sys.executable, [sys.executable, str(script), *argv])



def _try_acquire_lock(lock_path: Path):
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
    os.close(fd)
    return True


def _wait_for_speed_entry(registry_path: Path, key: str, timeout_s: float = 900.0):
    start = time.time()
    while time.time() - start < timeout_s:
        registry = load_registry(registry_path)
        entry = registry.get("entries", {}).get(key)
        if isinstance(entry, dict) and registry_entry_environment_matches(entry):
            return entry
        time.sleep(1.0)
    return None


def main():
    original_argv = sys.argv[1:]
    force_generic = as_bool(one(original_argv, "force_generic_engine", "False"))
    guard = str(one(original_argv, "generic_speed_guard", "auto")).strip().lower()
    require_speedup = as_bool(one(original_argv, "generic_require_speedup", "False"))
    registry_path = Path(one(original_argv, "generic_speed_registry", str(REGISTRY_DEFAULT))).expanduser()
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path

    clean_argv = strip_options(
        original_argv,
        {
            "force_generic_engine",
            "generic_speed_guard",
            "generic_require_speedup",
            "generic_speed_registry",
        },
    )

    # Respect an explicit request for the original unaccelerated/manual policy.
    # The generalized speed guard must never override fast_backend_policy=off/strict.
    explicit_fast_policy = str(one(clean_argv, "fast_backend_policy", "auto")).strip().lower()
    if explicit_fast_policy in {"off", "strict"}:
        print(
            f"[GENERALIZED_DISPATCH] explicit fast_backend_policy={explicit_fast_policy}; "
            "whole-model autotune bypassed.",
            flush=True,
        )
        exec_python(GENERIC, clean_argv)

    # Freeze the historical record path. Because this is a separate script
    # directory with its own original model/CUDA files, generic changes cannot
    # alter its kernels or dispatch.
    if not force_generic and is_frozen_amazon_v9(clean_argv):
        print(
            "[GENERALIZED_DISPATCH] frozen Amazon V9 path: "
            "hidden=256, heads=6, hops=[1,2,3], layers=1..6 "
            "(historical fastest record at layers=4).",
            flush=True,
        )
        exec_python(FROZEN, clean_argv)

    explicit_profile = one(clean_argv, "generic_backend_profile", None)
    if explicit_profile and str(explicit_profile).lower() != "auto":
        print(
            f"[GENERALIZED_DISPATCH] explicit generic profile={explicit_profile}; speed guard bypassed.",
            flush=True,
        )
        exec_python(GENERIC, clean_argv)

    if guard not in {"auto", "registry", "off", "refresh"}:
        raise SystemExit(
            "--generic_speed_guard must be one of auto|registry|off|refresh"
        )

    if guard == "off":
        print("[GENERALIZED_DISPATCH] generic speed guard disabled; using generic auto profile.", flush=True)
        exec_python(GENERIC, clean_argv)

    key, cfg = canonical_config(clean_argv)
    registry = load_registry(registry_path)
    entry = registry.get("entries", {}).get(key)
    env_match = registry_entry_environment_matches(entry) if isinstance(entry, dict) else False

    need_tune = (
        guard == "refresh"
        or (guard == "auto" and (not isinstance(entry, dict) or not env_match))
    )
    if isinstance(entry, dict) and not env_match and guard == "auto":
        print(
            "[GENERALIZED_DISPATCH] cached speed certificate belongs to a different "
            "Torch/CUDA/GPU environment; refreshing it.",
            flush=True,
        )
    if need_tune:
        dataset_key = str(cfg.get("dataset", "")).lower()
        # Amazon is the compute-heavy case: use a longer probe so a transient
        # 60-70 ms island cannot incorrectly certify a backend whose long-run
        # time is actually ~120-130 ms.
        probe_warmup = 8 if dataset_key == "amazon-ratings" else 3
        probe_steps = 24 if dataset_key == "amazon-ratings" else 10

        lock_path = registry_path.with_name(
            registry_path.name + f".{key}.autotune.lock"
        )
        got_lock = _try_acquire_lock(lock_path)

        if got_lock:
            try:
                # Another process may have completed the entry between our first
                # check and lock acquisition.
                registry = load_registry(registry_path)
                entry = registry.get("entries", {}).get(key)
                valid_now = (
                    isinstance(entry, dict)
                    and registry_entry_environment_matches(entry)
                    and guard != "refresh"
                )
                if not valid_now:
                    print(
                        f"[GENERALIZED_DISPATCH] tuning signature={key}; "
                        f"dataset={dataset_key}; warmup={probe_warmup}; measured={probe_steps}. "
                        "This happens outside formal epoch timing.",
                        flush=True,
                    )
                    cmd = [
                        sys.executable,
                        str(AUTOTUNE),
                        "--registry_path", str(registry_path),
                        "--min_speedup", "1.02",
                        "--probe_warmup", str(probe_warmup),
                        "--probe_steps", str(probe_steps),
                        "--",
                        *clean_argv,
                    ]
                    env = os.environ.copy()
                    env["PYTHONNOUSERSITE"] = "1"
                    rc = subprocess.call(cmd, env=env)
                    if rc != 0:
                        raise SystemExit(f"Generalized autotune failed with return code {rc}")
            finally:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
        else:
            print(
                f"[GENERALIZED_DISPATCH] another process is autotuning signature={key}; "
                "waiting for its shared certificate instead of launching duplicate benchmarks.",
                flush=True,
            )
            waited = _wait_for_speed_entry(registry_path, key, timeout_s=900.0)
            if waited is None:
                raise SystemExit(
                    f"Timed out waiting for autotune certificate for signature={key}"
                )

        registry = load_registry(registry_path)
        entry = registry.get("entries", {}).get(key)

    if not isinstance(entry, dict) or (
        guard == "registry" and not registry_entry_environment_matches(entry)
    ):
        if guard == "registry":
            raise SystemExit(
                f"No valid speed-registry entry for signature={key} in the current "
                "Torch/CUDA/GPU environment. Run with --generic_speed_guard auto or refresh first."
            )
        print("[GENERALIZED_DISPATCH] registry entry unavailable; using baseline profile.", flush=True)
        selected = "baseline"
        certified = False
    else:
        selected = str(entry.get("selected_profile", "baseline"))
        certified = bool(entry.get("certified_speedup", False))
        print(
            "[GENERALIZED_DISPATCH] "
            f"signature={key} selected_profile={selected} "
            f"certified_speedup={certified} "
            f"speedup={float(entry.get('best_candidate_speedup_vs_baseline', 1.0)):.4f}x",
            flush=True,
        )

    if require_speedup and not certified:
        raise SystemExit(
            "This configuration has no >=1.02x certified full-model speedup yet. "
            "The dispatcher refuses to claim acceleration because "
            "--generic_require_speedup True was requested."
        )

    run_argv = strip_options(clean_argv, {"generic_backend_profile"})
    run_argv += ["--generic_backend_profile", selected]
    exec_python(GENERIC, run_argv)


if __name__ == "__main__":
    main()
