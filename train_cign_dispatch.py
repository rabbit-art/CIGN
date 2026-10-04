#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from generic_dispatch import (
    is_frozen_amazon_v9,
    load_registry,
    registry_entry_environment_matches,
    strip_options,
)
from cign_dispatch_utils import (
    all_old,
    cign_speed_signature,
    current_switch_modes,
)

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "amazon_v9_frozen" / "trainb9.py"
GENERIC = ROOT / "generic_engine" / "train_cign.py"
AUTOTUNE = ROOT / "backend_autotune_cign.py"
REGISTRY_DEFAULT = ROOT / "cign_sixswitch_speed_registry.json"


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


_POSIX_LOCKS = {}
def _try_acquire_lock(lock_path: Path):
    if os.name == 'posix':
        import fcntl
        handle = open(lock_path, 'a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        _POSIX_LOCKS[str(lock_path)] = handle
        return True
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f'pid={os.getpid()}\n'.encode('utf-8'))
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
    modes = current_switch_modes()

    force_generic = as_bool(one(original_argv, "force_generic_engine", "False"))
    guard = str(one(original_argv, "generic_speed_guard", "auto")).strip().lower()
    require_speedup = as_bool(one(original_argv, "generic_require_speedup", "False"))

    registry_path = Path(
        one(
            original_argv,
            "generic_speed_registry",
            str(REGISTRY_DEFAULT),
        )
    ).expanduser()
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

    print(
        "[CIGN_SWITCH_DISPATCH] "
        f"alpha={modes['alpha_score']} "
        f"interaction={modes['interaction']} "
        f"weight={modes['alpha_weighting']}",
        flush=True,
    )

    explicit_fast_policy = str(
        one(clean_argv, "fast_backend_policy", "auto")
    ).strip().lower()

    if explicit_fast_policy in {"off", "strict"}:
        print(
            f"[CIGN_SWITCH_DISPATCH] explicit fast_backend_policy="
            f"{explicit_fast_policy}; autotune bypassed.",
            flush=True,
        )
        exec_python(GENERIC, clean_argv)

    # OLD/OLD/OLD is mathematically the existing model. Therefore when the
    # current clean_new dispatcher considers a configuration eligible for the
    # frozen Amazon V9 path, reuse it exactly. NEW formulas must never enter
    # the frozen old-formula model.
    if all_old() and not force_generic and is_frozen_amazon_v9(clean_argv):
        print(
            "[CIGN_SWITCH_DISPATCH] OLD/OLD/OLD -> existing frozen Amazon V9 "
            "fast path.",
            flush=True,
        )
        exec_python(FROZEN, clean_argv)

    explicit_profile = one(clean_argv, "generic_backend_profile", None)
    if explicit_profile and str(explicit_profile).lower() != "auto":
        print(
            f"[CIGN_SWITCH_DISPATCH] explicit profile={explicit_profile}; "
            "speed guard bypassed.",
            flush=True,
        )
        exec_python(GENERIC, clean_argv)

    if guard not in {"auto", "registry", "off", "refresh"}:
        raise SystemExit(
            "--generic_speed_guard must be one of auto|registry|off|refresh"
        )

    if guard == "off":
        print(
            "[CIGN_SWITCH_DISPATCH] speed guard disabled; using generic AUTO.",
            flush=True,
        )
        exec_python(GENERIC, clean_argv)

    key, cfg = cign_speed_signature(clean_argv)
    registry = load_registry(registry_path)
    entry = registry.get("entries", {}).get(key)
    env_match = (
        registry_entry_environment_matches(entry)
        if isinstance(entry, dict)
        else False
    )

    need_tune = (
        guard == "refresh"
        or (
            guard == "auto"
            and (not isinstance(entry, dict) or not env_match)
        )
    )

    if need_tune:
        dataset_key = str(cfg.get("dataset", "")).lower()
        probe_warmup = 8 if dataset_key == "amazon-ratings" else 3
        probe_steps = 24 if dataset_key == "amazon-ratings" else 10

        lock_path = registry_path.with_name(
            registry_path.name + f".{key}.autotune.lock"
        )
        got_lock = _try_acquire_lock(lock_path)

        if got_lock:
            try:
                registry = load_registry(registry_path)
                entry = registry.get("entries", {}).get(key)
                valid_now = (
                    isinstance(entry, dict)
                    and registry_entry_environment_matches(entry)
                    and guard != "refresh"
                )
                if not valid_now:
                    print(
                        f"[CIGN_SWITCH_DISPATCH] tuning signature={key}; "
                        f"warmup={probe_warmup}; measured={probe_steps}.",
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
                        raise SystemExit(
                            f"CIGN switch autotune failed with return code {rc}"
                        )
            finally:
                if os.name == 'posix':
                    _POSIX_LOCKS.pop(str(lock_path)).close()
                else:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
        else:
            print(
                f"[CIGN_SWITCH_DISPATCH] another split is autotuning {key}; "
                "waiting for the shared certificate.",
                flush=True,
            )
            waited = _wait_for_speed_entry(
                registry_path, key, timeout_s=900.0
            )
            if waited is None:
                raise SystemExit(
                    f"Timed out waiting for speed certificate {key}"
                )

        registry = load_registry(registry_path)
        entry = registry.get("entries", {}).get(key)

    if not isinstance(entry, dict) or (
        guard == "registry"
        and not registry_entry_environment_matches(entry)
    ):
        if guard == "registry":
            raise SystemExit(
                f"No valid CIGN switch registry entry for signature={key}."
            )
        selected = "baseline"
        certified = False
        speedup = None
    else:
        selected = str(entry.get("selected_profile", "baseline"))
        certified = bool(entry.get("certified_speedup", False))
        speedup = entry.get("best_candidate_speedup_vs_baseline")
        print(
            "[CIGN_SWITCH_DISPATCH] "
            f"signature={key} selected_profile={selected} "
            f"certified_speedup={certified} speedup={speedup}",
            flush=True,
        )

    if require_speedup and not certified:
        raise SystemExit(
            "No certified >=1.02x speedup for this switch combination."
        )

    run_argv = strip_options(clean_argv, {"generic_backend_profile"})
    run_argv += ["--generic_backend_profile", selected]
    exec_python(GENERIC, run_argv)


if __name__ == "__main__":
    main()
