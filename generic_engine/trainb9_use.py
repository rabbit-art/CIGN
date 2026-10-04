# ============================================================
# trainb9_use.py
# Fixed-hyperparameter multi-split runner, 100% sweep_train-style.
#
# IMPORTANT DESIGN:
#   This script does NOT train the model by itself.
#   It only launches your original train.py/trainb9.py as independent subprocesses.
#
# Therefore it matches your sweep_train3 / parallel-GPU sweep style:
#   split 0 -> new Python subprocess -> train_script
#   split 1 -> new Python subprocess -> train_script
#   ...
#
# Benefits:
#   1) Training logic is 100% from your original train script.
#   2) Each split has a fresh Python process / CUDA context / model init.
#   3) original mode: fixed seed, only split_index changes.
#   4) random_ratio mode: fixed model seed, only split_seed changes.
#   5) Parse BestVal/Test@BestVal and pure training-step mean/std.
#   6) Save summary log if mean result exceeds top-level thresholds.
#
# Usage:
#   Modify USER_CONFIG below, then run:
#       python multi_split_runner_100pct_sweep_style.py
# ============================================================

import concurrent.futures as futures
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev


# ============================================================
# User-editable configuration area
# ============================================================
# Amazon-ratings example parameters; no other dataset presets are distributed.
USER_CONFIG = {'train_script': 'trainb9.py',
 'parallel_gpu': False,
 'gpus': ['0'],
 'max_workers_per_gpu': 1,
 'device_arg_when_visible': 'cuda',
 'num_splits': 10,
 'split_start': 0,
 'fixed_model_seed': 42,
 'split_mode': 'random_ratio',
 'base_split_seed': 42,
 'base_config': {'dataset_name': 'Amazon-ratings',
                 'data_root': './data',
                 'use_undirected': True,
                 'split_mode': 'random_ratio',
                 'train_ratio': 0.5,
                 'val_ratio': 0.25,
                 'test_ratio': 0.25,
                 'split_seed': 42,
                 'hidden_dim': 256,
                 'num_layers': 4,
                 'dropout': 0.28,
                 'hop_scales': [1, 2, 3],
                 'gat_heads': 6,
                 'gat_dropout': 0.32,
                 'fast_backend_policy': 'auto',
                 'use_fused_gat': True,
                 'use_fused_gat_sideops': True,
                 'use_collapsed_fused_gat': True,
                 'use_fused_block_ops': True,
                 'use_fixed_csr_spmm': True,
                 'use_multihop_csr_pipeline': True,
                 'multihop_csr_algorithm': 2,
                 'use_fused_clifford_pack': True,
                 'use_tf32': True,
                 'use_selective_amp': True,
                 'selective_amp_dtype': 'bf16',
                 'enable_dirichlet_recording': False,
                 'init_gamma': 0.49,
                 'gamma_mode': 'vector',
                 'hop_gate_mode': 'node',
                 'operator_mode': 'adjacency',
                 'hybrid_alpha': 0.5,
                 'layer_combine': 'concat',
                 'lr': 0.003,
                 'weight_decay': 1e-05,
                 'epochs': 2000,
                 'patience': 200,
                 'print_every': 20,
                 'device': 'cuda',
                 'save_log': False,
                 'log_dir': 'logs',
                 'save_min_best_val': 0.0,
                 'save_min_best_test': 0.0},
 'single_run_log_dir': 'multi_split_single_run_logs_100pct',
 'summary_log_dir': 'multi_split_summary_logs_100pct',
 'parse_failed_log_dir': 'multi_split_parse_failed_logs_100pct',
 'save_summary_log': True,
 'save_min_mean_val': 0.0,
 'save_min_mean_test': 0.0,
 'print_subprocess_output': False}


# ============================================================
# Utilities
# ============================================================
def to_cli_args(config: dict):
    args = []
    for key, value in config.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            args.extend([flag, str(value)])
        elif isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend([str(v) for v in value])
        else:
            args.extend([flag, str(value)])
    return args


def sanitize_filename(text: str) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", ".", "="}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def mean_std(values):
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def combine_epoch_stats(results):
    """
    Combine per-run sample statistics into the exact pooled mean and sample std
    over all measured training epochs from all valid runs.
    """
    groups = [
        (int(r["trained_epochs"]), float(r["epoch_train_mean"]), float(r["epoch_train_std"]))
        for r in results
        if int(r["trained_epochs"]) > 0
    ]
    total_n = sum(n for n, _, _ in groups)
    if total_n == 0:
        return 0, 0.0, 0.0

    pooled_mean = sum(n * group_mean for n, group_mean, _ in groups) / total_n
    if total_n == 1:
        return total_n, pooled_mean, 0.0

    total_m2 = 0.0
    for n, group_mean, group_std in groups:
        within_m2 = (n - 1) * (group_std ** 2) if n > 1 else 0.0
        between_m2 = n * ((group_mean - pooled_mean) ** 2)
        total_m2 += within_m2 + between_m2

    pooled_std = math.sqrt(max(total_m2 / (total_n - 1), 0.0))
    return total_n, pooled_mean, pooled_std


def parse_train_output(stdout_text: str):
    best_epoch_pattern = re.compile(r"Best epoch\s*:\s*([0-9]+)")
    best_val_pattern = re.compile(r"Best validation\s+.*?:\s*([0-9]+(?:\.[0-9]+)?)%")
    test_pattern = re.compile(r"Test\s+.*?@\s*best val\s*:\s*([0-9]+(?:\.[0-9]+)?)%")

    trained_epochs_pattern = re.compile(r"Trained epochs\s*:\s*([0-9]+)")
    pure_train_total_pattern = re.compile(r"Pure training time total\s*:\s*([0-9]+(?:\.[0-9]+)?)s")
    epoch_train_mean_pattern = re.compile(r"Mean epoch training time\s*:\s*([0-9]+(?:\.[0-9]+)?)s")
    epoch_train_std_pattern = re.compile(r"Std epoch training time\s*:\s*([0-9]+(?:\.[0-9]+)?)s")

    # Compatibility fallbacks for older train scripts.
    old_train_time_pattern = re.compile(r"Training finished in\s*([0-9]+(?:\.[0-9]+)?)s")
    early_stop_pattern = re.compile(r"Early stopping at epoch\s*([0-9]+)")
    epoch_line_pattern = re.compile(r"Epoch\s+([0-9]+)\s*\|")

    best_epoch_matches = best_epoch_pattern.findall(stdout_text)
    best_val_matches = best_val_pattern.findall(stdout_text)
    test_matches = test_pattern.findall(stdout_text)
    trained_epochs_matches = trained_epochs_pattern.findall(stdout_text)
    pure_train_total_matches = pure_train_total_pattern.findall(stdout_text)
    epoch_train_mean_matches = epoch_train_mean_pattern.findall(stdout_text)
    epoch_train_std_matches = epoch_train_std_pattern.findall(stdout_text)

    if not best_val_matches or not test_matches:
        raise ValueError(
            "Could not parse BestVal/Test@BestVal from train output. "
            "Please check whether trainb9.py prints the expected final result lines."
        )

    best_epoch = int(best_epoch_matches[-1]) if best_epoch_matches else -1
    best_val = float(best_val_matches[-1])
    best_test = float(test_matches[-1])

    if trained_epochs_matches:
        trained_epochs = int(trained_epochs_matches[-1])
    else:
        early_stop_matches = early_stop_pattern.findall(stdout_text)
        epoch_line_matches = epoch_line_pattern.findall(stdout_text)
        if early_stop_matches:
            trained_epochs = int(early_stop_matches[-1])
        elif epoch_line_matches:
            trained_epochs = int(epoch_line_matches[-1])
        else:
            trained_epochs = -1

    if pure_train_total_matches:
        pure_train_time = float(pure_train_total_matches[-1])
    else:
        old_matches = old_train_time_pattern.findall(stdout_text)
        pure_train_time = float(old_matches[-1]) if old_matches else 0.0

    if epoch_train_mean_matches:
        epoch_train_mean = float(epoch_train_mean_matches[-1])
    elif pure_train_time > 0 and trained_epochs > 0:
        epoch_train_mean = pure_train_time / trained_epochs
    else:
        epoch_train_mean = 0.0

    epoch_train_std = float(epoch_train_std_matches[-1]) if epoch_train_std_matches else 0.0

    return {
        "best_epoch": best_epoch,
        "trained_epochs": trained_epochs,
        "best_val": best_val,
        "best_test": best_test,
        "pure_train_time": pure_train_time,
        "epoch_train_mean": epoch_train_mean,
        "epoch_train_std": epoch_train_std,
    }


# ============================================================
# Plan builder: exactly sweep_train3-style split logic
# ============================================================
def build_repeat_plan(config: dict):
    split_mode = USER_CONFIG["split_mode"]
    num_splits = int(USER_CONFIG["num_splits"])
    split_start = int(USER_CONFIG["split_start"])
    seed = int(USER_CONFIG["fixed_model_seed"])
    base_split_seed = int(USER_CONFIG["base_split_seed"])

    if split_mode == "original":
        plan = []
        for offset in range(num_splits):
            split_index = split_start + offset
            plan.append({
                "repeat_idx": offset + 1,
                "seed": seed,
                "split_index": split_index,
                "split_seed": base_split_seed,
                "display_split_seed": "not used",
            })
        return plan

    if split_mode == "random_ratio":
        plan = []
        for offset in range(num_splits):
            split_seed = base_split_seed + split_start + offset
            plan.append({
                "repeat_idx": offset + 1,
                "seed": seed,
                "split_index": 0,
                "split_seed": split_seed,
                "display_split_seed": str(split_seed),
            })
        return plan

    raise ValueError(f"Unknown split_mode={split_mode}.")


# ============================================================
# Logging
# ============================================================
def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_summary_text(results, valid_results, total_wall_time):
    base_config = USER_CONFIG["base_config"]
    val_scores = [r["best_val"] for r in valid_results]
    test_scores = [r["best_test"] for r in valid_results]
    pure_train_times = [r["pure_train_time"] for r in valid_results]
    epoch_train_means = [r["epoch_train_mean"] for r in valid_results]
    epoch_train_stds = [r["epoch_train_std"] for r in valid_results]
    trained_epochs_list = [r["trained_epochs"] for r in valid_results]

    mean_val, std_val = mean_std(val_scores)
    mean_test, std_test = mean_std(test_scores)
    mean_pure_train_time, std_pure_train_time = mean_std(pure_train_times)
    mean_epoch_train, std_across_split_means = mean_std(epoch_train_means)
    mean_within_run_std, std_within_run_std = mean_std(epoch_train_stds)
    mean_trained_epochs, std_trained_epochs = mean_std(trained_epochs_list)
    pooled_epoch_count, pooled_epoch_mean, pooled_epoch_std = combine_epoch_stats(valid_results)

    lines = []
    lines.append("=" * 120)
    lines.append("MULTI-SPLIT SUMMARY: PURE TRAINING-STEP TIMING")
    lines.append("=" * 120)
    lines.append(f"Train script              : {USER_CONFIG['train_script']}")
    lines.append(f"Parallel GPU              : {USER_CONFIG['parallel_gpu']}")
    lines.append(f"Dataset                   : {base_config.get('dataset_name')}")
    lines.append(f"Split mode                : {USER_CONFIG['split_mode']}")
    lines.append(f"Fixed model seed          : {USER_CONFIG['fixed_model_seed']}")
    lines.append(f"Num splits                : {USER_CONFIG['num_splits']}")
    lines.append("Timing scope              : zero_grad + train forward + loss + backward + optimizer.step only")
    lines.append("Excluded from timing      : evaluation, metrics, logging, data loading, early stopping, runner overhead")
    if USER_CONFIG["split_mode"] == "original":
        lines.append(
            f"Split index range         : {USER_CONFIG['split_start']} ... "
            f"{USER_CONFIG['split_start'] + USER_CONFIG['num_splits'] - 1}"
        )
        lines.append("Split seed                : not used in original mode")
    else:
        lines.append(
            f"Split seed range          : {USER_CONFIG['base_split_seed'] + USER_CONFIG['split_start']} ... "
            f"{USER_CONFIG['base_split_seed'] + USER_CONFIG['split_start'] + USER_CONFIG['num_splits'] - 1}"
        )
    lines.append("-" * 120)
    lines.append("BASE CONFIG:")
    lines.append(json.dumps(base_config, indent=2, ensure_ascii=False))
    lines.append("-" * 120)
    lines.append("SPLIT RESULTS:")
    for r in sorted(results, key=lambda x: x["repeat_idx"]):
        lines.append(
            f"Repeat {r['repeat_idx']:02d} | "
            f"gpu={r['gpu']} | "
            f"seed={r['seed']} | "
            f"split_index={r['split_index']} | "
            f"split_seed={r['display_split_seed']} | "
            f"BestEpoch={r['best_epoch']} | "
            f"TrainedEpochs={r['trained_epochs']} | "
            f"PureTrainTotal={r['pure_train_time']:.6f}s | "
            f"EpochTrainMean={r['epoch_train_mean']:.6f}s | "
            f"EpochTrainStd={r['epoch_train_std']:.6f}s | "
            f"BestVal={r['best_val']:.2f}% | "
            f"Test@BestVal={r['best_test']:.2f}% | "
            f"returncode={r['returncode']} | "
            f"log={r['log_path']}"
        )
    lines.append("-" * 120)
    lines.append(f"Valid runs                : {len(valid_results)}/{len(results)}")
    lines.append(f"Mean BestVal              : {mean_val:.2f}% ± {std_val:.2f}%")
    lines.append(f"Mean Test@BestVal         : {mean_test:.2f}% ± {std_test:.2f}%")
    lines.append(f"Mean TrainedEpochs        : {mean_trained_epochs:.2f} ± {std_trained_epochs:.2f}")
    lines.append(f"Mean PureTrainTotal       : {mean_pure_train_time:.6f}s ± {std_pure_train_time:.6f}s")
    lines.append(
        f"Mean EpochTrainTime       : {mean_epoch_train:.6f}s ± {std_across_split_means:.6f}s "
        f"(mean/std across split means)"
    )
    lines.append(
        f"Mean WithinRun EpochStd   : {mean_within_run_std:.6f}s ± {std_within_run_std:.6f}s"
    )
    lines.append(
        f"Pooled EpochTrainTime     : {pooled_epoch_mean:.6f}s ± {pooled_epoch_std:.6f}s "
        f"over {pooled_epoch_count} epochs"
    )
    lines.append("=" * 120)
    return "\n".join(lines), {
        "mean_val": mean_val,
        "std_val": std_val,
        "mean_test": mean_test,
        "std_test": std_test,
        "mean_pure_train_time": mean_pure_train_time,
        "std_pure_train_time": std_pure_train_time,
        "mean_epoch_train": mean_epoch_train,
        "std_across_split_means": std_across_split_means,
        "pooled_epoch_mean": pooled_epoch_mean,
        "pooled_epoch_std": pooled_epoch_std,
    }


# ============================================================
# Subprocess execution
# ============================================================
def run_one_task(task: dict):
    cmd = task["cmd"]
    cwd = task["cwd"]
    env = os.environ.copy()

    if task["gpu"] is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
        env.setdefault("PYTHONUNBUFFERED", "1")

    header_lines = []
    header_lines.append("=" * 120)
    header_lines.append(f"Repeat                  : {task['repeat_idx']}")
    header_lines.append(f"Physical GPU            : {task['gpu'] if task['gpu'] is not None else 'not forced'}")
    if task["gpu"] is not None:
        header_lines.append(f"CUDA_VISIBLE_DEVICES    : {env['CUDA_VISIBLE_DEVICES']}")
    header_lines.append("Command:")
    header_lines.append(" ".join(cmd))
    header_lines.append("=" * 120)
    header_text = "\n".join(header_lines) + "\n"

    start_wall = time.time()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    output_lines = []
    if USER_CONFIG["print_subprocess_output"] and not USER_CONFIG["parallel_gpu"]:
        print(header_text, end="")
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end="")
                sys.stdout.flush()
                output_lines.append(line)
            if line == "" and process.poll() is not None:
                break
        returncode = process.wait()
        stdout_text = "".join(output_lines)
    else:
        stdout_text, _ = process.communicate()
        returncode = process.returncode
        output_lines = [stdout_text]

    wall_elapsed = time.time() - start_wall
    full_text = header_text + stdout_text + "\n" + "=" * 120 + "\n"
    full_text += f"Return code: {returncode}\n"

    save_text(task["log_path"], full_text)

    parsed = {
        "best_epoch": -1,
        "trained_epochs": -1,
        "best_val": 0.0,
        "best_test": 0.0,
        "pure_train_time": 0.0,
        "epoch_train_mean": 0.0,
        "epoch_train_std": 0.0,
    }
    parse_failed_path = ""
    if returncode == 0:
        try:
            parsed = parse_train_output(stdout_text)
        except Exception as e:
            parse_failed_path = str(task["parse_failed_log_path"])
            save_text(task["parse_failed_log_path"], full_text + f"\nPARSE ERROR: {repr(e)}\n")

    result = dict(task["result_base"])
    result.update(parsed)
    result.update({
        "returncode": returncode,
        "runner_wall_elapsed": wall_elapsed,
        "log_path": str(task["log_path"]),
        "parse_failed_path": parse_failed_path,
    })
    return result


# ============================================================
# Main
# ============================================================
def main():
    script_dir = Path(__file__).resolve().parent
    train_py = script_dir / USER_CONFIG["train_script"]
    if not train_py.exists():
        raise FileNotFoundError(f"Cannot find train script: {train_py}")

    base_config = dict(USER_CONFIG["base_config"])
    base_config["split_mode"] = USER_CONFIG["split_mode"]

    single_run_log_dir = script_dir / USER_CONFIG["single_run_log_dir"]
    summary_log_dir = script_dir / USER_CONFIG["summary_log_dir"]
    parse_failed_log_dir = script_dir / USER_CONFIG["parse_failed_log_dir"]
    single_run_log_dir.mkdir(parents=True, exist_ok=True)
    summary_log_dir.mkdir(parents=True, exist_ok=True)
    parse_failed_log_dir.mkdir(parents=True, exist_ok=True)

    plan = build_repeat_plan(base_config)

    gpu_slots = [None]
    if USER_CONFIG["parallel_gpu"]:
        gpu_slots = []
        for gpu in USER_CONFIG["gpus"]:
            for _ in range(int(USER_CONFIG["max_workers_per_gpu"])):
                gpu_slots.append(str(gpu))
        if len(gpu_slots) == 0:
            raise ValueError("parallel_gpu=True but no GPU is provided.")

    tasks = []
    for idx, item in enumerate(plan, start=1):
        gpu = gpu_slots[(idx - 1) % len(gpu_slots)] if USER_CONFIG["parallel_gpu"] else None
        run_config = dict(base_config)
        run_config["seed"] = item["seed"]
        run_config["split_index"] = item["split_index"]
        run_config["split_seed"] = item["split_seed"]

        if USER_CONFIG["parallel_gpu"]:
            run_config["device"] = USER_CONFIG["device_arg_when_visible"]

        run_name = (
            f"repeat={item['repeat_idx']}"
            f"__dataset={run_config.get('dataset_name')}"
            f"__splitmode={run_config.get('split_mode')}"
            f"__split={run_config.get('split_index')}"
            f"__seed={run_config.get('seed')}"
            f"__gamma={run_config.get('init_gamma')}"
        )
        safe_run_name = sanitize_filename(run_name)
        log_path = single_run_log_dir / f"{safe_run_name}.txt"
        parse_failed_log_path = parse_failed_log_dir / f"PARSE_FAILED__{safe_run_name}.txt"

        cmd = [sys.executable, str(train_py)] + to_cli_args(run_config)
        tasks.append({
            "repeat_idx": item["repeat_idx"],
            "cmd": cmd,
            "cwd": script_dir,
            "gpu": gpu,
            "log_path": log_path,
            "parse_failed_log_path": parse_failed_log_path,
            "result_base": {
                "repeat_idx": item["repeat_idx"],
                "gpu": gpu if gpu is not None else "not forced",
                "seed": item["seed"],
                "split_index": item["split_index"],
                "split_seed": item["split_seed"],
                "display_split_seed": item["display_split_seed"],
            },
        })

    print("=" * 120)
    print("100% SWEEP-STYLE MULTI-SPLIT RUNNER")
    print("=" * 120)
    print(f"Train script           : {train_py}")
    print(f"Parallel GPU           : {USER_CONFIG['parallel_gpu']}")
    if USER_CONFIG["parallel_gpu"]:
        print(f"GPUs                   : {USER_CONFIG['gpus']}")
        print(f"Workers per GPU         : {USER_CONFIG['max_workers_per_gpu']}")
        print(f"Device passed to train  : {USER_CONFIG['device_arg_when_visible']}")
    print(f"Dataset                : {base_config.get('dataset_name')}")
    print(f"Split mode             : {USER_CONFIG['split_mode']}")
    print(f"Fixed model seed        : {USER_CONFIG['fixed_model_seed']}")
    print(f"Num splits             : {USER_CONFIG['num_splits']}")
    if USER_CONFIG["split_mode"] == "original":
        print(
            f"Split index range      : {USER_CONFIG['split_start']} ... "
            f"{USER_CONFIG['split_start'] + USER_CONFIG['num_splits'] - 1}"
        )
        print("Split seed             : not used in original mode")
    else:
        print(
            f"Split seed range       : {USER_CONFIG['base_split_seed'] + USER_CONFIG['split_start']} ... "
            f"{USER_CONFIG['base_split_seed'] + USER_CONFIG['split_start'] + USER_CONFIG['num_splits'] - 1}"
        )
    print(f"Single-run logs         : {single_run_log_dir}")
    print(f"Summary logs            : {summary_log_dir}")
    print("=" * 120)

    results = []
    total_start = time.time()

    if USER_CONFIG["parallel_gpu"]:
        max_workers = len(gpu_slots)
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task in tasks:
                print(
                    f"SUBMIT | Repeat={task['repeat_idx']}/{len(tasks)} | "
                    f"GPU={task['gpu']} | split_index={task['result_base']['split_index']} | "
                    f"split_seed={task['result_base']['display_split_seed']}"
                )
                future_to_task[executor.submit(run_one_task, task)] = task

            finished = 0
            for fut in futures.as_completed(future_to_task):
                result = fut.result()
                finished += 1
                results.append(result)
                print("-" * 120)
                print(
                    f"DONE [{finished}/{len(tasks)}] | "
                    f"Repeat={result['repeat_idx']} | GPU={result['gpu']} | "
                    f"split_index={result['split_index']} | split_seed={result['display_split_seed']} | "
                    f"BestEpoch={result['best_epoch']} | TrainedEpochs={result['trained_epochs']} | "
                    f"PureTrainTotal={result['pure_train_time']:.6f}s | EpochMean={result['epoch_train_mean']:.6f}s | EpochStd={result['epoch_train_std']:.6f}s | "
                    f"BestVal={result['best_val']:.2f}% | Test@BestVal={result['best_test']:.2f}% | "
                    f"returncode={result['returncode']}"
                )
                print(f"Log: {result['log_path']}")
                if result["parse_failed_path"]:
                    print(f"Parse failed log: {result['parse_failed_path']}")
                valid_now = [r for r in results if r["returncode"] == 0 and r["best_val"] > 0 and r["best_test"] > 0]
                if valid_now:
                    mean_test_now, std_test_now = mean_std([r["best_test"] for r in valid_now])
                    print(f"Running Mean Test@BestVal: {mean_test_now:.2f}% ± {std_test_now:.2f}%")
                print("-" * 120)
    else:
        for task in tasks:
            print("\n" + "#" * 120)
            print(
                f"RUN | Repeat={task['repeat_idx']}/{len(tasks)} | "
                f"split_index={task['result_base']['split_index']} | "
                f"split_seed={task['result_base']['display_split_seed']}"
            )
            print("#" * 120)
            result = run_one_task(task)
            results.append(result)
            print("-" * 120)
            print(
                f"Parsed result | "
                f"BestEpoch={result['best_epoch']} | "
                f"TrainedEpochs={result['trained_epochs']} | "
                f"PureTrainTotal={result['pure_train_time']:.6f}s | "
                f"EpochMean={result['epoch_train_mean']:.6f}s | "
                f"EpochStd={result['epoch_train_std']:.6f}s | "
                f"BestVal={result['best_val']:.2f}% | "
                f"Test@BestVal={result['best_test']:.2f}% | "
                f"returncode={result['returncode']}"
            )
            valid_now = [r for r in results if r["returncode"] == 0 and r["best_val"] > 0 and r["best_test"] > 0]
            if valid_now:
                mean_test_now, std_test_now = mean_std([r["best_test"] for r in valid_now])
                mean_val_now, std_val_now = mean_std([r["best_val"] for r in valid_now])
                print(
                    f"Running summary | Valid={len(valid_now)}/{len(results)} | "
                    f"MeanVal={mean_val_now:.2f}% ± {std_val_now:.2f}% | "
                    f"MeanTest={mean_test_now:.2f}% ± {std_test_now:.2f}%"
                )
            print("-" * 120)

    total_wall_time = time.time() - total_start
    results = sorted(results, key=lambda x: x["repeat_idx"])
    valid_results = [r for r in results if r["returncode"] == 0 and r["best_val"] > 0 and r["best_test"] > 0]
    summary_text, summary_stats = build_summary_text(results, valid_results, total_wall_time)
    print("\n" + summary_text)

    if USER_CONFIG["save_summary_log"]:
        mean_val = summary_stats["mean_val"]
        mean_test = summary_stats["mean_test"]
        if mean_val >= USER_CONFIG["save_min_mean_val"] and mean_test >= USER_CONFIG["save_min_mean_test"]:
            dataset = sanitize_filename(base_config.get("dataset_name", "dataset"))
            split_mode = sanitize_filename(USER_CONFIG["split_mode"])
            gamma = sanitize_filename(base_config.get("init_gamma", "gamma"))
            save_name = (
                f"SUMMARY__{dataset}__{split_mode}"
                f"__gamma={gamma}"
                f"__meanTest={mean_test:.2f}"
                f"__meanVal={mean_val:.2f}.txt"
            )
            summary_path = summary_log_dir / save_name
            save_text(summary_path, summary_text)
            print(f"SUMMARY_LOG_SAVED: {summary_path}")
        else:
            print(
                "SUMMARY_LOG_SAVED: skipped "
                f"because MeanVal={mean_val:.2f}% / MeanTest={mean_test:.2f}% "
                f"does not satisfy thresholds "
                f"{USER_CONFIG['save_min_mean_val']:.2f}% / {USER_CONFIG['save_min_mean_test']:.2f}%"
            )
    else:
        print("SUMMARY_LOG_SAVED: disabled")


if __name__ == "__main__":
    main()
