import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from Codes.path_utils import safe_path_component
from Codes.enums import resolve_model

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ.setdefault('MPLBACKEND', 'Agg')
PROJECT_ROOT = Path(__file__).resolve().parent


def change_console_code_page():
    """Change console code page to UTF-8."""
    # subprocess.run("chcp 65001", shell=True)


def _add_flag(cmd: List[str], flag: str, value: Any):
    """Append a CLI flag and value if value is not None."""
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _add_bool_flag(cmd: List[str], flag: str, enabled: bool):
    """Append a CLI boolean flag if enabled."""
    if enabled:
        cmd.append(flag)


def _add_bool_option(cmd: List[str], flag: str, value: Any):
    """Append a BooleanOptionalAction flag when explicitly configured."""
    if value is None:
        return
    cmd.append(flag if bool(value) else f"--no-{flag.removeprefix('--')}")


def build_cmd(config: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, str(PROJECT_ROOT / "ProjectControl_Loop.py")]

    # Federated learning arguments
    _add_flag(cmd, "--group", config.get("group"))
    _add_flag(cmd, "--num_clients", config.get("num_clients"))
    _add_flag(cmd, "--local_epochs", config.get("local_epochs"))
    _add_flag(cmd, "--rounds", config.get("rounds"))
    _add_flag(cmd, "--encryption_ratio", config.get("encryption_ratio"))
    _add_bool_flag(cmd, "--aggregate_BN", bool(config.get("aggregate_BN", False)))

    # Dataset arguments
    _add_flag(cmd, "--dataset", config.get("dataset"))
    _add_flag(cmd, "--DB_samples_per_client", config.get("DB_samples_per_client"))
    _add_flag(cmd, "--alpha", config.get("alpha"))
    _add_flag(cmd, "--seed", config.get("seed"))
    _add_bool_option(cmd, "--nonIID", config.get("nonIID"))
    _add_bool_flag(cmd, "--DB_forceCreate", bool(config.get("DB_forceCreate", False)))

    # Model arguments
    _add_flag(cmd, "--model", config.get("model"))
    _add_flag(cmd, "--temperature", config.get("temperature"))
    _add_flag(cmd, "--local_batch_size", config.get("local_batch_size"))
    _add_flag(cmd, "--eval_batch_size", config.get("eval_batch_size"))
    _add_flag(
        cmd,
        "--mask_gradient_batch_size",
        config.get("mask_gradient_batch_size"),
    )

    # Misc arguments
    _add_flag(cmd, "--exp_id", config.get("exp_id"))
    _add_flag(cmd, "--run_id", config.get("run_id"))
    _add_flag(cmd, "--checkpoint_root", config.get("checkpoint_root"))

    # Attack arguments
    attack_enabled = config.get("attack")
    if attack_enabled is None:
        legacy_attack_values = [
            config.get("run_gradient_attacks"),
            config.get("run_membership_attack"),
        ]
        if any(value is not None for value in legacy_attack_values):
            attack_enabled = any(bool(value) for value in legacy_attack_values)
    _add_bool_option(cmd, "--attack", attack_enabled)
    _add_flag(cmd, "--attack_interval", config.get("attack_interval"))
    _add_bool_option(cmd, "--run_mia", config.get("run_mia"))
    _add_bool_option(cmd, "--run_dlg", config.get("run_dlg"))
    _add_flag(cmd, "--attack_seed", config.get("attack_seed"))
    _add_flag(cmd, "--mia_sample_size", config.get("mia_sample_size"))
    _add_flag(cmd, "--mia_bootstrap_samples", config.get("mia_bootstrap_samples"))
    _add_bool_option(cmd, "--run_ilrg", config.get("run_ilrg"))
    _add_flag(cmd, "--ilrg_batch_size", config.get("ilrg_batch_size"))
    _add_flag(cmd, "--ilrg_num_batches", config.get("ilrg_num_batches"))
    _add_flag(cmd, "--ilrg_alpha", config.get("ilrg_alpha"))
    _add_flag(cmd, "--ilrg_mask_mode", config.get("ilrg_mask_mode"))
    _add_flag(cmd, "--dlg_num_samples", config.get("dlg_num_samples"))
    _add_flag(cmd, "--dlg_num_restarts", config.get("dlg_num_restarts"))
    _add_flag(cmd, "--dlg_iterations", config.get("dlg_iterations"))
    _add_flag(cmd, "--dlg_learning_rate", config.get("dlg_learning_rate"))
    _add_flag(cmd, "--dlg_optimizer", config.get("dlg_optimizer"))
    _add_flag(cmd, "--dlg_objective", config.get("dlg_objective"))
    _add_flag(cmd, "--dlg_tv_weight", config.get("dlg_tv_weight"))
    _add_flag(
        cmd,
        "--dlg_early_stopping_patience",
        config.get("dlg_early_stopping_patience"),
    )
    _add_flag(cmd, "--dlg_success_ssim", config.get("dlg_success_ssim"))
    _add_bool_option(cmd, "--dlg_compute_lpips", config.get("dlg_compute_lpips"))
    _add_bool_option(cmd, "--dlg_known_label", config.get("dlg_known_label"))
    _add_bool_option(cmd, "--run_ig", config.get("run_ig"))
    _add_flag(cmd, "--ig_num_samples", config.get("ig_num_samples"))
    _add_flag(cmd, "--ig_num_restarts", config.get("ig_num_restarts"))
    _add_flag(cmd, "--ig_iterations", config.get("ig_iterations"))
    _add_flag(cmd, "--ig_learning_rate", config.get("ig_learning_rate"))
    _add_flag(cmd, "--ig_tv_weight", config.get("ig_tv_weight"))
    _add_flag(
        cmd,
        "--ig_early_stopping_patience",
        config.get("ig_early_stopping_patience"),
    )
    _add_flag(cmd, "--ig_success_ssim", config.get("ig_success_ssim"))
    _add_bool_option(cmd, "--ig_compute_lpips", config.get("ig_compute_lpips"))
    _add_bool_option(cmd, "--ig_known_label", config.get("ig_known_label"))

    return cmd


def run_single_experiment(config: Dict[str, Any], results_dir: Path, timeout_s: Optional[int] = None):
    config = dict(config)
    if not config.get("run_id"):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config["run_id"] = safe_path_component(
            f"{config['exp_id']}-{timestamp}-{uuid.uuid4().hex[:10]}"
        )
    exp_id = config["exp_id"]
    safe_exp_id = safe_path_component(exp_id)
    group = config.get("group", "UNKNOWN")

    logs_dir = results_dir / "logs" / group
    logs_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = logs_dir / f"{safe_exp_id}.out.txt"
    stderr_path = logs_dir / f"{safe_exp_id}.err.txt"

    cmd = build_cmd(config)

    print("\n" + "=" * 80)
    print(f"Running: {exp_id}")
    print(f"Group:   {group}")
    print("Cmd:     " + " ".join(cmd))


    start_time = time.perf_counter()
    try:
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            proc = subprocess.run(
                cmd,
                check=True,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=timeout_s,
                cwd=PROJECT_ROOT,
            )
        elapsed = time.perf_counter() - start_time

        print("Status: success")
        print("=" * 80 + "\n")
        return {
            "status": "success",
            "time": elapsed,
            "config": config,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "checkpoint_path": str(
                PROJECT_ROOT / config.get("checkpoint_root", "Results/AttackCheckpoints") /
                resolve_model(str(config.get("model", "lenet5"))).name /
                str(group) / str(config["run_id"])
            ),
        }


    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start_time
        print("Status: timeout")
        print("=" * 80 + "\n")
        return {
            "status": "timeout",
            "time": elapsed,
            "config": config,
            "error": f"Timeout after {timeout_s}s",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start_time
        print("Status: failed")
        print("=" * 80 + "\n")
        return {
            "status": "failed",
            "time": elapsed,
            "config": config,
            "error": str(e),
            "returncode": e.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }



def _resolve_config_file(config_source: str) -> Path:
    supplied_path = Path(config_source)
    if supplied_path.suffix.lower() == ".json":
        return supplied_path if supplied_path.is_absolute() else PROJECT_ROOT / supplied_path
    return PROJECT_ROOT / "experiment_configs" / f"{config_source}_configs.json"


def run_experiment_group(config_source: str, start_idx: int = 0, end_idx: Optional[int] = None,
                         timeout_s: Optional[int] = None, skip_successful: bool = True,
                         dry_run: bool = False):
    config_file = _resolve_config_file(config_source)
    if not config_file.exists():
        print(f"Config file not found: {config_file}")
        return

    with config_file.open("r", encoding="utf-8") as f:
        configs = json.load(f)
    if not isinstance(configs, list):
        raise ValueError(f"Expected a JSON list in {config_file}")
    for index, config in enumerate(configs):
        missing = {"group", "exp_id"} - config.keys()
        if missing:
            raise ValueError(f"Config {index} in {config_file} is missing {sorted(missing)}")
    seen_exp_ids = set()
    duplicate_exp_ids = set()
    for config in configs:
        exp_id = str(config["exp_id"])
        if exp_id in seen_exp_ids:
            duplicate_exp_ids.add(exp_id)
        seen_exp_ids.add(exp_id)
    if duplicate_exp_ids:
        raise ValueError(
            f"Experiment IDs must be unique: {sorted(duplicate_exp_ids)}"
        )

    if end_idx is None:
        end_idx = len(configs)
    if not 0 <= start_idx <= end_idx <= len(configs):
        raise ValueError(
            f"Invalid range [{start_idx}:{end_idx}] for {len(configs)} configs"
        )

    configs_to_run = configs[start_idx:end_idx]
    group_name = config_file.stem.removesuffix("_configs")

    if dry_run:
        print(f"Validated {len(configs_to_run)} experiment(s) from {config_file}")
        for config in configs_to_run:
            print(f"[{config['exp_id']}] {subprocess.list2cmdline(build_cmd(config))}")
        return

    results_dir = PROJECT_ROOT / "Results" / "ExperimentRuns"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{group_name}_results.json"

    # Load previous results for resume
    previous = []
    done_success = set()
    if results_path.exists():
        try:
            previous = json.loads(results_path.read_text(encoding="utf-8"))
            for r in previous:
                if r.get("status") == "success":
                    done_success.add(r["config"]["exp_id"])
        except Exception:
            previous = []

    results = previous[:]

    print("\n" + "=" * 80)
    print(f"Running {len(configs_to_run)} experiments from {group_name}")
    print(f"Range: {start_idx} to {end_idx}")
    if skip_successful:
        print(f"Resume mode: ON (skip successful runs already in {results_path})")
    if timeout_s is not None:
        print(f"Timeout per run: {timeout_s}s")
    print("=" * 80 + "\n")

    for i, config in enumerate(configs_to_run, start=start_idx):
        exp_id = config["exp_id"]
        if skip_successful and exp_id in done_success:
            print(f"[{i + 1}/{end_idx}] SKIP (already successful): {exp_id}")
            continue

        print(f"[{i + 1}/{end_idx}] Starting: {exp_id}")
        result = run_single_experiment(config, results_dir=results_dir, timeout_s=timeout_s)
        results.append(result)

        # Save progress after every run
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    total_time = sum(r.get("time", 0.0) for r in results)

    print("\n" + "=" * 80)
    print(f"SUMMARY for {group_name}")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Timeout:    {timeout}")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    change_console_code_page()

    parser = argparse.ArgumentParser(description="Run FEDML-HE experiment configurations")
    parser.add_argument(
        "configs", nargs="+",
        help="Config group names (for example, attack) or JSON file paths",
    )
    parser.add_argument("--start", type=int, default=0, help="Inclusive config index")
    parser.add_argument("--end", type=int, default=None, help="Exclusive config index")
    parser.add_argument("--timeout", type=int, default=None, help="Per-run timeout in seconds")
    parser.add_argument("--rerun", action="store_true", help="Do not skip successful runs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only")
    args = parser.parse_args()

    config_sources = [
        source.strip()
        for value in args.configs
        for source in value.split(",")
        if source.strip()
    ]
    for source in config_sources:
        run_experiment_group(
            source,
            start_idx=args.start,
            end_idx=args.end,
            timeout_s=args.timeout,
            skip_successful=not args.rerun,
            dry_run=args.dry_run,
        )
