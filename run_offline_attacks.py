"""Standalone CLI for planning, executing, and reducing privacy attacks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from Codes.attack_checkpoints import AttackCheckpoint
from Codes.offline_attacks import (
    TaskFilter, execute_attack_run, plan_attack_run, reduce_attack_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FEDML-HE privacy attacks offline from validated checkpoints"
    )
    parser.add_argument("--checkpoint", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=("plan", "run", "reduce", "all"), default="all",
        help="Pipeline stage to perform (default: all)",
    )
    parser.add_argument("--attacks", nargs="+", choices=("mia", "ilrg", "dlg", "ig"), default=None)
    parser.add_argument("--rounds", nargs="+", type=int)
    parser.add_argument("--clients", nargs="+", type=int)
    parser.add_argument(
        "--samples", nargs="+", type=int,
        help="DLG/IG selection positions or frozen client-partition sample IDs",
    )
    parser.add_argument(
        "--batches", nargs="+", type=int,
        help="iLRG selection positions or frozen batch IDs",
    )
    parser.add_argument("--restarts", nargs="+", type=int)
    parser.add_argument(
        "--devices", nargs="+", default=["0"],
        help="CUDA device indices, or 'cpu' for tests/smoke runs",
    )
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-target-gradients", action="store_true")

    parser.add_argument("--attack-seed", type=int)
    parser.add_argument("--mia-sample-size", type=int)
    parser.add_argument("--mia-batch-size", type=int)
    parser.add_argument("--mia-bootstrap-samples", type=int)
    parser.add_argument(
        "--mia-signals", nargs="+",
        choices=("loss", "entropy", "modified_entropy"),
    )

    parser.add_argument("--ilrg-batch-size", type=int)
    parser.add_argument("--ilrg-num-batches", type=int)
    parser.add_argument("--ilrg-alpha", type=float)
    parser.add_argument("--ilrg-epsilon", type=float)
    parser.add_argument("--ilrg-mask-mode", choices=("partial", "strict"))

    parser.add_argument("--dlg-num-samples", type=int)
    parser.add_argument("--dlg-num-restarts", type=int)
    parser.add_argument("--dlg-iterations", type=int)
    parser.add_argument("--dlg-learning-rate", type=float)
    parser.add_argument("--dlg-optimizer", choices=("adam", "lbfgs"))
    parser.add_argument("--dlg-objective", choices=("l2", "normalized_l2", "cosine"))
    parser.add_argument("--dlg-tv-weight", type=float)
    parser.add_argument("--dlg-early-stopping-patience", type=int)
    parser.add_argument("--dlg-early-stopping-delta", type=float)
    parser.add_argument("--dlg-success-ssim", type=float)
    parser.add_argument("--dlg-compute-lpips", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dlg-known-label", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--ig-num-samples", type=int)
    parser.add_argument("--ig-num-restarts", type=int)
    parser.add_argument("--ig-iterations", type=int)
    parser.add_argument("--ig-learning-rate", type=float)
    parser.add_argument("--ig-tv-weight", type=float)
    parser.add_argument("--ig-early-stopping-patience", type=int)
    parser.add_argument("--ig-early-stopping-delta", type=float)
    parser.add_argument("--ig-success-ssim", type=float)
    parser.add_argument("--ig-compute-lpips", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ig-known-label", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _overrides(args: argparse.Namespace) -> dict:
    names = (
        "attacks", "attack_seed", "mia_sample_size", "mia_batch_size",
        "mia_bootstrap_samples", "mia_signals", "ilrg_batch_size",
        "ilrg_num_batches", "ilrg_alpha", "ilrg_epsilon", "ilrg_mask_mode",
        "dlg_num_samples", "dlg_num_restarts", "dlg_iterations",
        "dlg_learning_rate", "dlg_optimizer", "dlg_objective", "dlg_tv_weight",
        "dlg_early_stopping_patience", "dlg_early_stopping_delta",
        "dlg_success_ssim", "dlg_compute_lpips", "dlg_known_label",
        "ig_num_samples", "ig_num_restarts", "ig_iterations", "ig_learning_rate",
        "ig_tv_weight", "ig_early_stopping_patience", "ig_early_stopping_delta",
        "ig_success_ssim", "ig_compute_lpips", "ig_known_label",
    )
    values = {name: getattr(args, name) for name in names}
    values["cache_target_gradients"] = args.cache_target_gradients
    return values


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    filters = TaskFilter(
        rounds=None if args.rounds is None else set(args.rounds),
        clients=None if args.clients is None else set(args.clients),
        samples=None if args.samples is None else set(args.samples),
        batches=None if args.batches is None else set(args.batches),
        restarts=None if args.restarts is None else set(args.restarts),
    )
    exit_code = 0
    for checkpoint_path in args.checkpoint:
        checkpoint = AttackCheckpoint(checkpoint_path)
        output_root = args.output_root
        if output_root is not None and len(args.checkpoint) > 1:
            output_root = output_root / checkpoint.run_id
        run_root, manifest = plan_attack_run(
            checkpoint_path, overrides=_overrides(args), filters=filters,
            output_root=output_root,
        )
        print(f"Attack run: {manifest['attack_run_id']}")
        print(f"Checkpoint: {checkpoint.run_id}")
        print(f"Tasks: {len(manifest['tasks'])}")
        print(f"Output: {run_root}")
        if args.stage == "plan":
            continue
        if args.stage in {"run", "all"}:
            execution = execute_attack_run(
                run_root, manifest, devices=args.devices,
                workers_per_device=args.workers_per_device,
                resume=args.resume, force=args.force,
            )
            print("Execution: " + json.dumps(execution, sort_keys=True))
            if execution["failed"]:
                exit_code = 1
        if args.stage in {"reduce", "all"}:
            report = reduce_attack_run(run_root, manifest)
            print("Reduction: " + json.dumps(report["task_counts"], sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
