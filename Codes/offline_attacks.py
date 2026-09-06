"""Registry, task runner, and reducer for checkpoint-backed privacy attacks."""

from __future__ import annotations

import contextlib
import copy
import json
import multiprocessing
import os
import random
import shutil
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from Codes import excelHelper
from Codes.attack_checkpoints import (
    AttackCheckpoint, ArtifactNamespaces, CheckpointValidationError,
    PRODUCER_VERSION, SCHEMA_VERSION, atomic_write_json, atomic_write_text,
    hash_arrays, hash_file, hash_json, load_npz, write_npz,
)
from Codes.attacks_new import (
    MembershipInferenceAttack, perform_DLG_attack, perform_IG_attack,
    perform_iLRG_attack,
)
from Codes.enums import Config, resolve_dataset, resolve_model
from Codes.functions import dlg_mean_std
from Codes.trainEngine import TrainEngine


ATTACK_PROTOCOL_VERSION = "FEDML-HE-standardized-gradient-leakage-v1"
IMPLEMENTATION_VERSIONS = {
    "mia": "mia-threshold-v2-offline1",
    "ilrg": "ilrg-elementwise-v1-offline1",
    "dlg": "dlg-elementwise-v2-offline1",
    "ig": "ig-elementwise-v1-offline1",
}


class OfflineAttackError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resolved_defaults(checkpoint: AttackCheckpoint, overrides: Mapping[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(checkpoint.manifest["attack_defaults"])
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["attacks"] = list(overrides.get("attacks") or ("mia", "ilrg", "dlg", "ig"))
    config["mia_signals"] = list(overrides.get("mia_signals") or config["mia_signals"])
    config["cache_target_gradients"] = bool(overrides.get("cache_target_gradients", False))
    config["protocol_version"] = ATTACK_PROTOCOL_VERSION
    config["implementation_versions"] = IMPLEMENTATION_VERSIONS
    for key in ("mia_sample_size", "mia_batch_size", "dlg_num_samples", "dlg_num_restarts",
                "dlg_iterations", "ig_num_samples", "ig_num_restarts", "ig_iterations",
                "ilrg_num_batches"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("dlg_learning_rate", "ig_learning_rate", "ilrg_alpha", "ilrg_epsilon"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("dlg_tv_weight", "ig_tv_weight", "dlg_early_stopping_delta",
                "ig_early_stopping_delta"):
        if float(config[key]) < 0:
            raise ValueError(f"{key} cannot be negative")
    invalid_attacks = set(config["attacks"]) - set(ATTACK_REGISTRY)
    if invalid_attacks:
        raise ValueError(f"Unknown attacks: {sorted(invalid_attacks)}")
    invalid_signals = set(config["mia_signals"]) - {"loss", "entropy", "modified_entropy"}
    if invalid_signals:
        raise ValueError(f"Unknown MIA signals: {sorted(invalid_signals)}")
    return config


def _trainer_from_checkpoint(checkpoint: AttackCheckpoint, device: str) -> TrainEngine:
    manifest = checkpoint.manifest
    cfg = Config()
    cfg.temperature = float(manifest["resolved_configuration"].get("temperature", 4.0))
    cfg.trainStrategy = manifest["resolved_configuration"].get("trainStrategy")
    cfg.inputShape = tuple(manifest["dataset"]["input_dimensions"])
    cfg.classNum = int(manifest["dataset"]["num_classes"])
    cfg.local_batch_size = int(manifest["federated_learning"]["local_batch_size"])
    cfg.eval_batch_size = int(manifest["federated_learning"]["evaluation_batch_size"])
    cfg.mask_gradient_batch_size = int(manifest["federated_learning"]["mask_gradient_batch_size"])
    cfg.local_epochs = int(manifest["federated_learning"]["local_epochs"])
    cfg.model = resolve_model(manifest["model"]["name"])
    cfg.DB_dataset = resolve_dataset(manifest["dataset"]["name"])
    if device == "cpu":
        cfg.gpu_id = 0
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} requested, but CUDA is unavailable")
        device_index = int(device)
        torch.cuda.set_device(device_index)
        cfg.gpu_id = device_index
    empty_inputs = np.empty((0, *cfg.inputShape), dtype=np.float32)
    empty_labels = np.empty((0, cfg.classNum), dtype=np.float32)
    trainer = TrainEngine(cfg=cfg, testData=(empty_inputs, empty_labels), globalData=None)
    expected_names = [entry["name"] for entry in checkpoint.schema]
    actual_names = list(trainer.model.state_dict().keys())
    if actual_names != expected_names:
        raise CheckpointValidationError(
            "Reconstructed model state_dict schema differs from checkpoint"
        )
    for (name, tensor), entry in zip(trainer.model.state_dict().items(), checkpoint.schema):
        if list(tensor.shape) != entry["shape"] or str(tensor.detach().cpu().numpy().dtype) != entry["dtype"]:
            raise CheckpointValidationError(f"Reconstructed model schema mismatch for {name}")
    return trainer


@dataclass(frozen=True)
class TaskFilter:
    rounds: Optional[set[int]] = None
    clients: Optional[set[int]] = None
    samples: Optional[set[int]] = None
    batches: Optional[set[int]] = None
    restarts: Optional[set[int]] = None

    @staticmethod
    def _matches(selected: Optional[set[int]], *values: int) -> bool:
        return selected is None or any(int(value) in selected for value in values)


class AttackAdapter(ABC):
    name: str

    @abstractmethod
    def requirements(self) -> Dict[str, Sequence[str]]:
        raise NotImplementedError

    @abstractmethod
    def plan_tasks(
        self, checkpoint: AttackCheckpoint, config: Mapping[str, Any], filters: TaskFilter
    ) -> list[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def run_task(
        self,
        *,
        task: Mapping[str, Any],
        config: Mapping[str, Any],
        checkpoint: AttackCheckpoint,
        artifacts: ArtifactNamespaces,
        trainer: TrainEngine,
        output_dir: Path,
        cache_root: Path,
    ) -> Dict[str, Any]:
        raise NotImplementedError


def _base_task(
    checkpoint: AttackCheckpoint,
    config_hash: str,
    attack: str,
    round_index: int,
    client_id: int,
    **identity: Any,
) -> Dict[str, Any]:
    identity_payload = {
        "checkpoint_run_id": checkpoint.run_id,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "round": int(round_index),
        "client": int(client_id),
        "attack": attack,
        "implementation_version": IMPLEMENTATION_VERSIONS[attack],
        "protocol_version": ATTACK_PROTOCOL_VERSION,
        "attack_config_hash": config_hash,
        **identity,
    }
    return {**identity_payload, "task_id": hash_json(identity_payload)}


class MIAAdapter(AttackAdapter):
    name = "mia"

    def requirements(self):
        return {
            "server_visible": ("attacker_visible_state",),
            "evaluation_oracle": ("selections", "selection_metadata"),
        }

    def plan_tasks(self, checkpoint, config, filters):
        tasks = []
        config_hash = hash_json(config)
        for round_index in checkpoint.available_rounds():
            if not filters._matches(filters.rounds, round_index):
                continue
            round_manifest = checkpoint.load_round_manifest(round_index)
            for client_id_text in sorted(round_manifest["clients"], key=int):
                client_id = int(client_id_text)
                if not filters._matches(filters.clients, client_id):
                    continue
                selection_meta, arrays = checkpoint.load_selections(client_id)
                stored = int(selection_meta["mia"]["actual_member_count"])
                requested = int(config["mia_sample_size"])
                frozen_requested = int(selection_meta["mia"]["requested_sample_count"])
                if requested > stored and requested != frozen_requested:
                    raise OfflineAttackError(
                        f"MIA requested {requested} samples for client {client_id}, but checkpoint freezes {stored}"
                    )
                actual_count = min(requested, stored)
                tasks.append(_base_task(
                    checkpoint, config_hash, self.name, round_index, client_id,
                    signal=None, signals=list(config["mia_signals"]),
                    sample_or_batch_id=f"mia-{actual_count}", restart_id=None,
                    seed=int(config["attack_seed"] + client_id * 1000),
                ))
        return tasks

    def run_task(self, *, task, config, checkpoint, artifacts, trainer, output_dir, cache_root):
        attacker_state = artifacts.require("server_visible", "attacker_visible_state")
        selections = artifacts.require("evaluation_oracle", "selections")
        trainer.setAllWeights(attacker_state)
        count = min(int(config["mia_sample_size"]), len(selections["mia_member_inputs"]))
        signals = tuple(task["signals"])
        seed_base = int(config["attack_seed"] + task["client"] * 1000)
        mia = MembershipInferenceAttack(SimpleNamespace(), trainer)
        save_paths = {signal: str(output_dir / f"mia_{signal}.json") for signal in signals}
        started = time.perf_counter()
        results = mia.run_signals(
            member_data=(selections["mia_member_inputs"][:count], selections["mia_member_labels"][:count]),
            nonmember_data=(selections["mia_nonmember_inputs"][:count], selections["mia_nonmember_labels"][:count]),
            signals=signals,
            batch_size=int(config["mia_batch_size"]),
            save_paths=save_paths,
            num_bootstrap=int(config["mia_bootstrap_samples"]),
            bootstrap_seed=seed_base,
        )
        payload = {
            "round": task["round"], "client": task["client"],
            "MIA_TotalTime": time.perf_counter() - started,
        }
        for signal, metrics in results.items():
            info = metrics.additional_info
            payload.update({
                f"{signal}_AUC": info.get("auc"),
                f"{signal}_AUC_CI_Lower": info.get("auc_ci_lower"),
                f"{signal}_AUC_CI_Upper": info.get("auc_ci_upper"),
                f"{signal}_BootstrapSamples": info.get("bootstrap_samples"),
                f"{signal}_BootstrapScheme": info.get("bootstrap_scheme"),
                f"{signal}_OracleBestAcc": info.get("oracle_best_accuracy"),
                f"{signal}_BalancedAcc": info.get("balanced_accuracy"),
                f"{signal}_OptimalThreshold": info.get("optimal_threshold"),
                f"{signal}_Precision": info.get("precision"),
                f"{signal}_Recall": info.get("recall"),
                f"{signal}_F1": info.get("f1"),
                f"{signal}_Specificity": info.get("specificity"),
                f"{signal}_PrivacyAdvantage": info.get("privacy_advantage"),
                f"{signal}_TPRAtFPR1pct": info.get("tpr_at_fpr_1pct"),
                f"{signal}_TPRAtFPR5pct": info.get("tpr_at_fpr_5pct"),
                f"{signal}_TPRAtFPR10pct": info.get("tpr_at_fpr_10pct"),
                f"{signal}_AttackTime": metrics.attack_time,
                f"{signal}_SharedFeatureExtractionTime": info.get("shared_feature_extraction_time"),
                f"{signal}_SignalEvaluationTime": info.get("signal_evaluation_time"),
                f"{signal}_NumMember": info.get("num_member"),
                f"{signal}_NumNonmember": info.get("num_nonmember"),
                f"{signal}_ThresholdNote": info.get("threshold_note"),
                f"{signal}_FPR": json.dumps(metrics.fpr, separators=(",", ":")),
                f"{signal}_TPR": json.dumps(metrics.tpr, separators=(",", ":")),
            })
        return {
            "scientific_status": "available",
            "metrics": {signal: metrics.to_dict() for signal, metrics in results.items()},
            "legacy_sheet": "Attack_MIA",
            "legacy_payload": payload,
            "selection": {
                "member_sample_ids": selections["mia_member_sample_ids"][:count].tolist(),
                "nonmember_sample_ids": selections["mia_nonmember_sample_ids"][:count].tolist(),
                "selection_seed": artifacts.evaluation_oracle["selection_metadata"]["mia"]["selection_seed"],
                "selection_policy": "frozen checkpoint prefix",
            },
        }


class ILRGAdapter(AttackAdapter):
    name = "ilrg"

    def requirements(self):
        return {
            "server_visible": ("attacker_visible_state", "plaintext_masks"),
            "evaluation_oracle": ("actual_post_local_state", "selections", "selection_metadata"),
        }

    def plan_tasks(self, checkpoint, config, filters):
        tasks = []
        config_hash = hash_json(config)
        for round_index in checkpoint.available_rounds():
            if not filters._matches(filters.rounds, round_index):
                continue
            round_manifest = checkpoint.load_round_manifest(round_index)
            for client_id_text in sorted(round_manifest["clients"], key=int):
                client_id = int(client_id_text)
                if not filters._matches(filters.clients, client_id):
                    continue
                metadata, arrays = checkpoint.load_selections(client_id)
                stored_batch_size = int(metadata["ilrg"]["batch_size"])
                if int(config["ilrg_batch_size"]) != stored_batch_size:
                    raise OfflineAttackError(
                        "iLRG batch membership is frozen; --ilrg-batch-size must match "
                        f"the checkpoint value {stored_batch_size}"
                    )
                count = min(int(config["ilrg_num_batches"]), len(arrays["ilrg_batch_ids"]))
                for offset in range(count):
                    batch_id = int(arrays["ilrg_batch_ids"][offset])
                    if not filters._matches(filters.batches, batch_id, offset):
                        continue
                    tasks.append(_base_task(
                        checkpoint, config_hash, self.name, round_index, client_id,
                        batch_id=batch_id, batch_position=offset,
                        sample_or_batch_id=batch_id, restart_id=None,
                        seed=int(config["attack_seed"] + client_id * 1000),
                    ))
        return tasks

    def run_task(self, *, task, config, checkpoint, artifacts, trainer, output_dir, cache_root):
        selections = artifacts.require("evaluation_oracle", "selections")
        position = int(task["batch_position"])
        start, end = map(int, selections["ilrg_offsets"][position:position + 2])
        metrics = perform_iLRG_attack(
            trainer,
            client_model=artifacts.require("evaluation_oracle", "actual_post_local_state"),
            attacker_model=artifacts.require("server_visible", "attacker_visible_state"),
            batch_x=selections["ilrg_inputs"][start:end],
            batch_y=selections["ilrg_labels"][start:end],
            maskBoolNot=artifacts.require("server_visible", "plaintext_masks"),
            alpha=float(config["ilrg_alpha"]),
            mask_mode=config["ilrg_mask_mode"],
            epsilon=float(config["ilrg_epsilon"]),
            batch_id=int(task["batch_id"]),
            save_dir=str(output_dir),
        )
        info = metrics.additional_info
        boundary = selections["ilrg_batch_boundaries"][position]
        payload = {
            "round": task["round"], "client": task["client"],
            "batch_id": int(task["batch_id"]),
            "batch_start": int(boundary[0]), "batch_end": int(boundary[1]),
            "batch_size": metrics.batch_size, "attack_available": metrics.attack_available,
            "unavailable_reason": metrics.unavailable_reason, "mask_mode": info.get("mask_mode"),
            "attack_scope": info.get("attack_scope"), "alpha": info.get("alpha"),
            "true_labels": json.dumps(info.get("true_labels"), separators=(",", ":")),
            "predicted_labels": json.dumps(info.get("predicted_labels"), separators=(",", ":")),
            "true_counts": json.dumps(metrics.true_counts, separators=(",", ":")),
            "continuous_counts": json.dumps(metrics.continuous_counts, separators=(",", ":")),
            "predicted_counts": json.dumps(metrics.predicted_counts, separators=(",", ":")),
            "label_existence_accuracy": metrics.label_existence_accuracy,
            "label_number_accuracy": metrics.label_number_accuracy,
            "instance_recall": metrics.instance_recall, "count_mae": metrics.count_mae,
            "normalized_count_l1": metrics.normalized_count_l1,
            "count_cosine_similarity": metrics.count_cosine_similarity,
            "exact_count_vector": metrics.exact_count_vector,
            "label_precision": metrics.label_precision, "label_recall": metrics.label_recall,
            "label_f1": metrics.label_f1, "attack_time": metrics.attack_time,
            "visible_gradient_fraction": info.get("visible_gradient_fraction"),
            "final_weight_visible_fraction": info.get("final_weight_visible_fraction"),
            "final_bias_visible_fraction": info.get("final_bias_visible_fraction"),
            "usable_bias_equations": info.get("usable_bias_equations"),
            "recovered_embedding_classes": info.get("recovered_embedding_classes"),
            "mean_embedding_visible_fraction": info.get("mean_embedding_visible_fraction"),
            "system_rank": info.get("system_rank"),
            "system_condition_number": info.get("system_condition_number"),
            "residual_l2": info.get("residual_l2"),
            "target_gradient_state": info.get("target_gradient_state"),
            "inference_model_state": info.get("inference_model_state"),
            "final_layer_name": info.get("final_layer_name"),
        }
        return {
            "scientific_status": "available" if metrics.attack_available else "unavailable",
            "unavailable_reason": metrics.unavailable_reason,
            "metrics": metrics.to_dict(), "legacy_sheet": "Attack_iLRG",
            "legacy_payload": payload,
            "selection": {
                "sample_ids": selections["ilrg_sample_ids"][start:end].tolist(),
                "batch_id": int(task["batch_id"]),
            },
        }


def _target_cache(
    *, trainer: TrainEngine, actual_state: Sequence[np.ndarray], sample_x: np.ndarray,
    sample_y: np.ndarray, cache_root: Path, cache_key: str, enabled: bool,
) -> tuple[Optional[list[np.ndarray]], Dict[str, Any]]:
    if not enabled:
        return None, {"enabled": False, "hit": False, "runtime_seconds": 0.0}
    final_dir = cache_root / cache_key
    arrays_path = final_dir / "target_gradients.npz"
    manifest_path = final_dir / "manifest.json"
    complete_path = final_dir / "COMPLETE"
    if arrays_path.is_file() and manifest_path.is_file() and complete_path.is_file():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") == cache_key and complete_path.read_text(encoding="utf-8").strip() == metadata.get("file_sha256"):
            if hash_file(arrays_path) == metadata["file_sha256"]:
                arrays = load_npz(arrays_path)
                gradients = [arrays[f"gradient_{index:05d}"] for index in range(len(arrays))]
                return gradients, {"enabled": True, "hit": True, "runtime_seconds": 0.0, "cache_key": cache_key}
    start = time.perf_counter()
    trainer.setAllWeights(actual_state)
    gradients = trainer.compute_gradients(sample_x, sample_y)
    temp_dir = cache_root / f".{cache_key}.tmp-{os.getpid()}-{time.time_ns()}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        write_npz(temp_dir / "target_gradients.npz", {
            f"gradient_{index:05d}": gradient for index, gradient in enumerate(gradients)
        })
        metadata = {
            "cache_key": cache_key,
            "file_sha256": hash_file(temp_dir / "target_gradients.npz"),
            "created_at_unix": datetime.now(timezone.utc).timestamp(),
        }
        atomic_write_json(temp_dir / "manifest.json", metadata)
        atomic_write_text(temp_dir / "COMPLETE", metadata["file_sha256"] + "\n")
        try:
            os.replace(temp_dir, final_dir)
        except OSError:
            shutil.rmtree(temp_dir, ignore_errors=True)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return gradients, {
        "enabled": True, "hit": False,
        "runtime_seconds": time.perf_counter() - start, "cache_key": cache_key,
    }


class GradientAdapter(AttackAdapter):
    def __init__(self, name: str):
        self.name = name

    def requirements(self):
        return {
            "server_visible": ("attacker_visible_state", "plaintext_masks"),
            "evaluation_oracle": ("actual_post_local_state", "selections", "selection_metadata"),
        }

    def plan_tasks(self, checkpoint, config, filters):
        tasks = []
        config_hash = hash_json(config)
        sample_count_key = f"{self.name}_num_samples"
        restart_count_key = f"{self.name}_num_restarts"
        for round_index in checkpoint.available_rounds():
            if not filters._matches(filters.rounds, round_index):
                continue
            round_manifest = checkpoint.load_round_manifest(round_index)
            for client_id_text in sorted(round_manifest["clients"], key=int):
                client_id = int(client_id_text)
                if not filters._matches(filters.clients, client_id):
                    continue
                metadata, arrays = checkpoint.load_selections(client_id)
                stored = len(arrays[f"{self.name}_sample_ids"])
                requested = int(config[sample_count_key])
                if requested > stored:
                    raise OfflineAttackError(
                        f"{self.name.upper()} requested {requested} samples for client {client_id}, "
                        f"but checkpoint freezes {stored}"
                    )
                restarts = int(config[restart_count_key])
                seed_base = int(config["attack_seed"] + client_id * 1000)
                for position in range(requested):
                    sample_id = int(arrays[f"{self.name}_sample_ids"][position])
                    if not filters._matches(filters.samples, position, sample_id):
                        continue
                    sample_seed_base = seed_base + position * restarts
                    for restart in range(restarts):
                        if not filters._matches(filters.restarts, restart):
                            continue
                        tasks.append(_base_task(
                            checkpoint, config_hash, self.name, round_index, client_id,
                            sample_id=sample_id, selection_position=position,
                            sample_or_batch_id=sample_id, restart_id=restart,
                            sample_seed_base=sample_seed_base,
                            seed=sample_seed_base + restart,
                        ))
        return tasks

    def run_task(self, *, task, config, checkpoint, artifacts, trainer, output_dir, cache_root):
        masks = artifacts.require("server_visible", "plaintext_masks")
        trainable_indices = [entry["state_index"] for entry in checkpoint.trainable_schema]
        visible = sum(np.count_nonzero(masks[index]) for index in trainable_indices)
        total = sum(masks[index].size for index in trainable_indices)
        if visible == 0:
            return {
                "scientific_status": "unavailable",
                "unavailable_reason": "no trainable gradient coordinates are server-visible",
                "metrics": {
                    "attack": f"{self.name.upper()}_ElementWise",
                    "visible_gradient_fraction": 0.0,
                },
                "legacy_sheet": f"Attack_{self.name.upper()}",
                "legacy_payload": {
                    "round": task["round"], "client": task["client"],
                    "sample_id": task["sample_id"], "attack_time": 0.0,
                    "num_restarts": int(config[f"{self.name}_num_restarts"]),
                    "visible_gradient_fraction": 0.0,
                    "attack_scope": "single-sample gradient diagnostic",
                },
            }
        selections = artifacts.require("evaluation_oracle", "selections")
        position = int(task["selection_position"])
        sample_x = selections[f"{self.name}_inputs"][position:position + 1]
        sample_y = selections[f"{self.name}_labels"][position:position + 1]
        actual_state = artifacts.require("evaluation_oracle", "actual_post_local_state")
        sample_hash = hash_arrays({"input": sample_x, "label": sample_y})
        cache_key = hash_json({
            "checkpoint_hash": checkpoint.checkpoint_hash,
            "round": task["round"], "client": task["client"],
            "actual_state_hash": checkpoint.load_round_manifest(task["round"])["clients"][str(task["client"])]["actual_state_sha256"],
            "sample_hash": sample_hash, "loss": "mean_cross_entropy",
        })
        target_gradients, cache_info = _target_cache(
            trainer=trainer, actual_state=actual_state, sample_x=sample_x,
            sample_y=sample_y, cache_root=cache_root, cache_key=cache_key,
            enabled=bool(config["cache_target_gradients"]),
        )
        common = dict(
            trainer=trainer, client_model=actual_state,
            attacker_model=artifacts.require("server_visible", "attacker_visible_state"),
            sample_x=sample_x, sample_y=sample_y, maskBoolNot=masks,
            num_restarts=1, attack_seed=int(task["sample_seed_base"]),
            restart_offset=int(task["restart_id"]), sample_id=int(task["sample_id"]),
            precomputed_target_gradients=target_gradients, save_dir=str(output_dir),
        )
        if self.name == "dlg":
            metrics = perform_DLG_attack(
                **common,
                num_iterations=int(config["dlg_iterations"]),
                learning_rate=float(config["dlg_learning_rate"]),
                optimizer_name=config["dlg_optimizer"], objective=config["dlg_objective"],
                tv_weight=float(config["dlg_tv_weight"]),
                early_stopping_patience=int(config["dlg_early_stopping_patience"]),
                early_stopping_delta=float(config["dlg_early_stopping_delta"]),
                success_ssim_threshold=float(config["dlg_success_ssim"]),
                compute_lpips=bool(config["dlg_compute_lpips"]),
                known_label=bool(config["dlg_known_label"]),
            )
        else:
            metrics = perform_IG_attack(
                **common,
                num_iterations=int(config["ig_iterations"]),
                learning_rate=float(config["ig_learning_rate"]),
                tv_weight=float(config["ig_tv_weight"]),
                early_stopping_patience=int(config["ig_early_stopping_patience"]),
                early_stopping_delta=float(config["ig_early_stopping_delta"]),
                success_ssim_threshold=float(config["ig_success_ssim"]),
                compute_lpips=bool(config["ig_compute_lpips"]),
                known_label=bool(config["ig_known_label"]),
            )
        info = metrics.additional_info
        payload = {
            "round": task["round"], "client": task["client"],
            "sample_id": info.get("sample_id"), "true_label": info.get("true_label"),
            "inferred_label": info.get("inferred_label"),
            "label_inference_available": info.get("label_inference_available"),
            "label_inference_method": info.get("label_inference_method"),
            "known_label": info.get("known_label"),
            "idlg_inferred_label": info.get("idlg_inferred_label"),
            "idlg_label_inference_success": metrics.idlg_label_inference_success,
            "label_accuracy": metrics.label_accuracy, "success_rate": metrics.success_rate,
            "mse": metrics.mse, "psnr": metrics.psnr, "ssim": metrics.ssim,
            "cosine_sim": metrics.cosine_sim, "lpips": metrics.lpips,
            "attack_time": metrics.attack_time, "best_loss": info.get("best_loss"),
            "best_gradient_loss": info.get("best_gradient_loss"),
            "best_iteration": info.get("best_iteration"),
            "best_restart": info.get("best_restart"),
            "best_restart_seed": info.get("best_restart_seed"),
            "final_loss": info.get("final_loss"), "iterations_run": info.get("iterations_run"),
            "num_restarts": int(config[f"{self.name}_num_restarts"]),
            "visible_gradient_fraction": info.get("visible_gradient_fraction"),
            "optimizer": info.get("optimizer"), "objective": info.get("objective"),
            "learning_rate": info.get("learning_rate"), "tv_weight": info.get("tv_weight"),
            "success_ssim_threshold": info.get("success_ssim_threshold"),
            "attack_scope": info.get("attack_scope"),
        }
        if self.name == "ig":
            payload.update({
                "initialization": info.get("initialization"),
                "lr_decay_gamma": info.get("lr_decay_gamma"),
                "lr_decay_milestones": json.dumps(info.get("lr_decay_milestones"), separators=(",", ":")),
            })
        return {
            "scientific_status": "available", "metrics": metrics.to_dict(),
            "legacy_sheet": f"Attack_{self.name.upper()}", "legacy_payload": payload,
            "selection": {
                "sample_id": int(task["sample_id"]), "selection_position": position,
                "restart_id": int(task["restart_id"]), "restart_seed": int(task["seed"]),
                "stored_default_restart_seeds": selections[f"{self.name}_restart_seeds"][position].tolist(),
            },
            "target_gradient_cache": cache_info,
        }


ATTACK_REGISTRY: Dict[str, AttackAdapter] = {
    "mia": MIAAdapter(),
    "ilrg": ILRGAdapter(),
    "dlg": GradientAdapter("dlg"),
    "ig": GradientAdapter("ig"),
}


def plan_attack_run(
    checkpoint_path: Path,
    *,
    overrides: Mapping[str, Any],
    filters: TaskFilter,
    output_root: Optional[Path] = None,
) -> tuple[Path, Dict[str, Any]]:
    checkpoint = AttackCheckpoint(checkpoint_path)
    config = _resolved_defaults(checkpoint, overrides)
    config_hash = hash_json(config)
    filter_payload = _jsonable(filters.__dict__)
    selection_hash = hash_json(filter_payload)
    attack_run_id = f"attack-{config_hash[:16]}-{selection_hash[:8]}"
    if output_root is None:
        project_root = Path(__file__).resolve().parents[1]
        output_root = (
            project_root / "Results" / "OfflineAttacks" /
            checkpoint.manifest["model"]["name"] /
            checkpoint.manifest["resolved_configuration"].get("group", "default") /
            checkpoint.run_id
        )
    run_root = Path(output_root) / attack_run_id
    tasks = []
    for attack_name in config["attacks"]:
        tasks.extend(ATTACK_REGISTRY[attack_name].plan_tasks(checkpoint, config, filters))
    tasks.sort(key=lambda item: (
        item["round"], item["client"], item["attack"],
        str(item.get("signal", "")), str(item.get("sample_or_batch_id", "")),
        -1 if item.get("restart_id") is None else item["restart_id"],
    ))
    manifest = {
        "attack_run_id": attack_run_id,
        "created_at_unix": datetime.now(timezone.utc).timestamp(),
        "checkpoint_path": str(checkpoint.run_dir),
        "checkpoint_run_id": checkpoint.run_id,
        "checkpoint_schema_version": SCHEMA_VERSION,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
        "attack_config_hash": config_hash,
        "task_selection_hash": selection_hash,
        "resolved_attack_configuration": config,
        "filters": filter_payload,
        "tasks": tasks,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    existing = run_root / "attack_manifest.json"
    if existing.is_file():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        for key in (
                "checkpoint_hash", "attack_config_hash", "task_selection_hash",
                "attack_run_id"):
            if previous.get(key) != manifest.get(key):
                raise OfflineAttackError(f"Existing attack run has mismatched {key}")
    atomic_write_json(existing, manifest)
    return run_root, manifest


_WORKER_CHECKPOINTS: Dict[str, AttackCheckpoint] = {}


def _worker_checkpoint(path: str) -> AttackCheckpoint:
    checkpoint = _WORKER_CHECKPOINTS.get(path)
    if checkpoint is None:
        checkpoint = AttackCheckpoint(Path(path))
        _WORKER_CHECKPOINTS[path] = checkpoint
    return checkpoint


def _task_dir(run_root: Path, task: Mapping[str, Any]) -> Path:
    return run_root / "tasks" / task["attack"] / task["task_id"]


def _valid_success(task: Mapping[str, Any], run_root: Path) -> bool:
    path = _task_dir(run_root, task) / "result.json"
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        expected_hash = result.pop("result_hash")
    except Exception:
        return False
    return (
        result.get("status") == "succeeded"
        and result.get("task_id") == task["task_id"]
        and result.get("checkpoint_hash") == task["checkpoint_hash"]
        and result.get("attack_config_hash") == task["attack_config_hash"]
        and result.get("implementation_version") == task["implementation_version"]
        and hash_json(result) == expected_hash
    )


def _execute_task(
    task: Mapping[str, Any], run_root_text: str, checkpoint_path: str,
    config: Mapping[str, Any], attack_run_id: str, device: str,
) -> Dict[str, Any]:
    run_root = Path(run_root_text)
    output_dir = _task_dir(run_root, task)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_result = output_dir / "result.json"
    if previous_result.is_file():
        os.replace(previous_result, output_dir / "previous_result.json")
    started = time.perf_counter()
    atomic_write_json(output_dir / "status.json", {
        "status": "running", "task_id": task["task_id"], "pid": os.getpid(),
        "device": device, "started_at_unix": started,
    })
    log_path = output_dir / "task.log"
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle, \
                contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            checkpoint = _worker_checkpoint(checkpoint_path)
            if checkpoint.checkpoint_hash != task["checkpoint_hash"]:
                raise CheckpointValidationError("Task checkpoint hash no longer matches")
            np.random.seed(int(task.get("seed", config["attack_seed"])) % (2 ** 32))
            random.seed(int(task.get("seed", config["attack_seed"])))
            torch.manual_seed(int(task.get("seed", config["attack_seed"])))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(task.get("seed", config["attack_seed"])))
            artifacts = checkpoint.load_artifacts(task["round"], task["client"])
            adapter = ATTACK_REGISTRY[task["attack"]]
            requirements = adapter.requirements()
            for namespace, keys in requirements.items():
                for key in keys:
                    artifacts.require(namespace, key)
            artifacts = ArtifactNamespaces(
                server_visible={
                    key: artifacts.require("server_visible", key)
                    for key in requirements.get("server_visible", ())
                },
                evaluation_oracle={
                    key: artifacts.require("evaluation_oracle", key)
                    for key in requirements.get("evaluation_oracle", ())
                },
            )
            trainer = _trainer_from_checkpoint(checkpoint, device)
            payload = adapter.run_task(
                task=task, config=config, checkpoint=checkpoint,
                artifacts=artifacts, trainer=trainer, output_dir=output_dir,
                cache_root=run_root / "target_gradient_cache",
            )
            del trainer, artifacts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = {
            "status": "succeeded", "task_id": task["task_id"],
            "attack_run_id": attack_run_id, "checkpoint_run_id": task["checkpoint_run_id"],
            "checkpoint_schema_version": SCHEMA_VERSION,
            "checkpoint_hash": task["checkpoint_hash"],
            "attack_config_hash": task["attack_config_hash"],
            "implementation_version": task["implementation_version"],
            "protocol_version": task["protocol_version"],
            "attack": task["attack"], "round": task["round"], "client": task["client"],
            "sample_or_batch_id": task.get("sample_or_batch_id"),
            "restart_id": task.get("restart_id"), "signal": task.get("signal"),
            "seed": int(task.get("seed", config["attack_seed"])), "device": device,
            "runtime_seconds": time.perf_counter() - started,
            "resolved_attack_hyperparameters": config,
            **payload,
        }
    except BaseException as error:
        failure_traceback = traceback.format_exc()
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(failure_traceback)
        result = {
            "status": "failed", "task_id": task["task_id"],
            "attack_run_id": attack_run_id, "checkpoint_run_id": task["checkpoint_run_id"],
            "checkpoint_schema_version": SCHEMA_VERSION,
            "checkpoint_hash": task["checkpoint_hash"],
            "attack_config_hash": task["attack_config_hash"],
            "implementation_version": task["implementation_version"],
            "protocol_version": task["protocol_version"],
            "attack": task["attack"], "round": task["round"], "client": task["client"],
            "sample_or_batch_id": task.get("sample_or_batch_id"),
            "restart_id": task.get("restart_id"), "signal": task.get("signal"),
            "seed": int(task.get("seed", config["attack_seed"])), "device": device,
            "runtime_seconds": time.perf_counter() - started,
            "resolved_attack_hyperparameters": config,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "traceback": failure_traceback,
        }
    result["result_hash"] = hash_json(result)
    atomic_write_json(output_dir / "result.json", result)
    atomic_write_json(output_dir / "status.json", {
        "status": result["status"], "task_id": task["task_id"],
        "finished_at_unix": datetime.now(timezone.utc).timestamp(), "result_hash": result["result_hash"],
    })
    return {"task_id": task["task_id"], "status": result["status"]}


def execute_attack_run(
    run_root: Path, manifest: Mapping[str, Any], *, devices: Sequence[str],
    workers_per_device: int = 1, resume: bool = False, force: bool = False,
) -> Dict[str, int]:
    if not devices:
        raise ValueError("At least one device is required")
    if workers_per_device <= 0:
        raise ValueError("workers_per_device must be positive")
    tasks = []
    skipped = 0
    for task in manifest["tasks"]:
        succeeded = _valid_success(task, run_root)
        if succeeded and not force:
            if resume:
                skipped += 1
                continue
            raise OfflineAttackError(
                f"Task {task['task_id']} already succeeded; use --resume or --force"
            )
        tasks.append(task)
    if not tasks:
        return {"planned": len(manifest["tasks"]), "skipped": skipped, "succeeded": 0, "failed": 0}

    assignments = {device: [] for device in devices}
    for index, task in enumerate(tasks):
        assignments[devices[index % len(devices)]].append(task)
    spawn_context = multiprocessing.get_context("spawn")
    executors = []
    futures = {}
    interrupted = False
    try:
        for device, assigned in assignments.items():
            if not assigned:
                continue
            executor = ProcessPoolExecutor(
                max_workers=workers_per_device, mp_context=spawn_context
            )
            executors.append(executor)
            for task in assigned:
                future = executor.submit(
                    _execute_task, task, str(run_root), manifest["checkpoint_path"],
                    manifest["resolved_attack_configuration"], manifest["attack_run_id"],
                    str(device),
                )
                futures[future] = task
        succeeded = failed = 0
        for future in as_completed(futures):
            try:
                outcome = future.result()
                if outcome["status"] == "succeeded":
                    succeeded += 1
                else:
                    failed += 1
            except BaseException:
                failed += 1
        return {
            "planned": len(manifest["tasks"]), "skipped": skipped,
            "succeeded": succeeded, "failed": failed,
        }
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        for executor in executors:
            for process in list(getattr(executor, "_processes", {}).values()):
                if process.is_alive():
                    process.terminate()
        raise
    finally:
        for executor in executors:
            executor.shutdown(wait=not interrupted, cancel_futures=True)


def _load_task_results(run_root: Path, manifest: Mapping[str, Any]) -> tuple[list[Dict], list[Dict]]:
    succeeded = []
    missing_or_failed = []
    for task in manifest["tasks"]:
        path = _task_dir(run_root, task) / "result.json"
        if _valid_success(task, run_root):
            succeeded.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            status = "incomplete"
            if path.is_file():
                try:
                    status = json.loads(path.read_text(encoding="utf-8")).get("status", status)
                except Exception:
                    status = "corrupt"
            missing_or_failed.append({"task_id": task["task_id"], "status": status})
    return succeeded, missing_or_failed


def _mean_std_payload(records: Sequence[Mapping], metric_names: Sequence[str]) -> Dict[str, Any]:
    payload = {}
    for name in metric_names:
        mean_value, std_value = dlg_mean_std([record.get(name) for record in records])
        payload[f"{name}_mean"] = mean_value
        payload[f"{name}_std"] = std_value
    return payload


def _copy_best_reconstruction(run_root: Path, result: Mapping[str, Any]) -> Dict[str, str]:
    attack = result["attack"]
    source_dir = _task_dir(run_root, result)
    target_dir = (
        run_root / "reduced" / attack / f"round_{result['round']:04d}" /
        f"client_{result['client']:03d}" / f"sample_{int(result['sample_or_batch_id']):06d}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for suffix in ("reconstruction_comparison.png", "metrics.json"):
        source = source_dir / f"{attack}_{suffix}"
        if source.is_file():
            target = target_dir / source.name
            shutil.copy2(source, target)
            copied[suffix] = str(target.relative_to(run_root))
    return copied


def _write_combined_restart_curve(
    run_root: Path, results: Sequence[Mapping[str, Any]], best: Mapping[str, Any]
) -> Optional[str]:
    attack = best["attack"]
    target_dir = (
        run_root / "reduced" / attack / f"round_{best['round']:04d}" /
        f"client_{best['client']:03d}" / f"sample_{int(best['sample_or_batch_id']):06d}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    plotted = False
    plt.figure(figsize=(8, 5))
    for result in sorted(results, key=lambda item: int(item["restart_id"])):
        path = _task_dir(run_root, result) / f"{attack}_restart_artifacts.npz"
        if not path.is_file():
            continue
        arrays = load_npz(path)
        history = arrays["history"]
        plt.plot(history, lw=1.5, label=f"Restart {result['restart_id'] + 1} (seed {result['seed']})")
        plotted = True
    if not plotted:
        plt.close()
        return None
    if attack == "dlg":
        plt.yscale("log")
        plt.ylabel("Total Reconstruction Objective")
    else:
        plt.ylabel("IG Reconstruction Objective")
    plt.xlabel("Iteration")
    plt.title(f"{attack.upper()} Element-Wise Optimization Trajectories")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    path = target_dir / f"{attack}_loss_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return str(path.relative_to(run_root))


def reduce_attack_run(run_root: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    task_results, incomplete = _load_task_results(run_root, manifest)
    task_results.sort(key=lambda result: result["task_id"])
    sheet_records: Dict[str, list[Dict[str, Any]]] = {}
    reduced_records = []

    mia_groups: Dict[tuple, list[Mapping]] = {}
    ilrg_results = []
    gradient_groups: Dict[tuple, list[Mapping]] = {}
    for result in task_results:
        if result["attack"] == "mia":
            mia_groups.setdefault((result["round"], result["client"]), []).append(result)
        elif result["attack"] == "ilrg":
            ilrg_results.append(result)
        else:
            key = (result["attack"], result["round"], result["client"], result["sample_or_batch_id"])
            gradient_groups.setdefault(key, []).append(result)

    for key in sorted(mia_groups):
        results = sorted(mia_groups[key], key=lambda item: item["task_id"])
        payload = {"round": key[0], "client": key[1]}
        for result in results:
            payload.update(result["legacy_payload"])
        sheet_records.setdefault("Attack_MIA", []).append(payload)
        reduced_records.append({
            "record_type": "client", "attack": "mia", "round": key[0], "client": key[1],
            "task_ids": [item["task_id"] for item in results], "metrics": payload,
        })

    for result in sorted(ilrg_results, key=lambda item: (item["round"], item["client"], item["sample_or_batch_id"])):
        sheet_records.setdefault("Attack_iLRG", []).append(result["legacy_payload"])
        reduced_records.append({
            "record_type": "batch", "attack": "ilrg", "round": result["round"],
            "client": result["client"], "batch_id": result["sample_or_batch_id"],
            "task_id": result["task_id"], "metrics": result["legacy_payload"],
        })

    best_gradient_records = []
    for key in sorted(gradient_groups):
        results = sorted(gradient_groups[key], key=lambda item: int(item["restart_id"]))
        available = [item for item in results if item.get("scientific_status") == "available"]
        if available:
            best = min(available, key=lambda item: (float(item["metrics"]["best_loss"]), int(item["restart_id"])))
        else:
            best = results[0]
        payload = dict(best["legacy_payload"])
        payload["attack_time"] = float(sum(item.get("metrics", {}).get("time", 0.0) or 0.0 for item in results))
        payload["num_restarts"] = int(manifest["resolved_attack_configuration"][f"{key[0]}_num_restarts"])
        artifacts = _copy_best_reconstruction(run_root, best) if available else {}
        curve = _write_combined_restart_curve(run_root, results, best) if available else None
        if curve:
            artifacts["loss_curve"] = curve
        sheet = f"Attack_{key[0].upper()}"
        sheet_records.setdefault(sheet, []).append(payload)
        reduced = {
            "record_type": "sample", "attack": key[0], "round": key[1], "client": key[2],
            "sample_id": key[3], "best_task_id": best["task_id"],
            "restart_task_ids": [item["task_id"] for item in results],
            "scientific_status": best.get("scientific_status"),
            "unavailable_reason": best.get("unavailable_reason"),
            "metrics": payload, "artifacts": artifacts,
        }
        reduced_records.append(reduced)
        best_gradient_records.append(reduced)

    ilrg_summary_rows = []
    ilrg_grouped: Dict[tuple, list[Mapping]] = {}
    for result in ilrg_results:
        ilrg_grouped.setdefault((result["round"], result["client"]), []).append(result["legacy_payload"])
    ilrg_metrics = (
        "label_existence_accuracy", "label_number_accuracy", "instance_recall",
        "count_mae", "normalized_count_l1", "count_cosine_similarity",
        "exact_count_vector", "label_precision", "label_recall", "label_f1",
        "attack_time", "visible_gradient_fraction", "final_weight_visible_fraction",
        "final_bias_visible_fraction", "usable_bias_equations",
        "recovered_embedding_classes", "mean_embedding_visible_fraction",
    )
    for key in sorted(ilrg_grouped):
        records = ilrg_grouped[key]
        row = {
            "round": key[0], "client": key[1], "num_batches": len(records),
            "num_available": sum(int(bool(record["attack_available"])) for record in records),
            "availability_rate": float(np.mean([bool(record["attack_available"]) for record in records])),
            "total_attack_time": float(sum(record.get("attack_time", 0.0) for record in records)),
            **_mean_std_payload(records, ilrg_metrics),
        }
        ilrg_summary_rows.append(row)
    if ilrg_summary_rows:
        sheet_records["Attack_iLRG_Summary"] = ilrg_summary_rows

    gradient_metric_names = (
        "success_rate", "label_accuracy", "idlg_label_inference_success", "mse",
        "psnr", "ssim", "cosine_sim", "lpips", "attack_time",
        "visible_gradient_fraction",
    )
    for attack in ("dlg", "ig"):
        grouped: Dict[tuple, list[Mapping]] = {}
        for record in best_gradient_records:
            if record["attack"] == attack:
                grouped.setdefault((record["round"], record["client"]), []).append(record["metrics"])
        summary_rows = []
        for key in sorted(grouped):
            records = grouped[key]
            row = {
                "round": key[0], "client": key[1], "num_samples": len(records),
                "total_attack_time": float(sum(record.get("attack_time", 0.0) or 0.0 for record in records)),
                **_mean_std_payload(records, gradient_metric_names),
            }
            summary_rows.append(row)
        if summary_rows:
            sheet_records[f"Attack_{attack.upper()}_Summary"] = summary_rows

    provenance = {
        "checkpoint_run_id": manifest["checkpoint_run_id"],
        "checkpoint_schema_version": manifest["checkpoint_schema_version"],
        "checkpoint_hash": manifest["checkpoint_hash"],
        "attack_run_id": manifest["attack_run_id"],
        "attack_protocol_version": ATTACK_PROTOCOL_VERSION,
        "attack_config_hash": manifest["attack_config_hash"],
        "task_selection_hash": manifest["task_selection_hash"],
    }
    jsonl_lines = []
    for record in reduced_records:
        jsonl_lines.append(json.dumps({**provenance, **record}, sort_keys=True, separators=(",", ":")))
    atomic_write_text(run_root / "records.jsonl", "\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""))

    numeric_by_round: Dict[int, Dict[str, list[float]]] = {}
    numeric_experiment: Dict[str, list[float]] = {}
    for record in reduced_records:
        for name, value in record.get("metrics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                key = f"{record['attack']}.{name}"
                numeric_by_round.setdefault(int(record["round"]), {}).setdefault(key, []).append(float(value))
                numeric_experiment.setdefault(key, []).append(float(value))
    round_summaries = {
        str(round_index): {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": len(values)}
            for name, values in sorted(metrics.items())
        }
        for round_index, metrics in sorted(numeric_by_round.items())
    }
    experiment_summary = {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": len(values)}
        for name, values in sorted(numeric_experiment.items())
    }
    report = {
        **provenance, "generated_at_unix": datetime.now(timezone.utc).timestamp(),
        "task_counts": {"succeeded": len(task_results), "incomplete_or_failed": len(incomplete)},
        "incomplete_or_failed_tasks": incomplete,
        "client_summaries": {
            "ilrg": ilrg_summary_rows,
            "dlg": sheet_records.get("Attack_DLG_Summary", []),
            "ig": sheet_records.get("Attack_IG_Summary", []),
        },
        "round_summaries": round_summaries,
        "experiment_summary": experiment_summary,
        "resolved_attack_configuration": manifest["resolved_attack_configuration"],
    }
    atomic_write_json(run_root / "report.json", report)
    excelHelper.write_offline_attack_workbook(
        sheet_records, run_root / "offline_attacks.xlsx", provenance=provenance
    )
    return report
