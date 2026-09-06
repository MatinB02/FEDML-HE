"""Portable, validated checkpoints for offline privacy attacks.

The schema deliberately stores arrays in non-executable ``.npz`` containers
and keeps attacker-visible artifacts separate from evaluation-only oracle
artifacts.  It is baseline-neutral enough to be reused by the companion FL
implementations while recording FEDML-HE-specific mask metadata.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from Codes.attack_selection import plan_client_selections
from Codes.functions_mainAlg import flattener, reconstructor


SCHEMA_VERSION = "attack-checkpoint-v1"
PRODUCER_VERSION = "FEDML-HE-offline-attacks-1.0"
COMPLETE_SENTINEL = "CHECKPOINT_COMPLETE"
ROUND_COMPLETE_SENTINEL = "ROUND_COMPLETE"


class CheckpointValidationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in arrays.items():
        array = np.asarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(array.shape)))
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def hash_state(state: Sequence[np.ndarray], names: Sequence[str]) -> str:
    if len(state) != len(names):
        raise ValueError("State array/name count mismatch")
    return hash_arrays({f"{index:05d}:{name}": value for index, (name, value) in enumerate(zip(names, state))})


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return getattr(value, "name", value.__name__)
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an uncompressed, allow_pickle=False-compatible NPZ atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **{str(k): np.asarray(v) for k, v in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}
    except Exception as error:
        raise CheckpointValidationError(f"Cannot load safe array file {path}: {error}") from error


def state_to_arrays(state: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    return {f"state_{index:05d}": np.asarray(value) for index, value in enumerate(state)}


def arrays_to_state(arrays: Mapping[str, np.ndarray], count: int) -> list[np.ndarray]:
    expected = [f"state_{index:05d}" for index in range(count)]
    missing = [key for key in expected if key not in arrays]
    if missing or len(arrays) != count:
        raise CheckpointValidationError(
            f"State archive keys mismatch; missing={missing}, expected_count={count}, actual={len(arrays)}"
        )
    return [np.asarray(arrays[key]).copy() for key in expected]


def state_schema(model: torch.nn.Module) -> Tuple[list[Dict], list[Dict]]:
    parameters = dict(model.named_parameters())
    entries = []
    trainable = []
    for index, (name, tensor) in enumerate(model.state_dict().items()):
        classification = "parameter" if name in parameters else "buffer"
        item = {
            "index": index,
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.detach().cpu().numpy().dtype),
            "numel": int(tensor.numel()),
            "classification": classification,
            "trainable": bool(name in parameters and parameters[name].requires_grad),
        }
        entries.append(item)
        if item["trainable"]:
            trainable.append({
                "trainable_index": len(trainable),
                "state_index": index,
                "name": name,
                "shape": item["shape"],
                "dtype": item["dtype"],
                "numel": item["numel"],
            })
    return entries, trainable


def normalize_state(state: Sequence[np.ndarray], schema: Sequence[Mapping]) -> list[np.ndarray]:
    if len(state) != len(schema):
        raise ValueError(f"Expected {len(schema)} state arrays, received {len(state)}")
    normalized = []
    for value, entry in zip(state, schema):
        array = np.asarray(value)
        expected_shape = tuple(entry["shape"])
        if array.shape != expected_shape:
            raise ValueError(
                f"State shape mismatch for {entry['name']}: {array.shape} != {expected_shape}"
            )
        normalized.append(np.asarray(array, dtype=np.dtype(entry["dtype"])).copy())
    return normalized


def validate_state(
    state: Sequence[np.ndarray],
    schema: Sequence[Mapping],
    *,
    expected_hash: Optional[str] = None,
) -> str:
    if len(state) != len(schema):
        raise CheckpointValidationError(
            f"Expected {len(schema)} state arrays, received {len(state)}"
        )
    names = []
    for value, entry in zip(state, schema):
        array = np.asarray(value)
        if tuple(array.shape) != tuple(entry["shape"]):
            raise CheckpointValidationError(
                f"Shape mismatch for {entry['name']}: {array.shape} != {tuple(entry['shape'])}"
            )
        if str(array.dtype) != entry["dtype"]:
            raise CheckpointValidationError(
                f"Dtype mismatch for {entry['name']}: {array.dtype} != {entry['dtype']}"
            )
        names.append(entry["name"])
    actual_hash = hash_state(state, names)
    if expected_hash is not None and actual_hash != expected_hash:
        raise CheckpointValidationError(
            f"State hash mismatch: {actual_hash} != {expected_hash}"
        )
    return actual_hash


def construct_attacker_visible_state(
    client_state: Sequence[np.ndarray],
    exposed_flat_before_update: np.ndarray,
    encrypted_masks: Sequence[np.ndarray],
    schema: Sequence[Mapping],
) -> list[np.ndarray]:
    """Reproduce the former online smart-adversary model construction."""
    structure = [
        {"shape": tuple(entry["shape"]), "size": int(entry["numel"])}
        for entry in schema
    ]
    client_flat = flattener(client_state).copy()
    encrypted_flat = np.asarray(flattener(encrypted_masks), dtype=bool)
    exposed_flat = np.asarray(exposed_flat_before_update)
    if not (client_flat.shape == encrypted_flat.shape == exposed_flat.shape):
        raise ValueError(
            "Attacker construction shape mismatch: "
            f"client={client_flat.shape}, mask={encrypted_flat.shape}, exposed={exposed_flat.shape}"
        )
    client_flat[encrypted_flat] = exposed_flat[encrypted_flat]
    reconstructed = reconstructor(client_flat, structure)
    return normalize_state(reconstructed, schema)


def _state_delta(
    client_state: Sequence[np.ndarray], entering_state: Sequence[np.ndarray]
) -> list[np.ndarray]:
    return [
        np.asarray(client, dtype=np.float64) - np.asarray(entering, dtype=np.float64)
        for client, entering in zip(client_state, entering_state)
    ]


def _visible_payload_hash(
    client_state: Sequence[np.ndarray], plaintext_masks: Sequence[np.ndarray], names: Sequence[str]
) -> str:
    arrays = {}
    for index, (name, value, mask) in enumerate(zip(names, client_state, plaintext_masks)):
        arrays[f"{index:05d}:{name}"] = np.asarray(value).reshape(-1)[
            np.asarray(mask, dtype=bool).reshape(-1)
        ]
    return hash_arrays(arrays)


def _git_metadata(project_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args], cwd=project_root, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            return result.stdout
        except Exception:
            return None

    commit = (run("rev-parse", "HEAD") or "").strip() or None
    status = run("status", "--porcelain=v1")
    dirty = bool(status and status.strip())
    source_digest = hashlib.sha256()
    source_digest.update((commit or "no-git-commit").encode("utf-8"))
    for path in sorted(project_root.rglob("*.py")):
        if any(part in {"__pycache__", ".git", "Results", "Temp"} for part in path.parts):
            continue
        try:
            source_digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
            source_digest.update(bytes.fromhex(hash_file(path)))
        except OSError:
            continue
    return {
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "git_status_porcelain": [] if status is None else status.splitlines(),
        "code_version_identifier": source_digest.hexdigest(),
    }


def _package_versions() -> Dict[str, Optional[str]]:
    packages = [
        "numpy", "torch", "torchvision", "scikit-learn", "scikit-image",
        "matplotlib", "openpyxl", "tenseal", "lpips",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_metadata() -> Dict[str, Any]:
    cudnn_version = torch.backends.cudnn.version()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn": None if cudnn_version is None else str(cudnn_version),
        "numpy": np.__version__,
        "packages": _package_versions(),
    }


def resolved_config(cfg: Any) -> Dict[str, Any]:
    values = {}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        try:
            value = getattr(cfg, name)
        except Exception:
            continue
        if callable(value):
            continue
        if value is not None:
            values[name] = _jsonable(value)
    if hasattr(cfg, "resolved_args"):
        values["cli"] = _jsonable(cfg.resolved_args)
    return values


def _model_arguments(cfg: Any) -> Dict[str, Any]:
    candidates = {
        "num_classes": int(cfg.classNum),
        "in_channels": int(cfg.inputShape[-1]),
        "input_height": int(cfg.inputShape[0]),
        "input_width": int(cfg.inputShape[1]),
    }
    signature = inspect.signature(cfg.model)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return {
        key: value
        for key, value in candidates.items()
        if accepts_kwargs or key in signature.parameters
    }


def _attack_defaults(cfg: Any) -> Dict[str, Any]:
    names = (
        "attack_seed", "mia_sample_size", "mia_bootstrap_samples",
        "ilrg_batch_size", "ilrg_num_batches", "ilrg_alpha", "ilrg_mask_mode",
        "dlg_num_samples", "dlg_num_restarts", "dlg_iterations",
        "dlg_learning_rate", "dlg_optimizer", "dlg_objective", "dlg_tv_weight",
        "dlg_early_stopping_patience", "dlg_success_ssim", "dlg_compute_lpips",
        "dlg_known_label", "ig_num_samples", "ig_num_restarts", "ig_iterations",
        "ig_learning_rate", "ig_tv_weight", "ig_early_stopping_patience",
        "ig_success_ssim", "ig_compute_lpips", "ig_known_label",
    )
    values = {name: _jsonable(getattr(cfg, name)) for name in names}
    values["mia_batch_size"] = int(cfg.local_batch_size)
    values["mia_signals"] = ["loss", "entropy", "modified_entropy"]
    values["dlg_early_stopping_delta"] = 1e-7
    values["ig_early_stopping_delta"] = 1e-7
    values["ilrg_epsilon"] = 1e-12
    return values


def _seed_metadata(cfg: Any) -> Dict[str, Any]:
    return {
        "experiment_seed": int(cfg.seed),
        "python_random_seed": int(cfg.seed),
        "numpy_seed": int(cfg.seed),
        "torch_cpu_seed": int(cfg.seed),
        "torch_cuda_seed_all": int(cfg.seed),
        "partition_seed": int(cfg.seed),
        "partition_seed_offsets": {"iid": 100, "dirichlet": 200},
        "preprocessing_seed_offsets": {"train": 0, "test": 1000},
        "attack_seed": int(cfg.attack_seed),
    }


@dataclass
class PendingRound:
    round_index: int
    prepared_at: str
    copy_seconds: float
    global_entering: list[np.ndarray]
    exposed_before: np.ndarray
    encrypted_masks: list[np.ndarray]
    plaintext_masks: list[np.ndarray]
    global_mask: Optional[np.ndarray]
    mask_plan_metadata: Dict[str, Any]
    sensitivity_maps: list[list[np.ndarray]]
    client_states: list[list[np.ndarray]]
    attacker_states: list[list[np.ndarray]]
    client_weights: np.ndarray
    client_dataset_sizes: list[int]
    transmission_metadata: Sequence[Mapping[str, Any]]


class AttackCheckpointWriter:
    """Create one immutable attack-checkpoint run during FL training."""

    def __init__(
        self,
        *,
        base_dir: Path,
        project_root: Path,
        cfg: Any,
        model: torch.nn.Module,
        initial_state: Sequence[np.ndarray],
        client_data: Sequence[np.ndarray],
        client_labels: Sequence[np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        dataset_train_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.cfg = cfg
        self.run_dir = (
            Path(base_dir) / cfg.model.name / str(cfg.group) / str(cfg.run_id)
        ).resolve()
        if self.run_dir.exists():
            raise FileExistsError(f"Attack checkpoint run already exists: {self.run_dir}")
        self.schema, self.trainable_schema = state_schema(model)
        self.state_names = [entry["name"] for entry in self.schema]
        initial_state = normalize_state(initial_state, self.schema)
        self._pending: Optional[PendingRound] = None

        parent = self.run_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        temp_dir = parent / f".{self.run_dir.name}.tmp-{uuid.uuid4().hex}"
        temp_dir.mkdir()
        try:
            shared_dir = temp_dir / "shared" / "evaluation_oracle" / "selections"
            shared_dir.mkdir(parents=True)
            selection_entries = {}
            X_test, Y_test = test_data
            for client_id, (X_member, Y_member) in enumerate(zip(client_data, client_labels)):
                metadata, arrays = plan_client_selections(
                    (X_member, Y_member), (X_test, Y_test),
                    client_id=client_id,
                    attack_seed=cfg.attack_seed,
                    mia_sample_size=cfg.mia_sample_size,
                    ilrg_batch_size=cfg.ilrg_batch_size,
                    ilrg_num_batches=cfg.ilrg_num_batches,
                    dlg_num_samples=cfg.dlg_num_samples,
                    dlg_num_restarts=cfg.dlg_num_restarts,
                    ig_num_samples=cfg.ig_num_samples,
                    ig_num_restarts=cfg.ig_num_restarts,
                )
                file_name = f"client_{client_id:03d}.npz"
                metadata_name = f"client_{client_id:03d}.json"
                write_npz(shared_dir / file_name, arrays)
                metadata.update({
                    "array_hash": hash_arrays(arrays),
                    "partition_identity": {
                        "client_id": client_id,
                        "stable_id_scheme": "partition-position-v1",
                        "num_samples": int(len(X_member)),
                        "partition_hash": hash_arrays({
                            "inputs": np.asarray(X_member),
                            "labels": np.asarray(Y_member),
                        }),
                    },
                    "preprocessing": (
                        "Inputs are float32 normalized to [0,1], deterministically shuffled; "
                        "labels are one-hot encoded. Exact selected arrays are stored."
                    ),
                })
                atomic_write_json(shared_dir / metadata_name, metadata)
                selection_entries[str(client_id)] = {
                    "arrays": f"shared/evaluation_oracle/selections/{file_name}",
                    "metadata": f"shared/evaluation_oracle/selections/{metadata_name}",
                    "arrays_sha256": hash_file(shared_dir / file_name),
                    "metadata_sha256": hash_file(shared_dir / metadata_name),
                    "selection_hash": metadata["array_hash"],
                    "partition_hash": metadata["partition_identity"]["partition_hash"],
                }

            initial_path = temp_dir / "shared" / "server_visible" / "initial_state.npz"
            write_npz(initial_path, state_to_arrays(initial_state))
            fingerprint_train = dataset_train_data or (
                np.concatenate(client_data), np.concatenate(client_labels)
            )
            dataset_fingerprint = hash_arrays({
                "train_inputs": np.asarray(fingerprint_train[0]),
                "train_labels": np.asarray(fingerprint_train[1]),
                "test_inputs": np.asarray(X_test),
                "test_labels": np.asarray(Y_test),
            })
            total_parameters = int(sum(entry["numel"] for entry in self.schema))
            git = _git_metadata(self.project_root)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "producer_application": "FEDML-HE",
                "producer_version": PRODUCER_VERSION,
                "created_at": utc_now(),
                "experiment_id": str(cfg.exp_id),
                "run_id": str(cfg.run_id),
                "baseline": "FEDML-HE",
                "training_complete": False,
                "resolved_configuration": resolved_config(cfg),
                "dataset": {
                    "name": cfg.DB_dataset.name,
                    "num_classes": int(cfg.classNum),
                    "channels": int(cfg.inputShape[-1]),
                    "input_dimensions": list(cfg.inputShape),
                    "client_count": int(cfg.num_clients),
                    "samples_per_client": [int(len(values)) for values in client_data],
                    "iid": not bool(cfg.DB_nonIID),
                    "non_iid": bool(cfg.DB_nonIID),
                    "alpha": float(cfg.DB_nonIID_alpha),
                    "partition_seed": int(cfg.seed),
                    "fingerprint_sha256": dataset_fingerprint,
                    "preprocessing": {
                        "inputs": "astype(float32) / 255.0",
                        "shuffle": "NumPy RandomState permutation; train seed, test seed+1000",
                        "labels": "one-hot float32 unless already one-hot",
                        "cache_identity": f"{cfg.DB_dataset.value}_preprocessed_seed{cfg.seed}",
                    },
                },
                "model": {
                    "name": cfg.model.name,
                    "factory": f"{cfg.model.__module__}.{cfg.model.__qualname__}",
                    "construction_arguments": _model_arguments(cfg),
                    "state_dict_schema": self.schema,
                    "trainable_parameter_schema": self.trainable_schema,
                    "total_state_values": total_parameters,
                    "initial_state": {
                        "path": "shared/server_visible/initial_state.npz",
                        "file_sha256": hash_file(initial_path),
                        "state_sha256": hash_state(initial_state, self.state_names),
                    },
                },
                "federated_learning": {
                    "rounds": int(cfg.rounds),
                    "local_epochs": int(cfg.local_epochs),
                    "local_batch_size": int(cfg.local_batch_size),
                    "evaluation_batch_size": int(cfg.eval_batch_size),
                    "mask_gradient_batch_size": int(cfg.mask_gradient_batch_size),
                    "encryption_ratio": float(cfg.encryption_ratio),
                    "mask_consensus": "ordered interleaving, de-duplication, first quota entries",
                    "full_encryption_sentinel": True,
                    "always_encrypt_non_trainable_state": True,
                    "aggregate_BN": bool(cfg.aggregate_BN),
                    "client_optimizer": {
                        "name": "Adam",
                        "learning_rate": float(cfg.model.learning_rate_local),
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "amsgrad": False,
                        "reset_per_client_training_call": True,
                    },
                    "aggregation": "sample-count-weighted FedAvg over aligned encrypted/plaintext coordinates",
                },
                "seeds": _seed_metadata(cfg),
                "attack_defaults": _attack_defaults(cfg),
                "attack_round_policy": {
                    "enabled": bool(cfg.attack),
                    "attack_interval": int(cfg.attack_interval),
                    "condition": "round % attack_interval == 0 or round == rounds - 1",
                    "round_indexing": "zero-based",
                },
                "attacker_knowledge": {
                    "threat_model": "honest-but-curious aggregation server",
                    "server_visible": [
                        "entering global model", "final consensus mask and proposals",
                        "plaintext client transmission coordinates", "aggregation weights",
                        "previously exposed coordinates", "post-aggregation global model",
                    ],
                    "evaluation_oracle": [
                        "actual post-local client model", "selected private examples and labels",
                        "target gradients generated at the actual client model",
                    ],
                    "standardized_protocol": (
                        "DLG and IG invert one selected example. Their target gradient is generated "
                        "at the actual private client model; reconstruction starts at the exact "
                        "attacker-visible model; only plaintext gradient coordinates participate. "
                        "iLRG and MIA retain the repository's existing semantics. These oracle "
                        "artifacts are not claimed to be directly observed by the server."
                    ),
                    "paper_comparison": (
                        "The FEDML-HE paper describes an honest-but-curious server and an exposed "
                        "model composed from newly plaintext and previously exposed coordinates. "
                        "Its published reconstruction example uses a five-step local update; this "
                        "checkpoint intentionally preserves the repository's standardized single-"
                        "example gradient-leakage protocol for cross-baseline parity."
                    ),
                },
                "selection_artifacts": selection_entries,
                "environment": environment_metadata(),
                "source": git,
                "rounds": {},
            }
            manifest["checkpoint_hash"] = hash_json({
                key: value for key, value in manifest.items() if key != "checkpoint_hash"
            })
            atomic_write_json(temp_dir / "manifest.json", manifest)
            os.replace(temp_dir, self.run_dir)
            self.manifest = manifest
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _write_root_manifest(self) -> None:
        self.manifest["checkpoint_hash"] = hash_json({
            key: value for key, value in self.manifest.items() if key != "checkpoint_hash"
        })
        atomic_write_json(self.run_dir / "manifest.json", self.manifest)

    def prepare_round(
        self,
        *,
        round_index: int,
        global_entering: Sequence[np.ndarray],
        exposed_flat_before_update: np.ndarray,
        encrypted_masks: Sequence[np.ndarray],
        plaintext_masks: Sequence[np.ndarray],
        global_mask: Optional[np.ndarray],
        mask_plan: Mapping[str, Any],
        sensitivity_maps: Sequence[Sequence[np.ndarray]],
        client_states: Sequence[Sequence[np.ndarray]],
        client_weights: Sequence[float],
        client_dataset_sizes: Sequence[int],
        transmission_metadata: Sequence[Mapping[str, Any]],
    ) -> float:
        if self._pending is not None:
            raise RuntimeError("A round snapshot is already pending")
        start = time.perf_counter()
        entering = normalize_state(global_entering, self.schema)
        client_copies = [normalize_state(state, self.schema) for state in client_states]
        exposed_before = np.asarray(exposed_flat_before_update, dtype=np.float32).copy()
        encrypted = [np.asarray(mask, dtype=np.uint8).copy() for mask in encrypted_masks]
        plaintext = [np.asarray(mask, dtype=np.uint8).copy() for mask in plaintext_masks]
        for entry, enc, plain in zip(self.schema, encrypted, plaintext):
            if enc.shape != tuple(entry["shape"]) or plain.shape != tuple(entry["shape"]):
                raise ValueError(f"Mask shape mismatch for {entry['name']}")
            if not np.array_equal(enc + plain, np.ones_like(enc)):
                raise ValueError(f"Encrypted/plaintext masks are not complementary for {entry['name']}")
        if len(sensitivity_maps) != len(client_copies):
            raise ValueError(
                "sensitivity_maps must contain one map per selected client"
            )
        sensitivity_copies = []
        for client_index, client_map in enumerate(sensitivity_maps):
            if len(client_map) != len(self.trainable_schema):
                raise ValueError(
                    f"Client {client_index} sensitivity map has {len(client_map)} arrays; "
                    f"expected {len(self.trainable_schema)}"
                )
            copied_map = []
            for value, entry in zip(client_map, self.trainable_schema):
                array = np.asarray(value)
                if array.shape != tuple(entry["shape"]):
                    raise ValueError(
                        f"Sensitivity-map shape mismatch for client {client_index}, "
                        f"parameter {entry['name']}"
                    )
                if not np.issubdtype(array.dtype, np.floating):
                    raise TypeError(
                        f"Sensitivity map for {entry['name']} must be floating point"
                    )
                copied_map.append(array.copy())
            sensitivity_copies.append(copied_map)
        attacker_states = [
            construct_attacker_visible_state(state, exposed_before, encrypted, self.schema)
            for state in client_copies
        ]
        metadata = _jsonable(mask_plan.get("metadata", {}))
        copy_seconds = time.perf_counter() - start
        self._pending = PendingRound(
            round_index=int(round_index), prepared_at=utc_now(), copy_seconds=copy_seconds,
            global_entering=entering, exposed_before=exposed_before,
            encrypted_masks=encrypted, plaintext_masks=plaintext,
            global_mask=None if global_mask is None else np.asarray(global_mask, dtype=np.int64).copy(),
            mask_plan_metadata=metadata,
            sensitivity_maps=sensitivity_copies,
            client_states=client_copies, attacker_states=attacker_states,
            client_weights=np.asarray(client_weights, dtype=np.float64).copy(),
            client_dataset_sizes=[int(value) for value in client_dataset_sizes],
            transmission_metadata=[dict(item) for item in transmission_metadata],
        )
        return copy_seconds

    def finalize_round(
        self,
        *,
        global_after_aggregation: Sequence[np.ndarray],
        exposed_flat_after_update: np.ndarray,
        post_aggregation_metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if self._pending is None:
            raise RuntimeError("No pending round snapshot")
        pending = self._pending
        write_start = time.perf_counter()
        global_after = normalize_state(global_after_aggregation, self.schema)
        exposed_after = np.asarray(exposed_flat_after_update, dtype=np.float32).copy()
        round_name = f"round_{pending.round_index:04d}"
        final_dir = self.run_dir / round_name
        if final_dir.exists():
            raise FileExistsError(f"Round checkpoint already exists: {final_dir}")
        temp_dir = self.run_dir / f".{round_name}.tmp-{uuid.uuid4().hex}"
        temp_dir.mkdir()
        try:
            common_dir = temp_dir / "common_state"
            common_dir.mkdir()
            oracle_dir = temp_dir / "evaluation_oracle"
            oracle_dir.mkdir()
            state_files = {
                "global_entering": ("common_state/global_entering.npz", pending.global_entering),
                "global_after_aggregation": ("common_state/global_after_aggregation.npz", global_after),
            }
            common_states = {}
            for key, (relative, state) in state_files.items():
                path = temp_dir / relative
                write_npz(path, state_to_arrays(state))
                common_states[key] = {
                    "path": relative,
                    "file_sha256": hash_file(path),
                    "state_sha256": hash_state(state, self.state_names),
                }
            exposed_path = common_dir / "exposed_flat.npz"
            write_npz(exposed_path, {
                "before_round_ending_update": pending.exposed_before,
                "after_round_ending_update": exposed_after,
            })
            masks_path = common_dir / "visibility_masks.npz"
            write_npz(masks_path, {
                **{f"encrypted_{index:05d}": mask for index, mask in enumerate(pending.encrypted_masks)},
                **{f"plaintext_{index:05d}": mask for index, mask in enumerate(pending.plaintext_masks)},
            })
            sensitivity_maps_path = oracle_dir / "sensitivity_maps.npz"
            sensitivity_arrays = {
                f"client_{client_index:03d}_parameter_{parameter_index:05d}": value
                for client_index, client_map in enumerate(pending.sensitivity_maps)
                for parameter_index, value in enumerate(client_map)
            }
            write_npz(sensitivity_maps_path, sensitivity_arrays)
            global_mask_path = None
            global_mask_file_hash = None
            if pending.global_mask is not None:
                global_mask_path = "common_state/global_mask_indices.npz"
                write_npz(temp_dir / global_mask_path, {"indices": pending.global_mask})
                global_mask_file_hash = hash_file(temp_dir / global_mask_path)

            files = {}
            for path in common_dir.rglob("*"):
                if path.is_file():
                    files[path.relative_to(temp_dir).as_posix()] = hash_file(path)
            for path in oracle_dir.rglob("*"):
                if path.is_file():
                    files[path.relative_to(temp_dir).as_posix()] = hash_file(path)

            clients = {}
            for client_id, (actual, attacker) in enumerate(zip(pending.client_states, pending.attacker_states)):
                client_dir = temp_dir / f"client_{client_id:03d}"
                oracle_dir = client_dir / "evaluation_oracle"
                oracle_dir.mkdir(parents=True)
                actual_path = oracle_dir / "post_local_state.npz"
                write_npz(actual_path, state_to_arrays(actual))
                actual_hash = hash_state(actual, self.state_names)
                attacker_hash = hash_state(attacker, self.state_names)
                reconstructed = construct_attacker_visible_state(
                    actual, pending.exposed_before, pending.encrypted_masks, self.schema
                )
                if hash_state(reconstructed, self.state_names) != attacker_hash:
                    raise RuntimeError("Attacker state failed immediate reconstruction validation")
                delta = _state_delta(actual, pending.global_entering)
                client_manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "client_id": client_id,
                    "client_aggregation_weight": float(pending.client_weights[client_id]),
                    "client_dataset_size": pending.client_dataset_sizes[client_id],
                    "client_dataset_partition_identity": self.manifest["selection_artifacts"][str(client_id)]["partition_hash"],
                    "server_visible": {
                        "attacker_visible_state": {
                            "construction": (
                                "post_local plaintext coordinates plus exposed_flat_before_round_ending_update "
                                "on encrypted coordinates"
                            ),
                            "state_sha256": attacker_hash,
                            "stored": False,
                            "reconstructed_from": [
                                "evaluation_oracle/post_local_state.npz",
                                "../common_state/exposed_flat.npz",
                                "../common_state/visibility_masks.npz",
                            ],
                        },
                        "plaintext_payload": {
                            "representation": "derived from post-local state and plaintext visibility masks",
                            "payload_sha256": _visible_payload_hash(
                                actual, pending.plaintext_masks, self.state_names
                            ),
                        },
                        "protected_coordinates": {
                            "representation": "state_dict-level encrypted masks in common state",
                            "available_to_attacker": False,
                        },
                        "transmission_metadata": pending.transmission_metadata[client_id],
                    },
                    "evaluation_oracle": {
                        "post_local_state": {
                            "path": "evaluation_oracle/post_local_state.npz",
                            "file_sha256": hash_file(actual_path),
                            "state_sha256": actual_hash,
                        },
                        "model_delta": {
                            "representation": "derived exactly as post_local_state - global_entering_state in float64",
                            "state_sha256": hash_state(delta, self.state_names),
                        },
                        "selection_artifact": self.manifest["selection_artifacts"][str(client_id)],
                        "oracle_usage": (
                            "Private state/examples are supplied only to adapters that declare them; "
                            "DLG/IG/iLRG use them for target gradients and evaluation, and MIA uses frozen labels/examples."
                        ),
                    },
                    "local_training": {
                        "optimizer": self.manifest["federated_learning"]["client_optimizer"],
                        "local_epochs": int(self.cfg.local_epochs),
                        "batch_size": int(self.cfg.local_batch_size),
                        "batches_per_epoch": int(math.ceil(pending.client_dataset_sizes[client_id] / self.cfg.local_batch_size)),
                        "optimizer_steps": int(self.cfg.local_epochs * math.ceil(pending.client_dataset_sizes[client_id] / self.cfg.local_batch_size)),
                    },
                    "mask_metadata": {
                        "consensus_scope": "common_state",
                        "encrypted_values": int(sum(np.count_nonzero(mask) for mask in pending.encrypted_masks)),
                        "plaintext_values": int(sum(np.count_nonzero(mask) for mask in pending.plaintext_masks)),
                    },
                }
                atomic_write_json(client_dir / "client_manifest.json", client_manifest)
                for path in client_dir.rglob("*"):
                    if path.is_file():
                        files[path.relative_to(temp_dir).as_posix()] = hash_file(path)
                clients[str(client_id)] = {
                    "manifest": f"client_{client_id:03d}/client_manifest.json",
                    "manifest_sha256": hash_file(client_dir / "client_manifest.json"),
                    "actual_state_sha256": actual_hash,
                    "attacker_state_sha256": attacker_hash,
                }

            mask_mode = "all" if pending.global_mask is None else (
                "none" if len(pending.global_mask) == 0 else "indices"
            )
            round_manifest = {
                "schema_version": SCHEMA_VERSION,
                "round": pending.round_index,
                "round_indexing": "zero-based",
                "timing": {
                    "observation": (
                        "after all local client training and mask consensus; after logical transmissions; "
                        "before aggregation and before exposed_flat round-ending update"
                    ),
                    "prepared_at": pending.prepared_at,
                    "finalized_at": utc_now(),
                    "capture_copy_seconds": pending.copy_seconds,
                },
                "common_state": common_states,
                "exposed_flat": {
                    "path": "common_state/exposed_flat.npz",
                    "file_sha256": hash_file(exposed_path),
                    "before_sha256": hash_arrays({"exposed": pending.exposed_before}),
                    "after_sha256": hash_arrays({"exposed": exposed_after}),
                    "attacker_construction_uses": "before_round_ending_update",
                },
                "consensus_mask": {
                    "representation": mask_mode,
                    "global_indices_path": global_mask_path,
                    "global_indices_file_sha256": global_mask_file_hash,
                    "state_dict_visibility_masks_path": "common_state/visibility_masks.npz",
                    "state_dict_visibility_masks_file_sha256": hash_file(masks_path),
                    "encrypted_count": int(sum(np.count_nonzero(mask) for mask in pending.encrypted_masks)),
                    "plaintext_count": int(sum(np.count_nonzero(mask) for mask in pending.plaintext_masks)),
                    "layer_state_boundaries": np.cumsum(
                        [0] + [entry["numel"] for entry in self.schema], dtype=np.int64
                    ).tolist(),
                    "trainable_parameter_visibility": [
                        {
                            "name": entry["name"],
                            "state_index": entry["state_index"],
                            "visible_count": int(np.count_nonzero(pending.plaintext_masks[entry["state_index"]])),
                            "total_count": int(entry["numel"]),
                        }
                        for entry in self.trainable_schema
                    ],
                    "mask_plan": pending.mask_plan_metadata,
                    "sensitivity_maps": {
                        "namespace": "evaluation_oracle",
                        "available_to_server": False,
                        "path": "evaluation_oracle/sensitivity_maps.npz",
                        "file_sha256": hash_file(sensitivity_maps_path),
                        "client_count": len(pending.sensitivity_maps),
                        "parameter_order": [
                            entry["name"] for entry in self.trainable_schema
                        ],
                        "aggregation": "sample-count-weighted mean",
                        "aggregation_weights": pending.client_weights.tolist(),
                    },
                },
                "selected_clients": list(range(len(pending.client_states))),
                "aggregation_weights": pending.client_weights.tolist(),
                "client_dataset_sizes": pending.client_dataset_sizes,
                "server_visible_aggregation": _jsonable(post_aggregation_metadata),
                "clients": clients,
                "files": files,
            }
            for relative, expected in round_manifest["files"].items():
                actual_hash = hash_file(temp_dir / relative)
                if actual_hash != expected:
                    raise RuntimeError(f"Immediate file validation failed for {relative}")
            round_manifest["storage_bytes"] = int(sum(
                path.stat().st_size for path in temp_dir.rglob("*") if path.is_file()
            ))
            round_manifest["approx_storage_bytes_per_client"] = int(
                sum(
                    path.stat().st_size
                    for path in temp_dir.rglob("post_local_state.npz")
                ) / max(len(pending.client_states), 1)
            )
            round_manifest["round_hash"] = hash_json({
                key: value for key, value in round_manifest.items() if key != "round_hash"
            })
            atomic_write_json(temp_dir / "round_manifest.json", round_manifest)
            atomic_write_text(temp_dir / ROUND_COMPLETE_SENTINEL, round_manifest["round_hash"] + "\n")
            os.replace(temp_dir, final_dir)
            write_seconds = time.perf_counter() - write_start
            round_manifest["timing"]["write_and_validate_seconds"] = write_seconds
            round_manifest["timing"]["total_checkpoint_seconds"] = pending.copy_seconds + write_seconds
            round_manifest["round_hash"] = hash_json({
                key: value for key, value in round_manifest.items() if key != "round_hash"
            })
            atomic_write_json(final_dir / "round_manifest.json", round_manifest)
            atomic_write_text(final_dir / ROUND_COMPLETE_SENTINEL, round_manifest["round_hash"] + "\n")
            self.manifest["rounds"][str(pending.round_index)] = {
                "path": f"{round_name}/round_manifest.json",
                "round_hash": round_manifest["round_hash"],
                "storage_bytes": round_manifest["storage_bytes"],
                "checkpoint_seconds": round_manifest["timing"]["total_checkpoint_seconds"],
            }
            self._write_root_manifest()
            self._pending = None
            return round_manifest
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def finalize_experiment(self) -> None:
        if self._pending is not None:
            raise RuntimeError("Cannot finalize while a round snapshot is pending")
        self.manifest["training_complete"] = True
        self.manifest["completed_at"] = utc_now()
        self._write_root_manifest()
        atomic_write_text(self.run_dir / COMPLETE_SENTINEL, self.manifest["checkpoint_hash"] + "\n")


@dataclass(frozen=True)
class ArtifactNamespaces:
    server_visible: Mapping[str, Any]
    evaluation_oracle: Mapping[str, Any]

    def require(self, namespace: str, key: str) -> Any:
        if namespace not in {"server_visible", "evaluation_oracle"}:
            raise KeyError(f"Unknown artifact namespace: {namespace}")
        values = getattr(self, namespace)
        if key not in values:
            raise KeyError(f"Missing required artifact {namespace}.{key}")
        return values[key]


class AttackCheckpoint:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise CheckpointValidationError(f"Missing manifest: {manifest_path}")
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise CheckpointValidationError(f"Invalid manifest JSON: {error}") from error
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointValidationError(
                f"Unsupported schema {self.manifest.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
            )
        expected_hash = self.manifest.get("checkpoint_hash")
        actual_hash = hash_json({
            key: value for key, value in self.manifest.items() if key != "checkpoint_hash"
        })
        if expected_hash != actual_hash:
            raise CheckpointValidationError("Root manifest hash mismatch")
        sentinel = self.run_dir / COMPLETE_SENTINEL
        if not sentinel.is_file() or sentinel.read_text(encoding="utf-8").strip() != expected_hash:
            raise CheckpointValidationError("Checkpoint is incomplete or has an invalid completion sentinel")
        if not self.manifest.get("training_complete"):
            raise CheckpointValidationError("Checkpoint training run is not complete")
        self.checkpoint_hash = expected_hash
        self.schema = self.manifest["model"]["state_dict_schema"]
        self.trainable_schema = self.manifest["model"]["trainable_parameter_schema"]
        self.state_names = [entry["name"] for entry in self.schema]
        self._round_cache: Dict[int, Dict[str, Any]] = {}
        self._selection_cache: Dict[int, Tuple[Dict, Dict[str, np.ndarray]]] = {}

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    def available_rounds(self) -> list[int]:
        return sorted(int(value) for value in self.manifest["rounds"])

    def _verify_file(self, path: Path, expected_hash: str) -> None:
        if not path.is_file():
            raise CheckpointValidationError(f"Missing checkpoint file: {path}")
        actual_hash = hash_file(path)
        if actual_hash != expected_hash:
            raise CheckpointValidationError(
                f"File hash mismatch for {path}: {actual_hash} != {expected_hash}"
            )

    def load_round_manifest(self, round_index: int) -> Dict[str, Any]:
        round_index = int(round_index)
        if round_index in self._round_cache:
            return self._round_cache[round_index]
        entry = self.manifest["rounds"].get(str(round_index))
        if entry is None:
            raise CheckpointValidationError(f"Round {round_index} is not available")
        path = self.run_dir / entry["path"]
        try:
            round_manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise CheckpointValidationError(f"Invalid round manifest {path}: {error}") from error
        actual_hash = hash_json({
            key: value for key, value in round_manifest.items() if key != "round_hash"
        })
        if actual_hash != round_manifest.get("round_hash") or actual_hash != entry["round_hash"]:
            raise CheckpointValidationError(f"Round {round_index} manifest hash mismatch")
        sentinel = path.parent / ROUND_COMPLETE_SENTINEL
        if not sentinel.is_file() or sentinel.read_text(encoding="utf-8").strip() != actual_hash:
            raise CheckpointValidationError(f"Round {round_index} is incomplete")
        for relative, expected in round_manifest["files"].items():
            self._verify_file(path.parent / relative, expected)
        self._round_cache[round_index] = round_manifest
        return round_manifest

    def _load_state_descriptor(self, round_dir: Path, descriptor: Mapping[str, Any]) -> list[np.ndarray]:
        path = round_dir / descriptor["path"]
        self._verify_file(path, descriptor["file_sha256"])
        state = arrays_to_state(load_npz(path), len(self.schema))
        validate_state(state, self.schema, expected_hash=descriptor["state_sha256"])
        return state

    def load_selections(self, client_id: int) -> Tuple[Dict, Dict[str, np.ndarray]]:
        client_id = int(client_id)
        if client_id in self._selection_cache:
            return self._selection_cache[client_id]
        entry = self.manifest["selection_artifacts"].get(str(client_id))
        if entry is None:
            raise CheckpointValidationError(f"No selections for client {client_id}")
        array_path = self.run_dir / entry["arrays"]
        metadata_path = self.run_dir / entry["metadata"]
        self._verify_file(array_path, entry["arrays_sha256"])
        self._verify_file(metadata_path, entry["metadata_sha256"])
        arrays = load_npz(array_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if hash_arrays(arrays) != metadata["array_hash"]:
            raise CheckpointValidationError(f"Selection array hash mismatch for client {client_id}")
        self._selection_cache[client_id] = (metadata, arrays)
        return metadata, arrays

    def load_artifacts(self, round_index: int, client_id: int) -> ArtifactNamespaces:
        round_manifest = self.load_round_manifest(round_index)
        round_dir = self.run_dir / f"round_{round_index:04d}"
        client_entry = round_manifest["clients"].get(str(client_id))
        if client_entry is None:
            raise CheckpointValidationError(
                f"Client {client_id} does not exist in round {round_index}"
            )
        client_manifest_path = round_dir / client_entry["manifest"]
        self._verify_file(client_manifest_path, client_entry["manifest_sha256"])
        client_manifest = json.loads(client_manifest_path.read_text(encoding="utf-8"))
        actual_desc = client_manifest["evaluation_oracle"]["post_local_state"]
        actual_path = client_manifest_path.parent / actual_desc["path"]
        self._verify_file(actual_path, actual_desc["file_sha256"])
        actual_state = arrays_to_state(load_npz(actual_path), len(self.schema))
        validate_state(actual_state, self.schema, expected_hash=actual_desc["state_sha256"])

        exposed_desc = round_manifest["exposed_flat"]
        exposed_path = round_dir / exposed_desc["path"]
        self._verify_file(exposed_path, exposed_desc["file_sha256"])
        exposed_arrays = load_npz(exposed_path)
        exposed_before = exposed_arrays["before_round_ending_update"]
        exposed_after = exposed_arrays["after_round_ending_update"]
        if hash_arrays({"exposed": exposed_before}) != exposed_desc["before_sha256"]:
            raise CheckpointValidationError("Exposed-before hash mismatch")
        if hash_arrays({"exposed": exposed_after}) != exposed_desc["after_sha256"]:
            raise CheckpointValidationError("Exposed-after hash mismatch")

        mask_desc = round_manifest["consensus_mask"]
        mask_path = round_dir / mask_desc["state_dict_visibility_masks_path"]
        self._verify_file(mask_path, mask_desc["state_dict_visibility_masks_file_sha256"])
        mask_arrays = load_npz(mask_path)
        encrypted_masks = []
        plaintext_masks = []
        for index, entry in enumerate(self.schema):
            encrypted = mask_arrays[f"encrypted_{index:05d}"]
            plaintext = mask_arrays[f"plaintext_{index:05d}"]
            if encrypted.dtype != np.uint8 or plaintext.dtype != np.uint8:
                raise CheckpointValidationError("Visibility masks must use uint8")
            if encrypted.shape != tuple(entry["shape"]) or plaintext.shape != tuple(entry["shape"]):
                raise CheckpointValidationError(f"Visibility mask shape mismatch for {entry['name']}")
            if not np.array_equal(encrypted + plaintext, np.ones_like(encrypted)):
                raise CheckpointValidationError(f"Visibility masks overlap or have gaps for {entry['name']}")
            encrypted_masks.append(encrypted)
            plaintext_masks.append(plaintext)

        sensitivity_desc = mask_desc["sensitivity_maps"]
        sensitivity_path = round_dir / sensitivity_desc["path"]
        self._verify_file(sensitivity_path, sensitivity_desc["file_sha256"])
        sensitivity_arrays = load_npz(sensitivity_path)
        sensitivity_map = []
        for parameter_index, entry in enumerate(self.trainable_schema):
            key = f"client_{client_id:03d}_parameter_{parameter_index:05d}"
            if key not in sensitivity_arrays:
                raise CheckpointValidationError(
                    f"Missing sensitivity-map array: {key}"
                )
            value = sensitivity_arrays[key]
            if value.shape != tuple(entry["shape"]):
                raise CheckpointValidationError(
                    f"Sensitivity-map shape mismatch for {entry['name']}"
                )
            if not np.issubdtype(value.dtype, np.floating):
                raise CheckpointValidationError(
                    f"Sensitivity map for {entry['name']} must be floating point"
                )
            sensitivity_map.append(value)

        attacker_state = construct_attacker_visible_state(
            actual_state, exposed_before, encrypted_masks, self.schema
        )
        validate_state(
            attacker_state, self.schema,
            expected_hash=client_manifest["server_visible"]["attacker_visible_state"]["state_sha256"],
        )
        global_entering = self._load_state_descriptor(
            round_dir, round_manifest["common_state"]["global_entering"]
        )
        global_after = self._load_state_descriptor(
            round_dir, round_manifest["common_state"]["global_after_aggregation"]
        )
        visible_payload_hash = _visible_payload_hash(
            actual_state, plaintext_masks, self.state_names
        )
        expected_payload_hash = client_manifest["server_visible"][
            "plaintext_payload"
        ]["payload_sha256"]
        if visible_payload_hash != expected_payload_hash:
            raise CheckpointValidationError("Client plaintext payload hash mismatch")
        model_delta = _state_delta(actual_state, global_entering)
        expected_delta_hash = client_manifest["evaluation_oracle"]["model_delta"][
            "state_sha256"
        ]
        if hash_state(model_delta, self.state_names) != expected_delta_hash:
            raise CheckpointValidationError("Client model-delta hash mismatch")
        selection_metadata, selection_arrays = self.load_selections(client_id)

        server_visible = {
            "attacker_visible_state": attacker_state,
            "global_entering_state": global_entering,
            "global_after_aggregation_state": global_after,
            "exposed_flat_before_update": exposed_before,
            "exposed_flat_after_update": exposed_after,
            "encrypted_masks": encrypted_masks,
            "plaintext_masks": plaintext_masks,
            "aggregation_weight": client_manifest["client_aggregation_weight"],
            "transmission_metadata": client_manifest["server_visible"]["transmission_metadata"],
            "consensus_mask": mask_desc,
        }
        evaluation_oracle = {
            "actual_post_local_state": actual_state,
            "model_delta": model_delta,
            "sensitivity_map": sensitivity_map,
            "sensitivity_map_metadata": sensitivity_desc,
            "selections": selection_arrays,
            "selection_metadata": selection_metadata,
            "partition_identity": client_manifest["client_dataset_partition_identity"],
        }
        return ArtifactNamespaces(
            server_visible=server_visible,
            evaluation_oracle=evaluation_oracle,
        )


def generate_run_id(exp_id: str) -> str:
    from Codes.path_utils import safe_path_component

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{safe_path_component(exp_id, fallback='experiment')}__{stamp}-{uuid.uuid4().hex[:8]}"
