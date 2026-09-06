"""Deterministic privacy-attack sample selection.

This module intentionally has no model, plotting, or training imports so the
federated-learning process can freeze attack inputs without loading attack
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class MIASelection:
    member_indices: np.ndarray
    nonmember_indices: np.ndarray
    candidate_member_indices: np.ndarray


def _class_indices(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return (
        np.argmax(labels, axis=1).astype(np.int64, copy=False)
        if labels.ndim == 2
        else labels.astype(np.int64, copy=False).reshape(-1)
    )


def select_class_matched_mia_indices(
    member_labels: np.ndarray,
    nonmember_labels: np.ndarray,
    max_samples: int,
    seed: int,
) -> MIASelection:
    """Return the exact indices used by the historical online MIA selector."""
    if max_samples <= 0:
        raise ValueError("MIA sample count must be positive")

    member_classes = _class_indices(member_labels)
    nonmember_classes = _class_indices(nonmember_labels)
    rng = np.random.default_rng(seed)
    candidate_members = rng.permutation(len(member_classes))[
        : min(max_samples, len(member_classes))
    ]

    selected_members = []
    selected_nonmembers = []
    for class_id in np.unique(member_classes[candidate_members]):
        member_indices = candidate_members[
            member_classes[candidate_members] == class_id
        ]
        nonmember_indices = np.flatnonzero(nonmember_classes == class_id)
        count = min(len(member_indices), len(nonmember_indices))
        if count == 0:
            continue
        selected_members.extend(
            rng.choice(member_indices, count, replace=False).tolist()
        )
        selected_nonmembers.extend(
            rng.choice(nonmember_indices, count, replace=False).tolist()
        )

    if not selected_members:
        raise ValueError("No class-matched member/nonmember samples are available")
    member_indices = np.asarray(selected_members, dtype=np.int64)
    nonmember_indices = np.asarray(selected_nonmembers, dtype=np.int64)
    member_indices = member_indices[rng.permutation(len(member_indices))]
    nonmember_indices = nonmember_indices[rng.permutation(len(nonmember_indices))]
    return MIASelection(
        member_indices=member_indices,
        nonmember_indices=nonmember_indices,
        candidate_member_indices=np.asarray(candidate_members, dtype=np.int64),
    )


def sample_class_matched_mia_data(
    member_data: Tuple[np.ndarray, np.ndarray],
    nonmember_data: Tuple[np.ndarray, np.ndarray],
    max_samples: int,
    seed: int,
):
    """Return balanced member/nonmember arrays with identical class counts."""
    X_member, Y_member = member_data
    X_nonmember, Y_nonmember = nonmember_data
    selection = select_class_matched_mia_indices(
        Y_member, Y_nonmember, max_samples=max_samples, seed=seed
    )
    return (
        X_member[selection.member_indices],
        Y_member[selection.member_indices],
        X_nonmember[selection.nonmember_indices],
        Y_nonmember[selection.nonmember_indices],
    )


def plan_client_selections(
    member_data: Tuple[np.ndarray, np.ndarray],
    nonmember_data: Tuple[np.ndarray, np.ndarray],
    *,
    client_id: int,
    attack_seed: int,
    mia_sample_size: int,
    ilrg_batch_size: int,
    ilrg_num_batches: int,
    dlg_num_samples: int,
    dlg_num_restarts: int,
    ig_num_samples: int,
    ig_num_restarts: int,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Freeze all selections made by the former online orchestration block."""
    X_member, Y_member = member_data
    X_nonmember, Y_nonmember = nonmember_data
    seed_base = int(attack_seed + client_id * 1000)

    mia = select_class_matched_mia_indices(
        Y_member, Y_nonmember, max_samples=mia_sample_size, seed=seed_base
    )

    batch_starts = np.arange(0, len(X_member), ilrg_batch_size, dtype=np.int64)
    ilrg_count = min(ilrg_num_batches, len(batch_starts))
    ilrg_rng = np.random.default_rng(seed_base)
    selected_batch_ids = np.sort(
        ilrg_rng.choice(len(batch_starts), size=ilrg_count, replace=False)
    ).astype(np.int64, copy=False)
    ilrg_boundaries = np.asarray(
        [
            (
                int(batch_starts[batch_id]),
                min(int(batch_starts[batch_id]) + ilrg_batch_size, len(X_member)),
            )
            for batch_id in selected_batch_ids
        ],
        dtype=np.int64,
    )
    ilrg_positions = np.concatenate(
        [np.arange(start, end, dtype=np.int64) for start, end in ilrg_boundaries],
        dtype=np.int64,
    ) if len(ilrg_boundaries) else np.empty(0, dtype=np.int64)
    ilrg_offsets = np.cumsum(
        np.asarray([0] + [int(end - start) for start, end in ilrg_boundaries], dtype=np.int64)
    )

    dlg_count = min(dlg_num_samples, len(X_member))
    dlg_indices = np.random.default_rng(seed_base).choice(
        len(X_member), size=dlg_count, replace=False
    ).astype(np.int64, copy=False)
    dlg_restart_seeds = np.asarray(
        [
            [seed_base + position * dlg_num_restarts + restart
             for restart in range(dlg_num_restarts)]
            for position in range(dlg_count)
        ],
        dtype=np.int64,
    )

    ig_count = min(ig_num_samples, len(X_member))
    ig_indices = np.random.default_rng(seed_base).choice(
        len(X_member), size=ig_count, replace=False
    ).astype(np.int64, copy=False)
    ig_restart_seeds = np.asarray(
        [
            [seed_base + position * ig_num_restarts + restart
             for restart in range(ig_num_restarts)]
            for position in range(ig_count)
        ],
        dtype=np.int64,
    )

    metadata = {
        "selection_protocol_version": "FEDML-HE-online-selection-v1",
        "client_id": int(client_id),
        "attack_seed_base": seed_base,
        "mia": {
            "selection_seed": seed_base,
            "requested_sample_count": int(mia_sample_size),
            "actual_member_count": int(len(mia.member_indices)),
            "actual_nonmember_count": int(len(mia.nonmember_indices)),
            "member_split": "client_train_partition",
            "nonmember_split": "dataset_test",
            "class_matching": "identical selected counts per true class",
        },
        "ilrg": {
            "selection_seed": seed_base,
            "requested_num_batches": int(ilrg_num_batches),
            "actual_num_batches": int(len(selected_batch_ids)),
            "batch_size": int(ilrg_batch_size),
        },
        "dlg": {
            "selection_seed": seed_base,
            "attack_base_seed": seed_base,
            "requested_num_samples": int(dlg_num_samples),
            "actual_num_samples": int(len(dlg_indices)),
            "num_restarts": int(dlg_num_restarts),
        },
        "ig": {
            "selection_seed": seed_base,
            "attack_base_seed": seed_base,
            "requested_num_samples": int(ig_num_samples),
            "actual_num_samples": int(len(ig_indices)),
            "num_restarts": int(ig_num_restarts),
        },
    }

    arrays = {
        "mia_member_inputs": np.asarray(X_member[mia.member_indices]),
        "mia_member_labels": np.asarray(Y_member[mia.member_indices]),
        "mia_member_sample_ids": mia.member_indices,
        "mia_nonmember_inputs": np.asarray(X_nonmember[mia.nonmember_indices]),
        "mia_nonmember_labels": np.asarray(Y_nonmember[mia.nonmember_indices]),
        "mia_nonmember_sample_ids": mia.nonmember_indices,
        "mia_member_classes": _class_indices(Y_member[mia.member_indices]),
        "mia_nonmember_classes": _class_indices(Y_nonmember[mia.nonmember_indices]),
        "mia_candidate_member_ids": mia.candidate_member_indices,
        "ilrg_inputs": np.asarray(X_member[ilrg_positions]),
        "ilrg_labels": np.asarray(Y_member[ilrg_positions]),
        "ilrg_sample_ids": ilrg_positions,
        "ilrg_batch_ids": selected_batch_ids,
        "ilrg_batch_boundaries": ilrg_boundaries,
        "ilrg_offsets": ilrg_offsets,
        "dlg_inputs": np.asarray(X_member[dlg_indices]),
        "dlg_labels": np.asarray(Y_member[dlg_indices]),
        "dlg_sample_ids": dlg_indices,
        "dlg_selection_positions": np.arange(dlg_count, dtype=np.int64),
        "dlg_restart_seeds": dlg_restart_seeds,
        "ig_inputs": np.asarray(X_member[ig_indices]),
        "ig_labels": np.asarray(Y_member[ig_indices]),
        "ig_sample_ids": ig_indices,
        "ig_selection_positions": np.arange(ig_count, dtype=np.int64),
        "ig_restart_seeds": ig_restart_seeds,
    }
    return metadata, arrays
