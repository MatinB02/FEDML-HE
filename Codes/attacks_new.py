import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Codes.functions import calculate_ssim
from Codes.attack_selection import sample_class_matched_mia_data
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, recall_score, f1_score)

@dataclass
class AttackMetrics:
    """Container for attack evaluation metrics"""
    attack_name: str
    success_rate: float = 0.0
    mse: float = 0.0
    ssim: float = 0.0
    psnr: float = 0.0
    cosine_sim: float = 0.0
    lpips: Optional[float] = None
    attack_time: float = 0.0
    label_accuracy: float = 0.0
    idlg_label_inference_success: Optional[bool] = None
    convergence_iterations: int = 0
    additional_info: Dict = field(default_factory=dict)
    iterations: float = 0.0
    fpr: list = field(default_factory=list)
    tpr: list = field(default_factory=list)

    def to_dict(self):
        return {
            'attack': self.attack_name,
            'success_rate': self.success_rate,
            'mse': self.mse,
            'ssim': self.ssim,
            'psnr': self.psnr,
            'cosine_sim': self.cosine_sim,
            'lpips': self.lpips,
            'time': self.attack_time,
            'label_acc': self.label_accuracy,
            'idlg_label_inference_success': self.idlg_label_inference_success,
            'convergence_iters': self.convergence_iterations,
            'fpr': self.fpr,
            'tpr': self.tpr,
            **self.additional_info
        }

    def print_summary(self):
        print(f"\n{'=' * 70}")
        print(f"Attack: {self.attack_name}")
        print(f"{'=' * 70}")
        print(f"Success Rate:      {self.success_rate:.2%}")
        print(f"MSE:               {self.mse:.6f}")
        print(f"SSIM:              {self.ssim:.4f}")
        print(f"PSNR:              {self.psnr:.2f} dB")
        print(f"Cosine Similarity: {self.cosine_sim:.4f}")
        if self.lpips is not None:
            print(f"LPIPS:             {self.lpips:.4f}")
        print(f"Label Accuracy:    {self.label_accuracy:.2%}")
        print(f"Attack Time:       {self.attack_time:.2f}s")
        print(f"Convergence Iters: {self.convergence_iterations}")
        print(f"{'=' * 70}\n")


@dataclass
class ILRGMetrics:
    """Metrics and diagnostics for one masked batch iLRG evaluation."""
    attack_name: str = "iLRG_ElementWise"
    attack_available: bool = False
    unavailable_reason: Optional[str] = None
    batch_size: int = 0
    attack_time: float = 0.0
    true_counts: list = field(default_factory=list)
    continuous_counts: list = field(default_factory=list)
    predicted_counts: list = field(default_factory=list)
    label_existence_accuracy: Optional[float] = None
    label_number_accuracy: Optional[float] = None
    instance_recall: Optional[float] = None
    count_mae: Optional[float] = None
    normalized_count_l1: Optional[float] = None
    count_cosine_similarity: Optional[float] = None
    exact_count_vector: Optional[bool] = None
    label_precision: Optional[float] = None
    label_recall: Optional[float] = None
    label_f1: Optional[float] = None
    additional_info: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "attack": self.attack_name,
            "attack_available": self.attack_available,
            "unavailable_reason": self.unavailable_reason,
            "batch_size": self.batch_size,
            "attack_time": self.attack_time,
            "true_counts": self.true_counts,
            "continuous_counts": self.continuous_counts,
            "predicted_counts": self.predicted_counts,
            "label_existence_accuracy": self.label_existence_accuracy,
            "label_number_accuracy": self.label_number_accuracy,
            "instance_recall": self.instance_recall,
            "count_mae": self.count_mae,
            "normalized_count_l1": self.normalized_count_l1,
            "count_cosine_similarity": self.count_cosine_similarity,
            "exact_count_vector": self.exact_count_vector,
            "label_precision": self.label_precision,
            "label_recall": self.label_recall,
            "label_f1": self.label_f1,
            **self.additional_info,
        }

    def print_summary(self):
        availability = "available" if self.attack_available else "unavailable"
        print(f"\n{'=' * 70}")
        print(f"Attack: {self.attack_name} ({availability})")
        print(f"Batch size: {self.batch_size}")
        if self.attack_available:
            print(f"True counts:      {self.true_counts}")
            print(f"Predicted counts: {self.predicted_counts}")
            print(f"Label existence accuracy: {self.label_existence_accuracy:.2%}")
            print(f"Label number accuracy:    {self.label_number_accuracy:.2%}")
            print(f"Instance recall:          {self.instance_recall:.2%}")
            print(f"Normalized count L1:      {self.normalized_count_l1:.4f}")
        else:
            print(f"Reason: {self.unavailable_reason}")
        print(f"Attack time: {self.attack_time:.4f}s")
        print(f"{'=' * 70}\n")


def _save_restart_artifacts(save_dir, attack_name, restart_results):
    """Persist reducer inputs without pickle or Python object arrays."""
    histories = [np.asarray(result["history"], dtype=np.float64) for result in restart_results]
    history_offsets = np.cumsum(
        np.asarray([0] + [len(values) for values in histories], dtype=np.int64)
    )
    arrays = {
        "restart_ids": np.asarray([result["restart"] for result in restart_results], dtype=np.int64),
        "seeds": np.asarray([result["seed"] for result in restart_results], dtype=np.int64),
        "best_losses": np.asarray([result["best_loss"] for result in restart_results], dtype=np.float64),
        "best_gradient_losses": np.asarray(
            [result["best_gradient_loss"] for result in restart_results], dtype=np.float64
        ),
        "best_iterations": np.asarray(
            [result["best_iteration"] for result in restart_results], dtype=np.int64
        ),
        "final_losses": np.asarray(
            [result["final_loss"] for result in restart_results], dtype=np.float64
        ),
        "iterations_run": np.asarray(
            [result["iterations_run"] for result in restart_results], dtype=np.int64
        ),
        "inferred_labels": np.asarray(
            [result["inferred_label"] for result in restart_results], dtype=np.int64
        ),
        "images": np.stack([np.asarray(result["image"], dtype=np.float32) for result in restart_results]),
        "history": np.concatenate(histories) if histories else np.empty(0, dtype=np.float64),
        "history_offsets": history_offsets,
    }
    if attack_name == "dlg":
        gradients = [
            np.asarray(result["gradient_history"], dtype=np.float64)
            for result in restart_results
        ]
        arrays["gradient_history"] = (
            np.concatenate(gradients) if gradients else np.empty(0, dtype=np.float64)
        )
    else:
        cosine = [
            np.asarray(result["cosine_history"], dtype=np.float64)
            for result in restart_results
        ]
        tv = [np.asarray(result["tv_history"], dtype=np.float64) for result in restart_results]
        arrays["cosine_history"] = (
            np.concatenate(cosine) if cosine else np.empty(0, dtype=np.float64)
        )
        arrays["tv_history"] = np.concatenate(tv) if tv else np.empty(0, dtype=np.float64)
    np.savez(os.path.join(save_dir, f"{attack_name}_restart_artifacts.npz"), **arrays)


def _integer_counts_with_fixed_total(continuous_counts, total):
    """Project non-negative real counts onto integers summing exactly to total."""
    continuous = np.nan_to_num(
        np.asarray(continuous_counts, dtype=np.float64),
        nan=0.0,
        posinf=float(total),
        neginf=0.0,
    )
    continuous = np.clip(continuous, 0.0, float(total))
    counts = np.rint(continuous).astype(np.int64)

    while int(counts.sum()) < total:
        candidates = np.where(counts < total)[0]
        if len(candidates) == 0:
            break
        residuals = continuous[candidates] - counts[candidates]
        counts[candidates[int(np.argmax(residuals))]] += 1

    while int(counts.sum()) > total:
        candidates = np.where(counts > 0)[0]
        if len(candidates) == 0:
            break
        residuals = counts[candidates] - continuous[candidates]
        counts[candidates[int(np.argmax(residuals))]] -= 1

    return counts


def perform_iLRG_attack(
        trainer,
        client_model,
        attacker_model,
        batch_x,
        batch_y,
        maskBoolNot=None,
        alpha: float = 0.01,
        mask_mode: str = "partial",
        epsilon: float = 1e-12,
        batch_id: Optional[int] = None,
        save_dir: str = "./attack_results",
) -> ILRGMetrics:
    """Recover per-class batch counts from visible final-layer gradients.

    Target gradients are generated at ``client_model``. All inference is then
    performed at ``attacker_model`` and can use only entries marked visible by
    ``maskBoolNot``. In ``partial`` mode, hidden weight-gradient coordinates
    are omitted from each recovered embedding and hidden bias-gradient rows are
    omitted from the linear system. ``strict`` mode requires the complete final
    classifier weight and bias gradients.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if mask_mode not in {"partial", "strict"}:
        raise ValueError("mask_mode must be 'partial' or 'strict'")

    os.makedirs(save_dir, exist_ok=True)
    start_time = time.perf_counter()
    device = trainer.device

    if isinstance(batch_x, np.ndarray) and batch_x.dtype == object:
        batch_x = np.array(batch_x.tolist(), dtype=np.float32)
    else:
        batch_x = np.asarray(batch_x, dtype=np.float32)
    if isinstance(batch_y, np.ndarray) and batch_y.dtype == object:
        batch_y = np.array(batch_y.tolist(), dtype=np.float32)
    else:
        batch_y = np.asarray(batch_y)
    if len(batch_x) == 0 or len(batch_x) != len(batch_y):
        raise ValueError("iLRG expects a non-empty batch with matching inputs and labels")

    # Match DLG/IG: create the private target at the actual client state, then
    # restore the model state available to the attacker before doing inference.
    trainer.setAllWeights(client_model)
    raw_gradients = trainer.compute_gradients(batch_x, batch_y)
    target_gradients = [
        torch.as_tensor(gradient, device=device, dtype=torch.float32)
        if gradient is not None else None
        for gradient in raw_gradients
    ]

    trainer.setAllWeights(attacker_model)
    shadow_model = trainer.model
    shadow_model.eval()
    named_parameters = list(shadow_model.named_parameters())
    parameter_names = [name for name, _ in named_parameters]
    state_names = list(shadow_model.state_dict().keys())
    if len(target_gradients) != len(parameter_names):
        raise ValueError(
            f"Received {len(target_gradients)} target gradients for "
            f"{len(parameter_names)} trainable parameters"
        )

    if maskBoolNot is None:
        trainable_masks = [
            np.ones(tuple(gradient.shape), dtype=np.uint8)
            if gradient is not None else None
            for gradient in target_gradients
        ]
    elif len(maskBoolNot) == len(state_names):
        masks_by_name = dict(zip(state_names, maskBoolNot))
        trainable_masks = [masks_by_name[name] for name in parameter_names]
    elif len(maskBoolNot) == len(parameter_names):
        trainable_masks = list(maskBoolNot)
    else:
        raise ValueError(
            f"maskBoolNot has {len(maskBoolNot)} entries; expected either "
            f"{len(state_names)} state_dict entries or "
            f"{len(parameter_names)} trainable parameters"
        )

    parameter_masks = []
    visible_count = 0
    parameter_count = 0
    for name, gradient, mask in zip(parameter_names, target_gradients, trainable_masks):
        if gradient is None:
            parameter_masks.append(None)
            continue
        mask_tensor = torch.as_tensor(mask, device=device)
        if mask_tensor.shape != gradient.shape:
            raise ValueError(
                f"Mask shape {tuple(mask_tensor.shape)} for {name} does not "
                f"match gradient shape {tuple(gradient.shape)}"
            )
        visible_mask = mask_tensor == 1
        parameter_masks.append(visible_mask)
        visible_count += int(visible_mask.sum().item())
        parameter_count += visible_mask.numel()

    # Find the last actual Linear module, avoiding model-specific names such as
    # LeNet's model.fc3 and ResNet's model.fc.
    linear_modules = [
        (name, module)
        for name, module in shadow_model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if not linear_modules:
        raise ValueError("iLRG requires a final nn.Linear classifier")
    final_module_name, final_linear = linear_modules[-1]
    prefix = f"{final_module_name}." if final_module_name else ""
    weight_name = prefix + "weight"
    bias_name = prefix + "bias"
    if weight_name not in parameter_names or bias_name not in parameter_names:
        raise ValueError("iLRG requires a final Linear classifier with a trainable bias")

    weight_index = parameter_names.index(weight_name)
    bias_index = parameter_names.index(bias_name)
    weight_gradient = target_gradients[weight_index]
    bias_gradient = target_gradients[bias_index]
    weight_mask = parameter_masks[weight_index]
    bias_mask = parameter_masks[bias_index]
    num_classes = int(trainer.classCount)
    if (weight_gradient.ndim != 2 or
            tuple(weight_gradient.shape) != tuple(final_linear.weight.shape) or
            weight_gradient.shape[0] != num_classes or
            bias_gradient.ndim != 1 or bias_gradient.numel() != num_classes):
        raise ValueError("Final Linear gradient shapes are incompatible with iLRG")

    weight_visible_fraction = float(weight_mask.float().mean().item())
    bias_visible_fraction = float(bias_mask.float().mean().item())
    visible_gradient_fraction = (
        float(visible_count / parameter_count) if parameter_count else 0.0
    )
    batch_size = int(len(batch_x))
    true_indices = (
        np.argmax(batch_y, axis=1).astype(np.int64)
        if batch_y.ndim == 2 else batch_y.astype(np.int64).reshape(-1)
    )
    true_counts = np.bincount(true_indices, minlength=num_classes)[:num_classes]

    common_info = {
        "batch_id": batch_id,
        "alpha": float(alpha),
        "mask_mode": mask_mode,
        "attack_scope": "batch-gradient diagnostic",
        "target_gradient_state": "client_model",
        "inference_model_state": "attacker_model",
        "final_layer_name": final_module_name,
        "visible_gradient_fraction": visible_gradient_fraction,
        "final_weight_visible_fraction": weight_visible_fraction,
        "final_bias_visible_fraction": bias_visible_fraction,
    }

    unavailable_reason = None
    if mask_mode == "strict" and (not torch.all(weight_mask) or not torch.all(bias_mask)):
        unavailable_reason = "strict mode requires the complete final-layer gradient"
    elif not torch.any(bias_mask):
        unavailable_reason = "no final-layer bias-gradient equations are visible"

    recovered_probabilities = []
    embedding_coverages = []
    recovered_embedding_classes = 0
    attacker_weight = final_linear.weight.detach()
    attacker_bias = final_linear.bias.detach()

    if unavailable_reason is None:
        uniform_probability = torch.full(
            (num_classes,), 1.0 / num_classes, device=device
        )
        for class_index in range(num_classes):
            coordinate_mask = weight_mask[class_index]
            bias_is_usable = bool(bias_mask[class_index].item()) and (
                abs(float(bias_gradient[class_index].item())) > epsilon
            )
            if bias_is_usable and torch.any(coordinate_mask):
                recovered_embedding = torch.zeros(
                    weight_gradient.shape[1], device=device, dtype=torch.float32
                )
                recovered_embedding[coordinate_mask] = (
                    weight_gradient[class_index, coordinate_mask] /
                    bias_gradient[class_index]
                )
                logits = alpha * (
                    torch.mv(attacker_weight, recovered_embedding) + attacker_bias
                )
                recovered_probabilities.append(torch.softmax(logits, dim=0))
                embedding_coverages.append(float(coordinate_mask.float().mean().item()))
                recovered_embedding_classes += 1
            else:
                # No hidden gradient is substituted. A uniform distribution is
                # an explicit no-information prior for an unrecoverable class.
                recovered_probabilities.append(uniform_probability.clone())
                embedding_coverages.append(0.0)

        if recovered_embedding_classes == 0:
            unavailable_reason = "no class embedding can be recovered from visible gradients"

    if unavailable_reason is not None:
        metrics = ILRGMetrics(
            attack_available=False,
            unavailable_reason=unavailable_reason,
            batch_size=batch_size,
            attack_time=float(time.perf_counter() - start_time),
            true_counts=true_counts.astype(int).tolist(),
            additional_info={
                **common_info,
                "true_labels": np.flatnonzero(true_counts).astype(int).tolist(),
                "predicted_labels": [],
                "usable_bias_equations": int(bias_mask.sum().item()),
                "recovered_embedding_classes": int(recovered_embedding_classes),
                "mean_embedding_visible_fraction": (
                    float(np.mean(embedding_coverages)) if embedding_coverages else 0.0
                ),
                "system_rank": None,
                "system_condition_number": None,
                "residual_l2": None,
            },
        )
        with open(os.path.join(save_dir, "ilrg_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics.to_dict(), handle, indent=2)
        metrics.print_summary()
        return metrics

    probabilities = torch.stack(recovered_probabilities).detach().cpu().numpy()
    coefficients = [np.ones(num_classes, dtype=np.float64)]
    values = [float(batch_size)]
    usable_bias_indices = np.flatnonzero(bias_mask.detach().cpu().numpy())
    for output_class in usable_bias_indices:
        row = probabilities[:, output_class].astype(np.float64, copy=True)
        row[output_class] -= 1.0
        coefficients.append(row)
        values.append(float(batch_size * bias_gradient[output_class].item()))

    coefficient_matrix = np.stack(coefficients)
    value_vector = np.asarray(values, dtype=np.float64)
    continuous_counts = np.linalg.pinv(coefficient_matrix).dot(value_vector)
    continuous_counts = np.clip(continuous_counts, 0.0, float(batch_size))
    predicted_counts = _integer_counts_with_fixed_total(continuous_counts, batch_size)
    residual_l2 = float(np.linalg.norm(
        coefficient_matrix.dot(continuous_counts) - value_vector
    ))
    system_rank = int(np.linalg.matrix_rank(coefficient_matrix))
    condition_number = float(np.linalg.cond(coefficient_matrix))
    if not np.isfinite(condition_number):
        condition_number = None

    true_presence = true_counts > 0
    predicted_presence = predicted_counts > 0
    true_positive = int(np.sum(true_presence & predicted_presence))
    false_positive = int(np.sum(~true_presence & predicted_presence))
    false_negative = int(np.sum(true_presence & ~predicted_presence))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    label_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    count_norm_product = float(np.linalg.norm(true_counts) * np.linalg.norm(predicted_counts))
    count_cosine = (
        float(np.dot(true_counts, predicted_counts) / count_norm_product)
        if count_norm_product > 0 else 0.0
    )

    metrics = ILRGMetrics(
        attack_available=True,
        batch_size=batch_size,
        attack_time=float(time.perf_counter() - start_time),
        true_counts=true_counts.astype(int).tolist(),
        continuous_counts=continuous_counts.astype(float).tolist(),
        predicted_counts=predicted_counts.astype(int).tolist(),
        label_existence_accuracy=float(np.mean(true_presence == predicted_presence)),
        label_number_accuracy=float(np.mean(true_counts == predicted_counts)),
        instance_recall=float(np.minimum(true_counts, predicted_counts).sum() / batch_size),
        count_mae=float(np.mean(np.abs(true_counts - predicted_counts))),
        normalized_count_l1=float(
            np.abs(true_counts - predicted_counts).sum() / (2.0 * batch_size)
        ),
        count_cosine_similarity=count_cosine,
        exact_count_vector=bool(np.array_equal(true_counts, predicted_counts)),
        label_precision=float(precision),
        label_recall=float(recall),
        label_f1=float(label_f1),
        additional_info={
            **common_info,
            "true_labels": np.flatnonzero(true_presence).astype(int).tolist(),
            "predicted_labels": np.flatnonzero(predicted_presence).astype(int).tolist(),
            "usable_bias_equations": int(len(usable_bias_indices)),
            "recovered_embedding_classes": int(recovered_embedding_classes),
            "mean_embedding_visible_fraction": float(np.mean(embedding_coverages)),
            "system_rank": system_rank,
            "system_condition_number": condition_number,
            "residual_l2": residual_l2,
        },
    )

    with open(os.path.join(save_dir, "ilrg_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)
    metrics.print_summary()
    return metrics


def perform_DLG_attack(
        trainer,
        client_model,
        attacker_model,
        sample_x, sample_y,
        maskBoolNot=None,
        num_iterations: int = 1000,
        num_restarts: int = 3,
        learning_rate: float = 0.01,
        optimizer_name: str = "adam",
        objective: str = "l2",
        tv_weight: float = 1e-4,
        early_stopping_patience: int = 200,
        early_stopping_delta: float = 1e-7,
        attack_seed: int = 0,
        sample_id: Optional[int] = None,
        success_ssim_threshold: float = 0.5,
        compute_lpips: bool = False,
        known_label: bool = False,
        precomputed_target_gradients=None,
        restart_offset: int = 0,
        save_dir: str = './attack_results'
) -> AttackMetrics:
    """Run a reproducible element-wise DLG attack with multiple restarts."""
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if num_restarts <= 0:
        raise ValueError("num_restarts must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if optimizer_name not in {"adam", "lbfgs"}:
        raise ValueError("optimizer_name must be 'adam' or 'lbfgs'")
    if objective not in {"l2", "normalized_l2", "cosine"}:
        raise ValueError("objective must be 'l2', 'normalized_l2', or 'cosine'")

    os.makedirs(save_dir, exist_ok=True)
    device = trainer.device

    if isinstance(sample_x, np.ndarray) and sample_x.dtype == object:
        sample_x = np.array(sample_x.tolist(), dtype=np.float32)
    else:
        sample_x = np.array(sample_x, dtype=np.float32)
    if isinstance(sample_y, np.ndarray) and sample_y.dtype == object:
        sample_y = np.array(sample_y.tolist(), dtype=np.float32)
    else:
        sample_y = np.array(sample_y, dtype=np.float32)
    if len(sample_x) != 1 or len(sample_y) != 1:
        raise ValueError("perform_DLG_attack expects exactly one private sample")

    # The target is computed at the private client state. Dummy gradients are
    # subsequently computed only at the state available to the attacker.
    if restart_offset < 0:
        raise ValueError("restart_offset cannot be negative")
    if precomputed_target_gradients is None:
        trainer.setAllWeights(client_model)
        raw_gradients = trainer.compute_gradients(sample_x, sample_y)
    else:
        raw_gradients = precomputed_target_gradients
    target_gradients = [
        torch.as_tensor(gradient, device=device, dtype=torch.float32)
        if gradient is not None else None
        for gradient in raw_gradients
    ]

    trainer.setAllWeights(attacker_model)
    shadow_model = trainer.model
    shadow_model.eval()
    model_parameters = tuple(shadow_model.parameters())

    parameter_names = [name for name, _ in shadow_model.named_parameters()]
    state_names = list(shadow_model.state_dict().keys())
    if len(target_gradients) != len(parameter_names):
        raise ValueError(
            f"Received {len(target_gradients)} target gradients for "
            f"{len(parameter_names)} trainable parameters"
        )

    if maskBoolNot is None:
        trainable_masks = [
            np.ones(tuple(gradient.shape), dtype=np.uint8)
            if gradient is not None else None
            for gradient in target_gradients
        ]
    elif len(maskBoolNot) == len(state_names):
        masks_by_name = dict(zip(state_names, maskBoolNot))
        trainable_masks = [masks_by_name[name] for name in parameter_names]
    elif len(maskBoolNot) == len(parameter_names):
        trainable_masks = list(maskBoolNot)
    else:
        raise ValueError(
            f"maskBoolNot has {len(maskBoolNot)} entries; expected either "
            f"{len(state_names)} state_dict entries or "
            f"{len(parameter_names)} trainable parameters"
        )

    parameter_masks = []
    visible_count = 0
    parameter_count = 0
    for name, gradient, mask in zip(parameter_names, target_gradients, trainable_masks):
        if gradient is None:
            parameter_masks.append(None)
            continue
        mask_tensor = torch.as_tensor(mask, device=device)
        if mask_tensor.shape != gradient.shape:
            raise ValueError(
                f"Mask shape {tuple(mask_tensor.shape)} for {name} does not "
                f"match gradient shape {tuple(gradient.shape)}"
            )
        visible_mask = mask_tensor == 1
        parameter_masks.append(visible_mask)
        visible_count += int(visible_mask.sum().item())
        parameter_count += visible_mask.numel()

    if visible_count == 0:
        raise ValueError("DLG cannot run because no trainable gradient elements are visible")
    visible_gradient_fraction = visible_count / parameter_count

    true_label_idx = int(np.argmax(sample_y[0])) if sample_y.ndim == 2 else int(sample_y[0])
    num_classes = trainer.classCount
    if len(trainer.inputShape) == 3:
        h, w, c = trainer.inputShape
        torch_shape = (1, c, h, w)
    else:
        torch_shape = (1, 1, trainer.inputShape[0], trainer.inputShape[1])

    idlg_inferred_label = None
    label_inference_available = False
    for gradient, visible_mask in reversed(list(zip(target_gradients, parameter_masks))):
        if gradient is None or visible_mask is None or not torch.any(visible_mask):
            continue

        class_scores = None
        visible_classes = None
        if (gradient.ndim == 1 and gradient.numel() == num_classes and
                torch.all(visible_mask)):
            class_scores = gradient
            visible_classes = visible_mask
        elif (gradient.ndim >= 2 and gradient.shape[0] == num_classes and
              torch.all(visible_mask)):
            gradient_by_class = gradient.reshape(num_classes, -1)
            mask_by_class = visible_mask.reshape(num_classes, -1)
            class_scores = torch.where(
                mask_by_class, gradient_by_class, torch.zeros_like(gradient_by_class)
            ).sum(dim=1)
            visible_classes = mask_by_class.any(dim=1)

        if class_scores is not None and torch.any(visible_classes):
            exposed_scores = class_scores.masked_fill(~visible_classes, float('inf'))
            if torch.isfinite(exposed_scores).any():
                idlg_inferred_label = int(torch.argmin(exposed_scores).item())
                label_inference_available = True
                break

    idlg_label_inference_success = bool(
        label_inference_available and idlg_inferred_label == true_label_idx
    )
    if known_label:
        fixed_label = true_label_idx
        label_method = "known_label"
    elif label_inference_available:
        fixed_label = idlg_inferred_label
        label_method = "analytic_iDLG"
    else:
        fixed_label = None
        label_method = "optimized_soft_label"
    dummy_y = (
        torch.tensor([fixed_label], device=device, dtype=torch.long)
        if fixed_label is not None else None
    )
    print(
        f"\n[Element-Wise DLG] Sample: {sample_id} | Initial Label: "
        f"{fixed_label if fixed_label is not None else 'optimized'} | "
        f"True Label: {true_label_idx} | Label method: {label_method} | "
        f"iDLG Label: "
        f"{idlg_inferred_label if idlg_inferred_label is not None else 'unavailable'} | "
        f"iDLG Correct: {idlg_label_inference_success} | "
        f"Visible gradients: {visible_gradient_fraction:.2%}"
    )

    criterion = nn.CrossEntropyLoss()
    start_time = time.perf_counter()

    def tensor_to_image(tensor):
        image = tensor.detach().cpu().numpy()[0]
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        return image.astype(np.float32, copy=True)

    def compute_losses(dummy_x, dummy_label_logits=None):
        dummy_logits = shadow_model(dummy_x)
        if dummy_label_logits is None:
            loss_ce = criterion(dummy_logits, dummy_y)
        else:
            soft_labels = torch.softmax(dummy_label_logits, dim=1)
            loss_ce = -torch.sum(
                soft_labels * torch.log_softmax(dummy_logits, dim=1), dim=1
            ).mean()
        dummy_grads = torch.autograd.grad(
            loss_ce, model_parameters, create_graph=True, allow_unused=True
        )

        if objective == "cosine":
            dot = torch.zeros((), device=device)
            dummy_norm_sq = torch.zeros((), device=device)
            target_norm_sq = torch.zeros((), device=device)
        else:
            gradient_loss = torch.zeros((), device=device)

        for dummy_gradient, target_gradient, visible_mask in zip(
                dummy_grads, target_gradients, parameter_masks):
            if (dummy_gradient is None or target_gradient is None or
                    visible_mask is None or not torch.any(visible_mask)):
                continue
            visible_dummy = dummy_gradient[visible_mask]
            visible_target = target_gradient[visible_mask]
            if objective == "l2":
                gradient_loss = gradient_loss + torch.sum(
                    (visible_dummy - visible_target) ** 2
                )
            elif objective == "normalized_l2":
                gradient_loss = gradient_loss + (
                    torch.norm(visible_dummy - visible_target) /
                    (torch.norm(visible_target) + 1e-8)
                )
            else:
                dot = dot + torch.sum(visible_dummy * visible_target)
                dummy_norm_sq = dummy_norm_sq + torch.sum(visible_dummy ** 2)
                target_norm_sq = target_norm_sq + torch.sum(visible_target ** 2)

        if objective == "cosine":
            gradient_loss = 1.0 - dot / (
                torch.sqrt(dummy_norm_sq * target_norm_sq) + 1e-12
            )

        tv_loss = (
            torch.sum(torch.abs(dummy_x[:, :, :, :-1] - dummy_x[:, :, :, 1:])) +
            torch.sum(torch.abs(dummy_x[:, :, :-1, :] - dummy_x[:, :, 1:, :]))
        )
        return gradient_loss + tv_weight * tv_loss, gradient_loss, tv_loss

    restart_results = []
    overall_best = None
    print_every = max(1, num_iterations // 5)

    for local_restart in range(num_restarts):
        restart = int(restart_offset + local_restart)
        restart_seed = int(attack_seed + restart)
        generator = torch.Generator(device=device)
        generator.manual_seed(restart_seed)
        dummy_x = torch.rand(
            torch_shape, generator=generator, device=device, requires_grad=True
        )
        dummy_label_logits = None
        optimization_parameters = [dummy_x]
        if fixed_label is None:
            dummy_label_logits = torch.randn(
                (1, num_classes), generator=generator, device=device,
                requires_grad=True,
            )
            optimization_parameters.append(dummy_label_logits)

        if optimizer_name == "adam":
            optimizer = optim.Adam(optimization_parameters, lr=learning_rate)
            milestones = sorted(set([
                max(1, num_iterations // 2),
                max(1, (3 * num_iterations) // 4),
            ]))
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.5
            )
        else:
            optimizer = optim.LBFGS(
                optimization_parameters, lr=learning_rate, max_iter=1,
                history_size=100, line_search_fn="strong_wolfe"
            )
            scheduler = None

        history = []
        gradient_history = []
        restart_best_loss = float('inf')
        restart_best_gradient_loss = float('inf')
        restart_best_iteration = 0
        restart_best_image = tensor_to_image(dummy_x)
        restart_best_label = (
            fixed_label if fixed_label is not None
            else int(torch.argmax(dummy_label_logits, dim=1).item())
        )
        stale_iterations = 0

        for iteration in range(num_iterations):
            if optimizer_name == "adam":
                optimizer.zero_grad()
                total_loss, gradient_loss, _ = compute_losses(
                    dummy_x, dummy_label_logits
                )
                if not torch.isfinite(total_loss):
                    print(f"  Restart {restart} iteration {iteration}: non-finite loss; stopping")
                    break
                loss_val = float(total_loss.detach().item())
                gradient_loss_val = float(gradient_loss.detach().item())

                if loss_val < restart_best_loss - early_stopping_delta:
                    restart_best_loss = loss_val
                    restart_best_gradient_loss = gradient_loss_val
                    restart_best_iteration = iteration
                    restart_best_image = tensor_to_image(dummy_x)
                    if dummy_label_logits is not None:
                        restart_best_label = int(
                            torch.argmax(dummy_label_logits, dim=1).item()
                        )
                    stale_iterations = 0
                else:
                    stale_iterations += 1

                optimization_gradients = torch.autograd.grad(
                    total_loss, optimization_parameters
                )
                for parameter, gradient in zip(
                        optimization_parameters, optimization_gradients):
                    parameter.grad = gradient
                optimizer.step()
                with torch.no_grad():
                    dummy_x.clamp_(0.0, 1.0)
                scheduler.step()
            else:
                def closure():
                    optimizer.zero_grad()
                    closure_total, _, _ = compute_losses(
                        dummy_x, dummy_label_logits
                    )
                    if torch.isfinite(closure_total):
                        optimization_gradients = torch.autograd.grad(
                            closure_total, optimization_parameters
                        )
                        for parameter, gradient in zip(
                                optimization_parameters, optimization_gradients):
                            parameter.grad = gradient
                    return closure_total

                optimizer.step(closure)
                with torch.no_grad():
                    dummy_x.clamp_(0.0, 1.0)
                total_loss, gradient_loss, _ = compute_losses(
                    dummy_x, dummy_label_logits
                )
                if not torch.isfinite(total_loss):
                    print(f"  Restart {restart} iteration {iteration}: non-finite loss; stopping")
                    break
                loss_val = float(total_loss.detach().item())
                gradient_loss_val = float(gradient_loss.detach().item())
                if loss_val < restart_best_loss - early_stopping_delta:
                    restart_best_loss = loss_val
                    restart_best_gradient_loss = gradient_loss_val
                    restart_best_iteration = iteration
                    restart_best_image = tensor_to_image(dummy_x)
                    if dummy_label_logits is not None:
                        restart_best_label = int(
                            torch.argmax(dummy_label_logits, dim=1).item()
                        )
                    stale_iterations = 0
                else:
                    stale_iterations += 1

            history.append(loss_val)
            gradient_history.append(gradient_loss_val)
            if iteration % print_every == 0:
                print(
                    f"  Restart {local_restart + 1}/{num_restarts} (global {restart}) | "
                    f"Iteration {iteration:4d}/{num_iterations} | Loss: {loss_val:.6g}"
                )
            if 0 < early_stopping_patience <= stale_iterations:
                print(
                    f"  Restart {local_restart + 1} (global {restart}): early stopping at iteration {iteration} "
                    f"(best iteration {restart_best_iteration})"
                )
                break

        final_loss_value = history[-1] if history else float('inf')
        if optimizer_name == "adam" and history:
            final_total_loss, final_gradient_loss, _ = compute_losses(
                dummy_x, dummy_label_logits
            )
            if torch.isfinite(final_total_loss):
                final_loss_value = float(final_total_loss.detach().item())
                if final_loss_value < restart_best_loss - early_stopping_delta:
                    restart_best_loss = final_loss_value
                    restart_best_gradient_loss = float(
                        final_gradient_loss.detach().item()
                    )
                    restart_best_iteration = len(history)
                    restart_best_image = tensor_to_image(dummy_x)
                    if dummy_label_logits is not None:
                        restart_best_label = int(
                            torch.argmax(dummy_label_logits, dim=1).item()
                        )

        restart_result = {
            "restart": restart,
            "seed": restart_seed,
            "best_loss": restart_best_loss,
            "best_gradient_loss": restart_best_gradient_loss,
            "best_iteration": restart_best_iteration,
            "final_loss": final_loss_value,
            "iterations_run": len(history),
            "inferred_label": restart_best_label,
            "image": restart_best_image,
            "history": history,
            "gradient_history": gradient_history,
        }
        restart_results.append(restart_result)
        if overall_best is None or restart_best_loss < overall_best["best_loss"]:
            overall_best = restart_result

    attack_time = time.perf_counter() - start_time
    best_image = np.clip(overall_best["image"], 0.0, 1.0)
    inferred_label = int(overall_best["inferred_label"])
    original_image = sample_x[0]
    if (original_image.ndim == 3 and original_image.shape[0] in (1, 3) and
            original_image.shape[-1] not in (1, 3)):
        original_image = np.transpose(original_image, (1, 2, 0))
    original_image = np.clip(original_image.astype(np.float32), 0.0, 1.0)

    mse = float(np.mean((original_image - best_image) ** 2))
    psnr = 100.0 if mse < 1e-10 else float(20 * np.log10(1.0 / np.sqrt(mse)))
    ssim_val = float(calculate_ssim(original_image, best_image))
    flat_orig, flat_recon = original_image.flatten(), best_image.flatten()
    norm_prod = np.linalg.norm(flat_orig) * np.linalg.norm(flat_recon)
    cosine_sim = float(np.dot(flat_orig, flat_recon) / norm_prod) if norm_prod > 1e-9 else 0.0
    label_accuracy = 1.0 if inferred_label == true_label_idx else 0.0

    lpips_value = None
    if compute_lpips:
        try:
            import lpips as lpips_package

            loss_fn = lpips_package.LPIPS(net="alex").to(device).eval()

            def lpips_tensor(image):
                array = image
                if array.ndim == 2:
                    array = array[:, :, None]
                if array.shape[-1] == 1:
                    array = np.repeat(array, 3, axis=-1)
                tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)
                tensor = tensor.float()
                if min(tensor.shape[-2:]) < 64:
                    tensor = torch.nn.functional.interpolate(
                        tensor, size=(64, 64), mode="bilinear", align_corners=False
                    )
                return tensor * 2.0 - 1.0

            with torch.no_grad():
                lpips_value = float(
                    loss_fn(lpips_tensor(original_image), lpips_tensor(best_image)).item()
                )
        except Exception as error:
            print(f"  LPIPS unavailable; metric omitted: {error}")

    metrics = AttackMetrics(
        attack_name='DLG_ElementWise', mse=mse, psnr=psnr, ssim=ssim_val,
        label_accuracy=label_accuracy,
        idlg_label_inference_success=idlg_label_inference_success,
        success_rate=float(ssim_val >= success_ssim_threshold),
        cosine_sim=cosine_sim,
        lpips=lpips_value,
        convergence_iterations=int(overall_best["best_iteration"]),
        attack_time=attack_time,
        additional_info={
            "sample_id": None if sample_id is None else int(sample_id),
            "true_label": true_label_idx,
            "inferred_label": inferred_label,
            "label_inference_available": label_inference_available,
            "label_inference_method": label_method,
            "known_label": bool(known_label),
            "idlg_inferred_label": idlg_inferred_label,
            "idlg_label_inference_success": idlg_label_inference_success,
            "best_loss": float(overall_best["best_loss"]),
            "best_gradient_loss": float(overall_best["best_gradient_loss"]),
            "best_iteration": int(overall_best["best_iteration"]),
            "best_restart": int(overall_best["restart"]),
            "best_restart_seed": int(overall_best["seed"]),
            "final_loss": float(overall_best["final_loss"]),
            "iterations_run": int(overall_best["iterations_run"]),
            "num_restarts": int(num_restarts),
            "visible_gradient_fraction": float(visible_gradient_fraction),
            "optimizer": optimizer_name,
            "objective": objective,
            "learning_rate": float(learning_rate),
            "tv_weight": float(tv_weight),
            "success_ssim_threshold": float(success_ssim_threshold),
            "attack_scope": "single-sample gradient diagnostic",
        },
    )

    plt.figure(figsize=(8, 5))
    for result in restart_results:
        plt.plot(
            result["history"], lw=1.5,
            label=f"Restart {result['restart'] + 1} (seed {result['seed']})"
        )
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Total Reconstruction Objective')
    plt.title('DLG Element-Wise Optimization Trajectories')
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'dlg_loss_curve.png'), bbox_inches='tight', dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if original_image.ndim == 2 or original_image.shape[-1] == 1:
        axes[0].imshow(original_image.squeeze(), cmap='gray')
        axes[1].imshow(best_image.squeeze(), cmap='gray')
        axes[2].imshow(np.abs(original_image - best_image).squeeze(), cmap='hot')
    else:
        axes[0].imshow(original_image)
        axes[1].imshow(best_image)
        axes[2].imshow(np.abs(original_image - best_image))

    axes[0].set_title('Original Image')
    axes[1].set_title(f'Reconstructed\nSSIM: {ssim_val:.4f}')
    axes[2].set_title(f'Absolute Error\nMSE: {mse:.5f}')
    for ax in axes: ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dlg_reconstruction_comparison.png'), bbox_inches='tight', dpi=150)
    plt.close()

    with open(os.path.join(save_dir, "dlg_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)
    _save_restart_artifacts(save_dir, "dlg", restart_results)

    metrics.print_summary()
    return metrics


def perform_IG_attack(
        trainer,
        client_model,
        attacker_model,
        sample_x, sample_y,
        maskBoolNot=None,
        num_iterations: int = 4800,
        num_restarts: int = 1,
        learning_rate: float = 0.1,
        tv_weight: float = 1e-4,
        early_stopping_patience: int = 200,
        early_stopping_delta: float = 1e-7,
        attack_seed: int = 0,
        sample_id: Optional[int] = None,
        success_ssim_threshold: float = 0.5,
        compute_lpips: bool = False,
        known_label: bool = False,
        precomputed_target_gradients=None,
        restart_offset: int = 0,
        save_dir: str = './attack_results'
) -> AttackMetrics:
    """Run element-wise Inverting Gradients (IG) on one private sample.

    The target gradient is computed at ``client_model`` while reconstruction
    gradients are computed at ``attacker_model``. Only elements marked visible
    by ``maskBoolNot`` participate in the global cosine-similarity objective.
    The optimization follows Geiping et al.: Gaussian initialization, Adam on
    the signed image gradient, step decay, anisotropic mean TV, and projection
    to [0, 1]. The label workflow mirrors this project's DLG implementation.
    """
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if num_restarts <= 0:
        raise ValueError("num_restarts must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if tv_weight < 0:
        raise ValueError("tv_weight cannot be negative")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")

    os.makedirs(save_dir, exist_ok=True)
    device = trainer.device

    if isinstance(sample_x, np.ndarray) and sample_x.dtype == object:
        sample_x = np.array(sample_x.tolist(), dtype=np.float32)
    else:
        sample_x = np.array(sample_x, dtype=np.float32)
    if isinstance(sample_y, np.ndarray) and sample_y.dtype == object:
        sample_y = np.array(sample_y.tolist(), dtype=np.float32)
    else:
        sample_y = np.array(sample_y, dtype=np.float32)
    if len(sample_x) != 1 or len(sample_y) != 1:
        raise ValueError("perform_IG_attack expects exactly one private sample")

    # Match the DLG threat-model implementation: obtain the private target at
    # the client state, then perform all reconstruction at the server-visible
    # attacker state.
    if restart_offset < 0:
        raise ValueError("restart_offset cannot be negative")
    if precomputed_target_gradients is None:
        trainer.setAllWeights(client_model)
        raw_gradients = trainer.compute_gradients(sample_x, sample_y)
    else:
        raw_gradients = precomputed_target_gradients
    target_gradients = [
        torch.as_tensor(gradient, device=device, dtype=torch.float32)
        if gradient is not None else None
        for gradient in raw_gradients
    ]

    trainer.setAllWeights(attacker_model)
    shadow_model = trainer.model
    shadow_model.eval()
    model_parameters = tuple(shadow_model.parameters())
    parameter_names = [name for name, _ in shadow_model.named_parameters()]
    state_names = list(shadow_model.state_dict().keys())
    if len(target_gradients) != len(parameter_names):
        raise ValueError(
            f"Received {len(target_gradients)} target gradients for "
            f"{len(parameter_names)} trainable parameters"
        )

    if maskBoolNot is None:
        trainable_masks = [
            np.ones(tuple(gradient.shape), dtype=np.uint8)
            if gradient is not None else None
            for gradient in target_gradients
        ]
    elif len(maskBoolNot) == len(state_names):
        masks_by_name = dict(zip(state_names, maskBoolNot))
        trainable_masks = [masks_by_name[name] for name in parameter_names]
    elif len(maskBoolNot) == len(parameter_names):
        trainable_masks = list(maskBoolNot)
    else:
        raise ValueError(
            f"maskBoolNot has {len(maskBoolNot)} entries; expected either "
            f"{len(state_names)} state_dict entries or "
            f"{len(parameter_names)} trainable parameters"
        )

    parameter_masks = []
    visible_count = 0
    parameter_count = 0
    visible_target_norm_sq = torch.zeros((), device=device)
    for name, gradient, mask in zip(parameter_names, target_gradients, trainable_masks):
        if gradient is None:
            parameter_masks.append(None)
            continue
        mask_tensor = torch.as_tensor(mask, device=device)
        if mask_tensor.shape != gradient.shape:
            raise ValueError(
                f"Mask shape {tuple(mask_tensor.shape)} for {name} does not "
                f"match gradient shape {tuple(gradient.shape)}"
            )
        visible_mask = mask_tensor == 1
        parameter_masks.append(visible_mask)
        visible_count += int(visible_mask.sum().item())
        parameter_count += visible_mask.numel()
        if torch.any(visible_mask):
            visible_target_norm_sq += torch.sum(gradient[visible_mask] ** 2)

    if visible_count == 0:
        raise ValueError("IG cannot run because no trainable gradient elements are visible")
    if float(visible_target_norm_sq.detach().item()) <= 0.0:
        raise ValueError("IG cannot run because the visible target gradient has zero norm")
    visible_gradient_fraction = visible_count / parameter_count

    true_label_idx = int(np.argmax(sample_y[0])) if sample_y.ndim == 2 else int(sample_y[0])
    num_classes = trainer.classCount
    if len(trainer.inputShape) == 3:
        h, w, c = trainer.inputShape
        torch_shape = (1, c, h, w)
    else:
        torch_shape = (1, 1, trainer.inputShape[0], trainer.inputShape[1])

    # Always attempt iDLG, including known-label runs, so its independent
    # correctness remains measurable.
    idlg_inferred_label = None
    label_inference_available = False
    for gradient, visible_mask in reversed(list(zip(target_gradients, parameter_masks))):
        if gradient is None or visible_mask is None or not torch.any(visible_mask):
            continue

        class_scores = None
        visible_classes = None
        if (gradient.ndim == 1 and gradient.numel() == num_classes and
                torch.all(visible_mask)):
            class_scores = gradient
            visible_classes = visible_mask
        elif (gradient.ndim >= 2 and gradient.shape[0] == num_classes and
              torch.all(visible_mask)):
            gradient_by_class = gradient.reshape(num_classes, -1)
            mask_by_class = visible_mask.reshape(num_classes, -1)
            class_scores = torch.where(
                mask_by_class, gradient_by_class, torch.zeros_like(gradient_by_class)
            ).sum(dim=1)
            visible_classes = mask_by_class.any(dim=1)

        if class_scores is not None and torch.any(visible_classes):
            exposed_scores = class_scores.masked_fill(~visible_classes, float('inf'))
            if torch.isfinite(exposed_scores).any():
                idlg_inferred_label = int(torch.argmin(exposed_scores).item())
                label_inference_available = True
                break

    idlg_label_inference_success = bool(
        label_inference_available and idlg_inferred_label == true_label_idx
    )
    if known_label:
        fixed_label = true_label_idx
        label_method = "known_label"
    elif label_inference_available:
        fixed_label = idlg_inferred_label
        label_method = "analytic_iDLG"
    else:
        fixed_label = None
        label_method = "optimized_soft_label"
    dummy_y = (
        torch.tensor([fixed_label], device=device, dtype=torch.long)
        if fixed_label is not None else None
    )

    print(
        f"\n[Element-Wise IG] Sample: {sample_id} | Initial Label: "
        f"{fixed_label if fixed_label is not None else 'optimized'} | "
        f"True Label: {true_label_idx} | Label method: {label_method} | "
        f"iDLG Label: "
        f"{idlg_inferred_label if idlg_inferred_label is not None else 'unavailable'} | "
        f"iDLG Correct: {idlg_label_inference_success} | "
        f"Visible gradients: {visible_gradient_fraction:.2%}"
    )

    criterion = nn.CrossEntropyLoss()
    start_time = time.perf_counter()

    def tensor_to_image(tensor):
        image = tensor.detach().cpu().numpy()[0]
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        return image.astype(np.float32, copy=True)

    def compute_losses(dummy_x, dummy_label_logits=None):
        dummy_logits = shadow_model(dummy_x)
        if dummy_label_logits is None:
            loss_ce = criterion(dummy_logits, dummy_y)
        else:
            soft_labels = torch.softmax(dummy_label_logits, dim=1)
            loss_ce = -torch.sum(
                soft_labels * torch.log_softmax(dummy_logits, dim=1), dim=1
            ).mean()
        dummy_grads = torch.autograd.grad(
            loss_ce, model_parameters, create_graph=True, allow_unused=True
        )

        dot = torch.zeros((), device=device)
        dummy_norm_sq = torch.zeros((), device=device)
        for dummy_gradient, target_gradient, visible_mask in zip(
                dummy_grads, target_gradients, parameter_masks):
            if (dummy_gradient is None or target_gradient is None or
                    visible_mask is None or not torch.any(visible_mask)):
                continue
            visible_dummy = dummy_gradient[visible_mask]
            visible_target = target_gradient[visible_mask]
            dot += torch.sum(visible_dummy * visible_target)
            dummy_norm_sq += torch.sum(visible_dummy ** 2)

        cosine_loss = 1.0 - dot / (
            torch.sqrt(dummy_norm_sq * visible_target_norm_sq) + 1e-12
        )
        # Mean anisotropic TV matches the authors' implementation and keeps
        # the weight independent of image resolution.
        tv_loss = (
            torch.mean(torch.abs(dummy_x[:, :, :, :-1] - dummy_x[:, :, :, 1:])) +
            torch.mean(torch.abs(dummy_x[:, :, :-1, :] - dummy_x[:, :, 1:, :]))
        )
        return cosine_loss + tv_weight * tv_loss, cosine_loss, tv_loss

    restart_results = []
    overall_best = None
    print_every = max(1, num_iterations // 5)
    lr_milestones = sorted(set([
        max(1, (3 * num_iterations) // 8),
        max(1, (5 * num_iterations) // 8),
        max(1, (7 * num_iterations) // 8),
    ]))

    for local_restart in range(num_restarts):
        restart = int(restart_offset + local_restart)
        restart_seed = int(attack_seed + restart)
        generator = torch.Generator(device=device)
        generator.manual_seed(restart_seed)
        dummy_x = torch.randn(
            torch_shape, generator=generator, device=device, requires_grad=True
        )
        dummy_label_logits = None
        optimization_parameters = [dummy_x]
        if fixed_label is None:
            dummy_label_logits = torch.randn(
                (1, num_classes), generator=generator, device=device,
                requires_grad=True,
            )
            optimization_parameters.append(dummy_label_logits)

        optimizer = optim.Adam(optimization_parameters, lr=learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=0.1
        )

        history = []
        cosine_history = []
        tv_history = []
        restart_best_loss = float('inf')
        restart_best_gradient_loss = float('inf')
        restart_best_iteration = 0
        restart_best_image = tensor_to_image(dummy_x)
        restart_best_label = (
            fixed_label if fixed_label is not None
            else int(torch.argmax(dummy_label_logits, dim=1).item())
        )
        stale_iterations = 0

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            total_loss, cosine_loss, tv_loss = compute_losses(
                dummy_x, dummy_label_logits
            )
            if not torch.isfinite(total_loss):
                print(f"  Restart {restart} iteration {iteration}: non-finite loss; stopping")
                break

            loss_val = float(total_loss.detach().item())
            cosine_loss_val = float(cosine_loss.detach().item())
            tv_loss_val = float(tv_loss.detach().item())
            if loss_val < restart_best_loss - early_stopping_delta:
                restart_best_loss = loss_val
                restart_best_gradient_loss = cosine_loss_val
                restart_best_iteration = iteration
                restart_best_image = tensor_to_image(dummy_x)
                if dummy_label_logits is not None:
                    restart_best_label = int(
                        torch.argmax(dummy_label_logits, dim=1).item()
                    )
                stale_iterations = 0
            else:
                stale_iterations += 1

            optimization_gradients = torch.autograd.grad(
                total_loss, optimization_parameters
            )
            for parameter, gradient in zip(
                    optimization_parameters, optimization_gradients):
                parameter.grad = gradient
            if dummy_x.grad is None:
                raise RuntimeError("IG optimization produced no image gradient")
            # Signed Adam: only the image gradient is signed. If the label is
            # optimized, its logits retain their ordinary gradient.
            dummy_x.grad.sign_()
            optimizer.step()
            with torch.no_grad():
                dummy_x.clamp_(0.0, 1.0)
            scheduler.step()

            history.append(loss_val)
            cosine_history.append(cosine_loss_val)
            tv_history.append(tv_loss_val)
            if iteration % print_every == 0:
                print(
                    f"  Restart {local_restart + 1}/{num_restarts} (global {restart}) | "
                    f"Iteration {iteration:5d}/{num_iterations} | "
                    f"Loss: {loss_val:.6g} | Cosine loss: {cosine_loss_val:.6g}"
                )
            if 0 < early_stopping_patience <= stale_iterations:
                print(
                    f"  Restart {local_restart + 1} (global {restart}): early stopping at iteration {iteration} "
                    f"(best iteration {restart_best_iteration})"
                )
                break

        final_loss_value = history[-1] if history else float('inf')
        if history:
            final_total_loss, final_cosine_loss, _ = compute_losses(
                dummy_x, dummy_label_logits
            )
            if torch.isfinite(final_total_loss):
                final_loss_value = float(final_total_loss.detach().item())
                if final_loss_value < restart_best_loss - early_stopping_delta:
                    restart_best_loss = final_loss_value
                    restart_best_gradient_loss = float(
                        final_cosine_loss.detach().item()
                    )
                    restart_best_iteration = len(history)
                    restart_best_image = tensor_to_image(dummy_x)
                    if dummy_label_logits is not None:
                        restart_best_label = int(
                            torch.argmax(dummy_label_logits, dim=1).item()
                        )

        restart_result = {
            "restart": restart,
            "seed": restart_seed,
            "best_loss": restart_best_loss,
            "best_gradient_loss": restart_best_gradient_loss,
            "best_iteration": restart_best_iteration,
            "final_loss": final_loss_value,
            "iterations_run": len(history),
            "inferred_label": restart_best_label,
            "image": restart_best_image,
            "history": history,
            "cosine_history": cosine_history,
            "tv_history": tv_history,
        }
        restart_results.append(restart_result)
        if overall_best is None or restart_best_loss < overall_best["best_loss"]:
            overall_best = restart_result

    if overall_best is None or not np.isfinite(overall_best["best_loss"]):
        raise RuntimeError("IG failed to produce a finite reconstruction")

    attack_time = time.perf_counter() - start_time
    best_image = np.clip(overall_best["image"], 0.0, 1.0)
    inferred_label = int(overall_best["inferred_label"])
    original_image = sample_x[0]
    if (original_image.ndim == 3 and original_image.shape[0] in (1, 3) and
            original_image.shape[-1] not in (1, 3)):
        original_image = np.transpose(original_image, (1, 2, 0))
    original_image = np.clip(original_image.astype(np.float32), 0.0, 1.0)

    mse = float(np.mean((original_image - best_image) ** 2))
    psnr = 100.0 if mse < 1e-10 else float(20 * np.log10(1.0 / np.sqrt(mse)))
    ssim_val = float(calculate_ssim(original_image, best_image))
    flat_orig, flat_recon = original_image.flatten(), best_image.flatten()
    norm_prod = np.linalg.norm(flat_orig) * np.linalg.norm(flat_recon)
    cosine_sim = float(np.dot(flat_orig, flat_recon) / norm_prod) if norm_prod > 1e-9 else 0.0
    label_accuracy = 1.0 if inferred_label == true_label_idx else 0.0

    lpips_value = None
    if compute_lpips:
        try:
            import lpips as lpips_package

            loss_fn = lpips_package.LPIPS(net="alex").to(device).eval()

            def lpips_tensor(image):
                array = image
                if array.ndim == 2:
                    array = array[:, :, None]
                if array.shape[-1] == 1:
                    array = np.repeat(array, 3, axis=-1)
                tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)
                tensor = tensor.float()
                if min(tensor.shape[-2:]) < 64:
                    tensor = torch.nn.functional.interpolate(
                        tensor, size=(64, 64), mode="bilinear", align_corners=False
                    )
                return tensor * 2.0 - 1.0

            with torch.no_grad():
                lpips_value = float(
                    loss_fn(lpips_tensor(original_image), lpips_tensor(best_image)).item()
                )
        except Exception as error:
            print(f"  LPIPS unavailable; metric omitted: {error}")

    metrics = AttackMetrics(
        attack_name='IG_ElementWise', mse=mse, psnr=psnr, ssim=ssim_val,
        label_accuracy=label_accuracy,
        idlg_label_inference_success=idlg_label_inference_success,
        success_rate=float(ssim_val >= success_ssim_threshold),
        cosine_sim=cosine_sim,
        lpips=lpips_value,
        convergence_iterations=int(overall_best["best_iteration"]),
        attack_time=attack_time,
        additional_info={
            "sample_id": None if sample_id is None else int(sample_id),
            "true_label": true_label_idx,
            "inferred_label": inferred_label,
            "label_inference_available": label_inference_available,
            "label_inference_method": label_method,
            "known_label": bool(known_label),
            "idlg_inferred_label": idlg_inferred_label,
            "idlg_label_inference_success": idlg_label_inference_success,
            "best_loss": float(overall_best["best_loss"]),
            "best_gradient_loss": float(overall_best["best_gradient_loss"]),
            "best_iteration": int(overall_best["best_iteration"]),
            "best_restart": int(overall_best["restart"]),
            "best_restart_seed": int(overall_best["seed"]),
            "final_loss": float(overall_best["final_loss"]),
            "iterations_run": int(overall_best["iterations_run"]),
            "num_restarts": int(num_restarts),
            "visible_gradient_fraction": float(visible_gradient_fraction),
            "optimizer": "signed_adam",
            "objective": "global_cosine",
            "initialization": "gaussian",
            "learning_rate": float(learning_rate),
            "lr_decay_gamma": 0.1,
            "lr_decay_milestones": lr_milestones,
            "tv_weight": float(tv_weight),
            "early_stopping_patience": int(early_stopping_patience),
            "early_stopping_delta": float(early_stopping_delta),
            "success_ssim_threshold": float(success_ssim_threshold),
            "attack_scope": "single-sample gradient diagnostic",
        },
    )

    plt.figure(figsize=(8, 5))
    for result in restart_results:
        plt.plot(
            result["history"], lw=1.5,
            label=f"Restart {result['restart'] + 1} (seed {result['seed']})"
        )
    plt.xlabel('Iteration')
    plt.ylabel('IG Reconstruction Objective')
    plt.title('IG Element-Wise Optimization Trajectories')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'ig_loss_curve.png'), bbox_inches='tight', dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if original_image.ndim == 2 or original_image.shape[-1] == 1:
        axes[0].imshow(original_image.squeeze(), cmap='gray')
        axes[1].imshow(best_image.squeeze(), cmap='gray')
        axes[2].imshow(np.abs(original_image - best_image).squeeze(), cmap='hot')
    else:
        axes[0].imshow(original_image)
        axes[1].imshow(best_image)
        axes[2].imshow(np.abs(original_image - best_image))

    axes[0].set_title('Original Image')
    axes[1].set_title(f'Reconstructed\nSSIM: {ssim_val:.4f}')
    axes[2].set_title(f'Absolute Error\nMSE: {mse:.5f}')
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ig_reconstruction_comparison.png'), bbox_inches='tight', dpi=150)
    plt.close()

    with open(os.path.join(save_dir, "ig_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)
    _save_restart_artifacts(save_dir, "ig", restart_results)

    metrics.print_summary()
    return metrics


class MembershipInferenceAttack:
    """
    Label-aware threshold MIA with shared inference across distinct signals.

    Reported signals:
        - loss
        - entropy
        - modified_entropy

    True-class confidence is intentionally not reported: it equals exp(-loss)
    and therefore has exactly the same rankings and ROC curve as loss.

    Reports:
        - AUC
        - Oracle-best attack accuracy
        - Optimal threshold
        - Precision
        - Recall
        - F1-score
        - Specificity
        - Membership advantage
        - TPR at fixed low FPRs
    """

    def __init__(self, cfg, trainer):
        self.cfg = cfg
        self.trainer = trainer
        self.device = trainer.device

    def _extract_features(self, X: np.ndarray, Y: np.ndarray, batch_size: int = 128) -> Dict[str, np.ndarray]:
        if len(X) != len(Y):
            raise ValueError(f"X and Y contain {len(X)} and {len(Y)} samples")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.trainer.model.eval()
        losses = []
        entropies = []
        modified_entropies = []
        criterion = nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_x, batch_y_tensor = self.trainer._numpy_to_tensor(
                    np.asarray(X[i:i + batch_size]),
                    np.asarray(Y[i:i + batch_size]),
                )

                logits = self.trainer.model(batch_x)
                probs = torch.softmax(logits, dim=1)

                loss = criterion(logits, batch_y_tensor)
                confidence = probs.gather(1, batch_y_tensor[:, None]).squeeze(1)
                if not torch.allclose(
                        confidence, torch.exp(-loss), rtol=1e-5, atol=1e-7):
                    raise RuntimeError("True-class confidence is inconsistent with CE loss")

                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

                # Song & Mittal modified prediction entropy. It is label-aware
                # but not a monotonic transform of cross-entropy.
                log_probs = -torch.log(torch.clamp(probs, min=1e-30))
                reverse_probs = 1.0 - probs
                log_reverse_probs = -torch.log(torch.clamp(reverse_probs, min=1e-30))
                modified_probs = probs.clone()
                modified_logs = log_reverse_probs.clone()
                row_indices = torch.arange(len(batch_y_tensor), device=self.device)
                modified_probs[row_indices, batch_y_tensor] = reverse_probs[
                    row_indices, batch_y_tensor
                ]
                modified_logs[row_indices, batch_y_tensor] = log_probs[
                    row_indices, batch_y_tensor
                ]
                modified_entropy = torch.sum(modified_probs * modified_logs, dim=1)

                losses.extend(loss.cpu().numpy())
                entropies.extend(entropy.cpu().numpy())
                modified_entropies.extend(modified_entropy.cpu().numpy())

        return {
            "loss": np.asarray(losses, dtype=np.float64),
            "entropy": np.asarray(entropies, dtype=np.float64),
            "modified_entropy": np.asarray(modified_entropies, dtype=np.float64),
        }

    def _calculate_best_threshold_metrics(self, labels, scores):

        fpr, tpr, thresholds = roc_curve(labels, scores)

        best_acc = -1.0
        best_threshold = None
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_specificity = 0.0

        # Include every distinct score plus a finite threshold above the
        # maximum, ensuring the always-nonmember 50% baseline is available.
        threshold_candidates = np.concatenate((
            [np.nextafter(np.max(scores), np.inf)],
            np.unique(scores),
        ))
        for th in threshold_candidates:
            preds = (scores >= th).astype(np.int32)
            acc = np.mean(preds == labels)

            if acc > best_acc:
                best_acc = acc
                best_threshold = th
                best_precision = precision_score(labels, preds, zero_division=0)
                best_recall = recall_score(labels, preds, zero_division=0)
                best_f1 = f1_score(labels, preds, zero_division=0)
                negatives = labels == 0
                best_specificity = float(np.mean(preds[negatives] == 0))

        return (
            best_acc, best_threshold, best_precision, best_recall, best_f1,
            best_specificity, fpr, tpr,
        )

    @staticmethod
    def _bootstrap_auc_ci(
            scores_m, scores_nm, member_classes, nonmember_classes,
            num_bootstrap, seed):
        if num_bootstrap <= 0:
            return None, None
        rng = np.random.default_rng(seed)
        auc_values = np.empty(num_bootstrap, dtype=np.float64)
        labels = np.concatenate((
            np.ones(len(scores_m), dtype=np.int32),
            np.zeros(len(scores_nm), dtype=np.int32),
        ))
        member_groups = [
            np.flatnonzero(member_classes == class_id)
            for class_id in np.unique(member_classes)
        ]
        nonmember_groups = [
            np.flatnonzero(nonmember_classes == class_id)
            for class_id in np.unique(nonmember_classes)
        ]
        for index in range(num_bootstrap):
            sampled_member = np.concatenate([
                scores_m[rng.choice(group, size=len(group), replace=True)]
                for group in member_groups
            ])
            sampled_nonmember = np.concatenate([
                scores_nm[rng.choice(group, size=len(group), replace=True)]
                for group in nonmember_groups
            ])
            auc_values[index] = roc_auc_score(
                labels, np.concatenate((sampled_member, sampled_nonmember))
            )
        lower, upper = np.percentile(auc_values, [2.5, 97.5])
        return float(lower), float(upper)

    def _evaluate_signal(
            self, signal, member_features, nonmember_features,
            member_classes, nonmember_classes, shared_extraction_time,
            num_bootstrap, bootstrap_seed):
        evaluation_start = time.perf_counter()
        scores_m = -member_features[signal]
        scores_nm = -nonmember_features[signal]
        scores = np.concatenate((scores_m, scores_nm)).astype(np.float64)
        labels = np.concatenate((
            np.ones(len(scores_m), dtype=np.int32),
            np.zeros(len(scores_nm), dtype=np.int32),
        ))
        if not np.isfinite(scores).all():
            raise ValueError(f"MIA signal {signal!r} contains non-finite values")

        auc = float(roc_auc_score(labels, scores))
        (
            oracle_best_acc, optimal_threshold, precision, recall, f1,
            specificity, fpr, tpr,
        ) = self._calculate_best_threshold_metrics(labels, scores)

        def tpr_at_fpr(max_fpr):
            eligible = tpr[fpr <= max_fpr]
            return float(np.max(eligible)) if len(eligible) else 0.0

        auc_ci_lower, auc_ci_upper = self._bootstrap_auc_ci(
            scores_m, scores_nm, member_classes, nonmember_classes,
            num_bootstrap, bootstrap_seed
        )
        balanced_accuracy = float((recall + specificity) / 2.0)
        evaluation_time = time.perf_counter() - evaluation_start
        return AttackMetrics(
            attack_name=f"MIA_{signal}",
            success_rate=float(oracle_best_acc),
            attack_time=float(shared_extraction_time + evaluation_time),
            fpr=fpr.tolist(),
            tpr=tpr.tolist(),
            additional_info={
                "auc": auc,
                "auc_ci_lower": auc_ci_lower,
                "auc_ci_upper": auc_ci_upper,
                "bootstrap_samples": int(num_bootstrap),
                "bootstrap_scheme": "stratified by membership and true class",
                "oracle_best_accuracy": float(oracle_best_acc),
                "optimal_threshold": float(optimal_threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "specificity": float(specificity),
                "balanced_accuracy": balanced_accuracy,
                "privacy_advantage": float(np.max(tpr - fpr)),
                "tpr_at_fpr_1pct": tpr_at_fpr(0.01),
                "tpr_at_fpr_5pct": tpr_at_fpr(0.05),
                "tpr_at_fpr_10pct": tpr_at_fpr(0.10),
                "num_member": int(len(scores_m)),
                "num_nonmember": int(len(scores_nm)),
                "signal": signal,
                "shared_feature_extraction_time": float(shared_extraction_time),
                "signal_evaluation_time": float(evaluation_time),
                "threshold_note": (
                    "OracleBestAcc threshold fitted and evaluated on the same samples"
                ),
            },
        )

    def run_signals(
        self,
        member_data: Tuple[np.ndarray, np.ndarray],
        nonmember_data: Tuple[np.ndarray, np.ndarray],
        signals=("loss", "entropy", "modified_entropy"),
        batch_size: int = 128,
        save_paths: Optional[Dict[str, str]] = None,
        num_bootstrap: int = 1000,
        bootstrap_seed: int = 0,
    ) -> Dict[str, AttackMetrics]:
        supported_signals = {"loss", "entropy", "modified_entropy"}
        signals = tuple(signals)
        invalid = set(signals) - supported_signals
        if invalid:
            raise ValueError(
                f"Unsupported MIA signals {sorted(invalid)}; expected "
                f"a subset of {sorted(supported_signals)}"
            )
        if not signals:
            return {}

        X_m, Y_m = member_data
        X_nm, Y_nm = nonmember_data
        if len(X_m) == 0 or len(X_nm) == 0:
            raise ValueError("Membership inference requires non-empty datasets")

        extraction_start = time.perf_counter()
        member_features = self._extract_features(X_m, Y_m, batch_size)
        nonmember_features = self._extract_features(X_nm, Y_nm, batch_size)
        extraction_time = time.perf_counter() - extraction_start
        member_classes = (
            np.argmax(Y_m, axis=1) if np.asarray(Y_m).ndim == 2
            else np.asarray(Y_m, dtype=np.int64)
        )
        nonmember_classes = (
            np.argmax(Y_nm, axis=1) if np.asarray(Y_nm).ndim == 2
            else np.asarray(Y_nm, dtype=np.int64)
        )

        seed_offsets = {"loss": 0, "entropy": 1, "modified_entropy": 2}
        results = {}
        for signal in signals:
            metrics = self._evaluate_signal(
                signal,
                member_features,
                nonmember_features,
                member_classes,
                nonmember_classes,
                extraction_time,
                num_bootstrap,
                bootstrap_seed + seed_offsets[signal],
            )
            results[signal] = metrics
            if save_paths is not None and save_paths.get(signal) is not None:
                self._save_results(metrics, save_paths[signal])
        return results

    def _save_results(self, metrics, save_path):
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        plt.figure(figsize=(10, 8))

        auc = metrics.additional_info.get("auc")

        if auc is None:
            label = "ROC"
        else:
            label = f"ROC (AUC={auc:.3f})"

        plt.plot(metrics.fpr, metrics.tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Membership Inference Attack - {metrics.attack_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig( save_path.replace(".json", "_roc.png"), dpi=150, bbox_inches="tight")
        plt.close()
