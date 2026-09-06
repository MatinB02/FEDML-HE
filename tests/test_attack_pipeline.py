import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from openpyxl import load_workbook


# The test environment need not install scikit-image; the production function
# is exercised with a deterministic, bounded SSIM stand-in.
if "skimage.metrics" not in sys.modules:
    skimage_module = types.ModuleType("skimage")
    skimage_metrics_module = types.ModuleType("skimage.metrics")
    skimage_metrics_module.structural_similarity = (
        lambda image_a, image_b, **_: float(
            np.clip(1.0 - np.mean((image_a - image_b) ** 2), -1.0, 1.0)
        )
    )
    sys.modules["skimage"] = skimage_module
    sys.modules["skimage.metrics"] = skimage_metrics_module

from Codes import excelHelper
from Codes.attacks_new import (
    MembershipInferenceAttack,
    perform_DLG_attack,
    perform_IG_attack,
    perform_iLRG_attack,
    sample_class_matched_mia_data,
)


class TinyTrainer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.classCount = 2
        self.inputShape = (4, 4, 1)
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(16, 2))

    def _numpy_to_tensor(self, X, Y):
        x_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
        x_tensor = x_tensor.permute(0, 3, 1, 2).contiguous()
        labels = np.argmax(Y, axis=1) if np.asarray(Y).ndim == 2 else Y
        return x_tensor, torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def setAllWeights(self, weights):
        state = self.model.state_dict()
        for (name, tensor), value in zip(state.items(), weights):
            state[name] = torch.as_tensor(value, dtype=tensor.dtype)
        self.model.load_state_dict(state)

    def compute_gradients(self, X, Y):
        x_tensor, y_tensor = self._numpy_to_tensor(X, Y)
        self.model.zero_grad()
        loss = nn.CrossEntropyLoss()(self.model(x_tensor), y_tensor)
        loss.backward()
        gradients = [
            parameter.grad.detach().numpy().copy()
            for parameter in self.model.parameters()
        ]
        self.model.zero_grad()
        return gradients


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        return self.linear(inputs)


class MIAFeatureTrainer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = CountingModel()

    def _numpy_to_tensor(self, X, Y):
        labels = np.argmax(Y, axis=1) if np.asarray(Y).ndim == 2 else Y
        return (
            torch.from_numpy(np.asarray(X, dtype=np.float32)),
            torch.from_numpy(np.asarray(labels, dtype=np.int64)),
        )


class DummyConfig:
    pass


class AttackPipelineTests(unittest.TestCase):
    def test_class_matching_preserves_counts_and_seed(self):
        member_x = np.arange(24).reshape(12, 2)
        member_y = np.eye(3, dtype=np.float32)[[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]]
        nonmember_x = np.arange(100, 124).reshape(12, 2)
        nonmember_y = np.eye(3, dtype=np.float32)[[0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]]

        result_a = sample_class_matched_mia_data(
            (member_x, member_y), (nonmember_x, nonmember_y), 10, 7
        )
        result_b = sample_class_matched_mia_data(
            (member_x, member_y), (nonmember_x, nonmember_y), 10, 7
        )
        for array_a, array_b in zip(result_a, result_b):
            np.testing.assert_array_equal(array_a, array_b)
        member_classes = np.argmax(result_a[1], axis=1)
        nonmember_classes = np.argmax(result_a[3], axis=1)
        np.testing.assert_array_equal(
            np.bincount(member_classes, minlength=3),
            np.bincount(nonmember_classes, minlength=3),
        )

    def test_mia_extracts_once_and_reports_distinct_signals(self):
        trainer = MIAFeatureTrainer()
        with torch.no_grad():
            trainer.model.linear.weight.copy_(torch.tensor([[2.0, -1.0], [-1.0, 2.0]]))
            trainer.model.linear.bias.zero_()
        member_x = np.array([[2, 0], [0, 2], [1.5, 0], [0, 1.5]], dtype=np.float32)
        member_y = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]
        nonmember_x = np.array([[0, 2], [2, 0], [0.5, 1], [1, 0.5]], dtype=np.float32)
        nonmember_y = np.eye(2, dtype=np.float32)[[0, 1, 0, 1]]

        results = MembershipInferenceAttack(DummyConfig(), trainer).run_signals(
            (member_x, member_y),
            (nonmember_x, nonmember_y),
            batch_size=16,
            num_bootstrap=25,
            bootstrap_seed=11,
        )
        self.assertEqual(set(results), {"loss", "entropy", "modified_entropy"})
        self.assertEqual(trainer.model.forward_calls, 2)
        for metrics in results.values():
            self.assertGreaterEqual(metrics.additional_info["auc"], 0.0)
            self.assertLessEqual(metrics.additional_info["auc"], 1.0)
            self.assertIsNotNone(metrics.additional_info["auc_ci_lower"])
            self.assertEqual(
                metrics.additional_info["bootstrap_scheme"],
                "stratified by membership and true class",
            )

    def test_dlg_uses_attacker_state_and_writes_rich_metrics(self):
        trainer = TinyTrainer()
        client_model = [
            tensor.detach().numpy().copy()
            for tensor in trainer.model.state_dict().values()
        ]
        attacker_model = [value.copy() for value in client_model]
        attacker_model[0] = attacker_model[0] + 0.05
        masks = [np.ones_like(value, dtype=np.uint8) for value in attacker_model]
        masks[0].reshape(-1)[::2] = 0
        sample_x = np.linspace(0, 1, 16, dtype=np.float32).reshape(1, 4, 4, 1)
        sample_y = np.eye(2, dtype=np.float32)[[1]]

        with tempfile.TemporaryDirectory() as directory:
            metrics = perform_DLG_attack(
                trainer,
                client_model=client_model,
                attacker_model=attacker_model,
                sample_x=sample_x,
                sample_y=sample_y,
                maskBoolNot=masks,
                num_iterations=3,
                num_restarts=2,
                learning_rate=0.01,
                early_stopping_patience=0,
                attack_seed=123,
                sample_id=4,
                save_dir=directory,
            )
            self.assertTrue(Path(directory, "dlg_metrics.json").exists())
            self.assertTrue(Path(directory, "dlg_loss_curve.png").exists())
        for actual, expected in zip(trainer.model.state_dict().values(), attacker_model):
            np.testing.assert_allclose(actual.detach().numpy(), expected)
        self.assertTrue(all(
            parameter.grad is None for parameter in trainer.model.parameters()
        ))
        self.assertEqual(metrics.additional_info["num_restarts"], 2)
        self.assertEqual(metrics.additional_info["sample_id"], 4)
        self.assertAlmostEqual(
            metrics.additional_info["visible_gradient_fraction"],
            sum(np.count_nonzero(mask) for mask in masks) /
            sum(mask.size for mask in masks),
        )

        with tempfile.TemporaryDirectory() as directory:
            perform_DLG_attack(
                trainer,
                client_model=client_model,
                attacker_model=attacker_model,
                sample_x=sample_x,
                sample_y=sample_y,
                maskBoolNot=masks,
                num_iterations=1,
                num_restarts=1,
                learning_rate=0.01,
                optimizer_name="lbfgs",
                early_stopping_patience=0,
                save_dir=directory,
            )
        self.assertTrue(all(
            parameter.grad is None for parameter in trainer.model.parameters()
        ))

    def test_ilrg_uses_batch_client_target_attacker_state_and_partial_mask(self):
        trainer = TinyTrainer()
        client_model = [
            tensor.detach().numpy().copy()
            for tensor in trainer.model.state_dict().values()
        ]
        attacker_model = [value.copy() for value in client_model]
        attacker_model[0] = attacker_model[0] + 0.05
        masks = [np.ones_like(value, dtype=np.uint8) for value in attacker_model]
        masks[0][0, ::2] = 0
        masks[1][0] = 0
        batch_x = np.stack([
            np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4, 1),
            np.linspace(1, 0, 16, dtype=np.float32).reshape(4, 4, 1),
            np.full((4, 4, 1), 0.25, dtype=np.float32),
            np.full((4, 4, 1), 0.75, dtype=np.float32),
        ])
        batch_y = np.eye(2, dtype=np.float32)[[0, 1, 1, 1]]

        with tempfile.TemporaryDirectory() as directory:
            metrics = perform_iLRG_attack(
                trainer,
                client_model=client_model,
                attacker_model=attacker_model,
                batch_x=batch_x,
                batch_y=batch_y,
                maskBoolNot=masks,
                alpha=0.01,
                mask_mode="partial",
                batch_id=2,
                save_dir=directory,
            )
            self.assertTrue(Path(directory, "ilrg_metrics.json").exists())

        for actual, expected in zip(trainer.model.state_dict().values(), attacker_model):
            np.testing.assert_allclose(actual.detach().numpy(), expected)
        self.assertTrue(metrics.attack_available)
        self.assertEqual(metrics.batch_size, 4)
        self.assertEqual(metrics.true_counts, [1, 3])
        self.assertEqual(sum(metrics.predicted_counts), 4)
        self.assertEqual(metrics.additional_info["batch_id"], 2)
        self.assertEqual(metrics.additional_info["usable_bias_equations"], 1)
        self.assertAlmostEqual(metrics.additional_info["final_bias_visible_fraction"], 0.5)
        self.assertLess(metrics.additional_info["final_weight_visible_fraction"], 1.0)
        self.assertGreaterEqual(metrics.normalized_count_l1, 0.0)
        self.assertLessEqual(metrics.normalized_count_l1, 1.0)

    def test_ilrg_strict_mode_reports_masked_head_as_unavailable(self):
        trainer = TinyTrainer()
        model = [
            tensor.detach().numpy().copy()
            for tensor in trainer.model.state_dict().values()
        ]
        masks = [np.ones_like(value, dtype=np.uint8) for value in model]
        masks[0][0, 0] = 0
        batch_x = np.full((2, 4, 4, 1), 0.5, dtype=np.float32)
        batch_y = np.eye(2, dtype=np.float32)[[0, 1]]

        with tempfile.TemporaryDirectory() as directory:
            metrics = perform_iLRG_attack(
                trainer,
                client_model=model,
                attacker_model=model,
                batch_x=batch_x,
                batch_y=batch_y,
                maskBoolNot=masks,
                mask_mode="strict",
                save_dir=directory,
            )

        self.assertFalse(metrics.attack_available)
        self.assertIn("complete final-layer gradient", metrics.unavailable_reason)
        self.assertEqual(metrics.predicted_counts, [])
        self.assertIsNone(metrics.label_number_accuracy)

    def test_dlg_optimizes_label_when_analytic_label_is_partly_hidden(self):
        trainer = TinyTrainer()
        model = [
            tensor.detach().numpy().copy()
            for tensor in trainer.model.state_dict().values()
        ]
        masks = [np.ones_like(value, dtype=np.uint8) for value in model]
        masks[0][0, 0] = 0
        masks[-1][0] = 0
        sample_x = np.full((1, 4, 4, 1), 0.5, dtype=np.float32)
        sample_y = np.eye(2, dtype=np.float32)[[1]]

        with tempfile.TemporaryDirectory() as directory:
            metrics = perform_DLG_attack(
                trainer,
                client_model=model,
                attacker_model=model,
                sample_x=sample_x,
                sample_y=sample_y,
                maskBoolNot=masks,
                num_iterations=2,
                num_restarts=1,
                early_stopping_patience=0,
                save_dir=directory,
            )
        self.assertFalse(metrics.additional_info["label_inference_available"])
        self.assertEqual(
            metrics.additional_info["label_inference_method"],
            "optimized_soft_label",
        )

    def test_dlg_known_label_still_reports_idlg_result(self):
        trainer = TinyTrainer()
        model = [
            tensor.detach().numpy().copy()
            for tensor in trainer.model.state_dict().values()
        ]
        masks = [np.ones_like(value, dtype=np.uint8) for value in model]
        masks[0][0, 0] = 0
        masks[-1][0] = 0
        sample_x = np.full((1, 4, 4, 1), 0.5, dtype=np.float32)
        sample_y = np.eye(2, dtype=np.float32)[[1]]

        with tempfile.TemporaryDirectory() as directory:
            metrics = perform_DLG_attack(
                trainer,
                client_model=model,
                attacker_model=model,
                sample_x=sample_x,
                sample_y=sample_y,
                maskBoolNot=masks,
                num_iterations=2,
                num_restarts=1,
                early_stopping_patience=0,
                known_label=True,
                save_dir=directory,
            )
        info = metrics.additional_info
        self.assertEqual(info["label_inference_method"], "known_label")
        self.assertTrue(info["known_label"])
        self.assertFalse(info["label_inference_available"])
        self.assertIsNone(info["idlg_inferred_label"])
        self.assertFalse(info["idlg_label_inference_success"])
        self.assertFalse(metrics.idlg_label_inference_success)
        self.assertEqual(info["inferred_label"], 1)
        self.assertEqual(metrics.label_accuracy, 1.0)

    def test_ig_uses_client_target_attacker_state_and_idlg(self):
        trainer = TinyTrainer()
        client_model = [
            tensor.detach().numpy().copy()
            for tensor in trainer.model.state_dict().values()
        ]
        attacker_model = [value.copy() for value in client_model]
        attacker_model[0] = attacker_model[0] + 0.05
        masks = [np.ones_like(value, dtype=np.uint8) for value in attacker_model]
        masks[0].reshape(-1)[::2] = 0
        sample_x = np.linspace(0, 1, 16, dtype=np.float32).reshape(1, 4, 4, 1)
        sample_y = np.eye(2, dtype=np.float32)[[1]]

        with tempfile.TemporaryDirectory() as directory:
            metrics = perform_IG_attack(
                trainer,
                client_model=client_model,
                attacker_model=attacker_model,
                sample_x=sample_x,
                sample_y=sample_y,
                maskBoolNot=masks,
                num_iterations=3,
                num_restarts=2,
                learning_rate=0.1,
                attack_seed=123,
                sample_id=4,
                save_dir=directory,
            )
            self.assertTrue(Path(directory, "ig_metrics.json").exists())
            self.assertTrue(Path(directory, "ig_loss_curve.png").exists())
            self.assertTrue(
                Path(directory, "ig_reconstruction_comparison.png").exists()
            )
        for actual, expected in zip(trainer.model.state_dict().values(), attacker_model):
            np.testing.assert_allclose(actual.detach().numpy(), expected)
        self.assertTrue(all(
            parameter.grad is None for parameter in trainer.model.parameters()
        ))
        info = metrics.additional_info
        self.assertEqual(metrics.attack_name, "IG_ElementWise")
        self.assertEqual(info["label_inference_method"], "analytic_iDLG")
        self.assertTrue(info["idlg_label_inference_success"])
        self.assertEqual(info["optimizer"], "signed_adam")
        self.assertEqual(info["objective"], "global_cosine")
        self.assertEqual(info["initialization"], "gaussian")
        self.assertEqual(info["lr_decay_milestones"], [1, 2])
        self.assertAlmostEqual(
            info["visible_gradient_fraction"],
            sum(np.count_nonzero(mask) for mask in masks) /
            sum(mask.size for mask in masks),
        )

    def test_ig_label_fallback_and_known_label_override(self):
        trainer = TinyTrainer()
        model = [
            tensor.detach().numpy().copy()
            for tensor in trainer.model.state_dict().values()
        ]
        masks = [np.ones_like(value, dtype=np.uint8) for value in model]
        masks[0][0, 0] = 0
        masks[-1][0] = 0
        sample_x = np.full((1, 4, 4, 1), 0.5, dtype=np.float32)
        sample_y = np.eye(2, dtype=np.float32)[[1]]

        with tempfile.TemporaryDirectory() as directory:
            inferred_metrics = perform_IG_attack(
                trainer,
                client_model=model,
                attacker_model=model,
                sample_x=sample_x,
                sample_y=sample_y,
                maskBoolNot=masks,
                num_iterations=2,
                num_restarts=1,
                save_dir=str(Path(directory, "inferred")),
            )
            known_metrics = perform_IG_attack(
                trainer,
                client_model=model,
                attacker_model=model,
                sample_x=sample_x,
                sample_y=sample_y,
                maskBoolNot=masks,
                num_iterations=2,
                num_restarts=1,
                known_label=True,
                save_dir=str(Path(directory, "known")),
            )

        inferred_info = inferred_metrics.additional_info
        self.assertFalse(inferred_info["label_inference_available"])
        self.assertEqual(
            inferred_info["label_inference_method"], "optimized_soft_label"
        )
        known_info = known_metrics.additional_info
        self.assertEqual(known_info["label_inference_method"], "known_label")
        self.assertTrue(known_info["known_label"])
        self.assertFalse(known_info["label_inference_available"])
        self.assertFalse(known_metrics.idlg_label_inference_success)
        self.assertEqual(known_info["inferred_label"], 1)
        self.assertEqual(known_metrics.label_accuracy, 1.0)

    def test_ig_early_stopping_stops_stale_optimization(self):
        trainer = TinyTrainer()
        model = [
            tensor.detach().numpy().copy()
            for tensor in trainer.model.state_dict().values()
        ]
        masks = [np.ones_like(value, dtype=np.uint8) for value in model]
        sample_x = np.full((1, 4, 4, 1), 0.5, dtype=np.float32)
        sample_y = np.eye(2, dtype=np.float32)[[1]]

        with tempfile.TemporaryDirectory() as directory:
            metrics = perform_IG_attack(
                trainer,
                client_model=model,
                attacker_model=model,
                sample_x=sample_x,
                sample_y=sample_y,
                maskBoolNot=masks,
                num_iterations=10,
                num_restarts=1,
                early_stopping_patience=1,
                early_stopping_delta=1e9,
                save_dir=directory,
            )

        self.assertEqual(metrics.additional_info["iterations_run"], 2)
        self.assertEqual(metrics.additional_info["early_stopping_patience"], 1)

    def test_excel_schemas_and_global_accuracy_combine(self):
        cfg = DummyConfig()
        cfg.num_clients = 1
        cfg.rounds = 2
        cfg.currentEdge = 0

        with tempfile.TemporaryDirectory() as directory:
            base = str(Path(directory, "round_%d"))
            for round_index, accuracy in enumerate((51.25, 53.75)):
                cfg.currentRound = round_index
                cfg.excelAddr = base % round_index
                excelHelper.create(cfg)
                excelHelper.update(cfg, {"global_test_acc": accuracy})
                excelHelper.update(cfg, {"Attack_MIA": {
                    "round": round_index,
                    "client": 0,
                    "loss_AUC": 0.6 + round_index * 0.1,
                    "modified_entropy_AUC": 0.61 + round_index * 0.1,
                }})
                excelHelper.update(cfg, {"Attack_DLG": {
                    "round": round_index,
                    "client": 0,
                    "sample_id": 3,
                    "ssim": 0.25,
                    "best_restart_seed": 123,
                }})
                excelHelper.update(cfg, {"Attack_DLG_Summary": {
                    "round": round_index,
                    "client": 0,
                    "num_samples": 10,
                    "ssim_mean": 0.25,
                    "ssim_std": 0.02,
                }})
                excelHelper.update(cfg, {"Attack_IG": {
                    "round": round_index,
                    "client": 0,
                    "sample_id": 3,
                    "ssim": 0.3,
                    "optimizer": "signed_adam",
                }})
                excelHelper.update(cfg, {"Attack_IG_Summary": {
                    "round": round_index,
                    "client": 0,
                    "num_samples": 10,
                    "ssim_mean": 0.3,
                    "ssim_std": 0.01,
                }})
                excelHelper.update(cfg, {"Attack_iLRG": {
                    "round": round_index,
                    "client": 0,
                    "batch_id": 2,
                    "batch_size": 8,
                    "predicted_counts": "[3,5]",
                    "label_number_accuracy": 0.5,
                    "final_bias_visible_fraction": 0.5,
                }})
                excelHelper.update(cfg, {"Attack_iLRG_Summary": {
                    "round": round_index,
                    "client": 0,
                    "num_batches": 3,
                    "num_available": 2,
                    "availability_rate": 2 / 3,
                    "label_number_accuracy_mean": 0.5,
                }})

            target = str(Path(directory, "combined"))
            excelHelper.combineExcels(
                cfg,
                baseAddr=base,
                targetSaveAddr=target,
                sheetList=[
                    "global_test_acc", "Attack_MIA", "Attack_iLRG",
                    "Attack_iLRG_Summary", "Attack_DLG",
                    "Attack_DLG_Summary", "Attack_IG", "Attack_IG_Summary",
                ],
            )
            workbook = load_workbook(target + ".xlsx", read_only=True)
            global_sheet = workbook["global_test_acc"]
            self.assertEqual(global_sheet["A1"].value, "Round")
            self.assertEqual(global_sheet["B1"].value, "Global Test Accuracy")
            self.assertEqual(global_sheet["A2"].value, 0)
            self.assertEqual(global_sheet["B3"].value, 53.75)
            mia_sheet = workbook["Attack_MIA"]
            self.assertIn("modified_entropy_AUC", [cell.value for cell in mia_sheet[1]])
            self.assertNotIn("confidence_AUC", [cell.value for cell in mia_sheet[1]])
            dlg_sheet = workbook["Attack_DLG"]
            self.assertIn("best_restart_seed", [cell.value for cell in dlg_sheet[1]])
            summary_sheet = workbook["Attack_DLG_Summary"]
            self.assertEqual(summary_sheet["C2"].value, 10)
            ig_sheet = workbook["Attack_IG"]
            self.assertIn("lr_decay_milestones", [cell.value for cell in ig_sheet[1]])
            ig_summary_sheet = workbook["Attack_IG_Summary"]
            self.assertEqual(ig_summary_sheet["C2"].value, 10)
            ilrg_sheet = workbook["Attack_iLRG"]
            self.assertIn("predicted_counts", [cell.value for cell in ilrg_sheet[1]])
            ilrg_summary_sheet = workbook["Attack_iLRG_Summary"]
            self.assertEqual(ilrg_summary_sheet["C2"].value, 3)
            workbook.close()


if __name__ == "__main__":
    unittest.main()
