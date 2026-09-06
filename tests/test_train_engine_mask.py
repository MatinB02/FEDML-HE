import unittest
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn


if "skimage.metrics" not in sys.modules:
    skimage_module = types.ModuleType("skimage")
    skimage_metrics_module = types.ModuleType("skimage.metrics")
    skimage_metrics_module.structural_similarity = lambda *_args, **_kwargs: 0.0
    sys.modules["skimage"] = skimage_module
    sys.modules["skimage.metrics"] = skimage_metrics_module

from Codes.trainEngine import TrainEngine
from Codes.functions import maskFilter
from Codes.functions_mainAlg import build_consensus_mask, indexMask_to_BinaryMask


class TrainEngineMaskTests(unittest.TestCase):
    def test_dataset_conversion_caches_contiguous_nchw_and_class_indices(self):
        images = np.arange(5 * 4 * 3 * 3, dtype=np.float32).reshape(5, 4, 3, 3)
        labels = np.eye(3, dtype=np.float32)[[0, 1, 2, 1, 0]]

        image_tensor, label_tensor = TrainEngine.prepare_data_tensors(
            images.astype(object), labels.astype(object)
        )

        self.assertEqual(image_tensor.shape, (5, 3, 4, 3))
        self.assertEqual(image_tensor.dtype, torch.float32)
        self.assertTrue(image_tensor.is_contiguous())
        self.assertEqual(label_tensor.dtype, torch.long)
        torch.testing.assert_close(
            label_tensor, torch.tensor([0, 1, 2, 1, 0])
        )

        trainer = TrainEngine.__new__(TrainEngine)
        trainer.device = torch.device('cpu')
        cached_images, cached_labels = trainer.prepare_device_tensors(
            images, labels
        )
        self.assertEqual(cached_images.device, trainer.device)
        self.assertEqual(cached_labels.device, trainer.device)

    def test_microbatch_gradients_match_full_batch_mean(self):
        torch.manual_seed(7)
        trainer = TrainEngine.__new__(TrainEngine)
        trainer.device = torch.device('cpu')
        trainer.model = nn.Linear(4, 3)
        trainer.trainableVarsList = list(trainer.model.named_parameters())
        trainer._criterion = nn.CrossEntropyLoss()
        trainer.maskGradientBatchSize = 2
        trainer.batchSize = 2
        trainer.loss = None

        images = np.arange(20, dtype=np.float32).reshape(5, 4) / 10.0
        labels = np.array([0, 2, 1, 2, 0], dtype=np.int64)
        x_t = torch.from_numpy(images)
        y_t = torch.from_numpy(labels)
        expected = torch.autograd.grad(
            trainer._criterion(trainer.model(x_t), y_t),
            list(trainer.model.parameters()),
        )

        actual = trainer.compute_gradients(images, labels)

        for actual_gradient, expected_gradient in zip(actual, expected):
            np.testing.assert_allclose(
                actual_gradient,
                expected_gradient.detach().numpy(),
                rtol=1e-5,
                atol=1e-6,
            )
        self.assertTrue(
            all(parameter.grad is None for parameter in trainer.model.parameters())
        )

    def test_evaluation_uses_inference_mode_and_eval_batch_size(self):
        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []
                self.inference_modes = []

            def forward(self, inputs):
                self.batch_sizes.append(len(inputs))
                self.inference_modes.append(torch.is_inference_mode_enabled())
                return torch.stack((-inputs[:, 0], inputs[:, 0]), dim=1)

        trainer = TrainEngine.__new__(TrainEngine)
        trainer.device = torch.device('cpu')
        trainer.model = RecordingModel()
        trainer.evalBatchSize = 3
        images = torch.tensor([[-1.0], [1.0], [-2.0], [2.0], [3.0], [-3.0], [4.0]])
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 1])

        accuracy = trainer.evaluate(images, labels)

        self.assertEqual(accuracy, 100.0)
        self.assertEqual(trainer.model.batch_sizes, [3, 3, 1])
        self.assertTrue(all(trainer.model.inference_modes))

    def test_training_evaluates_only_after_final_local_epoch(self):
        trainer = TrainEngine.__new__(TrainEngine)
        trainer.device = torch.device('cpu')
        trainer.model = nn.Linear(4, 3)
        trainer._criterion = nn.CrossEntropyLoss()
        trainer._optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.trainableVarsList = list(trainer.model.named_parameters())
        trainer.batchSize = 2
        trainer.evalBatchSize = 4
        trainer.classCount = 3
        trainer.loss = None
        trainer.cfg = SimpleNamespace(
            local_epochs=3,
            currentEpoch=0,
            currentEdge=0,
            currentRound=0,
        )
        images = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10.0
        labels = torch.tensor([0, 1, 2, 1, 0])
        trainer.test_tensors = (images, labels)

        with patch('Codes.trainEngine.which_gpu'), patch(
                'Codes.trainEngine.excelHelper.update'
        ) as metrics_update, patch.object(
                trainer, 'evaluate', wraps=trainer.evaluate
        ) as evaluate:
            trainer.train((images, labels))

        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(metrics_update.call_count, 2)
        self.assertEqual(trainer.epoch_number, 3)

    def test_mask_filter_selects_exact_global_top_k_with_stable_ties(self):
        sensitivities = [
            np.array([1.0, 9.0, 5.0]),
            np.array([9.0, 2.0, 9.0, 3.0, 5.0]),
        ]

        selected = maskFilter(sensitivities, 0.5)

        # Four of eight values, descending by score. Equal scores use the
        # lower flat index, so index 7 is excluded at the threshold.
        np.testing.assert_array_equal(selected, np.array([1, 3, 5, 2]))

    def test_mask_filter_handles_zero_and_full_ratios(self):
        sensitivities = [np.array([3.0, 1.0]), np.array([2.0])]
        np.testing.assert_array_equal(
            maskFilter(sensitivities, 0.0), np.empty(0, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            maskFilter(sensitivities, 1.0), np.arange(3, dtype=np.int64)
        )

    def test_trainable_indices_map_to_state_and_always_include_bn_buffers(self):
        model = nn.Sequential(
            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
        )
        trainer = TrainEngine.__new__(TrainEngine)
        trainer.model = model
        trainer.trainableVarsList = list(model.named_parameters())
        trainer._build_flat_index_maps()

        mapped = trainer.mapTrainableToAllVars(np.array([0, 3]))
        expected = np.sort(np.concatenate((
            trainer.trainable_indices_in_all_flat[[0, 3]],
            trainer.non_trainable_indices,
        )))
        np.testing.assert_array_equal(mapped, expected)
        self.assertTrue(set(trainer.non_trainable_indices).issubset(mapped))

        no_trainable = trainer.mapTrainableToAllVars(
            np.empty(0, dtype=np.int64)
        )
        np.testing.assert_array_equal(no_trainable, trainer.non_trainable_indices)
        self.assertIsNone(trainer.mapTrainableToAllVars(
            np.arange(trainer.trainable_indices_in_all_flat.size)
        ))

    def test_fedml_he_sensitivity_is_mean_absolute_per_sample_gradient(self):
        torch.manual_seed(5)
        trainer = TrainEngine.__new__(TrainEngine)
        trainer.device = torch.device('cpu')
        trainer.model = nn.Linear(2, 2)
        trainer.trainableVarsList = list(trainer.model.named_parameters())
        trainer._criterion = nn.CrossEntropyLoss()
        images = torch.tensor([[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0]])
        labels = torch.tensor([0, 1, 0])

        expected = [torch.zeros_like(parameter) for parameter in trainer.model.parameters()]
        for sample_index in range(len(labels)):
            loss = trainer._criterion(
                trainer.model(images[sample_index:sample_index + 1]),
                labels[sample_index:sample_index + 1],
            )
            gradients = torch.autograd.grad(loss, tuple(trainer.model.parameters()))
            for accumulator, gradient in zip(expected, gradients):
                accumulator.add_(gradient.abs())
        expected = [value / len(labels) for value in expected]

        sensitivity = trainer.proposeMask((images, labels))

        for actual, expected_value in zip(sensitivity, expected):
            np.testing.assert_allclose(
                actual, expected_value.detach().numpy(), rtol=1e-6, atol=1e-7
            )
        self.assertTrue(all(
            parameter.grad is None for parameter in trainer.model.parameters()
        ))

    def test_full_encryption_sentinel_builds_all_one_masks(self):
        global_mask = build_consensus_mask([None, None])
        model_structure = [
            {'shape': (2, 2), 'size': 4},
            {'shape': (3,), 'size': 3},
        ]

        mask, inverse_mask = indexMask_to_BinaryMask(
            global_mask, model_structure
        )

        self.assertIsNone(global_mask)
        for layer_mask, layer_inverse in zip(mask, inverse_mask):
            np.testing.assert_array_equal(layer_mask, np.ones_like(layer_mask))
            np.testing.assert_array_equal(layer_inverse, np.zeros_like(layer_inverse))

    def test_consensus_rejects_mixed_full_and_partial_proposals(self):
        with self.assertRaisesRegex(AssertionError, "cannot mix"):
            build_consensus_mask([None, np.array([0], dtype=np.int32)])


if __name__ == "__main__":
    unittest.main()
