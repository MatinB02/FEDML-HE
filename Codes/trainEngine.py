# -*- coding: utf-8 -*-
# Codes/trainEngine.py  –  migrated from TensorFlow 1.x to PyTorch

import inspect
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from Codes import excelHelper
from Codes.functions import which_gpu
from Codes.enums import Config
# from Codes.attacks import ComprehensivePrivacyEvaluation


# ─────────────────────────────────────────────────────────────
#  Helper: resolve the device once, shared by the whole file
# ─────────────────────────────────────────────────────────────
def _get_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


class TrainEngine:
    """
    Drop-in replacement for the TF1 TrainEngine.

    Key differences from the TF1 version
    ──────────────────────────────────────
    • No static graph, no Session.  The model is a plain nn.Module.
    • setAllWeights / getAllWeights use PyTorch state_dict mechanics,
      so there is NO graph-rebuild overhead across rounds.
    • resetOptimizer is trivially correct: we simply re-instantiate
      the optimizer object at the start of every train() call.
    • allVarsList  ↔  list(model.state_dict().keys())   (strings)
      trainableVarsList ↔  parameters that require_grad
    • self.sess, self.X, self.Y, self.learningRatePH, self.loss,
      self.trainableVarsList, self.modelLogits  are kept as attributes
      so that call-sites in ProjectControl_Loop.py that reference them
      continue to work with minimal edits.  Where a TF symbol is
      referenced externally (e.g. tf.gradients / sess.run), the
      PyTorch equivalent is provided through small shim methods.
    """

    def __init__(
        self,
        cfg: Config,
        testData=None,
        globalData=None,
    ):
        self.trainData_last = None
        self.cfg = cfg
        self.temperature = cfg.temperature
        self.trainStrategy = cfg.trainStrategy
        self.inputShape = cfg.inputShape
        self.classCount = cfg.classNum
        self.globalData = globalData
        self.batchSize = cfg.local_batch_size
        self.evalBatchSize = getattr(cfg, "eval_batch_size", 1024)
        self.maskGradientBatchSize = getattr(
            cfg, "mask_gradient_batch_size", self.batchSize
        )
        self.epochCount = cfg.local_epochs

        # ── device ──────────────────────────────────────────────────────
        self.device = _get_device(getattr(cfg, "gpu_id", 0))

        self.test_tensors = self.prepare_device_tensors(
            testData[0], testData[1]
        )

        # ── build the model ─────────────────────────────────────────────
        # cfg.model is a ModelsReach descriptor; we call .build() to get
        # the actual nn.Module.  Adjust if your descriptor uses a
        # different factory method.
        # self.model: nn.Module = cfg.model.build(
        #     input_shape=self.inputShape,
        #     num_classes=self.classCount,
        # ).to(self.device)

        # Models share the core arguments, but only architectures with a
        # fixed-size classifier (for example LeNet5) need the input height and
        # width.  Pass each model only the dataset arguments it supports.
        in_channels = self.inputShape[-1]  # NHWC -> channels
        model_kwargs = {
            "num_classes": self.classCount,
            "in_channels": in_channels,
            "input_height": self.inputShape[0],
            "input_width": self.inputShape[1],
        }
        model_signature = inspect.signature(cfg.model)
        accepts_arbitrary_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in model_signature.parameters.values()
        )
        if not accepts_arbitrary_kwargs:
            model_kwargs = {
                name: value
                for name, value in model_kwargs.items()
                if name in model_signature.parameters
            }

        self.model = cfg.model(**model_kwargs).to(self.device)

        self.learningRate = cfg.model.learning_rate_local   # float

        # ── loss function ────────────────────────────────────────────────
        # CrossEntropyLoss expects class-index labels (Long), not one-hot.
        # If your labels ARE one-hot, see _to_class_indices() below.
        self._criterion = nn.CrossEntropyLoss()

        # ── TF1 compatibility shims ───────────────────────────────────────
        # These attributes are referenced in ProjectControl_Loop.py.
        # They are kept as no-ops / aliases so the call-sites compile.
        self.sess = None        # no Session in PyTorch
        self.X = None           # no placeholder
        self.Y = None           # no placeholder
        self.learningRatePH = self.learningRate  # plain float
        self.loss = None        # will be set during forward pass (see _compute_loss)

        # trainableVarsList  → list of (name, Parameter) tuples,
        # mirroring the TF1 convention of a list you can iterate.
        self.trainableVarsList = list(self.model.named_parameters())

        # allVarsList includes BN running stats (non-trainable in PyTorch sense)
        self.allVarsList = list(self.model.state_dict().keys())

        # The model structure is fixed for the lifetime of a TrainEngine, so
        # compute these flat-index mappings once instead of once per client.
        self._build_flat_index_maps()

        # timing attributes
        self.time_train = None
        self.time_getWeights = None
        self.time_getLogits = None
        self.time_getDerivative = None

        self.epoch_number = 0

    # ──────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_flat_index_maps(self) -> None:
        """Cache state-dict indices for trainable and non-trainable values."""
        trainable_names = {name for name, _ in self.trainableVarsList}
        trainable_ranges = []
        non_trainable_ranges = []
        current_pos = 0

        for name, tensor in self.model.state_dict().items():
            next_pos = current_pos + tensor.numel()
            indices = np.arange(current_pos, next_pos, dtype=np.int32)
            if name in trainable_names:
                trainable_ranges.append(indices)
            else:
                non_trainable_ranges.append(indices)
            current_pos = next_pos

        self.trainable_indices_in_all_flat = (
            np.concatenate(trainable_ranges)
            if trainable_ranges
            else np.empty(0, dtype=np.int32)
        )
        self.non_trainable_indices = (
            np.concatenate(non_trainable_ranges)
            if non_trainable_ranges
            else np.empty(0, dtype=np.int32)
        )
    @staticmethod
    def _to_class_indices(Y) -> torch.Tensor:
        """
        Accept either one-hot (N, C) float arrays or class-index (N,) arrays
        and return a Long tensor of class indices.
        """
        if isinstance(Y, torch.Tensor):
            return Y.argmax(dim=1).long() if Y.ndim == 2 else Y.long()
        labels = np.asarray(Y)
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        return torch.from_numpy(labels.astype(np.int64, copy=False)).long()

    @classmethod
    def prepare_data_tensors(cls, X, Y, pin_memory=False):
        """Convert a dataset to reusable contiguous NCHW tensors."""
        if isinstance(X, torch.Tensor):
            x_t = X.detach()
            if x_t.dtype != torch.float32:
                x_t = x_t.float()
        else:
            images = np.asarray(X, dtype=np.float32)
            x_t = torch.from_numpy(images)

        if x_t.ndim == 4 and x_t.shape[-1] in (1, 3):
            x_t = x_t.permute(0, 3, 1, 2)
        x_t = x_t.contiguous()
        y_t = cls._to_class_indices(Y).contiguous()

        should_pin = pin_memory and torch.cuda.is_available()
        if should_pin and x_t.device.type == "cpu" and not x_t.is_pinned():
            x_t = x_t.pin_memory()
        if should_pin and y_t.device.type == "cpu" and not y_t.is_pinned():
            y_t = y_t.pin_memory()
        return x_t, y_t

    def prepare_device_tensors(self, X, Y):
        """Create reusable NCHW tensors resident on the training device."""
        tensors = self.prepare_data_tensors(X, Y, pin_memory=True)
        return tuple(
            tensor.to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )
            for tensor in tensors
        )

    def _numpy_to_tensor(self, X, Y):
        x_t, y_t = self.prepare_data_tensors(X, Y)
        non_blocking = self.device.type == "cuda"
        return (
            x_t.to(self.device, non_blocking=non_blocking),
            y_t.to(self.device, non_blocking=non_blocking),
        )

    def _compute_loss(self, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """Forward pass → returns scalar loss.  Also caches self.loss."""
        logits = self.model(x_t)
        loss = self._criterion(logits, y_t)
        self.loss = loss          # keep reference for external callers
        return loss

    # ──────────────────────────────────────────────────────────────────────
    #  Optimizer reset  (the core problem that motivated the migration)
    # ──────────────────────────────────────────────────────────────────────
    def resetOptimizer(self):
        """
        Re-instantiate the optimizer from scratch.
        In PyTorch this is trivial and perfectly correct – no slot variables
        survive to contaminate the next client's training.
        """
        self._optimizer = optim.Adam(
            self.model.parameters(), lr=self.learningRate
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Weight get / set  (replaces TF1 assign_ops + weight_placeholders)
    # ──────────────────────────────────────────────────────────────────────
    def getAllWeights(self, saveWeights=False, saveTiming=True) -> list:
        """
        Return all model state (including BN running mean/var) as a list of
        numpy arrays, in the same order as self.allVarsList (state_dict keys).
        Mirrors the TF1 signature exactly.
        """
        t0 = time.perf_counter()

        weights = [
            v.detach().cpu().numpy()
            for v in self.model.state_dict().values()
        ]

        if saveTiming:
            self.time_getWeights = time.perf_counter() - t0

        if saveWeights and hasattr(self.cfg, "saveAddr") and self.cfg.saveAddr is not None:
            np.savez(self.cfg.saveAddr + "_weights.npz", *weights)

        return weights

    def getBN(self) -> list:
        """
        Return the non-trainable weights (e.g. BatchNorm running mean/var and
        num_batches_tracked) as a list of numpy arrays, in state_dict order.
        """
        trainable_names = {name for name, _ in self.trainableVarsList}
        return [
            v.detach().cpu().numpy()
            for name, v in self.model.state_dict().items()
            if name not in trainable_names
        ]


    def getTrainableWeights(self, saveWeights=False, saveTiming=True) -> list:
        """
        Return only trainable (requires_grad) parameters as a list of numpy
        arrays, in the same order as self.trainableVarsList.
        """
        t0 = time.perf_counter()

        weights = [
            p.detach().cpu().numpy()
            for _, p in self.trainableVarsList
        ]

        if saveTiming:
            self.time_getWeights = time.perf_counter() - t0

        if saveWeights and hasattr(self.cfg, "saveAddr") and self.cfg.saveAddr is not None:
            np.savez(self.cfg.saveAddr + '_trainableWeights.npz', *weights)

        return weights

    def setAllWeights(self, weights: list):
        """
        Load a list of numpy arrays (one per state_dict entry) back into the
        model.  Also resets the optimizer, exactly as the TF1 version did.

        This replaces the TF1 assign_ops + weight_placeholders machinery.
        """
        self.resetOptimizer()               # 1. fresh optimizer (no stale momentum)

        # 2. load weights
        state_keys = list(self.model.state_dict().keys())
        assert len(state_keys) == len(weights), (
            f"Mismatch: {len(state_keys)} state_dict entries vs "
            f"{len(weights)} supplied arrays"
        )
        new_state = OrderedDict(
            {k: torch.tensor(v, dtype=self.model.state_dict()[k].dtype)
             for k, v in zip(state_keys, weights)}
        )
        self.model.load_state_dict(new_state)

    def setBN(self, weights: list):
        """
        Replace the non-trainable weights (e.g. BatchNorm running mean/var and
        num_batches_tracked) of the model with the given list of numpy arrays,
        matched by position to getBN()'s output order.
        """
        trainable_names = {name for name, _ in self.trainableVarsList}
        state_dict = self.model.state_dict()
        non_trainable_names = [name for name in state_dict.keys() if name not in trainable_names]

        assert len(non_trainable_names) == len(weights), (
            f"Mismatch: {len(non_trainable_names)} non-trainable entries vs "
            f"{len(weights)} supplied arrays"
        )

        for name, w in zip(non_trainable_names, weights):
            state_dict[name] = torch.tensor(
                w, dtype=state_dict[name].dtype, device=state_dict[name].device
            )
        self.model.load_state_dict(state_dict)

    # ──────────────────────────────────────────────────────────────────────
    #  FEDML-HE mask index mapping
    # ──────────────────────────────────────────────────────────────────────

    def mapTrainableToAllVars(self, trainable_indices):
        """Map flat trainable indices into the complete state-dict index space.

        Every non-trainable state value, including BatchNorm running statistics
        and counters, is always included. ``None`` is the transport pipeline's
        existing full-encryption sentinel.
        """
        indices = np.asarray(trainable_indices)
        if indices.ndim != 1:
            raise ValueError("trainable_indices must be one-dimensional")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("trainable_indices must contain integers")

        indices = indices.astype(np.int64, copy=False)
        trainable_count = self.trainable_indices_in_all_flat.size
        if np.any(indices < 0) or np.any(indices >= trainable_count):
            raise IndexError(
                f"trainable index outside valid range [0, {trainable_count})"
            )
        if np.unique(indices).size != indices.size:
            raise ValueError("trainable_indices must not contain duplicates")

        mapped_trainable = self.trainable_indices_in_all_flat[indices]
        encrypted_indices = np.concatenate((
            mapped_trainable.astype(np.int64, copy=False),
            self.non_trainable_indices.astype(np.int64, copy=False),
        ))
        encrypted_indices.sort()

        total_state_values = trainable_count + self.non_trainable_indices.size
        if encrypted_indices.size == total_state_values:
            return None
        return encrypted_indices

    # ──────────────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────────────

    def train(self, trainData):
        which_gpu()
        X_train, Y_train = trainData

        # Reuse the cached NCHW tensors for training and mask proposal.
        X_train, Y_train = self.prepare_data_tensors(X_train, Y_train)
        self.trainData_last = (X_train, Y_train)

        trainAccuracy, testAccuracy = 0.0, 0.0
        self.epoch_number = 0
        self.time_train = 0

        print("^" * 60)
        print("    ", end="")
        loss_localSum = 0

        while self.epoch_number < self.cfg.local_epochs:
            self.cfg.currentEpoch = self.epoch_number
            examplesNum = len(X_train)
            loss_localSum = 0

            self.model.train()
            for start_idx in range(0, examplesNum, self.batchSize):
                end_idx = min(start_idx + self.batchSize, examplesNum)
                batch_x = X_train[start_idx:end_idx]
                batch_y = Y_train[start_idx:end_idx]

                x_t, y_t = self._numpy_to_tensor(batch_x, batch_y)

                t_start = time.perf_counter()
                self._optimizer.zero_grad()
                loss = self._compute_loss(x_t, y_t)
                loss.backward()
                self._optimizer.step()
                self.time_train += time.perf_counter() - t_start

                loss_localSum += loss.item()

            # ── evaluate ──────────────────────────────────────────────────
            if self.epoch_number == self.cfg.local_epochs - 1:
                trainAccuracy = self.evaluate(X_train, Y_train)
                testAccuracy = self.evaluate_test()
                print(
                    f"\r  Client {self.cfg.currentEdge}, "
                    f"Iter {self.cfg.currentRound}, "
                    f"Epoch {self.epoch_number} >> "
                    f"Train Acc: {trainAccuracy:2.2f},   "
                    f"Test Acc: {testAccuracy:2.2f}",
                    end=""
                )
                excelHelper.update(cfg=self.cfg, dataDic={
                    "Test Accuracy": testAccuracy,
                    "Train Accuracy": trainAccuracy,
                    "Loss": loss_localSum,
                    "Test Accuracy MAX": testAccuracy,
                    "Train Accuracy MAX": trainAccuracy,
                })

            # One local epoch is exactly one complete pass over this client's
            # dataset, independent of train or test accuracy.
            self.epoch_number += 1

        excelHelper.update(cfg=self.cfg, dataDic={"time_train": self.time_train})
        print(
            "\n  Maximum Accuracy >>  Train: %2.1f ,Test: %2.1f"
            % (trainAccuracy, testAccuracy)
        )
        return trainAccuracy, testAccuracy, loss_localSum

    # ──────────────────────────────────────────────────────────────────────
    #  Evaluation
    # ──────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def evaluate(self, X_data, Y_data) -> float:
        self.model.eval()
        examplesNum = len(X_data)
        totalCorrect = 0

        for offset in range(0, examplesNum, self.evalBatchSize):
            end = min(offset + self.evalBatchSize, examplesNum)
            x_t, y_t = self._numpy_to_tensor(
                X_data[offset:end], Y_data[offset:end]
            )
            logits = self.model(x_t)
            preds = logits.argmax(dim=1)
            totalCorrect += (preds == y_t).sum().item()

        return 100.0 * totalCorrect / examplesNum

    def evaluate_test(self) -> float:
        return self.evaluate(*self.test_tensors)

    # ──────────────────────────────────────────────────────────────────────
    #  Gradient computation  (replaces tf.gradients + sess.run)
    # ──────────────────────────────────────────────────────────────────────

    def compute_gradients(
        self,
        X,
        Y,
    ) -> list:
        """
        Compute per-parameter gradients of mean cross-entropy over (X, Y),
        using bounded microbatches to limit activation memory.

        Returns a list of numpy arrays in the same order as
        self.trainableVarsList – directly usable by the existing attack code.

        This replaces the TF1 pattern:
            grads = tf.gradients(trainer.loss, trainer.trainableVarsList)
            gradients = trainer.sess.run(grads, feed_dict={...})
        """
        self.model.eval()
        X, Y = self.prepare_data_tensors(X, Y)
        total_examples = len(X)
        if total_examples == 0:
            raise ValueError("Cannot compute gradients for an empty dataset")

        parameters = [param for _, param in self.trainableVarsList]
        accumulated = [torch.zeros_like(param) for param in parameters]
        batch_size = getattr(
            self, "maskGradientBatchSize", getattr(self, "batchSize", 128)
        )

        self.model.zero_grad(set_to_none=True)
        for start in range(0, total_examples, batch_size):
            end = min(start + batch_size, total_examples)
            x_t, y_t = self._numpy_to_tensor(X[start:end], Y[start:end])
            loss = self._compute_loss(x_t, y_t)
            microbatch_gradients = torch.autograd.grad(
                loss,
                parameters,
                allow_unused=True,
            )
            weight = (end - start) / total_examples
            for accumulator, gradient in zip(accumulated, microbatch_gradients):
                if gradient is not None:
                    accumulator.add_(gradient.detach(), alpha=weight)

        gradients = [
            gradient.detach().cpu().numpy().copy()
            for gradient in accumulated
        ]
        self.model.zero_grad(set_to_none=True)
        return gradients

    # ──────────────────────────────────────────────────────────────────────
    #  FedML-HE local sensitivity-map proposal
    # ──────────────────────────────────────────────────────────────────────

    def proposeMask(self, data, debug=False):
        """Calculate the FedML-HE sensitivity map for one client's data.

        For sample ``k`` and parameter ``w_m``, the paper defines
        ``J_m(y_k) = d/dy_k (d loss / d w_m)``. Cross-entropy with a soft
        target is linear in each target coordinate, so equality of mixed
        partials makes this the parameter gradient of the sample's negative
        true-class log-probability. Thus the implementation averages absolute
        per-sample cross-entropy gradients.
        """
        sample_count = len(data[1])

        x_t, y_t = self._numpy_to_tensor(*data)
        limit_k = min(len(y_t), sample_count)
        if limit_k == 0:
            raise ValueError("Cannot calculate sensitivity from an empty dataset")

        parameters = [parameter for _, parameter in self.trainableVarsList]
        accumulated = [torch.zeros_like(parameter) for parameter in parameters]
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        try:
            for sample_index in range(limit_k):
                if debug and sample_index % 10 == 0:
                    print(
                        f"--- Processing sensitivity sample: "
                        f"{sample_index + 1}/{limit_k} ---"
                    )

                logits = self.model(x_t[sample_index:sample_index + 1])
                sample_loss = self._criterion(
                    logits, y_t[sample_index:sample_index + 1]
                )
                gradients = torch.autograd.grad(
                    sample_loss,
                    parameters,
                    allow_unused=True,
                )
                for accumulator, gradient in zip(accumulated, gradients):
                    if gradient is not None:
                        accumulator.add_(gradient.detach().abs())

            sensitivity_map = [
                (value / float(limit_k)).detach().cpu().numpy().copy()
                for value in accumulated
            ]
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return sensitivity_map
