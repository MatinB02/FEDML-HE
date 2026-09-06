import enum
import os
import pickle
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from datetime import datetime

from torchvision.datasets import MNIST, FashionMNIST


class DatasetType(enum.Enum):
    """Supported dataset types"""
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    CINIC10 = "cinic10"
    SVHN = "svhn"
    MNIST = "mnist"
    FMNIST = "fmnist"


class DatasetLoader:
    """
    Handles loading, preprocessing, and partitioning of datasets for federated learning.

    Features:
    - Deterministic data loading and shuffling with seed control
    - Support for IID and Non-IID (Dirichlet) partitioning
    - Caching of preprocessed datasets for faster loading
    - Flexible client data allocation
    - PyTorch-friendly (no TensorFlow dependency)
    """

    DATASET_CONFIG = {
        DatasetType.CIFAR10:  {'num_classes': 10,  'input_shape': (32, 32, 3), 'train_samples': 50000, 'test_samples': 10000},
        DatasetType.CIFAR100: {'num_classes': 100, 'input_shape': (32, 32, 3), 'train_samples': 50000, 'test_samples': 10000},
        DatasetType.CINIC10:  {'num_classes': 10,  'input_shape': (32, 32, 3), 'train_samples': 90000, 'test_samples': 90000},
        DatasetType.SVHN:     {'num_classes': 10,  'input_shape': (32, 32, 3), 'train_samples': 73257, 'test_samples': 26032},
        DatasetType.MNIST:    {'num_classes': 10,  'input_shape': (28, 28, 1), 'train_samples': 60000, 'test_samples': 10000},
        DatasetType.FMNIST:   {'num_classes': 10,  'input_shape': (28, 28, 1), 'train_samples': 60000, 'test_samples': 10000},
    }

    def __init__(
            self,
            dataset_type: DatasetType,
            data_dir: str,
            seed: int = 42,
            cache_dir: Optional[str] = None
    ):
        """
        Initialize the dataset loader.

        Args:
            dataset_type: Type of dataset to load
            data_dir: Base directory containing raw datasets
            seed: Random seed for reproducibility
            cache_dir: Directory to cache preprocessed datasets
        """
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.seed = seed
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")

        self.config = self.DATASET_CONFIG[dataset_type]
        self.num_classes = self.config['num_classes']
        self.input_shape = self.config['input_shape']
        self.train_samples = self.config['train_samples']

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        os.makedirs(self.cache_dir, exist_ok=True)

    # ==========================================================================
    # RAW DATA LOADING
    # ==========================================================================

    def _load_raw_train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw training data based on dataset type."""
        images, labels = None, None

        if self.dataset_type == DatasetType.MNIST:
            dataset = MNIST(root=self.data_dir, train=True, download=True)
            images = np.expand_dims(dataset.data.numpy(), -1)   # (N, 28, 28, 1)
            labels = dataset.targets.numpy()

        elif self.dataset_type == DatasetType.FMNIST:
            dataset = FashionMNIST(root=self.data_dir, train=True, download=True)
            images = np.expand_dims(dataset.data.numpy(), -1)   # (N, 28, 28, 1)
            labels = dataset.targets.numpy()

        elif self.dataset_type == DatasetType.CIFAR10:
            images_list, labels_list = [], []
            for i in range(1, 6):
                batch_path = os.path.join(self.data_dir, 'Cifar', f'data_batch_{i}')
                batch = self._unpickle(batch_path)
                images_list.append(batch[b'data'])
                labels_list.append(batch[b'labels'])
            images = np.concatenate(images_list, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.concatenate(labels_list, axis=0)

        elif self.dataset_type == DatasetType.CIFAR100:
            batch_path = os.path.join(self.data_dir, 'cifar100', 'train')
            batch = self._unpickle(batch_path)
            images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.array(batch[b'fine_labels'])

        elif self.dataset_type == DatasetType.CINIC10:
            data_path = os.path.join(self.data_dir, 'CINIC10', 'Train.npy')
            data = np.load(data_path, allow_pickle=True).item()
            images = data['TrainImage']
            labels = data['TrainLabel']

        elif self.dataset_type == DatasetType.SVHN:
            mat_path = os.path.join(self.data_dir, 'SVHN', 'train_32x32.mat')
            mat = scipy.io.loadmat(mat_path)
            images = np.moveaxis(mat['X'], -1, 0)   # (N, 32, 32, 3)
            labels = mat['y'][:, 0] - 1             # 1-10 → 0-9

        if images is None or labels is None:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        return images, labels

    def _load_raw_test(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw test data based on dataset type."""
        images, labels = None, None

        if self.dataset_type == DatasetType.MNIST:
            dataset = MNIST(root=self.data_dir, train=False, download=True)
            images = np.expand_dims(dataset.data.numpy(), -1)
            labels = dataset.targets.numpy()

        elif self.dataset_type == DatasetType.FMNIST:
            dataset = FashionMNIST(root=self.data_dir, train=False, download=True)
            images = np.expand_dims(dataset.data.numpy(), -1)
            labels = dataset.targets.numpy()

        elif self.dataset_type == DatasetType.CIFAR10:
            batch_path = os.path.join(self.data_dir, 'Cifar', 'test_batch')
            batch = self._unpickle(batch_path)
            images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.array(batch[b'labels'])

        elif self.dataset_type == DatasetType.CIFAR100:
            batch_path = os.path.join(self.data_dir, 'cifar100', 'test')
            batch = self._unpickle(batch_path)
            images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = np.array(batch[b'fine_labels'])

        elif self.dataset_type == DatasetType.CINIC10:
            data_path = os.path.join(self.data_dir, 'CINIC10', 'Test.npy')
            data = np.load(data_path, allow_pickle=True).item()
            images = data['TestImage']
            labels = data['TestLabel']

        elif self.dataset_type == DatasetType.SVHN:
            mat_path = os.path.join(self.data_dir, 'SVHN', 'test_32x32.mat')
            mat = scipy.io.loadmat(mat_path)
            images = np.moveaxis(mat['X'], -1, 0)
            labels = mat['y'][:, 0] - 1

        if images is None or labels is None:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        return images, labels

    @staticmethod
    def _unpickle(file_path: str) -> dict:
        """Unpickle CIFAR data files."""
        with open(file_path, 'rb') as f:
            return pickle.load(f, encoding='bytes')

    # ==========================================================================
    # PREPROCESSING & CACHING
    # ==========================================================================

    def load_and_preprocess(self, force_reload: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the dataset. Uses cached version if available.

        Args:
            force_reload: If True, ignore cache and reload from raw data

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        cache_path = os.path.join(
            self.cache_dir,
            f"{self.dataset_type.value}_preprocessed_seed{self.seed}.npz"
        )

        if not force_reload and os.path.exists(cache_path):
            print(f"✅ Loading preprocessed dataset from cache: {cache_path}")
            data = SafeFileIO.safe_load_npz(cache_path)
            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.X_test  = data['X_test']
            self.y_test  = data['y_test']
            print(f"   Train: {self.X_train.shape}, Test: {self.X_test.shape}")
            return self.X_train, self.y_train, self.X_test, self.y_test

        print(f"📂 Loading raw dataset: {self.dataset_type.value}")
        X_train_raw, y_train_raw = self._load_raw_train()
        X_test_raw,  y_test_raw  = self._load_raw_test()

        print("🔄 Preprocessing dataset...")
        self.X_train, self.y_train = self._preprocess(X_train_raw, y_train_raw, is_train=True)
        self.X_test,  self.y_test  = self._preprocess(X_test_raw,  y_test_raw,  is_train=False)

        print(f"💾 Saving preprocessed dataset to cache: {cache_path}")
        SafeFileIO.safe_save_npz(
            cache_path,
            allow_pickle=False,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test
        )

        print(f"✅ Dataset ready - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self.X_train, self.y_train, self.X_test, self.y_test

    def _preprocess(
            self,
            X: np.ndarray,
            y: np.ndarray,
            is_train: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data: normalize to [0,1], shuffle deterministically,
        and convert labels to one-hot encoding.

        Args:
            X: Input images
            y: Labels (integer class indices)
            is_train: Affects the RNG seed offset used for shuffling

        Returns:
            Preprocessed (X, y) with y in one-hot format
        """
        X = X.astype(np.float32) / 255.0

        seed_offset = 0 if is_train else 1000
        rng = np.random.RandomState(self.seed + seed_offset)
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Already one-hot? — skip conversion
        if len(y.shape) > 1 and y.shape[1] == self.num_classes:
            return X, y

        y_onehot = np.zeros((len(y), self.num_classes), dtype=np.float32)
        y_onehot[np.arange(len(y)), y.astype(int)] = 1.0
        return X, y_onehot

    # ==========================================================================
    # DATA PARTITIONING
    # ==========================================================================

    def partition_data(
            self,
            num_clients: int,
            samples_per_client: List[int],
            alpha: float = 0.5,
            force_create: bool = False,
            is_iid: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Partition training data among clients.

        Args:
            num_clients: Number of clients
            samples_per_client: List of sample counts for each client
            alpha: Dirichlet concentration parameter (lower = more non-IID)
            force_create: If True, ignore cache and re-create partition
            is_iid: If True, use IID partitioning; if False, use Dirichlet

        Returns:
            Tuple of (client_data_list, client_labels_list)
        """
        if self.X_train is None:
            raise ValueError("Dataset not loaded. Call load_and_preprocess() first.")

        if len(samples_per_client) != num_clients:
            raise ValueError(
                f"samples_per_client length ({len(samples_per_client)}) "
                f"must equal num_clients ({num_clients})"
            )

        partition_type = "iid" if is_iid else f"noniid_alpha{alpha:.2f}"
        samples_str = "_".join(map(str, samples_per_client))
        cache_path = os.path.join(
            self.cache_dir,
            f"{self.dataset_type.value}_{partition_type}_{num_clients}clients"
            f"_{samples_str}_seed{self.seed}.npz"
        )

        if os.path.exists(cache_path) and not force_create:
            print(f"✅ Loading partitioned data from cache: {cache_path}")
            data = SafeFileIO.safe_load_npz(cache_path, allow_pickle=True)
            client_data   = list(data['client_data'])
            client_labels = list(data['client_labels'])
            print(f"   Loaded {len(client_data)} client partitions")
            self._save_distribution_plot(client_labels, num_clients, alpha, is_iid)
            return client_data, client_labels

        print(f"📊 Partitioning data: {partition_type}, {num_clients} clients")

        if is_iid:
            client_data, client_labels = self._partition_iid(num_clients, samples_per_client)
        else:
            client_data, client_labels = self._partition_dirichlet(num_clients, samples_per_client, alpha)

        print(f"💾 Saving partitioned data to cache: {cache_path}")
        SafeFileIO.safe_save_npz(
            cache_path,
            allow_pickle=True,
            client_data=np.array(client_data, dtype=object),
            client_labels=np.array(client_labels, dtype=object),
            num_clients=num_clients,
            alpha=alpha if not is_iid else np.nan,
            is_iid=is_iid,
            seed=self.seed
        )

        self._print_partition_summary(client_labels, is_iid, alpha)
        self._save_distribution_plot(client_labels, num_clients, alpha, is_iid)

        return client_data, client_labels

    def _partition_iid(
            self,
            num_clients: int,
            samples_per_client: List[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create IID partition — each client receives a balanced class distribution.
        """
        rng = np.random.RandomState(self.seed + 100)
        y_indices = np.argmax(self.y_train, axis=1)

        class_indices = [np.where(y_indices == c)[0] for c in range(self.num_classes)]
        for idx in class_indices:
            rng.shuffle(idx)

        client_data, client_labels = [], []

        for client_id in range(num_clients):
            n_samples = samples_per_client[client_id]
            samples_per_class = n_samples // self.num_classes
            remainder = n_samples % self.num_classes

            client_indices = []
            for class_id in range(self.num_classes):
                n_class_samples = samples_per_class + (1 if class_id < remainder else 0)
                start = client_id * samples_per_class
                end   = start + n_class_samples
                if end <= len(class_indices[class_id]):
                    client_indices.extend(class_indices[class_id][start:end])

            rng.shuffle(client_indices)
            client_data.append(self.X_train[client_indices])
            client_labels.append(self.y_train[client_indices])

        return client_data, client_labels

    def _partition_dirichlet(
            self,
            num_clients: int,
            samples_per_client: List[int],
            alpha: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create Non-IID partition using per-class Dirichlet sampling.

        Args:
            num_clients: Number of clients
            samples_per_client: Target sample count per client
            alpha: Dirichlet concentration (lower = more heterogeneous)
        """
        rng = np.random.RandomState(self.seed + 200)
        y_indices = np.argmax(self.y_train, axis=1)

        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        print(f"\n🔄 Applying Dirichlet partitioning (alpha={alpha})...")

        for class_id in range(self.num_classes):
            class_idx = np.where(y_indices == class_id)[0]
            rng.shuffle(class_idx)
            n = len(class_idx)
            if n == 0:
                continue

            proportions = rng.dirichlet([alpha] * num_clients)
            proportions = np.maximum(proportions, 1e-10)
            proportions /= proportions.sum()

            counts = (proportions * n).astype(int)
            remainder = n - counts.sum()
            if remainder > 0:
                counts[np.argsort(proportions)[-remainder:]] += 1
            elif remainder < 0:
                nonzero = np.where(counts > 0)[0]
                remove = rng.choice(nonzero, size=min(-remainder, len(nonzero)), replace=False)
                counts[remove] -= 1

            current = 0
            for client_id, count in enumerate(counts):
                if count > 0:
                    client_indices[client_id].extend(class_idx[current:current + count].tolist())
                    current += count

        for idx_list in client_indices:
            rng.shuffle(idx_list)

        print("\n📊 Initial distribution:")
        for i, idx in enumerate(client_indices):
            print(f"   Client {i}: {len(idx)} samples")

        print("\n🔧 Adjusting to requested sizes...")
        result_data, result_labels = [], []

        for client_id in range(num_clients):
            target  = samples_per_client[client_id]
            current = len(client_indices[client_id])

            if current == target:
                indices = np.array(client_indices[client_id])

            elif current > target:
                indices = np.array(client_indices[client_id])
                rng.shuffle(indices)
                indices = indices[:target]
                print(f"   Client {client_id}: Removed {current - target} samples")

            else:
                needed = target - current
                stolen: List[int] = []

                # Strategy 1: steal excess from other clients
                donors = [
                    (d, len(client_indices[d]) - samples_per_client[d])
                    for d in range(num_clients)
                    if d != client_id and len(client_indices[d]) > samples_per_client[d]
                ]
                donors.sort(key=lambda x: x[1], reverse=True)

                for donor_id, excess in donors:
                    if len(stolen) >= needed:
                        break
                    take = min(excess, needed - len(stolen))
                    stolen.extend(client_indices[donor_id][:take])
                    client_indices[donor_id] = client_indices[donor_id][take:]

                # Strategy 2: sample from entire dataset
                if len(stolen) < needed:
                    remaining = needed - len(stolen)
                    assigned = set(client_indices[client_id] + stolen)
                    available = [i for i in range(len(self.X_train)) if i not in assigned]

                    if len(available) >= remaining:
                        extra = rng.choice(available, size=remaining, replace=False)
                    else:
                        extra = rng.choice(len(self.X_train), size=remaining, replace=True)
                        print(f"   ⚠️  Client {client_id}: Had to sample with replacement")

                    stolen.extend(extra.tolist())

                indices = np.array(client_indices[client_id] + stolen)
                rng.shuffle(indices)
                print(f"   Client {client_id}: Added {len(stolen)} samples")

            result_data.append(self.X_train[indices])
            result_labels.append(self.y_train[indices])

        print("\n✅ Final distribution:")
        for client_id in range(num_clients):
            class_counts = np.sum(result_labels[client_id], axis=0).astype(int)
            total = len(result_labels[client_id])
            print(f"   Client {client_id}: {total:5d} samples | Classes: {class_counts}")

        return result_data, result_labels

    # ==========================================================================
    # GLOBAL DATASET
    # ==========================================================================

    def create_global_dataset(
            self,
            num_samples: int,
            alpha: Optional[float] = None,
            is_iid: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a shared global dataset (IID or Non-IID) accessible to all clients.

        Args:
            num_samples: Number of samples
            alpha: Dirichlet concentration (required when is_iid=False)
            is_iid: If True, balanced class distribution; otherwise Dirichlet

        Returns:
            Tuple of (X_global, y_global)
        """
        if self.X_train is None:
            raise ValueError("Dataset not loaded. Call load_and_preprocess() first.")
        if num_samples > len(self.X_train):
            raise ValueError(
                f"Requested {num_samples} samples but only {len(self.X_train)} available"
            )

        dist_type = "iid" if is_iid else f"noniid_alpha{alpha:.2f}"
        cache_path = os.path.join(
            self.cache_dir,
            f"{self.dataset_type.value}_global_{dist_type}_{num_samples}samples_seed{self.seed}.npz"
        )

        if os.path.exists(cache_path):
            print(f"✅ Loading global dataset from cache: {cache_path}")
            data = SafeFileIO.safe_load_npz(cache_path)
            X_global = data['X_global']
            y_global = data['y_global']
            print(f"   Loaded {len(X_global)} samples")
            self._print_global_distribution(y_global, is_iid, alpha)
            return X_global, y_global

        print(f"📊 Creating global dataset: {dist_type}, {num_samples} samples")
        rng = np.random.RandomState(self.seed + 300)

        if is_iid:
            X_global, y_global = self._create_global_iid(num_samples, rng)
        else:
            if alpha is None:
                raise ValueError("alpha must be specified for non-IID global dataset")
            X_global, y_global = self._create_global_noniid(num_samples, alpha, rng)

        print(f"💾 Saving global dataset to cache: {cache_path}")
        SafeFileIO.safe_save_npz(
            cache_path,
            allow_pickle=False,
            X_global=X_global,
            y_global=y_global,
            num_samples=num_samples,
            alpha=alpha if not is_iid else np.nan,
            is_iid=is_iid,
            seed=self.seed
        )

        self._print_global_distribution(y_global, is_iid, alpha)
        return X_global, y_global

    def _create_global_iid(
            self,
            num_samples: int,
            rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balanced class distribution for the global dataset."""
        samples_per_class = num_samples // self.num_classes
        remainder = num_samples % self.num_classes

        y_indices = np.argmax(self.y_train, axis=1)
        class_indices = [np.where(y_indices == c)[0] for c in range(self.num_classes)]
        for idx in class_indices:
            rng.shuffle(idx)

        selected = []
        for class_id in range(self.num_classes):
            n = samples_per_class + (1 if class_id < remainder else 0)
            avail = class_indices[class_id]
            selected.extend(avail[:n] if n <= len(avail) else avail)

        selected = np.array(selected)
        rng.shuffle(selected)
        return self.X_train[selected], self.y_train[selected]

    def _create_global_noniid(
            self,
            num_samples: int,
            alpha: float,
            rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dirichlet-distributed global dataset."""
        class_proportions = rng.dirichlet([alpha] * self.num_classes)
        samples_per_class = (class_proportions * num_samples).astype(int)

        diff = num_samples - samples_per_class.sum()
        if diff > 0:
            for _ in range(diff):
                samples_per_class[rng.randint(0, self.num_classes)] += 1
        elif diff < 0:
            for _ in range(-diff):
                samples_per_class[np.argmax(samples_per_class)] -= 1

        y_indices = np.argmax(self.y_train, axis=1)
        class_indices = [np.where(y_indices == c)[0] for c in range(self.num_classes)]
        for idx in class_indices:
            rng.shuffle(idx)

        selected = []
        for class_id in range(self.num_classes):
            n = samples_per_class[class_id]
            if n > 0:
                avail = class_indices[class_id]
                selected.extend(avail[:n] if n <= len(avail) else avail)

        selected = np.array(selected)
        rng.shuffle(selected)
        return self.X_train[selected], self.y_train[selected]

    # ==========================================================================
    # HELPERS
    # ==========================================================================

    def get_client_data(
            self,
            client_id: int,
            client_data_list: List[np.ndarray],
            client_labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return data for a specific client."""
        if client_id >= len(client_data_list):
            raise ValueError(f"Client ID {client_id} out of range (max: {len(client_data_list) - 1})")
        return client_data_list[client_id], client_labels_list[client_id]

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the test set."""
        if self.X_test is None:
            raise ValueError("Dataset not loaded. Call load_and_preprocess() first.")
        return self.X_test, self.y_test

    def _print_partition_summary(self, client_labels: List[np.ndarray], is_iid: bool, alpha: float):
        print("\n" + "=" * 60)
        print(f"📊 Partition Summary ({'IID' if is_iid else f'Non-IID α={alpha}'})")
        print("=" * 60)
        for client_id, labels in enumerate(client_labels):
            class_counts = np.sum(labels, axis=0).astype(int)
            total = len(labels)
            print(f"Client {client_id:2d}: {total:5d} samples | Class dist: {class_counts}")
        print("=" * 60 + "\n")

    def _print_global_distribution(self, y_global: np.ndarray, is_iid: bool, alpha: Optional[float]):
        class_counts = np.sum(y_global, axis=0).astype(int)
        total = len(y_global)
        print("\n" + "=" * 60)
        print(f"🌍 Global Dataset Summary ({'IID' if is_iid else f'Non-IID α={alpha}'})")
        print("=" * 60)
        print(f"Total samples: {total}")
        print(f"Class distribution: {class_counts}")
        print(f"Class percentages: {(class_counts / total * 100).astype(int)}%")
        print("=" * 60 + "\n")

    def _save_distribution_plot(
            self,
            client_labels: List[np.ndarray],
            num_clients: int,
            alpha: float,
            is_iid: bool
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = (
            f"./Plots/DB/iid_{self.dataset_type.name}_seed{self.seed}"
            f"_ClientsCount{num_clients}_{timestamp}.png"
            if is_iid else
            f"./Plots/DB/noniid_{self.dataset_type.name}_seed{self.seed}"
            f"_ClientsCount{num_clients}_alpha{alpha}_{timestamp}.png"
        )
        title = (
            "IID Data Distribution"
            if is_iid else
            f"Non-IID Data Distribution (α={alpha})"
        )
        self.plot_client_distribution(client_labels, save_path=save_path, title=title)

    def plot_client_distribution(
            self,
            client_labels: List[np.ndarray],
            save_path: Optional[str] = None,
            title: Optional[str] = None
    ):
        """Bar chart of class distribution across clients."""
        num_clients = len(client_labels)
        class_counts = np.zeros((num_clients, self.num_classes))
        for i, labels in enumerate(client_labels):
            class_counts[i] = np.sum(labels, axis=0)

        fig, ax = plt.subplots(figsize=(12, 6))
        x     = np.arange(num_clients)
        width = 0.8 / self.num_classes

        for class_id in range(self.num_classes):
            offset = (class_id - self.num_classes / 2) * width
            ax.bar(x + offset, class_counts[:, class_id], width, label=f'Class {class_id}')

        ax.set_xlabel('Client ID')
        ax.set_ylabel('Number of Samples')
        ax.set_title(title or f'Data Distribution Across {num_clients} Clients')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in range(num_clients)])
        ax.legend(ncol=self.num_classes, loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")

        plt.close(fig)


# ==============================================================================
# SAFE FILE I/O
# ==============================================================================

class SafeFileIO:
    """Atomic, corruption-safe numpy .npz file I/O."""

    @staticmethod
    def safe_save_npz(filepath: str, allow_pickle: bool = False, **arrays) -> bool:
        """Atomically save arrays to .npz (write to temp → verify → rename)."""
        dir_path = os.path.dirname(filepath) or '.'
        os.makedirs(dir_path, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(dir=dir_path, prefix='.tmp_', suffix='.npz')
        os.close(fd)

        try:
            np.savez_compressed(temp_path, **arrays)
            SafeFileIO._verify_npz(temp_path, list(arrays.keys()), allow_pickle=allow_pickle)
            os.replace(temp_path, filepath)
            SafeFileIO._sync_directory(dir_path)
            return True
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise IOError(f"Failed to save {filepath}: {e}")

    @staticmethod
    def _verify_npz(filepath: str, expected_keys: List[str], allow_pickle: bool = False):
        with zipfile.ZipFile(filepath, 'r') as zf:
            corrupt = zf.testzip()
            if corrupt is not None:
                raise ValueError(f"Corrupted file in archive: {corrupt}")

        with np.load(filepath, allow_pickle=allow_pickle) as data:
            for key in expected_keys:
                if key not in data:
                    raise KeyError(f"Missing key: {key}")
                _ = data[key].shape   # trigger actual load

    @staticmethod
    def _sync_directory(dir_path: str):
        try:
            flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
            fd = os.open(dir_path, flags)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except (OSError, AttributeError):
            pass

    @staticmethod
    def safe_load_npz(filepath: str, allow_pickle: bool = True) -> Dict[str, np.ndarray]:
        """Load .npz with corruption detection."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                corrupt = zf.testzip()
                if corrupt is not None:
                    raise ValueError(f"Corrupted file in archive: {corrupt}")

            data = np.load(filepath, allow_pickle=allow_pickle)
            result = {key: data[key] for key in data.files}
            data.close()
            return result

        except (zipfile.BadZipFile, EOFError, ValueError, KeyError) as e:
            raise IOError(f"Corrupted or invalid NPZ file: {filepath} - {e}")
