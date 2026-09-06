# -*- coding: utf-8 -*-
import shutil
import seaborn as sns
import numpy as np
import torch  # <-- Migrated from tensorflow.compat.v1 as tf
from scipy import signal
from scipy.ndimage import uniform_filter
try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ModuleNotFoundError:  # Minimal environments use an equivalent local implementation.
    skimage_ssim = None
from pathlib import Path


def foldersInit(cfg):
    ROOT_DIR = Path(__file__).parent
    CURRENT_DIR = ROOT_DIR.parents[0]
    createDIR = CURRENT_DIR / 'Temp' / cfg.model.name
    createDIR.mkdir(parents=True, exist_ok=True)
    # createDIR = CURRENT_DIR / 'Plots' / 'DB_Distributions'
    createDIR.mkdir(parents=True, exist_ok=True)
    createDIR = ROOT_DIR.parents[1] / 'Datasets' / 'GeneratedDBs'
    createDIR.mkdir(parents=True, exist_ok=True)

def mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

def calculate_ssim(img1, img2):
    if skimage_ssim is None:
        def fallback_ssim(first, second):
            first = np.asarray(first, dtype=np.float64)
            second = np.asarray(second, dtype=np.float64)
            if first.shape != second.shape:
                raise ValueError("SSIM inputs must have identical shapes")
            ux = uniform_filter(first, size=3)
            uy = uniform_filter(second, size=3)
            uxx = uniform_filter(first * first, size=3)
            uyy = uniform_filter(second * second, size=3)
            uxy = uniform_filter(first * second, size=3)
            covariance_normalization = 9.0 / 8.0
            vx = covariance_normalization * (uxx - ux * ux)
            vy = covariance_normalization * (uyy - uy * uy)
            vxy = covariance_normalization * (uxy - ux * uy)
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            score = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
                (ux * ux + uy * uy + c1) * (vx + vy + c2)
            )
            return float(np.mean(score[1:-1, 1:-1]))

        if img1.ndim == 3:
            return float(np.mean([
                fallback_ssim(img1[..., channel], img2[..., channel])
                for channel in range(img1.shape[-1])
            ]))
        return fallback_ssim(img1, img2)
    if img1.ndim == 3:
        return skimage_ssim(img1, img2, channel_axis=-1, data_range=1.0, win_size=3)
    else:
        return skimage_ssim(img1, img2, data_range=1.0, win_size=3)


def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))


def delete_all_in_folder(folder_path):
    path = Path(folder_path)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item)
    print(f"All files and folders in '{folder_path}' have been deleted.")


def which_gpu():
    """Quick check of current GPU using PyTorch"""
    import os

    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append(f"CUDA:{i} ({torch.cuda.get_device_name(i)})")

    print(f"\n>>> Using GPUs: {gpus if gpus else 'CPU only'} (CUDA_VISIBLE_DEVICES={cuda_visible})\n")
    return gpus


def confidence_Balance(labels):
    class_counts = np.sum(labels, axis=0)
    total_samples = len(labels)

    distributionList = []
    for cls_idx, count in enumerate(class_counts):
        distributionList.append((count / total_samples) * 100)
    score = np.sqrt(np.var(distributionList))
    return score


def maskFilter(model, p):
    """Return flat indices of the globally most-sensitive parameters.

    ``model`` is a sequence of trainable-parameter sensitivity arrays. The
    returned indices address their concatenation in the same order. Selection
    is global, has exact cardinality ``floor(p * parameter_count)``, and
    resolves threshold ties by lower flat index for deterministic runs.
    """
    if not np.isfinite(p) or not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must be a finite value between 0 and 1")

    arrays = [np.asarray(layer) for layer in model]
    if any(not np.issubdtype(layer.dtype, np.number) for layer in arrays):
        raise TypeError("Sensitivity values must be numeric")

    sensitivities = (
        np.concatenate([np.ravel(layer) for layer in arrays])
        if arrays else np.empty(0, dtype=np.float32)
    )
    parameter_count = sensitivities.size
    selected_count = int(parameter_count * float(p))

    if selected_count == 0:
        return np.empty(0, dtype=np.int64)
    if selected_count == parameter_count:
        return np.arange(parameter_count, dtype=np.int64)
    if not np.all(np.isfinite(sensitivities)):
        raise ValueError("Sensitivity values must all be finite")

    scores = np.abs(sensitivities)
    partition_start = parameter_count - selected_count
    threshold = np.partition(scores, partition_start)[partition_start]
    above_threshold = np.flatnonzero(scores > threshold)
    remaining = selected_count - above_threshold.size
    at_threshold = np.flatnonzero(scores == threshold)[:remaining]
    selected = np.concatenate((above_threshold, at_threshold))

    # Highest sensitivity first; flat index is the deterministic tie-breaker.
    order = np.lexsort((selected, -scores[selected]))
    return selected[order].astype(np.int64, copy=False)


def dlg_mean_std(values):
    values = np.asarray(
        [value for value in values if value is not None],
        dtype=np.float64,
    )
    if len(values) == 0:
        return None, None
    ddof = 1 if len(values) > 1 else 0
    return float(np.mean(values)), float(np.std(values, ddof=ddof))
