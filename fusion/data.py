"""Whole-image normalization and class-balanced train/test partitions."""

from pathlib import Path

import numpy as np
import torch


def make_split(labels, seed, train_per_class=20):
    generator = np.random.RandomState(seed)
    train = np.full(labels.size, -1, dtype=np.int32)
    test = train.copy()
    classes = int(labels.max())
    if set(np.unique(labels)) - {0} != set(range(1, classes + 1)):
        raise ValueError("Ground-truth labels must be consecutive integers starting at 1; use 0 for unlabeled pixels.")
    for category in range(1, classes + 1):
        indices = np.flatnonzero(labels.reshape(-1) == category)
        if len(indices) <= train_per_class:
            raise ValueError(f"Class {category} has insufficient pixels for disjoint train/test sets.")
        generator.shuffle(indices)
        train[indices[:train_per_class]] = category - 1
        test[indices[train_per_class:]] = category - 1
    return train.reshape(labels.shape), test.reshape(labels.shape)


def normalize(hsi, lidar):
    hsi = hsi.astype(np.float32)
    lidar = lidar.astype(np.float32)
    low = np.percentile(hsi, 2, axis=(0, 1))
    high = np.percentile(hsi, 98, axis=(0, 1))
    hsi = (np.clip((hsi - low) / np.maximum(high - low, 1e-8), 0, 1) * 255).astype(np.uint8).astype(np.float32) / 255
    lidar = (lidar - lidar.min((0, 1))) / np.maximum(lidar.max((0, 1)) - lidar.min((0, 1)), 1e-8)
    return hsi, lidar


def load_scene(data_root, dataset, seed, device):
    directory = Path(data_root) / dataset
    path = directory / "arrays.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Prepare the dataset using datasets/prepare.py first.")
    with np.load(path, allow_pickle=False) as arrays:
        hsi, lidar, labels = arrays["hsi"], arrays["lidar"], arrays["gt"].astype(np.int32)
    if lidar.ndim == 2:
        lidar = lidar[..., None]
    if hsi.ndim != 3 or lidar.ndim != 3 or hsi.shape[:2] != labels.shape or lidar.shape[:2] != labels.shape:
        raise ValueError("HSI, LiDAR and labels must share the same spatial dimensions.")
    if not np.isfinite(hsi).all() or not np.isfinite(lidar).all():
        raise ValueError("Dataset contains non-finite values.")
    train, test = make_split(labels, seed)
    hsi, lidar = normalize(hsi, lidar)
    hsi_tensor = torch.from_numpy(np.ascontiguousarray(hsi.transpose(2, 0, 1)))[None].to(device)
    lidar_tensor = torch.from_numpy(np.ascontiguousarray(lidar.transpose(2, 0, 1)))[None].to(device)
    train_tensor = torch.from_numpy(train.astype(np.int64))[None].to(device)
    return hsi_tensor, lidar_tensor, train_tensor, test, int(labels.max())
