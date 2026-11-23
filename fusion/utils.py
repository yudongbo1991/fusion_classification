"""Reproducible execution and classification metrics."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def classification_metrics(labels, predictions, classes):
    matrix = np.bincount(labels * classes + predictions, minlength=classes * classes).reshape(classes, classes)
    count = matrix.sum()
    if not count:
        raise ValueError("No labeled test pixels were found.")
    totals = matrix.sum(1)
    accuracy = np.divide(matrix.diagonal(), totals, out=np.zeros(classes, dtype=float), where=totals > 0)
    oa = float(matrix.trace() / count)
    expected = float(np.dot(matrix.sum(0).astype(float), totals) / count ** 2)
    return {
        "OA": oa, "AA": float(accuracy.mean()),
        "Kappa": (oa - expected) / (1 - expected) if expected < 1 else 1.0,
        "class_accuracy": accuracy.tolist(), "confusion_matrix": matrix.tolist(),
        "test_pixels": int(count),
    }
