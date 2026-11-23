"""Convert co-registered MATLAB arrays into a training dataset."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATASETS = {
    "muufl": ((325, 220, 64, 11), ("MUF_HSI.mat", "MUF_LiDAR.mat", "MUF_gt.mat")),
    "houston": ((349, 1905, 144, 15), ("Houston.mat", "Houston_LR.mat", "Houston_gt.mat")),
    "trento": ((166, 600, 63, 6), ("Italy_hsi.mat", "Italy_lidar.mat", "allgrd.mat")),
}


def numeric_arrays(value, prefix=""):
    if isinstance(value, dict):
        for name, item in value.items():
            if not name.startswith("__"):
                yield from numeric_arrays(item, f"{prefix}.{name}".strip("."))
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        yield prefix, value


def load_array(path, key, role, expected):
    try:
        arrays = dict(numeric_arrays(loadmat(path, simplify_cells=True)))
    except NotImplementedError:
        import h5py

        arrays = {}
        with h5py.File(path, "r") as handle:
            def visit(name, item):
                if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
                    arrays[name.replace("/", ".")] = np.asarray(item).T
            handle.visititems(visit)
    if key:
        selected = [part.strip() for part in key.split(",")]
        if any(part not in arrays for part in selected):
            raise ValueError(f"Unknown array '{key}' in {path}. Available keys: {list(arrays)}")
        if len(selected) > 1:
            return np.stack([arrays[part] for part in selected], axis=-1)
        return arrays[selected[0]]
    height, width, channels, classes = expected
    candidates = []
    for name, array in arrays.items():
        if role == "hsi":
            accepted = array.shape == (height, width, channels)
        elif role == "lidar":
            accepted = (array.ndim in [2, 3] and array.shape[:2] == (height, width)
                        and (array.ndim == 2 or array.shape[-1] <= 3))
        else:
            accepted = (array.shape == (height, width) and np.isfinite(array).all()
                        and np.equal(array, np.floor(array)).all()
                        and array.max() == classes)
        if accepted:
            candidates.append((name, array))
    if len(candidates) != 1:
        raise ValueError(f"Select --{role}-key for {path}; candidate arrays: {[name for name, _ in candidates]}")
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/prepared"))
    for name in ["hsi", "lidar", "labels"]:
        parser.add_argument(f"--{name}-file", type=Path)
        parser.add_argument(f"--{name}-key")
    args = parser.parse_args()
    expected, names = DATASETS[args.dataset]
    directory = args.raw_dir or Path("datasets/raw") / args.dataset
    paths = [args.hsi_file or directory / names[0], args.lidar_file or directory / names[1],
             args.labels_file or directory / names[2]]
    hsi, lidar, labels = [load_array(path, key, role, expected) for path, key, role in zip(
        paths, [args.hsi_key, args.lidar_key, args.labels_key], ["hsi", "lidar", "labels"],
    )]
    labels = np.asarray(labels).squeeze()
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Ground-truth labels must be finite integers.")
    labels = labels.astype(np.int32)
    labels[labels < 0] = 0
    if lidar.ndim == 2:
        lidar = lidar[..., None]
    height, width, channels, classes = expected
    if hsi.shape != (height, width, channels) or lidar.shape[:2] != (height, width) or labels.shape != (height, width):
        raise ValueError("Unexpected dataset dimensions. See datasets/README.md.")
    if labels.max() != classes or not np.isfinite(hsi).all() or not np.isfinite(lidar).all():
        raise ValueError("Invalid dataset values or class labels.")
    if args.dataset == "muufl" and lidar.shape[-1] != 2:
        raise ValueError("MUUFL requires two LiDAR channels: elevation followed by intensity.")
    if args.dataset == "houston" and lidar.shape[-1] != 1:
        raise ValueError("Houston requires one elevation channel.")
    if args.dataset == "trento" and lidar.shape[-1] != 2:
        raise ValueError("Trento requires the two supplied LiDAR channels.")
    output = args.output_root / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "arrays.npz", hsi=hsi, lidar=lidar, gt=labels)
    print(f"Prepared {args.dataset}: {output / 'arrays.npz'}")


if __name__ == "__main__":
    main()
