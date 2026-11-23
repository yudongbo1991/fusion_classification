"""Evaluate a trained HCDR-Fusion checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch

from fusion import HCDRFusion
from fusion.data import load_scene
from fusion.utils import classification_metrics, save_json, set_seed


def evaluate(args):
    path = args.checkpoint or Path("outputs") / args.dataset / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["dataset"] != args.dataset:
        raise ValueError("The checkpoint belongs to a different dataset.")
    seed = int(checkpoint["seed"])
    set_seed(seed)
    device = torch.device(args.device)
    hsi, lidar, _, test_labels, classes = load_scene(args.data_root, args.dataset, seed, device)
    if (hsi.shape[1], lidar.shape[1], classes) != (
        checkpoint["hsi_channels"], checkpoint["lidar_channels"], checkpoint["classes"],
    ):
        raise ValueError("Dataset channels or classes differ from those used for training.")
    model = HCDRFusion(hsi.shape[1], lidar.shape[1], classes, args.dataset).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        logits, _ = model(hsi, lidar)
        predictions = logits.argmax(1)[0].cpu().numpy()
    selected = test_labels >= 0
    result = classification_metrics(test_labels[selected], predictions[selected], classes)
    result.update(dataset=args.dataset, seed=seed, checkpoint_epoch=int(checkpoint["epoch"]))
    output = Path(args.output or Path("outputs") / args.dataset)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "metrics.json", result)
    np.savez_compressed(output / "predictions.npz", prediction=predictions.astype(np.int32) + 1,
                        test_mask=selected)
    print(f"OA: {100 * result['OA']:.2f}% | AA: {100 * result['AA']:.2f}% | Kappa: {result['Kappa']:.4f}")
    print(f"Saved evaluation: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["muufl", "houston", "trento"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
