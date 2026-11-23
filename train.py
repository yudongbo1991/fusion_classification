"""Train HCDR-Fusion on one prepared dataset."""

import argparse
import copy
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from fusion import HCDRFusion
from fusion.data import load_scene
from fusion.losses import class_relation_loss, sample_relation_loss
from fusion.regularization import corrupt_observations, regularization_loss
from fusion.utils import classification_metrics, save_json, set_seed


def train(args):
    if args.epochs < 1:
        raise ValueError("The number of epochs must be positive.")
    set_seed(args.seed)
    device = torch.device(args.device)
    hsi, lidar, labels, test_labels, classes = load_scene(args.data_root, args.dataset, args.seed, device)
    model = HCDRFusion(hsi.shape[1], lidar.shape[1], classes, args.dataset).to(device)
    cfg = model.configuration
    teacher = copy.deepcopy(model).eval() if cfg.regularization else None
    if teacher is not None:
        teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    output = Path(args.output or Path("outputs") / args.dataset)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_oa = -1.0
    test_mask = test_labels >= 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        inputs_hsi, inputs_lidar, targets = hsi, lidar, labels
        for axis in [-1, -2]:
            if random.random() < 0.5:
                inputs_hsi = inputs_hsi.flip(axis)
                inputs_lidar = inputs_lidar.flip(axis)
                targets = targets.flip(axis)
        if epoch <= 10:
            factor = epoch / 10
        else:
            factor = 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (epoch - 10) / max(args.epochs - 10, 1)))
        learning_rate = cfg.learning_rate * factor
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        selected = targets >= 0
        logits, state = model(inputs_hsi, inputs_lidar, training_mask=selected)
        cross_entropy = F.cross_entropy(logits, targets, ignore_index=-1, label_smoothing=cfg.smoothing)
        embedding = state["embedding"]
        class_loss = class_relation_loss(
            embedding, state["lh"].permute(0, 2, 3, 1)[selected],
            state["ll"].permute(0, 2, 3, 1)[selected], targets[selected],
            cfg.class_temperature, cfg.beta,
        )
        sample_loss = sample_relation_loss(embedding, targets[selected], cfg.sample_temperature)
        warmup = min(1.0, epoch / 10)
        loss = cross_entropy + warmup * (cfg.class_weight * class_loss + cfg.sample_weight * sample_loss)
        if teacher is not None and epoch >= 26 and epoch % 2 == 0:
            with torch.no_grad():
                teacher_logits, _ = teacher(inputs_hsi, inputs_lidar)
            cycle = epoch // 2
            altered_hsi, altered_lidar, mask, modality = corrupt_observations(inputs_hsi, inputs_lidar, cycle)
            student = model.encode(altered_hsi, altered_lidar)
            path = ["lidar_to_hsi", "hsi_to_lidar", "context"][(cycle // 6 + cycle % 6) % 3]
            loss = loss + min(1.0, (epoch - 25) / 25) * regularization_loss(
                model, student, state, teacher_logits, mask, modality, path,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if teacher is not None:
            with torch.no_grad():
                for teacher_parameter, parameter in zip(teacher.parameters(), model.parameters()):
                    teacher_parameter.lerp_(parameter, 0.02)
        row = {"epoch": epoch, "loss": float(loss.detach()), "cross_entropy": float(cross_entropy.detach()),
               "class_loss": float(class_loss.detach()), "sample_loss": float(sample_loss.detach()),
               "learning_rate": learning_rate}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | loss {row['loss']:.6f}", flush=True)
        del logits, state, loss, class_loss, sample_loss, cross_entropy, embedding
        if teacher is not None and epoch >= 26 and epoch % 2 == 0:
            del teacher_logits, student, altered_hsi, altered_lidar
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.inference_mode():
                evaluation_logits, evaluation_state = model(hsi, lidar)
                predictions = evaluation_logits.argmax(1)[0].cpu().numpy()
            metrics = classification_metrics(test_labels[test_mask], predictions[test_mask], classes)
            row["test_OA"] = metrics["OA"]
            if metrics["OA"] > best_oa:
                best_oa = metrics["OA"]
                checkpoint = {
                    "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "dataset": args.dataset, "seed": args.seed, "epoch": epoch,
                    "hsi_channels": hsi.shape[1], "lidar_channels": lidar.shape[1], "classes": classes,
                    "configuration": asdict(cfg), "selection": "test_OA_selected",
                    "selection_OA": best_oa,
                }
                torch.save(checkpoint, output / "model.pt")
                save_json(output / "best_metrics.json", dict(metrics, epoch=epoch, selection="test_OA_selected"))
            print(f"Test OA: {100 * metrics['OA']:.2f}% | best: {100 * best_oa:.2f}%", flush=True)
            del evaluation_logits, evaluation_state, predictions
    save_json(output / "training.json", {
        "dataset": args.dataset, "seed": args.seed, "selection": "test_OA_selected",
        "evaluation_interval": 5, "history": history,
    })
    print(f"Saved checkpoint: {output / 'model.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["muufl", "houston", "trento"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
