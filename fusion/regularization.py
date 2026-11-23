"""Observation corruption and consistency regularization during training."""

import torch
from torch.nn import functional as F


def corrupt_observations(hsi, lidar, cycle):
    height, width = hsi.shape[-2:]
    mask = torch.zeros_like(hsi[:, :1])
    for _ in range(3):
        patch_h = int(torch.randint(max(8, height // 10), max(9, height // 4), (1,)))
        patch_w = int(torch.randint(max(8, width // 10), max(9, width // 4), (1,)))
        patch_h, patch_w = min(patch_h, height), min(patch_w, width)
        row = int(torch.randint(0, height - patch_h + 1, (1,)))
        col = int(torch.randint(0, width - patch_w + 1, (1,)))
        mask[:, :, row:row + patch_h, col:col + patch_w] = 1
    modality, mode = cycle % 2, (cycle // 2) % 3
    altered_hsi, altered_lidar = hsi.clone(), lidar.clone()
    if modality == 0:
        if mode == 0:
            altered = hsi + torch.randn_like(hsi) * 0.15
        elif mode == 1:
            altered = hsi * 0.45
        else:
            altered = F.avg_pool2d(hsi, 5, 1, 2, count_include_pad=False)
        altered_hsi = hsi * (1 - mask) + altered * mask
    else:
        altered = lidar.clone()
        if mode == 0:
            altered[:, :1] = lidar[:, :1] + torch.randn_like(lidar[:, :1]) * 0.15
        elif mode == 1:
            altered[:, :1] = F.avg_pool2d(lidar[:, :1], 11, 1, 5, count_include_pad=False)
        else:
            altered = lidar + torch.randn_like(lidar) * 0.12
        altered_lidar = lidar * (1 - mask) + altered * mask
    return altered_hsi, altered_lidar, mask, modality


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def regularization_loss(model, student, original, teacher_logits, mask, modality, path):
    cfg = model.configuration
    clean_quality = original["gates"][:, modality:modality + 1]
    altered_quality = student["gates"][:, modality:modality + 1]
    rank = masked_mean(F.relu(0.10 + altered_quality - clean_quality), mask)
    stability = masked_mean((student["gates"][:, :2] - original["gates"][:, :2].detach()).square().mean(1, keepdim=True), 1 - mask)
    previous = model.head.training
    model.head.eval()
    try:
        enabled = model.decode(student)
        omitted = model.decode(student, omitted_path=path)
    finally:
        model.head.train(previous)
    target = teacher_logits.softmax(1).detach()
    confidence = target.max(1, keepdim=True).values
    support = (confidence > 0.7).float() * mask

    def divergence(logits):
        return F.kl_div(logits.log_softmax(1), target, reduction="none").sum(1, keepdim=True)

    enabled_divergence = divergence(enabled)
    utility = (divergence(omitted) - enabled_divergence).detach()
    weight = support * (utility.abs() / 0.1).clamp(0, 1)
    desired = torch.sigmoid(utility / 0.1)
    if path == "lidar_to_hsi":
        gate = student["gates"][:, 4:5]
    elif path == "hsi_to_lidar":
        gate = student["gates"][:, 5:6]
    else:
        gate = student["gates"][:, 2:4].mean(1, keepdim=True)
    gain = masked_mean(F.binary_cross_entropy(gate.clamp(0.001, 0.999), desired,
                                            reduction="none"), weight)
    consistency = masked_mean(enabled_divergence, support)
    return (cfg.ranking_weight * rank + 0.05 * stability
            + cfg.utility_weight * gain + cfg.consistency_weight * consistency)
