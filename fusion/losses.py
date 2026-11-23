"""Complementarity-enhanced class and sample relational losses."""

import torch
from torch.nn import functional as F


def relational_embeddings(features):
    normalized = F.normalize(features, dim=-1, eps=1e-6)
    origin = normalized.detach().mean(0, keepdim=True)
    return F.normalize(normalized - origin, dim=-1, eps=1e-6)


def class_relation_loss(fused, hsi, lidar, labels, temperature, beta):
    classes = int(labels.max().item()) + 1
    membership = F.one_hot(labels, classes).to(fused.dtype)
    counts = membership.sum(0)
    if (counts < 2).any():
        raise ValueError("Relational losses require at least two training samples per class.")
    with torch.no_grad():
        def separability(features):
            normalized = F.normalize(features.detach(), dim=-1, eps=1e-6)
            centers = membership.t() @ normalized / counts[:, None]
            residual = (normalized - centers[labels]).square().sum(-1)
            spread = (membership.t() @ residual) / counts
            between = (centers[:, None] - centers[None]).square().sum(-1)
            return between / (between + spread[:, None] + spread[None] + 1e-6)

        difference = (separability(hsi) - separability(lidar)).abs()
        weights = 1 + beta * difference[labels]
        weights.scatter_(1, labels[:, None], 1.0)
    embeddings = relational_embeddings(fused)
    totals = membership.t() @ embeddings
    centers = F.normalize(totals / counts[:, None], dim=-1, eps=1e-6)
    logits = embeddings @ centers.t() / temperature
    positive = F.normalize((totals[labels] - embeddings) / (counts[labels, None] - 1),
                           dim=-1, eps=1e-6)
    positive_logits = (embeddings * positive).sum(-1) / temperature
    logits = logits.scatter(1, labels[:, None], positive_logits[:, None]) + weights.log()
    return F.cross_entropy(logits, labels)


def sample_relation_loss(fused, labels, temperature):
    classes = int(labels.max().item()) + 1
    membership = F.one_hot(labels, classes).to(fused.dtype)
    counts = membership.sum(0)
    if (counts < 2).any():
        raise ValueError("Relational losses require at least two training samples per class.")
    identity = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positive = (labels[:, None] == labels[None, :]) & ~identity
    embeddings = relational_embeddings(fused)
    similarities = embeddings @ embeddings.t() / temperature
    class_membership = membership.t().bool()[None, :, :] & ~identity[:, None, :]
    available = counts[None, :] - membership
    scores = torch.logsumexp(similarities[:, None, :].masked_fill(~class_membership, -torch.inf), -1) - available.log()
    positive_affinity = torch.zeros_like(similarities).masked_fill(~positive, -torch.inf)
    positive_normalizer = torch.logsumexp(positive_affinity, 1)
    positive_scores = torch.logsumexp(similarities + positive_affinity, 1) - positive_normalizer
    scores = scores.scatter(1, labels[:, None], positive_scores[:, None])
    return F.cross_entropy(scores, labels)
