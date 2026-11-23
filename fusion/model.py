"""Hierarchical context modeling and multimodal-guided context return."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import get_configuration
from .wkv import biwkv


def convolution(input_channels, output_channels, kernel=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel, padding=kernel // 2,
                  groups=groups, bias=False),
        nn.GroupNorm(8, output_channels),
        nn.GELU(),
    )


class GroupedConvolution(nn.Conv2d):
    def forward(self, inputs):
        with torch.backends.cudnn.flags(allow_tf32=False):
            return super().forward(inputs)


def guided_read(values, own, other, own_grid, other_grid, quality, other_quality,
                acceptance, strength, minimum):
    batch, channels, height, width = values.shape
    target_height, target_width = own.shape[-2:]
    cy = (torch.arange(target_height, device=values.device, dtype=values.dtype) + 0.5) * (height / target_height) - 0.5
    cx = (torch.arange(target_width, device=values.device, dtype=values.dtype) + 0.5) * (width / target_width) - 0.5
    y0, x0 = cy.floor(), cx.floor()
    dy, dx = cy - y0, cx - x0
    query = F.normalize(own, dim=1).flatten(2)
    foreign_query = F.normalize(other, dim=1).flatten(2)
    keys = F.normalize(own_grid, dim=1).flatten(2)
    foreign_keys = F.normalize(other_grid, dim=1).flatten(2)
    own_weight = quality.flatten(2)
    cross_weight = (other_quality * acceptance).flatten(2)
    total = (own_weight + cross_weight).clamp_min(1e-8)
    numerator = values.new_zeros((batch, channels, target_height * target_width))
    denominator = values.new_zeros((batch, 1, target_height * target_width))
    flattened = values.flatten(2)
    for offset_y, wy in [(0, 1 - dy), (1, dy)]:
        yy = (y0.long() + offset_y).clamp(0, height - 1)
        for offset_x, wx in [(0, 1 - dx), (1, dx)]:
            xx = (x0.long() + offset_x).clamp(0, width - 1)
            indices = (yy[:, None] * width + xx[None, :]).reshape(-1)
            base = (wy[:, None] * wx[None, :]).reshape(1, 1, -1)
            distance = (query - keys[:, :, indices]).square().sum(1, keepdim=True)
            foreign_distance = (foreign_query - foreign_keys[:, :, indices]).square().sum(1, keepdim=True)
            joint = (own_weight * distance + cross_weight * foreign_distance) / total
            weight = base * torch.exp(-strength * joint)
            numerator = numerator + weight * flattened[:, :, indices]
            denominator = denominator + weight
    return (numerator / denominator.clamp_min(minimum)).reshape(
        batch, channels, target_height, target_width,
    )


class GlobalContext(nn.Module):
    def __init__(self, dimension, decay_bound, operator):
        super().__init__()
        self.operator = operator
        self.norm = nn.LayerNorm(dimension)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.rec = nn.Linear(dimension, dimension, bias=False)
        self.out = nn.Linear(dimension, dimension, bias=False)
        self.decay = nn.Parameter(torch.linspace(-decay_bound, decay_bound, dimension))
        self.first = nn.Parameter(torch.ones(dimension))
        self.ff = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension * 2),
            nn.GELU(), nn.Linear(dimension * 2, dimension),
        )

    def forward(self, features, quality):
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        length = height * width
        normalized = self.norm(tokens)
        key = self.key(normalized) + quality.flatten(2).transpose(1, 2).clamp_min(0.02).log()
        value = self.value(normalized)

        def column_major(tensor):
            return tensor.reshape(batch, height, width, channels).transpose(1, 2).reshape(batch, length, channels).contiguous()

        decay, first = self.decay / length, self.first / length
        row = biwkv(decay, first, key, value, self.operator)
        column = biwkv(decay, first, column_major(key), column_major(value), self.operator)
        column = column.reshape(batch, width, height, channels).transpose(1, 2).reshape(batch, length, channels)
        mixed = (row + column) * 0.5
        output = self.out(torch.sigmoid(self.rec(normalized)) * mixed)
        return output.transpose(1, 2).reshape(batch, channels, height, width)


class HCDRFusion(nn.Module):
    def __init__(self, hsi_channels, lidar_channels, classes, dataset):
        super().__init__()
        self.configuration = get_configuration(dataset)
        cfg = self.configuration
        dim = cfg.dimension
        self.hstem = nn.Sequential(convolution(hsi_channels, dim), convolution(dim, dim))
        self.lstem = nn.Sequential(convolution(lidar_channels + 2, dim), convolution(dim, dim))
        self.hlocal = nn.Sequential(convolution(dim, dim, 5, dim), convolution(dim, dim), convolution(dim, dim, 3, dim))
        self.llocal = nn.Sequential(convolution(dim, dim, 5, dim), convolution(dim, dim), convolution(dim, dim, 3, dim))
        self.control = nn.Sequential(convolution(dim * 4 + 4, dim), nn.Conv2d(dim, 8, 1))
        nn.init.zeros_(self.control[-1].weight)
        nn.init.zeros_(self.control[-1].bias)
        self.hg = GlobalContext(dim, cfg.decay_bound, cfg.operator)
        self.lg = GlobalContext(dim, cfg.decay_bound, cfg.operator)
        self.l_to_h = convolution(dim, dim)
        self.h_to_l = convolution(dim, dim)
        self.head = nn.Sequential(convolution(dim * 4, dim * 2), nn.Dropout2d(cfg.dropout), nn.Conv2d(dim * 2, classes, 1))
        if cfg.compact_head:
            previous = self.head[0][0]
            with torch.random.fork_rng(devices=[]):
                projection = nn.Conv2d(dim * 2, dim * 2, 1, bias=False)
            with torch.no_grad():
                projection.weight.copy_(previous.weight[:, dim * 2:])
            self.head[0][0] = projection
        for branch, dilation in [(self.hlocal, cfg.hsi_dilation), (self.llocal, cfg.lidar_dilation)]:
            branch[0][0].dilation = (dilation, dilation)
            branch[0][0].padding = (2 * dilation, 2 * dilation)
        if cfg.lidar_groups:
            previous = self.llocal[0][0]
            with torch.random.fork_rng(devices=[]):
                grouped = GroupedConvolution(dim, dim, 5, padding=previous.padding,
                                             dilation=previous.dilation, groups=cfg.lidar_groups, bias=False)
            with torch.no_grad():
                grouped.weight.zero_()
                index = torch.arange(dim)
                grouped.weight[index, index % (dim // cfg.lidar_groups)] = previous.weight[:, 0]
            self.llocal[0][0] = grouped
        for branch, kernel in [(self.hlocal, cfg.hsi_kernel), (self.llocal, cfg.lidar_kernel)]:
            if kernel == 5:
                previous = branch[2][0]
                with torch.random.fork_rng(devices=[]):
                    refinement = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
                with torch.no_grad():
                    refinement.weight.zero_()
                    refinement.weight[:, :, 1:4, 1:4] = previous.weight
                branch[2][0] = refinement

    def feature(self, module, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False, preserve_rng_state=False)
        return module(*args)

    def pool(self, features, quality, step=None):
        step = self.configuration.stride if step is None else step
        pooled_quality = F.avg_pool2d(quality, step, step, ceil_mode=True, count_include_pad=False)
        pooled = F.avg_pool2d(features * quality, step, step, ceil_mode=True,
                             count_include_pad=False) / pooled_quality.clamp_min(0.02)
        return pooled, pooled_quality

    def regional(self, module, features):
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        output = tokens + 0.2 * module.ff(tokens)
        return output.transpose(1, 2).reshape(batch, channels, height, width)

    def read(self, *args, strength):
        minimum = 1e-30 if self.configuration.dataset == "trento" else 1e-12
        if self.training and torch.is_grad_enabled():
            return checkpoint(guided_read, *args, strength, minimum,
                              use_reentrant=False, preserve_rng_state=False)
        return guided_read(*args, strength, minimum)

    def encode(self, hsi, lidar):
        cfg = self.configuration
        height = lidar[:, :1]
        residual = height - F.avg_pool2d(height, 5, 1, 2, count_include_pad=False)
        h = self.feature(self.hstem, hsi)
        l = self.feature(self.lstem, torch.cat([lidar, residual, residual.abs()], 1))
        local_h = self.feature(self.hlocal, h)
        local_l = self.feature(self.llocal, l)
        mean = F.avg_pool2d(hsi, 5, 1, 2, count_include_pad=False)
        deviation = (hsi - mean).square().mean(1, keepdim=True).sqrt()
        magnitude = hsi.square().mean(1, keepdim=True).sqrt()
        descriptors = torch.cat([deviation, magnitude, residual, residual.abs()], 1)
        raw = torch.sigmoid(self.feature(self.control, torch.cat([h, l, local_h, local_l, descriptors], 1)))
        context = raw[:, 2:4].mean((-2, -1), keepdim=True).expand_as(raw[:, 2:4])
        quality = raw[:, :2].mean((-2, -1), keepdim=True).expand_as(raw[:, :2])
        acceptance = raw[:, 4:6].mean((-2, -1), keepdim=True).expand_as(raw[:, 4:6])
        gates = torch.cat([quality, context, acceptance, raw[:, 6:]], 1)
        qh, ql = 0.05 + 0.95 * gates[:, :1], 0.05 + 0.95 * gates[:, 1:2]
        lh = h + gates[:, 6:7] * local_h
        ll = l + gates[:, 7:8] * local_l
        ph, pqh = self.pool(lh, qh)
        pl, pql = self.pool(ll, ql)
        regional_h = self.feature(self.regional, self.hg, ph)
        regional_l = self.feature(self.regional, self.lg, pl)
        deep_h, dqh = self.pool(regional_h, pqh, 2)
        deep_l, dql = self.pool(regional_l, pql, 2)
        increment_h = self.feature(self.hg, deep_h, dqh)
        increment_l = self.feature(self.lg, deep_l, dql)
        ch, _ = self.pool(h, qh)
        cl, _ = self.pool(l, ql)
        kh, _ = self.pool(ch, pqh, 2)
        kl, _ = self.pool(cl, pql, 2)
        alh = F.avg_pool2d(gates[:, 4:5], cfg.stride, cfg.stride, ceil_mode=True, count_include_pad=False)
        ahl = F.avg_pool2d(gates[:, 5:6], cfg.stride, cfg.stride, ceil_mode=True, count_include_pad=False)
        coarse_h = regional_h + self.read(increment_h, ch, cl, kh, kl, pqh, pql, alh, strength=cfg.regional_strength)
        coarse_l = regional_l + self.read(increment_l, cl, ch, kl, kh, pql, pqh, ahl, strength=cfg.regional_strength)
        state = dict(h=h, l=l, lh=lh, ll=ll, gates=gates, qh=qh, ql=ql,
                     read_inputs=(ch, cl, coarse_h, coarse_l))
        state.update(self.context_reads(state))
        return state

    def context_reads(self, state, omitted_path=None):
        ch, cl, coarse_h, coarse_l = state["read_inputs"]
        gates = state["gates"]
        alh, ahl = gates[:, 4:5], gates[:, 5:6]
        if omitted_path == "lidar_to_hsi":
            alh = alh * 0
        if omitted_path == "hsi_to_lidar":
            ahl = ahl * 0
        values = torch.cat([coarse_h, coarse_l], 1)
        dim = coarse_h.shape[1]
        hread = self.read(values, state["h"], state["l"], ch, cl, state["qh"], state["ql"], alh,
                          strength=self.configuration.local_strength)
        lread = self.read(values, state["l"], state["h"], cl, ch, state["ql"], state["qh"], ahl,
                          strength=self.configuration.local_strength)
        return dict(gh=hread[:, :dim], gl_to_h=hread[:, dim:],
                    gh_to_l=lread[:, :dim], gl=lread[:, dim:])

    def classify(self, state, omitted_path=None):
        if omitted_path in ["lidar_to_hsi", "hsi_to_lidar"]:
            state = state | self.context_reads(state, omitted_path)
        gates = state["gates"]
        dh, dl, alh, ahl = gates[:, 2:3], gates[:, 3:4], gates[:, 4:5], gates[:, 5:6]
        if omitted_path == "context":
            dh, dl = dh * 0, dl * 0
        if omitted_path == "lidar_to_hsi":
            alh = alh * 0
        if omitted_path == "hsi_to_lidar":
            ahl = ahl * 0
        fh = state["lh"] + dh * (state["gh"] + alh * self.feature(self.l_to_h, state["gl_to_h"]))
        fl = state["ll"] + dl * (state["gl"] + ahl * self.feature(self.h_to_l, state["gh_to_l"]))
        features = [fh, fl] if self.configuration.compact_head else [state["h"], state["l"], fh, fl]
        hidden = self.feature(self.head[0], torch.cat(features, 1))
        return self.head[2](self.head[1](hidden)), hidden

    def decode(self, state, omitted_path=None):
        return self.classify(state, omitted_path)[0]

    def forward(self, hsi, lidar, training_mask=None):
        state = self.encode(hsi, lidar)
        logits, hidden = self.classify(state)
        if training_mask is not None:
            state["embedding"] = hidden.permute(0, 2, 3, 1)[training_mask]
        return logits, state
