"""Bidirectional weighted key-value aggregation."""

from functools import lru_cache
from pathlib import Path

import torch


def tensor_biwkv(decay, first, key, value):
    length = key.shape[1]
    position = torch.arange(length, device=key.device, dtype=key.dtype).view(1, length, 1)
    left_key = key + position * decay
    right_key = key - position * decay
    left_max = left_key.amax(1, keepdim=True).detach()
    right_max = right_key.amax(1, keepdim=True).detach()
    left_weight = (left_key - left_max).exp()
    right_weight = (right_key - right_max).exp()

    def prefix_exclusive(tensor):
        return torch.cat([torch.zeros_like(tensor[:, :1]), tensor[:, :-1].cumsum(1)], 1)

    def suffix_exclusive(tensor):
        return prefix_exclusive(tensor.flip(1)).flip(1)

    left_num = prefix_exclusive(left_weight * value)
    left_den = prefix_exclusive(left_weight)
    right_num = suffix_exclusive(right_weight * value)
    right_den = suffix_exclusive(right_weight)
    left_scale = left_max - (position - 1) * decay
    right_scale = right_max + (position + 1) * decay
    diagonal = key + first
    normalizer = torch.maximum(torch.maximum(left_scale, right_scale), diagonal).detach()
    left = (left_scale - normalizer).exp()
    right = (right_scale - normalizer).exp()
    current = (diagonal - normalizer).exp()
    numerator = left * left_num + right * right_num + current * value
    denominator = left * left_den + right * right_den + current
    return numerator / denominator.clamp_min(torch.finfo(key.dtype).tiny)


@lru_cache(maxsize=1)
def cuda_extension():
    from torch.utils.cpp_extension import load

    directory = Path(__file__).resolve().parent / "cuda"
    return load(
        name="hcdr_bi_wkv",
        sources=[str(directory / "bi_wkv.cpp"), str(directory / "bi_wkv_kernel.cu")],
        verbose=False,
        extra_cuda_cflags=[
            "--use_fast_math", "-O3", "-Xptxas=-O3",
            "-DTmax=262144", "-allow-unsupported-compiler",
        ],
    )


class CudaBiWKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decay, first, key, value):
        ctx.save_for_backward(decay, first, key, value)
        output = cuda_extension().bi_wkv_forward(
            decay.float().contiguous(), first.float().contiguous(),
            key.float().contiguous(), value.float().contiguous(),
        )
        return output.to(key.dtype)

    @staticmethod
    def backward(ctx, gradient):
        tensors = ctx.saved_tensors
        gradients = cuda_extension().bi_wkv_backward(
            *(tensor.float().contiguous() for tensor in tensors),
            gradient.float().contiguous(),
        )
        return tuple(grad.to(tensor.dtype) for grad, tensor in zip(gradients, tensors))


def biwkv(decay, first, key, value, operator):
    if operator == "cuda" and key.is_cuda:
        return CudaBiWKV.apply(decay, first, key, value)
    return tensor_biwkv(decay, first, key, value)
