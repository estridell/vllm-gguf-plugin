from __future__ import annotations

from collections.abc import Sequence

import torch

_BLOCK_SIZE = 128


def _validate_shape(shape: Sequence[int], elements: int) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in shape)
    if not shape or any(dim < 0 for dim in shape):
        raise ValueError(f"Invalid tensor shape: {shape}")
    expected = 1
    for dim in shape:
        expected *= dim
    if expected != elements:
        raise ValueError(f"Shape {shape} has {expected} elements, expected {elements}")
    return shape


def _packed_blocks(packed: torch.Tensor, block_bytes: int) -> torch.Tensor:
    if packed.dtype is not torch.uint8:
        raise TypeError(f"Packed ternary data must be torch.uint8, got {packed.dtype}")
    if packed.shape[-1] % block_bytes != 0:
        raise ValueError(
            f"Packed row width {packed.shape[-1]} is not divisible by {block_bytes}"
        )
    return packed.contiguous().reshape(-1, block_bytes)


def _fp16_bytes(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float16).contiguous()
    return values.view(torch.uint8).reshape(-1, 2)


def dequant_q1_0(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    blocks = _packed_blocks(packed, 18)
    d = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    signs = ((blocks[:, 2:].unsqueeze(-1) >> shifts) & 1).reshape(-1, 128)
    values = torch.where(signs.bool(), d, -d)
    return values.reshape(_validate_shape(shape, values.numel()))


def dequant_q2_0(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    blocks = _packed_blocks(packed, 34)
    d = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    codes = ((blocks[:, 2:].unsqueeze(-1) >> shifts) & 3).reshape(-1, 128)
    values = (codes.to(torch.float32) - 1.0) * d
    return values.reshape(_validate_shape(shape, values.numel()))


def _quant_blocks(values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if not values.is_floating_point():
        raise TypeError(
            f"Ternary quantization requires floating-point input, got {values.dtype}"
        )
    if values.ndim == 0 or values.shape[-1] % _BLOCK_SIZE != 0:
        raise ValueError(
            f"Tensor row width must be divisible by {_BLOCK_SIZE}, got {values.shape}"
        )
    output_prefix = tuple(values.shape[:-1])
    blocks_per_row = values.shape[-1] // _BLOCK_SIZE
    blocks = values.to(torch.float32).contiguous().reshape(-1, _BLOCK_SIZE)
    return blocks, (*output_prefix, blocks_per_row)


def quantize_q1_0(values: torch.Tensor) -> torch.Tensor:
    blocks, output_shape = _quant_blocks(values)
    # Prism ggml uses the mean absolute value, computed in float32, as d.
    d = blocks.abs().sum(dim=-1, dtype=torch.float32) / _BLOCK_SIZE
    bits = (blocks >= 0).to(torch.uint8).reshape(-1, 16, 8)
    shifts = torch.arange(8, device=values.device, dtype=torch.uint8)
    qs = torch.sum(bits << shifts, dim=-1, dtype=torch.uint8)
    packed = torch.cat((_fp16_bytes(d), qs), dim=-1)
    return packed.reshape(*output_shape[:-1], output_shape[-1] * 18)


def quantize_q2_0(values: torch.Tensor) -> torch.Tensor:
    blocks, output_shape = _quant_blocks(values)
    # Prism ggml and gguf-py both use the unrounded float32 absolute maximum.
    d = blocks.abs().amax(dim=-1, keepdim=True)
    inverse = torch.where(d == 0, 0, 1.0 / d)
    normalized = blocks * inverse
    rounded = normalized.sign() * torch.floor(normalized.abs() + 0.5)
    codes = rounded.clamp(-1, 2).add_(1.0).to(torch.uint8)
    codes = codes.reshape(-1, 32, 4)
    shifts = torch.tensor([0, 2, 4, 6], device=values.device, dtype=torch.uint8)
    qs = torch.sum(codes << shifts, dim=-1, dtype=torch.uint8)
    packed = torch.cat((_fp16_bytes(d.reshape(-1)), qs), dim=-1)
    return packed.reshape(*output_shape[:-1], output_shape[-1] * 34)
