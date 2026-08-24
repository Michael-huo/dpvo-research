"""Models for JEPA-to-DPVO representation bridge validation."""

from __future__ import annotations

import math
import hashlib
import json
from functools import reduce
from operator import mul
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional


def _shape_tuple(values: Sequence[int]) -> tuple[int, ...]:
    shape = tuple(int(value) for value in values)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"invalid shape: {shape}")
    return shape


def _product(shape: Sequence[int]) -> int:
    return int(reduce(mul, shape, 1))


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


class SpatialDecoderBlock(nn.Module):
    """Memory-conscious depthwise/pointwise spatial refinement at fixed channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = math.gcd(channels, 32)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.block(tensor)


class JepaFMapAdapter(nn.Module):
    """Minimal token transformer and spatial decoder baseline."""

    def __init__(
        self,
        *,
        input_tokens: int = 576,
        input_dim: int = 768,
        token_grid: Sequence[int] = (24, 24),
        hidden_dim: int = 256,
        transformer_layers: int = 2,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        decoder_sizes: Sequence[Sequence[int]] = ((60, 94), (120, 188)),
        output_channels: int = 128,
        output_shape: Sequence[int] = (128, 120, 188),
        max_parameters: int = 20_000_000,
    ) -> None:
        super().__init__()
        self.input_tokens = int(input_tokens)
        self.input_dim = int(input_dim)
        self.token_grid = _shape_tuple(token_grid)
        self.hidden_dim = int(hidden_dim)
        self.decoder_sizes = tuple(_shape_tuple(size) for size in decoder_sizes)
        self.output_shape = _shape_tuple(output_shape)
        if len(self.token_grid) != 2 or _product(self.token_grid) != self.input_tokens:
            raise ValueError("token_grid must contain exactly input_tokens positions")
        if len(self.output_shape) != 3 or self.output_shape[0] != int(output_channels):
            raise ValueError("output_shape channel dimension must match output_channels")
        if not self.decoder_sizes or self.decoder_sizes[-1] != self.output_shape[1:]:
            raise ValueError("last decoder size must equal output spatial shape")
        if self.hidden_dim % int(attention_heads):
            raise ValueError("hidden_dim must be divisible by attention_heads")

        self.token_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, self.input_tokens, self.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(attention_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))
        self.input_refinement = SpatialDecoderBlock(self.hidden_dim)
        self.decoder_blocks = nn.ModuleList(SpatialDecoderBlock(self.hidden_dim) for _ in self.decoder_sizes)
        self.channel_projection = nn.Conv2d(self.hidden_dim, int(output_channels), kernel_size=1)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        parameter_count = count_parameters(self)
        if parameter_count >= int(max_parameters):
            raise ValueError(
                f"Adapter parameter limit exceeded: {parameter_count:,} >= {int(max_parameters):,}"
            )
        self.parameter_count = parameter_count
        self.max_parameters = int(max_parameters)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = (self.input_tokens, self.input_dim)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"expected JEPA tokens [B,{expected[0]},{expected[1]}], got {tuple(tokens.shape)}")
        hidden = self.token_projection(tokens) + self.position_embedding
        hidden = self.transformer(hidden)
        batch = hidden.shape[0]
        height, width = self.token_grid
        spatial = hidden.transpose(1, 2).reshape(batch, self.hidden_dim, height, width)
        spatial = self.input_refinement(spatial)
        for size, block in zip(self.decoder_sizes, self.decoder_blocks):
            spatial = functional.interpolate(spatial, size=size, mode="bilinear", align_corners=False)
            spatial = block(spatial)
        output = self.channel_projection(spatial)
        if tuple(output.shape[1:]) != self.output_shape:
            raise RuntimeError(f"adapter output shape drift: expected {self.output_shape}, got {tuple(output.shape[1:])}")
        return output


class LowRankLinearBaseline(nn.Module):
    """Global flatten-to-flatten linear capacity baseline, not a deployment model."""

    def __init__(
        self,
        *,
        input_shape: Sequence[int] = (576, 768),
        output_shape: Sequence[int] = (128, 120, 188),
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.input_shape = _shape_tuple(input_shape)
        self.output_shape = _shape_tuple(output_shape)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        self.input_dim = _product(self.input_shape)
        self.output_dim = _product(self.output_shape)
        self.input_projection = nn.Linear(self.input_dim, self.rank, bias=False)
        self.output_projection = nn.Linear(self.rank, self.output_dim, bias=True)
        self.parameter_count = count_parameters(self)

    @staticmethod
    def parameter_count_for_shapes(
        input_shape: Sequence[int], output_shape: Sequence[int], rank: int,
    ) -> int:
        input_dim = _product(_shape_tuple(input_shape))
        output_dim = _product(_shape_tuple(output_shape))
        return input_dim * int(rank) + int(rank) * output_dim + output_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != len(self.input_shape) + 1 or tuple(tokens.shape[1:]) != self.input_shape:
            raise ValueError(f"expected input [B,{self.input_shape}], got {tuple(tokens.shape)}")
        flat = tokens.reshape(tokens.shape[0], self.input_dim)
        output = self.output_projection(self.input_projection(flat))
        return output.reshape(tokens.shape[0], *self.output_shape)


class RandomPredictionBaseline(nn.Module):
    """Standard-normal metric floor; reproducibility is controlled by the run seed."""

    def __init__(self, output_shape: Sequence[int] = (128, 120, 188)) -> None:
        super().__init__()
        self.output_shape = _shape_tuple(output_shape)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 1:
            raise ValueError("tokens must have a batch dimension")
        return torch.randn(
            (tokens.shape[0], *self.output_shape),
            device=tokens.device,
            dtype=tokens.dtype,
        )


class MeanFMapBaseline(nn.Module):
    """Dataset-prior baseline using the train-set mean teacher at each position."""

    def __init__(self, mean_fmap: torch.Tensor) -> None:
        super().__init__()
        if mean_fmap.ndim != 3 or not torch.isfinite(mean_fmap).all().item():
            raise ValueError("mean_fmap must be a finite [C,H,W] tensor")
        self.register_buffer("mean_fmap", mean_fmap.detach().float().contiguous())
        self.output_shape = tuple(mean_fmap.shape)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 1:
            raise ValueError("tokens must have a batch dimension")
        return self.mean_fmap.to(dtype=tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1, -1)


def model_config_sha256(model_config: dict[str, Any]) -> str:
    encoded = json.dumps(model_config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_config_for_kind(config: dict[str, Any], kind: str, capacity: str = "small") -> dict[str, Any]:
    if kind == "adapter":
        profiles = config["capacity_profiles"]
        if capacity not in profiles:
            raise ValueError(f"unknown adapter capacity: {capacity}")
        return {**config["adapter"], **profiles[capacity]}
    if kind == "low_rank_linear":
        if capacity != "small":
            raise ValueError("LowRank Linear does not use adapter capacity profiles")
        values = dict(config["low_rank_linear"])
        values.pop("role", None)
        values.pop("deployment_model", None)
        return values
    raise ValueError(f"unsupported trainable model kind: {kind}")


def build_trainable_model(kind: str, model_config: dict[str, Any]) -> nn.Module:
    if kind == "adapter":
        return JepaFMapAdapter(**model_config)
    if kind == "low_rank_linear":
        return LowRankLinearBaseline(**model_config)
    raise ValueError(f"unsupported trainable model kind: {kind}")
