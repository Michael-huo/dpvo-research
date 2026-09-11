"""Canonical bracketed robust-transport predictor contracts for Exp6 H2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import FrameIdentity, canonical_sha256


PREDICTOR_ARCHITECTURE = "robust_transport_block5_residual_spatial"
H2_PROTOCOL = "exp6_h2_robust_transport_bracketed_query"
SPLIT_METHOD = "interval_contiguous_60_20_20_drop_shared_boundaries_v1"


@dataclass(frozen=True)
class HiddenQuery:
    identity: FrameIdentity
    ordinal: int
    alpha: float
    delta_t_seconds: float

    def payload(self) -> dict[str, Any]:
        return {"identity": self.identity.public_dict(), "ordinal": int(self.ordinal),
                "alpha": float(self.alpha), "delta_t_seconds": float(self.delta_t_seconds)}


@dataclass(frozen=True)
class AnchorInterval:
    interval_index: int
    anchor0: FrameIdentity
    anchor1: FrameIdentity
    hidden: tuple[HiddenQuery, ...]

    def __post_init__(self) -> None:
        if not self.hidden or self.anchor0.timestamp_ns >= self.anchor1.timestamp_ns:
            raise ValueError("anchor interval must contain ordered hidden queries")
        if self.anchor0.key == self.anchor1.key:
            raise ValueError("anchor interval endpoints must be distinct")
        denominator = self.anchor1.timestamp_ns - self.anchor0.timestamp_ns
        for query in self.hidden:
            if not self.anchor0.timestamp_ns < query.identity.timestamp_ns < self.anchor1.timestamp_ns:
                raise ValueError("hidden query timestamp lies outside its bracket")
            expected = (query.identity.timestamp_ns - self.anchor0.timestamp_ns) / denominator
            if not np.isclose(query.alpha, expected, rtol=0.0, atol=1e-15):
                raise ValueError("query alpha was not derived from integer timestamps")

    @property
    def anchor_keys(self) -> tuple[str, str]:
        return self.anchor0.key, self.anchor1.key

    def payload(self) -> dict[str, Any]:
        return {"interval_index": int(self.interval_index),
                "anchor0": self.anchor0.public_dict(), "anchor1": self.anchor1.public_dict(),
                "hidden": [query.payload() for query in self.hidden]}


def _identity(value: Any) -> FrameIdentity:
    return value.identity if hasattr(value, "identity") else value


def build_anchor_intervals(records: Sequence[Any], roles: Mapping[str, str]) -> tuple[AnchorInterval, ...]:
    identities = [_identity(value) for value in records]
    if len({item.key for item in identities}) != len(identities):
        raise ValueError("frame identities must be unique")
    anchors = [index for index, item in enumerate(identities) if roles[item.key] == "anchor"]
    intervals: list[AnchorInterval] = []
    for left, right in zip(anchors[:-1], anchors[1:]):
        between = identities[left + 1:right]
        hidden = tuple(item for item in between if roles[item.key] == "hidden")
        if not hidden:
            continue
        if len(hidden) != len(between):
            raise ValueError("bracket contains an unexpected non-hidden identity")
        anchor0, anchor1 = identities[left], identities[right]
        delta_ns = anchor1.timestamp_ns - anchor0.timestamp_ns
        queries = tuple(HiddenQuery(
            item, ordinal, (item.timestamp_ns - anchor0.timestamp_ns) / delta_ns,
            delta_ns / 1e9,
        ) for ordinal, item in enumerate(hidden, start=1))
        intervals.append(AnchorInterval(len(intervals), anchor0, anchor1, queries))
    if not intervals:
        raise ValueError("schedule contains no complete hidden anchor interval")
    hidden_keys = [query.identity.key for interval in intervals for query in interval.hidden]
    if len(hidden_keys) != len(set(hidden_keys)):
        raise ValueError("a hidden identity belongs to multiple intervals")
    return tuple(intervals)


def split_anchor_intervals(intervals: Sequence[AnchorInterval], train_fraction: float = .60,
                           validation_fraction: float = .20
                           ) -> tuple[dict[str, tuple[AnchorInterval, ...]], dict[str, Any]]:
    count = len(intervals); train_cut = int(count * train_fraction)
    validation_cut = int(count * (train_fraction + validation_fraction))
    if not 0 < train_cut < validation_cut < count - 1:
        raise ValueError("too few intervals for boundary-gapped split")
    parts = {"train": tuple(intervals[:train_cut]),
             "validation": tuple(intervals[train_cut + 1:validation_cut]),
             "test": tuple(intervals[validation_cut + 1:])}
    anchors = {name: {key for row in values for key in row.anchor_keys}
               for name, values in parts.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if anchors[left] & anchors[right]:
            raise RuntimeError(f"split anchor leakage: {left}/{right}")
    def summary(values: Sequence[AnchorInterval]) -> dict[str, Any]:
        queries = [query for interval in values for query in interval.hidden]
        return {"interval_count": len(values), "hidden_query_count": len(queries),
                "first_anchor_identity": values[0].anchor0.key,
                "last_anchor_identity": values[-1].anchor1.key,
                "timestamp_start_ns": values[0].anchor0.timestamp_ns,
                "timestamp_end_ns": values[-1].anchor1.timestamp_ns,
                "interval_identity_sha256": canonical_sha256([row.payload() for row in values])}
    payload = {"method": SPLIT_METHOD, "source_interval_count": count,
               "train_cut": train_cut, "validation_cut": validation_cut,
               "dropped_boundary_interval_indices": [train_cut, validation_cut],
               "parts": {name: summary(values) for name, values in parts.items()}}
    payload["split_sha256"] = canonical_sha256(payload)
    return parts, payload


def interval_mapping_payload(intervals: Sequence[AnchorInterval]) -> dict[str, Any]:
    rows = {query.identity.key: {
        "anchor0_identity": interval.anchor0.key, "anchor1_identity": interval.anchor1.key,
        "alpha": float(query.alpha), "delta_t_seconds": float(query.delta_t_seconds),
        "hidden_ordinal": int(query.ordinal),
    } for interval in intervals for query in interval.hidden}
    if len(rows) != sum(len(interval.hidden) for interval in intervals):
        raise RuntimeError("hidden-to-bracket mapping is not one-to-one")
    return {"rows": rows, "mapping_sha256": canonical_sha256(rows)}


def effective_records(records: Sequence[Any], intervals: Sequence[AnchorInterval], *,
                      candidate_cap: int | None = None) -> tuple[tuple[Any, ...], dict[str, Any]]:
    selected = tuple(records if candidate_cap is None else records[:candidate_cap])
    keys = {_identity(value).key for value in selected}
    complete = [interval for interval in intervals if interval.anchor1.key in keys]
    if not complete:
        raise ValueError("population contains no complete bracket")
    closing = complete[-1].anchor1
    closing_index = next(index for index, value in enumerate(records)
                         if _identity(value).key == closing.key)
    values = tuple(records[:closing_index + 1])
    return values, {"requested_candidate_cap": candidate_cap,
                    "source_candidate_count": len(records),
                    "effective_candidate_count": len(values),
                    "last_complete_anchor_identity": closing.key,
                    "last_complete_anchor_candidate_index": closing.candidate_index,
                    "trailing_excluded_count": len(selected) - len(values),
                    "source_after_effective_count": len(records) - len(values),
                    "effective_candidate_identity_sha256": canonical_sha256([
                        _identity(value).key for value in values])}


class FiLMSpatialResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels); self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels); self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(value)
        hidden = hidden * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        hidden = self.conv1(F.gelu(hidden)); hidden = self.conv2(F.gelu(self.norm2(hidden)))
        return value + hidden


class RobustTransportBlock5Predictor(nn.Module):
    """Canonical 3.09M residual predictor over deployable robust transport."""

    def __init__(self, feature_dim: int = 768, hidden_dim: int = 256,
                 difference_dim: int = 128, reliability_dim: int = 32,
                 residual_blocks: int = 2, groups: int = 16,
                 time_hidden_dim: int = 128) -> None:
        super().__init__()
        if (feature_dim, hidden_dim, difference_dim, reliability_dim, residual_blocks,
                groups, time_hidden_dim) != (768, 256, 128, 32, 2, 16, 128):
            raise ValueError("canonical Exp6 H2 predictor architecture is frozen")
        self.feature_dim = feature_dim; self.hidden_dim = hidden_dim
        self.difference_dim = difference_dim; self.reliability_dim = reliability_dim
        self.residual_blocks_count = residual_blocks
        self.transport_projection = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.difference_projection = nn.Conv2d(feature_dim, difference_dim, 1)
        self.reliability_projection = nn.Conv2d(3, reliability_dim, 1)
        self.fusion = nn.Sequential(nn.Conv2d(hidden_dim + difference_dim + reliability_dim, hidden_dim, 1),
                                    nn.GroupNorm(groups, hidden_dim), nn.GELU())
        self.time_mlp = nn.Sequential(nn.Linear(2, time_hidden_dim), nn.GELU(),
                                      nn.Linear(time_hidden_dim, residual_blocks * 2 * hidden_dim))
        self.blocks = nn.ModuleList([FiLMSpatialResidualBlock(hidden_dim, groups)
                                     for _ in range(residual_blocks)])
        self.output = nn.Conv2d(hidden_dim, feature_dim, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, transport: torch.Tensor, warped_difference: torch.Tensor,
                coverage0: torch.Tensor, coverage1: torch.Tensor,
                fused_confidence: torch.Tensor, alpha: torch.Tensor,
                delta_t_seconds: torch.Tensor) -> torch.Tensor:
        if transport.ndim != 4 or transport.shape != warped_difference.shape:
            raise ValueError("transport/difference must be equal [B,768,H,W]")
        batch, channels, height, width = transport.shape
        if channels != self.feature_dim:
            raise ValueError("predictor requires raw 768D block5 fields")
        reliability = torch.cat((coverage0, coverage1, fused_confidence), dim=1)
        if reliability.shape != (batch, 3, height, width):
            raise ValueError("reliability must be coverage0/coverage1/fused-confidence")
        alpha = torch.as_tensor(alpha, device=transport.device, dtype=transport.dtype).reshape(batch)
        delta = torch.as_tensor(delta_t_seconds, device=transport.device,
                                dtype=transport.dtype).reshape(batch)
        if bool(((alpha <= 0) | (alpha >= 1) | (delta <= 0)).any().item()):
            raise ValueError("bracketed queries require 0 < alpha < 1 and delta_t > 0")
        hidden = self.fusion(torch.cat((self.transport_projection(transport),
                                        self.difference_projection(warped_difference),
                                        self.reliability_projection(reliability)), dim=1))
        conditioning = self.time_mlp(torch.stack((alpha, delta), dim=1)).reshape(
            batch, len(self.blocks), 2, self.hidden_dim)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, conditioning[:, index, 0], conditioning[:, index, 1])
        return transport + self.output(hidden)


def prediction_loss(prediction: torch.Tensor, target: torch.Tensor,
                    valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction/target must be equal [B,C,H,W]")
    mask = valid_mask.to(device=prediction.device, dtype=torch.bool)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(prediction.shape[-2:]):
        raise ValueError("valid token mask does not match spatial field")
    selected_prediction = prediction.float().permute(0, 2, 3, 1)[:, mask]
    selected_target = target.float().permute(0, 2, 3, 1)[:, mask]
    cosine = F.cosine_similarity(selected_prediction, selected_target, dim=-1, eps=1e-8)
    smooth_l1 = F.smooth_l1_loss(selected_prediction, selected_target)
    mse = F.mse_loss(selected_prediction, selected_target)
    norm_ratio = selected_prediction.norm(dim=-1).mean() / selected_target.norm(dim=-1).mean().clamp_min(1e-8)
    return {"total": 1 - cosine.mean() + .1 * smooth_l1, "cosine": cosine.mean(),
            "smooth_l1": smooth_l1, "mse": mse, "norm_ratio": norm_ratio}


def predictor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def predictor_metadata(model: RobustTransportBlock5Predictor) -> dict[str, Any]:
    return {"architecture": PREDICTOR_ARCHITECTURE, "protocol": H2_PROTOCOL,
            "feature_dim": model.feature_dim, "hidden_dim": model.hidden_dim,
            "difference_dim": model.difference_dim, "reliability_dim": model.reliability_dim,
            "reliability_channels": 3, "residual_blocks": model.residual_blocks_count,
            "group_norm_groups": 16, "time_mlp": [2, 128, 1024],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "input": "raw_robust_transport_raw_warp_difference_coverage0_coverage1_fused_confidence_alpha_delta_t",
            "output": "768D_residual_over_robust_transport",
            "output_projection_zero_initialized": True, "dynamic_non_square_grid": True}
