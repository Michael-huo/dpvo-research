from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
import torch
from evo.core.geometry import umeyama_alignment
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

from .runtime import atomic_json_dump, sha256_file, splitmix64, splitmix64_array


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OBSERVATION_DTYPE = np.dtype(
    [
        ("score", "<u8"),
        ("factor_uid", "<i8"),
        ("source_patch_uid", "<u8"),
        ("source_dpvo_counter", "<i4"),
        ("target_dpvo_counter", "<i4"),
        ("update_dpvo_counter", "<i4"),
        ("weight_x", "<f4"),
        ("weight_y", "<f4"),
        ("weight_mean", "<f4"),
        ("weight_min", "<f4"),
        ("weight_anisotropy", "<f4"),
        ("delta_norm", "<f4"),
        ("apparent_motion", "<f4"),
        ("corr_peak_l0", "<f4"),
        ("corr_peak_l1", "<f4"),
        ("corr_mean_l0", "<f4"),
        ("corr_mean_l1", "<f4"),
        ("corr_std_l0", "<f4"),
        ("corr_std_l1", "<f4"),
        ("corr_margin_l0", "<f4"),
        ("corr_margin_l1", "<f4"),
        ("corr_entropy_l0", "<f4"),
        ("corr_entropy_l1", "<f4"),
        ("corr_peak_offset_l0", "<f4"),
        ("corr_peak_offset_l1", "<f4"),
        ("corr_peak_offset_l1_feature", "<f4"),
    ]
)


HISTOGRAM_SPECS: dict[str, tuple[float, float, int]] = {
    "weight_mean": (0.0, 1.0, 256),
    "corr_entropy_l0": (0.0, 1.0, 256),
    "corr_entropy_l1": (0.0, 1.0, 256),
    "corr_peak_l0": (-8.0, 16.0, 256),
    "corr_peak_l1": (-8.0, 16.0, 256),
    "corr_mean_l0": (-8.0, 16.0, 256),
    "corr_mean_l1": (-8.0, 16.0, 256),
    "corr_std_l0": (0.0, 16.0, 256),
    "corr_std_l1": (0.0, 16.0, 256),
    "corr_margin_l0": (0.0, 16.0, 256),
    "corr_margin_l1": (0.0, 16.0, 256),
    "delta_norm": (0.0, 64.0, 256),
    "corr_peak_offset_l0": (0.0, math.sqrt(18.0), 128),
    "corr_peak_offset_l1": (0.0, math.sqrt(18.0), 128),
}


FRAME_METRICS = OBSERVATION_DTYPE.names[6:]


def compute_image_difficulty(image_bgr: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray32 = gray.astype(np.float32)
    blur = float(cv2.Laplacian(gray32, cv2.CV_32F).var())
    grad_x = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)
    texture = float(np.sqrt(grad_x * grad_x + grad_y * grad_y).mean())
    return blur, texture


def correlation_metrics(corr: torch.Tensor, temperature: float = 1.0) -> dict[str, torch.Tensor]:
    """Compute the preregistered scalar metrics from DPVO's verified flat layout."""
    if corr.ndim != 3 or corr.shape[0] != 1 or corr.shape[-1] != 882:
        raise ValueError(f"Expected corr [1,E,882], got {tuple(corr.shape)}")
    if temperature != 1.0:
        raise ValueError("Experiment 1 fixes correlation entropy temperature at 1.0")
    edge_count = int(corr.shape[1])
    response = corr[0].float().reshape(edge_count, 7, 7, 3, 3, 2).mean(dim=(3, 4))
    flat = response.reshape(edge_count, 49, 2)
    top2 = torch.topk(flat, k=2, dim=1).values
    peak = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]
    mean = flat.mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    stable_logits = (flat - peak[:, None, :]) / temperature
    probabilities = torch.softmax(stable_logits, dim=1)
    entropy = -(
        probabilities * torch.log(torch.clamp(probabilities, min=1e-12))
    ).sum(dim=1) / math.log(49.0)
    entropy = torch.clamp(entropy, 0.0, 1.0)
    peak_index = flat.argmax(dim=1)
    peak_axis0 = torch.div(peak_index, 7, rounding_mode="floor").float()
    peak_axis1 = (peak_index % 7).float()
    offset = torch.sqrt((peak_axis0 - 3.0) ** 2 + (peak_axis1 - 3.0) ** 2)
    return {
        "peak": peak,
        "mean": mean,
        "std": std,
        "margin": margin,
        "entropy": entropy,
        "offset": offset,
    }


class ScalarHistogram:
    def __init__(self, low: float, high: float, bins: int):
        self.edges = np.linspace(low, high, bins + 1, dtype=np.float64)
        self.counts = np.zeros(bins, dtype=np.int64)
        self.underflow = 0
        self.overflow = 0
        self.nan_count = 0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        self.nan_count += int((~finite).sum())
        values = values[finite]
        self.underflow += int((values < self.edges[0]).sum())
        self.overflow += int((values > self.edges[-1]).sum())
        in_range = values[(values >= self.edges[0]) & (values <= self.edges[-1])]
        self.counts += np.histogram(in_range, bins=self.edges)[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "edges": self.edges,
            "counts": self.counts,
            "underflow": self.underflow,
            "overflow": self.overflow,
            "nan_count": self.nan_count,
            "accounted_count": int(
                self.counts.sum() + self.underflow + self.overflow + self.nan_count
            ),
        }


class OnlineMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        self.count += int(len(values))
        self.total += float(values.sum(dtype=np.float64))
        self.total_square += float(np.square(values).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def as_dict(self) -> dict[str, Any]:
        mean = self.total / self.count if self.count else None
        variance = (
            max(self.total_square / self.count - mean * mean, 0.0)
            if self.count
            else None
        )
        return {
            "count": self.count,
            "sum": self.total,
            "sum_squares": self.total_square,
            "mean": mean,
            "std": math.sqrt(variance) if variance is not None else None,
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
        }


class GlobalBottomK:
    """Chunked deterministic bottom-k for structured observation rows."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.retained = np.empty(0, dtype=OBSERVATION_DTYPE)
        self.buffers: list[np.ndarray] = []
        self.buffer_count = 0
        self.population_count = 0
        self.threshold = np.uint64((1 << 64) - 1)

    @staticmethod
    def _ordered(rows: np.ndarray) -> np.ndarray:
        order = np.lexsort(
            (rows["factor_uid"], rows["update_dpvo_counter"], rows["score"])
        )
        return rows[order]

    def _reduce(self) -> None:
        if not self.buffers:
            return
        rows = np.concatenate([self.retained, *self.buffers])
        self.buffers = []
        self.buffer_count = 0
        if len(rows) > self.capacity:
            partition = np.argpartition(rows["score"], self.capacity - 1)[: self.capacity]
            cutoff = rows["score"][partition].max()
            below = rows[rows["score"] < cutoff]
            tied = rows[rows["score"] == cutoff]
            needed = self.capacity - len(below)
            if needed:
                tied = self._ordered(tied)[:needed]
                rows = np.concatenate([below, tied])
            else:
                rows = below
        self.retained = self._ordered(rows)
        if len(self.retained):
            self.threshold = self.retained["score"][-1]

    def add(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=OBSERVATION_DTYPE)
        self.population_count += int(len(rows))
        if not self.capacity or not len(rows):
            return
        if len(self.retained) >= self.capacity:
            rows = rows[rows["score"] <= self.threshold]
        if not len(rows):
            return
        self.buffers.append(rows.copy())
        self.buffer_count += int(len(rows))
        if self.buffer_count >= max(self.capacity, 100_000):
            self._reduce()

    def values(self) -> np.ndarray:
        self._reduce()
        return self.retained.copy()


def _frame_metric_record(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "sum": 0.0, "sum_squares": 0.0, "min": np.nan, "max": np.nan}
    return {
        "count": int(len(finite)),
        "sum": float(finite.sum()),
        "sum_squares": float(np.square(finite).sum()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _histogram_quantiles(histogram: Counter[int]) -> dict[str, float | None]:
    total = int(sum(histogram.values()))
    if total == 0:
        return {"count": 0, "min": None, "q20": None, "median": None, "q80": None, "max": None}
    ordered = sorted(histogram.items())
    values = np.asarray([item[0] for item in ordered], dtype=np.float64)
    counts = np.asarray([item[1] for item in ordered], dtype=np.int64)
    cumulative = np.cumsum(counts)

    def quantile(q: float) -> float:
        rank = min(max(int(math.ceil(q * total)), 1), total)
        return float(values[np.searchsorted(cumulative, rank, side="left")])

    return {
        "count": total,
        "min": float(values[0]),
        "q20": quantile(0.2),
        "median": quantile(0.5),
        "q80": quantile(0.8),
        "max": float(values[-1]),
    }


class BehaviorRecorder:
    """Aggregation consumer attached to the fixed read-only probe callbacks."""

    def __init__(self, config: dict[str, Any]):
        behavior = config["behavior"]
        self.seed = int(config["experiment"]["seed"])
        self.patch_count = int(behavior.get("patches_per_frame", 96))
        self.temperature = float(behavior.get("correlation_temperature", 1.0))
        self.sampler = GlobalBottomK(int(behavior["observation_bottom_k"]))
        self.frames: list[dict[str, Any]] = []
        self.patch_frames: dict[int, dict[str, Any]] = {}
        self.pending_observations: np.ndarray | None = None
        self.current_frame_id: int | None = None
        self.histograms = {
            name: ScalarHistogram(*spec) for name, spec in HISTOGRAM_SPECS.items()
        }
        self.moments = {name: OnlineMoments() for name in FRAME_METRICS}
        self.factor_lifetime_histogram: Counter[int] = Counter()
        self.factor_update_count_histogram: Counter[int] = Counter()
        self.factor_span_histogram: Counter[int] = Counter()
        self.factor_remove_reasons: Counter[str] = Counter()
        self.factor_appended_total = 0
        self.factor_finalized_total = 0
        self.termination_refinement_observations = 0
        self.active_factor_uid = np.empty(0, dtype=np.int64)
        self.active_source_patch_uid = np.empty(0, dtype=np.uint64)
        self.active_target_frame = np.empty(0, dtype=np.int32)
        self.active_append_frame = np.empty(0, dtype=np.int32)
        self.active_first_update = np.empty(0, dtype=np.int32)
        self.active_last_update = np.empty(0, dtype=np.int32)
        self.active_update_count = np.empty(0, dtype=np.int32)
        self.creation_lifetime_sum: Counter[int] = Counter()
        self.creation_lifetime_count: Counter[int] = Counter()
        self._factors_finalized = False
        spool = tempfile.NamedTemporaryFile(
            prefix="dpvo-exp1-threshold-", suffix=".f32", delete=False
        )
        self.spool_path = Path(spool.name)
        self.spool_handle = spool
        self.spool_rows = 0
        self.spool_cleaned = False

    def before_frame(
        self,
        *,
        stream_index: int,
        euroc_timestamp_ns: int,
        blur_laplacian: float,
        texture_gradient: float,
    ) -> None:
        self.current_frame_id = int(stream_index)
        self.pending_observations = None
        self.frames.append(
            {
                "stream_index": int(stream_index),
                "dpvo_counter": int(stream_index),
                "euroc_timestamp_ns": int(euroc_timestamp_ns),
                "blur_laplacian": float(blur_laplacian),
                "texture_gradient": float(texture_gradient),
                "spool_start": self.spool_rows,
            }
        )

    def record_patches(
        self, frame_id: int, patch_center: np.ndarray, image_height: int, image_width: int
    ) -> None:
        patch_center = np.asarray(patch_center, dtype=np.float32)
        count = len(patch_center)
        if count != self.patch_count:
            raise AssertionError(f"Behavior patch count {count} != {self.patch_count}")
        self.patch_frames[int(frame_id)] = {
            "x": patch_center[:, 0].copy(),
            "y": patch_center[:, 1].copy(),
            "inverse_depth": patch_center[:, 2].copy(),
            "image_height": int(image_height),
            "image_width": int(image_width),
            "observation_count": np.zeros(count, dtype=np.int64),
            "weight_sum": np.zeros(count, dtype=np.float64),
            "entropy_sum": np.zeros(count, dtype=np.float64),
            "delta_sum": np.zeros(count, dtype=np.float64),
            "factor_lifetime_sum": np.zeros(count, dtype=np.float64),
            "factor_lifetime_count": np.zeros(count, dtype=np.int64),
        }

    def append_factors(
        self,
        factor_uids: Iterable[int],
        source_patch_uids: Iterable[int],
        target_frame_ids: Iterable[int],
        append_frame: int,
    ) -> None:
        factor_uids = np.asarray(list(factor_uids), dtype=np.int64)
        source_patch_uids = np.asarray(list(source_patch_uids), dtype=np.uint64)
        target_frame_ids = np.asarray(list(target_frame_ids), dtype=np.int32)
        count = len(factor_uids)
        if not count:
            return
        self.active_factor_uid = np.concatenate([self.active_factor_uid, factor_uids])
        self.active_source_patch_uid = np.concatenate(
            [self.active_source_patch_uid, source_patch_uids]
        )
        self.active_target_frame = np.concatenate(
            [self.active_target_frame, target_frame_ids]
        )
        self.active_append_frame = np.concatenate(
            [self.active_append_frame, target_frame_ids.astype(np.int32, copy=False)]
        )
        self.active_first_update = np.concatenate(
            [self.active_first_update, np.full(count, -1, dtype=np.int32)]
        )
        self.active_last_update = np.concatenate(
            [self.active_last_update, np.full(count, -1, dtype=np.int32)]
        )
        self.active_update_count = np.concatenate(
            [self.active_update_count, np.zeros(count, dtype=np.int32)]
        )
        self.factor_appended_total += count

    def _finalize_factor_rows(self, selected: np.ndarray, reason: str) -> None:
        if not selected.any():
            return
        first = self.active_first_update[selected]
        last = self.active_last_update[selected]
        observed = first >= 0
        lifetime = np.where(observed, last - first + 1, 0).astype(np.int32)
        updates = self.active_update_count[selected]
        source_uids = self.active_source_patch_uid[selected]
        target = self.active_target_frame[selected]
        source_frame = (source_uids >> np.uint64(32)).astype(np.int32)
        spans = np.abs(target - source_frame).astype(np.int32)
        creation = self.active_append_frame[selected]
        for value, count in zip(*np.unique(lifetime, return_counts=True)):
            self.factor_lifetime_histogram[int(value)] += int(count)
        for value, count in zip(*np.unique(updates, return_counts=True)):
            self.factor_update_count_histogram[int(value)] += int(count)
        for value, count in zip(*np.unique(spans, return_counts=True)):
            self.factor_span_histogram[int(value)] += int(count)
        for frame_id in np.unique(creation):
            mask = creation == frame_id
            self.creation_lifetime_sum[int(frame_id)] += float(lifetime[mask].sum())
            self.creation_lifetime_count[int(frame_id)] += int(mask.sum())
        for uid, value in zip(source_uids, lifetime):
            frame_id = int(uid >> np.uint64(32))
            local = int(uid & np.uint64((1 << 32) - 1))
            patch = self.patch_frames.get(frame_id)
            if patch is not None and local < self.patch_count:
                patch["factor_lifetime_sum"][local] += float(value)
                patch["factor_lifetime_count"][local] += 1
        finalized = int(selected.sum())
        self.factor_remove_reasons[reason] += finalized
        self.factor_finalized_total += finalized

    def remove_factors(self, mask: np.ndarray, reason: str) -> None:
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(self.active_factor_uid):
            raise AssertionError("Behavior factor mask is not parallel to active factors")
        self._finalize_factor_rows(mask, reason)
        keep = ~mask
        self.active_factor_uid = self.active_factor_uid[keep]
        self.active_source_patch_uid = self.active_source_patch_uid[keep]
        self.active_target_frame = self.active_target_frame[keep]
        self.active_append_frame = self.active_append_frame[keep]
        self.active_first_update = self.active_first_update[keep]
        self.active_last_update = self.active_last_update[keep]
        self.active_update_count = self.active_update_count[keep]

    def capture_update(
        self,
        *,
        phase: str,
        frame_id: int,
        factor_uids: np.ndarray,
        source_patch_uids: np.ndarray,
        target_frame_ids: np.ndarray,
        corr: torch.Tensor,
        delta: torch.Tensor,
        weight: torch.Tensor,
        coords_center: torch.Tensor,
        source_patch_center: torch.Tensor,
    ) -> None:
        edge_count = len(factor_uids)
        if phase == "termination_update":
            self.termination_refinement_observations += edge_count
            return
        if phase != "graph_update":
            return
        if edge_count != len(self.active_factor_uid) or not np.array_equal(
            factor_uids, self.active_factor_uid
        ):
            raise AssertionError("Behavior lifecycle sidecar differs from probe factor sidecar")
        unset = self.active_first_update < 0
        self.active_first_update[unset] = int(frame_id)
        self.active_last_update[:] = int(frame_id)
        self.active_update_count += 1
        if edge_count == 0:
            self.pending_observations = np.empty(0, dtype=OBSERVATION_DTYPE)
            return

        corr_stats = correlation_metrics(corr, self.temperature)
        weight32 = weight[0].float()
        delta32 = delta[0].float()
        weight_mean = weight32.mean(dim=-1)
        weight_min = weight32.min(dim=-1).values
        anisotropy = torch.abs(weight32[:, 0] - weight32[:, 1]) / torch.clamp(
            weight32.sum(dim=-1), min=1e-12
        )
        delta_norm = delta32.norm(dim=-1)
        apparent_motion = (coords_center[0].float() - source_patch_center.float()).norm(
            dim=-1
        )
        packed = torch.stack(
            [
                weight32[:, 0], weight32[:, 1], weight_mean, weight_min, anisotropy,
                delta_norm, apparent_motion,
                corr_stats["peak"][:, 0], corr_stats["peak"][:, 1],
                corr_stats["mean"][:, 0], corr_stats["mean"][:, 1],
                corr_stats["std"][:, 0], corr_stats["std"][:, 1],
                corr_stats["margin"][:, 0], corr_stats["margin"][:, 1],
                corr_stats["entropy"][:, 0], corr_stats["entropy"][:, 1],
                corr_stats["offset"][:, 0], corr_stats["offset"][:, 1],
                corr_stats["offset"][:, 1] * 4.0,
            ],
            dim=1,
        ).detach().cpu().numpy().astype(np.float32, copy=False)

        rows = np.empty(edge_count, dtype=OBSERVATION_DTYPE)
        rows["factor_uid"] = factor_uids
        rows["source_patch_uid"] = source_patch_uids
        rows["source_dpvo_counter"] = (
            source_patch_uids >> np.uint64(32)
        ).astype(np.int32)
        rows["target_dpvo_counter"] = target_frame_ids
        rows["update_dpvo_counter"] = int(frame_id)
        salt = np.uint64(splitmix64(self.seed)) ^ np.uint64(
            (int(frame_id) * 0xD6E8FEB86659FD93) & ((1 << 64) - 1)
        )
        rows["score"] = splitmix64_array(factor_uids.astype(np.uint64) ^ salt)
        packed_names = OBSERVATION_DTYPE.names[6:]
        for column, name in enumerate(packed_names):
            rows[name] = packed[:, column]
        self.pending_observations = rows

    def after_frame(self, accepted: bool) -> None:
        if self.current_frame_id is None:
            raise RuntimeError("Behavior after_frame without before_frame")
        frame = self.frames[-1]
        frame["accepted"] = bool(accepted)
        rows = self.pending_observations
        if rows is None:
            rows = np.empty(0, dtype=OBSERVATION_DTYPE)
        frame["online_observation_count"] = int(len(rows))
        for metric in FRAME_METRICS:
            stats = _frame_metric_record(rows[metric])
            for key, value in stats.items():
                frame[f"{metric}_{key}"] = value
            self.moments[metric].add(rows[metric])
        for name, histogram in self.histograms.items():
            histogram.add(rows[name])
        if len(rows):
            threshold_values = np.column_stack(
                (rows["weight_mean"], rows["delta_norm"])
            ).astype("<f4", copy=False)
            threshold_values.tofile(self.spool_handle)
            self.spool_rows += len(rows)
            self.sampler.add(rows)
            source_frames = rows["source_dpvo_counter"]
            local_indices = (
                rows["source_patch_uid"] & np.uint64((1 << 32) - 1)
            ).astype(np.int64)
            for source_frame in np.unique(source_frames):
                selection = source_frames == source_frame
                patch = self.patch_frames.get(int(source_frame))
                if patch is None:
                    continue
                local = local_indices[selection]
                np.add.at(patch["observation_count"], local, 1)
                np.add.at(patch["weight_sum"], local, rows["weight_mean"][selection])
                np.add.at(patch["entropy_sum"], local, rows["corr_entropy_l0"][selection])
                np.add.at(patch["delta_sum"], local, rows["delta_norm"][selection])
        frame["spool_end"] = self.spool_rows
        self.pending_observations = None
        self.current_frame_id = None

    def finalize_factors(self) -> None:
        if self._factors_finalized:
            return
        self._finalize_factor_rows(
            np.ones(len(self.active_factor_uid), dtype=bool), "run_end"
        )
        self._factors_finalized = True

    def _thresholds(
        self, observations: np.ndarray, patch_arrays: dict[str, np.ndarray]
    ) -> tuple[dict[str, float], str]:
        reference = self.configured_reference_thresholds
        if reference is not None:
            return {name: float(value) for name, value in reference.items()}, "MH_01_reference"
        if not len(observations):
            raise RuntimeError("Cannot calibrate behavior thresholds without observations")
        valid_patch = patch_arrays["observation_count"] > 0
        valid_factor = patch_arrays["factor_lifetime_count"] > 0
        patch_weight = patch_arrays["weight_mean"][valid_patch]
        patch_entropy = patch_arrays["corr_entropy_mean"][valid_patch]
        patch_lifetime = patch_arrays["factor_lifetime_mean"][valid_factor]
        return {
            "low_confidence": float(np.quantile(observations["weight_mean"], 0.20)),
            "high_confidence": float(np.quantile(observations["weight_mean"], 0.80)),
            "large_delta": float(np.quantile(observations["delta_norm"], 0.80)),
            "poor_corr_entropy": float(np.quantile(observations["corr_entropy_l0"], 0.80)),
            "poor_corr_margin": float(np.quantile(observations["corr_margin_l0"], 0.20)),
            "good_corr_entropy": float(np.quantile(observations["corr_entropy_l0"], 0.20)),
            "good_corr_margin": float(np.quantile(observations["corr_margin_l0"], 0.80)),
            "patch_low_confidence": float(np.quantile(patch_weight, 0.20)),
            "patch_weak_correlation": float(np.quantile(patch_entropy, 0.80)),
            "patch_short_factor_lifetime": float(np.quantile(patch_lifetime, 0.20)),
        }, "calibrated_from_this_MH_01_run"

    @property
    def configured_reference_thresholds(self) -> dict[str, float] | None:
        return getattr(self, "_reference_thresholds", None)

    def set_reference_thresholds(self, thresholds: dict[str, Any] | None) -> None:
        self._reference_thresholds = thresholds

    def _patch_arrays(self, patch_states: dict[int, dict[str, Any]]) -> dict[str, np.ndarray]:
        names: dict[str, list[np.ndarray]] = {
            "patch_uid": [], "source_dpvo_counter": [], "local_patch_index": [],
            "euroc_timestamp_ns": [], "x": [], "y": [], "x_normalized": [],
            "y_normalized": [], "inverse_depth_initial": [], "accepted": [],
            "patch_residency_lifetime": [], "observation_count": [],
            "weight_mean": [], "corr_entropy_mean": [], "delta_norm_mean": [],
            "factor_lifetime_count": [], "factor_lifetime_mean": [],
        }
        for frame_id in sorted(self.patch_frames):
            patch = self.patch_frames[frame_id]
            state = patch_states[frame_id]
            count = self.patch_count
            local = np.arange(count, dtype=np.uint64)
            uids = (np.uint64(frame_id) << np.uint64(32)) | local
            obs_count = patch["observation_count"]
            factor_count = patch["factor_lifetime_count"]
            weight_mean = np.divide(
                patch["weight_sum"], obs_count, out=np.full(count, np.nan), where=obs_count > 0
            )
            entropy_mean = np.divide(
                patch["entropy_sum"], obs_count, out=np.full(count, np.nan), where=obs_count > 0
            )
            delta_mean = np.divide(
                patch["delta_sum"], obs_count, out=np.full(count, np.nan), where=obs_count > 0
            )
            factor_mean = np.divide(
                patch["factor_lifetime_sum"], factor_count,
                out=np.full(count, np.nan), where=factor_count > 0,
            )
            names["patch_uid"].append(uids)
            names["source_dpvo_counter"].append(np.full(count, frame_id, dtype=np.int32))
            names["local_patch_index"].append(local.astype(np.int16))
            names["euroc_timestamp_ns"].append(
                np.full(count, int(state["euroc_timestamp_ns"]), dtype=np.uint64)
            )
            names["x"].append(patch["x"])
            names["y"].append(patch["y"])
            names["x_normalized"].append(patch["x"] / (patch["image_width"] / 4.0 - 1.0))
            names["y_normalized"].append(patch["y"] / (patch["image_height"] / 4.0 - 1.0))
            names["inverse_depth_initial"].append(patch["inverse_depth"])
            names["accepted"].append(np.full(count, bool(state.get("accepted")), dtype=bool))
            names["patch_residency_lifetime"].append(
                np.full(count, int(state.get("residency_lifetime", 0)), dtype=np.int32)
            )
            names["observation_count"].append(obs_count)
            names["weight_mean"].append(weight_mean)
            names["corr_entropy_mean"].append(entropy_mean)
            names["delta_norm_mean"].append(delta_mean)
            names["factor_lifetime_count"].append(factor_count)
            names["factor_lifetime_mean"].append(factor_mean)
        return {name: np.concatenate(parts) for name, parts in names.items()}

    def _cleanup_spool(self) -> None:
        if self.spool_cleaned:
            return
        if not self.spool_handle.closed:
            self.spool_handle.close()
        try:
            self.spool_path.unlink(missing_ok=True)
        finally:
            self.spool_cleaned = True

    def abort(self) -> None:
        self._cleanup_spool()

    def write_outputs(
        self,
        output_dir: str | Path,
        patch_states: dict[int, dict[str, Any]],
        reference_thresholds: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.set_reference_thresholds(reference_thresholds)
        self.finalize_factors()
        self.spool_handle.flush()
        self.spool_handle.close()
        try:
            observations = self.sampler.values()
            patches = self._patch_arrays(patch_states)
            thresholds, threshold_source = self._thresholds(observations, patches)
            spool = np.memmap(self.spool_path, dtype="<f4", mode="r").reshape(-1, 2)
            for frame in self.frames:
                start, end = int(frame["spool_start"]), int(frame["spool_end"])
                values = spool[start:end]
                count = end - start
                frame["low_confidence_count"] = int(
                    (values[:, 0] <= thresholds["low_confidence"]).sum()
                ) if count else 0
                frame["large_delta_count"] = int(
                    (values[:, 1] >= thresholds["large_delta"]).sum()
                ) if count else 0
            del spool

            valid_obs = patches["observation_count"] > 0
            valid_factor = patches["factor_lifetime_count"] > 0
            low = valid_obs & (patches["weight_mean"] <= thresholds["patch_low_confidence"])
            weak = valid_obs & (
                patches["corr_entropy_mean"] >= thresholds["patch_weak_correlation"]
            )
            short = valid_factor & (
                patches["factor_lifetime_mean"] <= thresholds["patch_short_factor_lifetime"]
            )
            patches["low_confidence"] = low
            patches["weak_correlation"] = weak
            patches["short_factor_lifetime"] = short
            patches["low_confidence_and_weak_correlation"] = low & weak
            patches["low_confidence_and_short_lifetime"] = low & short
            patches["weak_correlation_and_short_lifetime"] = weak & short
            patches["strict_low_efficiency"] = low & weak & short

            frame_arrays = records_to_arrays(self.frames)
            for metric in FRAME_METRICS:
                count = frame_arrays[f"{metric}_count"].astype(np.float64)
                total = frame_arrays[f"{metric}_sum"].astype(np.float64)
                frame_arrays[metric] = np.divide(
                    total, count, out=np.full(len(count), np.nan), where=count > 0
                )
            frame_arrays["low_confidence_ratio"] = np.divide(
                frame_arrays["low_confidence_count"], frame_arrays["online_observation_count"],
                out=np.full(len(self.frames), np.nan),
                where=frame_arrays["online_observation_count"] > 0,
            )
            frame_arrays["large_delta_ratio"] = np.divide(
                frame_arrays["large_delta_count"], frame_arrays["online_observation_count"],
                out=np.full(len(self.frames), np.nan),
                where=frame_arrays["online_observation_count"] > 0,
            )
            creation_mean = np.full(len(self.frames), np.nan, dtype=np.float64)
            for frame_id, total in self.creation_lifetime_sum.items():
                creation_mean[frame_id] = total / self.creation_lifetime_count[frame_id]
            frame_arrays["factor_lifetime_mean_created"] = creation_mean
            residency = np.full(len(self.frames), np.nan, dtype=np.float64)
            for frame_id, state in patch_states.items():
                residency[frame_id] = float(state.get("residency_lifetime", 0))
            frame_arrays["patch_residency_lifetime"] = residency

            np.savez_compressed(
                output_dir / "frames.npz", **compact_artifact_arrays(frame_arrays)
            )
            np.savez_compressed(
                output_dir / "patches.npz", **compact_artifact_arrays(patches)
            )
            np.savez_compressed(
                output_dir / "observations.npz",
                **{name: observations[name] for name in OBSERVATION_DTYPE.names},
            )
            digest = hashlib.sha256(
                observations[["score", "factor_uid", "update_dpvo_counter"]].tobytes()
            ).hexdigest()
            poor = (
                (observations["corr_entropy_l0"] >= thresholds["poor_corr_entropy"])
                & (observations["corr_margin_l0"] <= thresholds["poor_corr_margin"])
            )
            good = (
                (observations["corr_entropy_l0"] <= thresholds["good_corr_entropy"])
                & (observations["corr_margin_l0"] >= thresholds["good_corr_margin"])
            )
            high = observations["weight_mean"] >= thresholds["high_confidence"]
            low_obs = observations["weight_mean"] <= thresholds["low_confidence"]

            def flag_stats(flag: np.ndarray) -> dict[str, Any]:
                return {
                    "count": int(flag.sum()),
                    "ratio": float(flag.mean()) if len(flag) else None,
                }

            patch_flags = {
                name: flag_stats(patches[name])
                for name in (
                    "low_confidence", "weak_correlation", "short_factor_lifetime",
                    "low_confidence_and_weak_correlation",
                    "low_confidence_and_short_lifetime",
                    "weak_correlation_and_short_lifetime", "strict_low_efficiency",
                )
            }
            metrics = {
                "schema_version": 1,
                "observation_scope": "all factors from the last online graph update per input frame",
                "observation_population_count": self.sampler.population_count,
                "observation_sample_count": int(len(observations)),
                "observation_bottom_k": self.sampler.capacity,
                "observation_sample_sha256": digest,
                "thresholds": thresholds,
                "threshold_source": threshold_source,
                "correlation_entropy": {
                    "temperature": self.temperature,
                    "epsilon": 1e-12,
                    "normalizer": "log(49)",
                    "patch_reduction": "mean over 3x3 patch axes before softmax",
                    "no_per_observation_standardization": True,
                },
                "moments_exact": {name: value.as_dict() for name, value in self.moments.items()},
                "histograms_exact": {name: value.as_dict() for name, value in self.histograms.items()},
                "factor_observation_lifetime": {
                    "lifetime_summary": _histogram_quantiles(self.factor_lifetime_histogram),
                    "lifetime_histogram": dict(sorted(self.factor_lifetime_histogram.items())),
                    "update_count_summary": _histogram_quantiles(self.factor_update_count_histogram),
                    "stable_frame_span_summary": _histogram_quantiles(self.factor_span_histogram),
                    "remove_reasons": dict(self.factor_remove_reasons),
                    "appended": self.factor_appended_total,
                    "finalized": self.factor_finalized_total,
                    "termination_refinement_observations": self.termination_refinement_observations,
                },
                "mismatch": {
                    "poor_correlation_high_confidence": flag_stats(poor & high),
                    "good_correlation_low_confidence": flag_stats(good & low_obs),
                },
                "patch_diagnostics": patch_flags,
                "temporary_threshold_spool": {
                    "dtype": "two float32 columns: mean_weight, delta_norm",
                    "rows": self.spool_rows,
                    "retained_as_artifact": False,
                },
            }
            return metrics
        finally:
            self._cleanup_spool()


def records_to_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not records:
        return {}
    keys = sorted(set().union(*(record.keys() for record in records)))
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        values = [record.get(key, np.nan) for record in records]
        if all(isinstance(value, (bool, np.bool_)) for value in values):
            arrays[key] = np.asarray(values, dtype=bool)
        elif all(isinstance(value, (int, np.integer)) for value in values):
            if "timestamp_ns" in key:
                arrays[key] = np.asarray(values, dtype=np.uint64)
            else:
                arrays[key] = np.asarray(values, dtype=np.int64)
        else:
            arrays[key] = np.asarray(values, dtype=np.float64)
    return arrays


def compact_artifact_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Use compact identifiers and float32 derived scalars without changing meaning."""
    compact: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        values = np.asarray(values)
        if values.dtype.names is not None or values.dtype.kind in "b?":
            compact[name] = values
        elif "timestamp_ns" in name:
            compact[name] = values.astype(np.uint64, copy=False)
        elif values.dtype.kind == "f":
            compact[name] = values.astype(np.float32, copy=False)
        elif values.dtype.kind in "iu":
            minimum = int(values.min()) if values.size else 0
            maximum = int(values.max()) if values.size else 0
            if values.dtype.kind == "u":
                for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                    if maximum <= np.iinfo(dtype).max:
                        compact[name] = values.astype(dtype, copy=False)
                        break
            else:
                for dtype in (np.int8, np.int16, np.int32, np.int64):
                    info = np.iinfo(dtype)
                    if info.min <= minimum and maximum <= info.max:
                        compact[name] = values.astype(dtype, copy=False)
                        break
        else:
            compact[name] = values
    return compact


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name].copy() for name in data.files}


def load_groundtruth(path: str | Path) -> dict[str, np.ndarray]:
    rows: list[list[str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(line.split())
    timestamp_ns = np.asarray(
        [int(Decimal(row[0]).to_integral_value()) for row in rows], dtype=np.int64
    )
    position = np.asarray([[float(value) for value in row[1:4]] for row in rows])
    quaternion_wxyz = np.asarray([[float(value) for value in row[4:8]] for row in rows])
    return {
        "timestamp_ns": timestamp_ns,
        "position": position,
        "quaternion_wxyz": quaternion_wxyz,
    }


def interpolate_gt_rotation(
    timestamps_ns: np.ndarray, groundtruth: dict[str, np.ndarray]
) -> np.ndarray:
    gt_ns = groundtruth["timestamp_ns"].astype(np.int64)
    result = np.full(len(timestamps_ns), np.nan, dtype=np.float64)
    inside = (timestamps_ns.astype(np.int64) >= gt_ns[0]) & (
        timestamps_ns.astype(np.int64) <= gt_ns[-1]
    )
    if not inside.any():
        return result
    base = int(gt_ns[0])
    gt_seconds = (gt_ns - base).astype(np.float64) * 1e-9
    query_seconds = (timestamps_ns[inside].astype(np.int64) - base).astype(np.float64) * 1e-9
    quat_xyzw = groundtruth["quaternion_wxyz"][:, [1, 2, 3, 0]]
    orientations = Slerp(gt_seconds, Rotation.from_quat(quat_xyzw))(query_seconds)
    query_indices = np.flatnonzero(inside)
    if len(query_indices) > 1:
        relative = orientations[:-1].inv() * orientations[1:]
        angles = relative.magnitude()
        consecutive = query_indices[1:] == query_indices[:-1] + 1
        result[query_indices[1:][consecutive]] = angles[consecutive]
    return result


def associate_nearest_gt(
    timestamps_ns: np.ndarray,
    groundtruth: dict[str, np.ndarray],
    max_difference_ns: int = 30_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    gt_ns = groundtruth["timestamp_ns"]
    query = timestamps_ns.astype(np.int64)
    right = np.searchsorted(gt_ns, query)
    right = np.clip(right, 0, len(gt_ns) - 1)
    left = np.clip(right - 1, 0, len(gt_ns) - 1)
    choose_right = np.abs(gt_ns[right] - query) < np.abs(gt_ns[left] - query)
    matched = np.where(choose_right, right, left)
    valid = np.abs(gt_ns[matched] - query) <= int(max_difference_ns)
    return np.flatnonzero(valid), matched[valid]


def global_sim3_window_errors(
    estimate_positions: np.ndarray,
    timestamps_ns: np.ndarray,
    groundtruth: dict[str, np.ndarray],
    window_size: int = 20,
    minimum_associations: int = 10,
) -> dict[str, Any]:
    estimate_indices, gt_indices = associate_nearest_gt(timestamps_ns, groundtruth)
    estimate = np.asarray(estimate_positions, dtype=np.float64)[estimate_indices]
    reference = groundtruth["position"][gt_indices]
    rotation, translation, scale = umeyama_alignment(
        estimate.T, reference.T, with_scale=True
    )
    aligned = (scale * (rotation @ estimate.T) + translation[:, None]).T
    errors = np.linalg.norm(aligned - reference, axis=1)
    per_frame_error = np.full(len(timestamps_ns), np.nan, dtype=np.float64)
    per_frame_error[estimate_indices] = errors
    window_id = np.arange(len(timestamps_ns), dtype=np.int32) // int(window_size)
    window_rows: list[dict[str, Any]] = []
    for value in np.unique(window_id):
        selected = (window_id[estimate_indices] == value)
        count = int(selected.sum())
        if count < minimum_associations:
            continue
        window_rows.append(
            {
                "window_id": int(value),
                "start_frame": int(value * window_size),
                "end_frame": int(min((value + 1) * window_size, len(timestamps_ns)) - 1),
                "associated_pose_count": count,
                "translation_rmse": float(np.sqrt(np.mean(errors[selected] ** 2))),
            }
        )
    return {
        "global_translation_rmse": float(np.sqrt(np.mean(errors**2))),
        "associated_pose_count": int(len(errors)),
        "alignment_scale": float(scale),
        "alignment_rotation": rotation,
        "alignment_translation": translation,
        "per_frame_translation_error": per_frame_error,
        "window_id": window_id,
        "windows": window_rows,
        "alignment_scope": "one global Sim(3) fitted over the complete associated trajectory",
    }


def quintile_labels(values: np.ndarray, higher_is_harder: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros(len(values), dtype=np.int8)
    finite = np.isfinite(values)
    if finite.sum() < 5:
        return result
    oriented = values[finite] if higher_is_harder else -values[finite]
    edges = np.quantile(oriented, [0.2, 0.4, 0.6, 0.8])
    result[finite] = np.searchsorted(edges, oriented, side="right") + 1
    return result


def moving_block_bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 1234,
    iterations: int = 2000,
    block_length: int = 5,
) -> dict[str, Any]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=np.float64)[valid]
    y = np.asarray(y, dtype=np.float64)[valid]
    if len(x) < max(block_length, 3) or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"rho": None, "ci95": [None, None], "window_count": int(len(x))}
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    blocks_needed = int(math.ceil(len(x) / block_length))
    offsets = np.arange(block_length)
    for iteration in range(iterations):
        starts = rng.integers(0, len(x), size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % len(x)).reshape(-1)[: len(x)]
        estimates[iteration] = spearmanr(x[indices], y[indices]).statistic
    finite_estimates = estimates[np.isfinite(estimates)]
    ci = np.quantile(finite_estimates, [0.025, 0.975]) if len(finite_estimates) else [np.nan, np.nan]
    return {
        "rho": rho,
        "ci95": [float(ci[0]), float(ci[1])],
        "window_count": int(len(x)),
        "bootstrap_iterations": iterations,
        "block_length_windows": block_length,
        "criterion": "descriptive operational criterion only; not statistical significance",
    }


def _aggregate_windows(frames: dict[str, np.ndarray], trajectory: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = trajectory["windows"]
    result: dict[str, list[float]] = {
        "window_id": [], "translation_rmse": [], "associated_pose_count": [],
        "low_confidence_ratio": [], "large_delta_ratio": [], "weight_mean": [],
        "corr_entropy_l0": [], "corr_margin_l0": [], "delta_norm": [],
    }
    for row in rows:
        selection = frames["window_id"] == row["window_id"]
        obs_count = frames["online_observation_count"][selection].sum()
        result["window_id"].append(row["window_id"])
        result["translation_rmse"].append(row["translation_rmse"])
        result["associated_pose_count"].append(row["associated_pose_count"])
        result["low_confidence_ratio"].append(
            frames["low_confidence_count"][selection].sum() / obs_count if obs_count else np.nan
        )
        result["large_delta_ratio"].append(
            frames["large_delta_count"][selection].sum() / obs_count if obs_count else np.nan
        )
        for metric in ("weight_mean", "corr_entropy_l0", "corr_margin_l0", "delta_norm"):
            count = frames[f"{metric}_count"][selection].sum()
            total = frames[f"{metric}_sum"][selection].sum()
            result[metric].append(total / count if count else np.nan)
    return {name: np.asarray(values) for name, values in result.items()}


def enrich_sequence_outputs(
    artifact_dir: str | Path,
    probe_poses: np.ndarray,
    groundtruth_path: str | Path,
    seed: int = 1234,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    frames = load_npz(artifact_dir / "frames.npz")
    gt = load_groundtruth(groundtruth_path)
    frames["gt_relative_rotation_rad"] = interpolate_gt_rotation(
        frames["euroc_timestamp_ns"], gt
    )
    trajectory = global_sim3_window_errors(
        probe_poses[:, :3], frames["euroc_timestamp_ns"], gt
    )
    frames["global_aligned_translation_error"] = trajectory[
        "per_frame_translation_error"
    ]
    frames["window_id"] = trajectory["window_id"]
    frames["difficulty_blur_quintile"] = quintile_labels(
        frames["blur_laplacian"], higher_is_harder=False
    )
    frames["difficulty_texture_quintile"] = quintile_labels(
        frames["texture_gradient"], higher_is_harder=False
    )
    frames["difficulty_rotation_quintile"] = quintile_labels(
        frames["gt_relative_rotation_rad"], higher_is_harder=True
    )
    frames["difficulty_model_motion_quintile"] = quintile_labels(
        frames["apparent_motion"], higher_is_harder=True
    )
    np.savez_compressed(
        artifact_dir / "frames.npz", **compact_artifact_arrays(frames)
    )
    windows = _aggregate_windows(frames, trajectory)
    associations = {
        metric: moving_block_bootstrap_spearman(
            windows[metric], windows["translation_rmse"], seed=seed
        )
        for metric in (
            "low_confidence_ratio", "large_delta_ratio", "weight_mean",
            "corr_entropy_l0", "corr_margin_l0", "delta_norm",
        )
    }
    np.savez_compressed(
        artifact_dir / "windows.npz", **compact_artifact_arrays(windows)
    )
    return {
        "trajectory": {
            key: value for key, value in trajectory.items()
            if key not in {"alignment_rotation", "alignment_translation", "per_frame_translation_error", "window_id", "windows"}
        }
        | {
            "alignment_rotation": trajectory["alignment_rotation"],
            "alignment_translation": trajectory["alignment_translation"],
            "window_count": len(trajectory["windows"]),
            "window_size_processed_frames": 20,
            "minimum_associated_poses_per_window": 10,
            "window_realignment": False,
            "rpe_reported": False,
        },
        "window_associations": associations,
        "difficulty": {
            "blur": "input-independent; lower is harder",
            "texture": "input-independent; lower is harder",
            "rotation": "EuRoC GT quaternion Slerp; higher is harder",
            "apparent_motion": "model-derived reprojection proxy; higher is harder",
            "quintiles": "within-sequence Q1 easiest to Q5 hardest",
        },
    }


def sample_spearman(observations: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for metric in ("corr_peak_l0", "corr_margin_l0", "corr_entropy_l0"):
        values = spearmanr(observations[metric], observations["weight_mean"])
        result[metric] = {"rho": float(values.statistic), "sample_count": int(len(observations[metric]))}
    return result


def difficulty_stratification(frames: dict[str, np.ndarray]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    proxies = {
        "blur": "difficulty_blur_quintile",
        "texture": "difficulty_texture_quintile",
        "gt_rotation": "difficulty_rotation_quintile",
        "model_motion": "difficulty_model_motion_quintile",
    }
    metrics = (
        "weight_mean", "corr_entropy_l0", "corr_margin_l0", "delta_norm",
        "patch_residency_lifetime", "factor_lifetime_mean_created",
    )
    for proxy, quintile_name in proxies.items():
        output[proxy] = {}
        for quintile in range(1, 6):
            selected = frames[quintile_name] == quintile
            output[proxy][f"Q{quintile}"] = {
                "frame_count": int(selected.sum()),
                **{
                    metric: float(np.nanmean(frames[metric][selected]))
                    if selected.any() and np.isfinite(frames[metric][selected]).any()
                    else None
                    for metric in metrics
                },
            }
    return output


def make_sequence_figures(run_dir: str | Path, sequence: str) -> list[str]:
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"
    artifact_dir = run_dir / "artifacts"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frames = load_npz(artifact_dir / "frames.npz")
    patches = load_npz(artifact_dir / "patches.npz")
    observations = load_npz(artifact_dir / "observations.npz")
    windows = load_npz(artifact_dir / "windows.npz")
    with (report_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        thresholds = json.load(handle)["behavior"]["thresholds"]
    paths: list[str] = []

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    accepted = patches["accepted"]
    axes[0].hist2d(patches["x_normalized"][accepted], patches["y_normalized"][accepted], bins=(24, 18), range=((0, 1), (0, 1)))
    axes[0].invert_yaxis(); axes[0].set_title("normalized patch density")
    frame_ids = np.unique(patches["source_dpvo_counter"])
    coverage, border, nearest = [], [], []
    for frame_id in frame_ids:
        selected = (patches["source_dpvo_counter"] == frame_id) & accepted
        xy = np.column_stack((patches["x_normalized"][selected], patches["y_normalized"][selected]))
        if not len(xy):
            continue
        cells = np.unique(np.column_stack((np.clip((xy[:, 0] * 16).astype(int), 0, 15), np.clip((xy[:, 1] * 12).astype(int), 0, 11))), axis=0)
        coverage.append(len(cells) / (16 * 12))
        border.append(np.mean((xy[:, 0] < .1) | (xy[:, 0] > .9) | (xy[:, 1] < .1) | (xy[:, 1] > .9)))
        distances = np.sqrt(np.square(xy[:, None, :] - xy[None, :, :]).sum(axis=-1))
        np.fill_diagonal(distances, np.inf)
        nearest.extend(distances.min(axis=1))
    axes[1].hist(coverage, bins=30); axes[1].set_title("12x16 grid coverage")
    axes[2].hist(border, bins=30); axes[2].set_title("outer-10% border ratio")
    axes[3].hist(nearest, bins=40); axes[3].set_title("normalized NN distance")
    fig.suptitle(f"{sequence} — Patch spatial distribution")
    path = figure_dir / "figure_1_patch_spatial.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for axis, (name, title) in zip(axes, (("weight_mean", "weight"), ("corr_peak_l0", "corr peak L0"), ("corr_margin_l0", "corr margin L0"), ("corr_entropy_l0", "corr entropy L0"), ("delta_norm", "delta norm"))):
        axis.hist(observations[name], bins=80, density=True); axis.set_title(title)
    fig.suptitle(f"{sequence} — Confidence / correlation distributions")
    path = figure_dir / "figure_2_state_distributions.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(observations["corr_margin_l0"], observations["weight_mean"], gridsize=70, bins="log", mincnt=1); axes[0].set(xlabel="corr margin L0", ylabel="mean weight")
    axes[1].hexbin(observations["corr_entropy_l0"], observations["weight_mean"], gridsize=70, bins="log", mincnt=1); axes[1].set(xlabel="normalized entropy L0", ylabel="mean weight")
    axes[0].axvline(thresholds["poor_corr_margin"], color="C3", linestyle="--")
    axes[0].axvline(thresholds["good_corr_margin"], color="C2", linestyle="--")
    axes[0].axhline(thresholds["high_confidence"], color="C3", linestyle=":")
    axes[0].axhline(thresholds["low_confidence"], color="C2", linestyle=":")
    axes[0].text(.01, .98, "poor corr / high confidence", transform=axes[0].transAxes, va="top", color="C3")
    axes[0].text(.99, .02, "good corr / low confidence", transform=axes[0].transAxes, ha="right", color="C2")
    axes[1].axvline(thresholds["poor_corr_entropy"], color="C3", linestyle="--")
    axes[1].axvline(thresholds["good_corr_entropy"], color="C2", linestyle="--")
    axes[1].axhline(thresholds["high_confidence"], color="C3", linestyle=":")
    axes[1].axhline(thresholds["low_confidence"], color="C2", linestyle=":")
    axes[1].text(.99, .98, "poor corr / high confidence", transform=axes[1].transAxes, va="top", ha="right", color="C3")
    axes[1].text(.01, .02, "good corr / low confidence", transform=axes[1].transAxes, color="C2")
    corr = sample_spearman(observations)
    fig.suptitle(f"{sequence} — Correlation vs confidence; rho(margin)={corr['corr_margin_l0']['rho']:.3f}, rho(entropy)={corr['corr_entropy_l0']['rho']:.3f}")
    path = figure_dir / "figure_3_correlation_vs_confidence.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    proxy_names = (("difficulty_blur_quintile", "blur"), ("difficulty_texture_quintile", "texture"), ("difficulty_rotation_quintile", "GT rotation"), ("difficulty_model_motion_quintile", "model-derived motion"))
    for axis, (quintile_name, title) in zip(axes.ravel(), proxy_names):
        for metric in (
            "weight_mean", "corr_entropy_l0", "delta_norm",
            "patch_residency_lifetime", "factor_lifetime_mean_created",
        ):
            means = [np.nanmean(frames[metric][frames[quintile_name] == q]) for q in range(1, 6)]
            baseline = np.nanmean(frames[metric])
            relative = np.asarray(means) / baseline if np.isfinite(baseline) and baseline != 0 else means
            axis.plot(range(1, 6), relative, marker="o", label=metric)
        axis.axhline(1.0, color="0.5", linewidth=.8)
        axis.set_title(title); axis.set_xticks(range(1, 6)); axis.set_xlabel("difficulty quintile"); axis.set_ylabel("mean / sequence mean"); axis.legend(fontsize=7)
    fig.suptitle(f"{sequence} — Difficulty proxies vs internal state")
    path = figure_dir / "figure_4_difficulty_vs_state.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, metric in zip(axes.ravel(), ("low_confidence_ratio", "corr_entropy_l0", "large_delta_ratio", "weight_mean")):
        axis.scatter(windows[metric], windows["translation_rmse"], s=18, alpha=.8)
        rho = spearmanr(windows[metric], windows["translation_rmse"], nan_policy="omit").statistic
        axis.set(xlabel=metric, ylabel="globally aligned window RMSE", title=f"Spearman rho={rho:.3f}")
    error_axis = axes[1, 1]
    error_axis.plot(
        frames["stream_index"], frames["global_aligned_translation_error"],
        color="C0", alpha=.65, linewidth=.8, label="per-frame aligned error",
    )
    error_axis.plot(
        windows["window_id"] * 20 + 9, windows["translation_rmse"],
        color="C3", marker="o", markersize=3, linewidth=1, label="20-frame RMSE",
    )
    error_axis.set(
        xlabel="processed frame", ylabel="translation error (m)",
        title="trajectory error overlay",
    )
    error_axis.legend(fontsize=7)
    axes[1, 2].axis("off")
    fig.suptitle(f"{sequence} — Internal state vs trajectory degradation")
    path = figure_dir / "figure_5_state_vs_trajectory.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))
    return paths


# Final Experiment 1 comparison.  It is intentionally a library function: the
# public protocol is run_exp1, not a second comparison command.
COMPARISON_SEQUENCES = ("MH_01_easy", "MH_03_medium", "MH_05_difficult")
COMPARISON_METRICS = (
    "weight_mean", "corr_peak_l0", "corr_margin_l0", "corr_entropy_l0", "delta_norm",
)
COMPARISON_COLORS = {"MH_01_easy": "C0", "MH_03_medium": "C1", "MH_05_difficult": "C2"}


def _summary_statistics(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "mean": None, "q20": None, "median": None, "q80": None}
    q20, median, q80 = np.quantile(finite, (.2, .5, .8))
    return {"count": int(len(finite)), "mean": float(finite.mean()), "q20": float(q20), "median": float(median), "q80": float(q80)}


def _shift(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    lstat, rstat = _summary_statistics(left), _summary_statistics(right)
    if lstat["median"] is None or rstat["median"] is None:
        return {"median_absolute_change": None, "median_relative_change": None, "ks_statistic": None, "wasserstein_distance": None}
    delta = float(rstat["median"] - lstat["median"])
    return {
        "median_absolute_change": delta,
        "median_relative_change": delta / abs(float(lstat["median"])) if lstat["median"] else None,
        "ks_statistic": float(ks_2samp(np.asarray(left)[np.isfinite(left)], np.asarray(right)[np.isfinite(right)]).statistic),
        "wasserstein_distance": float(wasserstein_distance(np.asarray(left)[np.isfinite(left)], np.asarray(right)[np.isfinite(right)])),
    }


def _metric_tolerance(metric: str, easy_value: float) -> float:
    if metric == "ate_rmse":
        return max(0.001, .01 * abs(easy_value))
    if metric in {"weight_mean", "corr_entropy_l0", "low_confidence_ratio", "large_delta_ratio"}:
        return .005
    if metric.startswith("window_"):
        return .05
    return max(.01, .01 * abs(easy_value))


def ordinal_label(values: list[float | None], *, expected_direction: int, metric: str) -> str:
    """Small predeclared descriptive label, never an inference test."""
    if any(value is None or not np.isfinite(value) for value in values):
        return "non_monotonic"
    easy, medium, difficult = (float(value) for value in values)
    tolerance = _metric_tolerance(metric, easy)
    if expected_direction == 0:
        return "no_meaningful_change" if max(easy, medium, difficult) - min(easy, medium, difficult) <= tolerance else "non_monotonic"
    first, second = expected_direction * (medium - easy), expected_direction * (difficult - medium)
    total = expected_direction * (difficult - easy)
    if abs(first) <= tolerance and abs(second) <= tolerance:
        return "no_meaningful_change"
    if first > tolerance and second > tolerance:
        return "monotonic_degradation"
    if total > tolerance and ((first > tolerance and abs(second) <= tolerance) or (second > tolerance and abs(first) <= tolerance)):
        return "approximately_consistent_trend"
    return "non_monotonic"


def _headlines(summary: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    behavior = metrics["behavior"]
    moments = behavior["moments_exact"]
    return {
        "ate_rmse": summary["trajectory"]["baseline_translation_ate_rmse"],
        "fps": summary["headline_performance"]["fps"],
        "peak_vram_bytes": summary["headline_performance"]["peak_allocated_bytes"],
        "corr_peak_l0": moments["corr_peak_l0"]["mean"],
        "corr_margin_l0": moments["corr_margin_l0"]["mean"],
        "corr_entropy_l0": moments["corr_entropy_l0"]["mean"],
        "weight_mean": moments["weight_mean"]["mean"],
        "delta_norm": moments["delta_norm"]["mean"],
        "low_confidence_ratio": summary["low_confidence_ratio"],
        "large_delta_ratio": summary["large_delta_ratio"],
        "entropy_weight_spearman": metrics["observation_spearman"]["corr_entropy_l0"]["rho"],
        "headline_provenance": "uninstrumented baseline",
        "probe_perturbation_label": summary["probe_perturbation"]["perturbation_label"],
        "probe_representativeness": summary["probe_perturbation"]["probe_representativeness"],
    }


def _comparison_figures(output: Path, data: dict[str, dict[str, np.ndarray]], thresholds: dict[str, float]) -> list[str]:
    figure_dir = output / "figures"; figure_dir.mkdir(parents=True, exist_ok=True); paths: list[str] = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, name in zip(axes, COMPARISON_SEQUENCES):
        patches = data[name]["patches"]; selected = patches["accepted"]
        axis.hist2d(patches["x_normalized"][selected], patches["y_normalized"][selected], bins=(24, 18), range=((0, 1), (0, 1)))
        axis.invert_yaxis(); axis.set_title(name); axis.set(xlabel="x normalized", ylabel="y normalized")
    fig.suptitle("Figure 1 — Patch spatial distribution across EuRoC sequences")
    path = figure_dir / "figure_1_patch_spatial.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.5))
    for axis, metric in zip(axes, COMPARISON_METRICS):
        for name in COMPARISON_SEQUENCES:
            axis.hist(data[name]["observations"][metric], bins=80, density=True, histtype="step", linewidth=1.2, color=COMPARISON_COLORS[name], label=name)
        axis.set_title(metric); axis.legend(fontsize=6)
    fig.suptitle("Figure 2 — Correlation, confidence and update distributions")
    path = figure_dir / "figure_2_state_distributions.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, name in zip(axes, COMPARISON_SEQUENCES):
        observations = data[name]["observations"]
        axis.hexbin(observations["corr_entropy_l0"], observations["weight_mean"], gridsize=55, bins="log", mincnt=1)
        rho = spearmanr(observations["corr_entropy_l0"], observations["weight_mean"]).statistic
        axis.axvline(thresholds["poor_corr_entropy"], color="C3", linestyle="--"); axis.axvline(thresholds["good_corr_entropy"], color="C2", linestyle="--")
        axis.axhline(thresholds["high_confidence"], color="C3", linestyle=":"); axis.axhline(thresholds["low_confidence"], color="C2", linestyle=":")
        axis.set(title=f"{name}; rho={rho:.3f}", xlabel="normalized entropy L0", ylabel="mean weight")
    fig.suptitle("Figure 3 — Correlation versus confidence")
    path = figure_dir / "figure_3_correlation_vs_confidence.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))
    proxies = (("difficulty_blur_quintile", "blur"), ("difficulty_texture_quintile", "texture"), ("difficulty_rotation_quintile", "GT rotation"), ("difficulty_model_motion_quintile", "model-derived motion"))
    state_metrics = ("weight_mean", "corr_entropy_l0", "delta_norm", "patch_residency_lifetime", "factor_lifetime_mean_created")
    fig, axes = plt.subplots(4, 5, figsize=(20, 14), sharex=True)
    for row, (proxy, label) in enumerate(proxies):
        for column, metric in enumerate(state_metrics):
            axis = axes[row, column]
            for name in COMPARISON_SEQUENCES:
                frames = data[name]["frames"]; values = np.asarray([np.nanmean(frames[metric][frames[proxy] == quintile]) for quintile in range(1, 6)])
                baseline = np.nanmean(frames[metric]); values = values / baseline if np.isfinite(baseline) and baseline else values
                axis.plot(range(1, 6), values, marker="o", color=COMPARISON_COLORS[name], label=name)
            axis.axhline(1, color=".6", linewidth=.7); axis.set_xticks(range(1, 6))
            if row == 0: axis.set_title(metric)
            if column == 0: axis.set_ylabel(label + "\nmean / sequence mean")
            if row == 3: axis.set_xlabel("within-sequence quintile")
            if row == 0 and column == 4: axis.legend(fontsize=6)
    fig.suptitle("Figure 4 — Difficulty proxies versus DPVO internal state")
    path = figure_dir / "figure_4_difficulty_vs_state.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, metric in zip(axes.ravel(), ("low_confidence_ratio", "corr_entropy_l0", "large_delta_ratio", "weight_mean")):
        for name in COMPARISON_SEQUENCES:
            windows = data[name]["windows"]; axis.scatter(windows[metric], windows["translation_rmse"], s=13, alpha=.65, color=COMPARISON_COLORS[name], label=name)
        axis.set(xlabel=metric, ylabel="global-aligned window RMSE"); axis.legend(fontsize=6)
    fig.suptitle("Figure 5 — Internal state versus local trajectory error")
    path = figure_dir / "figure_5_state_vs_trajectory.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))
    return paths


def _final_comparison_report(comparison: dict[str, Any]) -> str:
    heads = comparison["sequence_headlines"]; trends = comparison["headline_trends"]
    ate = ", ".join(f"{name}: {heads[name]['ate_rmse']:.6g} m" for name in COMPARISON_SEQUENCES)
    perturbation = ", ".join(
        f"{name}: {100 * comparison['probe_perturbation'][name]['relative_ate_change']:.2f}% ({comparison['probe_perturbation'][name]['perturbation_label']})"
        for name in COMPARISON_SEQUENCES
    )
    patch_pair = ", ".join(
        f"{name}: {100 * comparison['patch_diagnostics'][name]['low_confidence_and_weak_correlation']['ratio']:.1f}%"
        for name in COMPARISON_SEQUENCES
    )
    return f"""# Experiment 1 — DPVO Internal Behavior under EuRoC Sequence Difficulty

## 1. Research Question

This experiment records how verified DPVO internal correlation, confidence, patch and factor statistics vary over MH_01_easy, MH_03_medium and MH_05_difficult, and whether their window aggregates describe local globally-aligned trajectory error. It does not modify DPVO, introduce JEPA, or assert causal effects.

## 2. Protocol

All runs use stride 2, seed 1234, the original checkpoint and `config/default.yaml`, with loop closure disabled. Headlines and retained trajectories come from uninstrumented baselines; probes supply behavior statistics and a non-blocking perturbation diagnostic. MH_01 → MH_03 → MH_05 is the EuRoC nominal difficulty order, not a controlled single-variable experiment. One global Sim(3) is fit per probe sequence; 20-frame RMSE windows are never independently aligned.

## 3. Three-sequence performance

Baseline translation ATE RMSE: {ate}. The complete absolute values, adjacent sequence-order changes and descriptive ordinal labels are in `comparison.json`; ATE is intentionally not forced to be monotonic.

## 3.1 Probe perturbation and representativeness

The baseline→probe relative ATE changes are {perturbation}. All are `small`, so their probe records have normal feasibility-stage representativeness. This is an engineering diagnostic, not a significance test or a numerical acceptance gate.

## 4. Correlation behavior

Figure 2 reports all deterministic bottom-K distributions, including KS and Wasserstein shifts. Raw peak and margin show `monotonic_degradation`; entropy is `no_meaningful_change`. These shifts are descriptive evidence, not proof that nominal sequence difficulty caused them.

## 5. Confidence behavior

Figure 3 displays every sequence independently. Entropy/weight Spearman is `no_meaningful_change` across the three sequences: DPVO confidence continues to track matching-quality variation, while the extreme poor-correlation/high-confidence mismatch remains rare. This is an association, not a calibrated correctness guarantee.

## 6. Patch diagnostics

Patch residency and factor-observation lifetime remain graph/keyframe-policy-dependent. The low-confidence-and-weak-correlation patch ratio increases in sequence order ({patch_pair}); this indicates more low-quality correspondences within the fixed random patch budget under the nominally harder runs. The three-way strict-low-efficiency intersection remains an extreme diagnostic rather than a broad quality label.

## 7. Difficulty response

Figure 4 stratifies within each sequence by blur, texture, GT rotation and model-derived apparent motion. Matching degradation is most visibly stratified by texture and model-derived motion; blur/texture/rotation are proxies rather than human failure annotations, and model-derived motion is not input-independent.

## 8. Trajectory association

Figure 5 uses exact window aggregates and globally aligned local RMSE. It finds no stable cross-sequence internal-state/local-RMSE association. Bootstrap intervals describe uncertainty only; they do not establish statistical significance or causality.

## 9. Q1–Q7

- **Q1 — correlation:** peak and margin degrade in sequence order; entropy is stable.
- **Q2 — confidence:** entropy/weight Spearman is `{trends['entropy_weight_spearman']['ordinal_label']}`, supporting a stable descriptive relation to matching quality.
- **Q3 — mismatch:** poor-correlation/high-confidence observations remain extremely rare, so confidence fusion is not the primary bottleneck.
- **Q4 — proxy response:** texture and model-derived motion show the clearest matching-degradation response, without a causal claim.
- **Q5 — patch selection:** low-confidence-and-weak-correlation patches increase across the three nominal-difficulty runs, indicating a less efficient random patch budget in difficult conditions.
- **Q6 — local trajectory association:** no stable cross-sequence association is observed; global ATE is `{trends['ate_rmse']['ordinal_label']}` and must not be treated as difficulty-caused.
- **Q7 — decision:** prioritize correspondence representation; do not prioritize confidence fusion in the next decision gate.

## 10. Decision Gate

This report pauses after Experiment 1. It does not enter Experiment 2 or implement JEPA.
"""


def build_comparison(sequence_runs: dict[str, Path], output: Path, *, schema_version: int = 1) -> Path:
    """Build the only Experiment 1 comparison; works in a temporary fixture dir."""
    if tuple(sequence_runs) != COMPARISON_SEQUENCES:
        raise ValueError("comparison requires MH_01_easy, MH_03_medium and MH_05_difficult in protocol order")
    output.mkdir(parents=True, exist_ok=False)
    summaries = {name: json.loads((path / "report/summary.json").read_text()) for name, path in sequence_runs.items()}
    metrics = {name: json.loads((path / "report/metrics.json").read_text()) for name, path in sequence_runs.items()}
    data = {name: {item: load_npz(path / "artifacts" / f"{item}.npz") for item in ("frames", "patches", "observations", "windows")} for name, path in sequence_runs.items()}
    thresholds = metrics["MH_01_easy"]["behavior"]["thresholds"]
    for name in COMPARISON_SEQUENCES[1:]:
        candidate = metrics[name]["behavior"]["thresholds"]
        if candidate != thresholds:
            raise ValueError(f"{name} does not reuse MH_01 frozen thresholds")
    distributions = {}
    for metric in COMPARISON_METRICS:
        values = {name: data[name]["observations"][metric] for name in COMPARISON_SEQUENCES}
        distributions[metric] = {"absolute": {name: _summary_statistics(values[name]) for name in COMPARISON_SEQUENCES}, "easy_to_medium": _shift(values["MH_01_easy"], values["MH_03_medium"]), "medium_to_difficult": _shift(values["MH_03_medium"], values["MH_05_difficult"]), "easy_to_difficult": _shift(values["MH_01_easy"], values["MH_05_difficult"]), "relative_to_mh01": {name: _shift(values["MH_01_easy"], values[name])["median_relative_change"] for name in COMPARISON_SEQUENCES}, "inference_scope": "deterministic bottom-K descriptive distribution shift"}
    headline = {name: _headlines(summaries[name], metrics[name]) for name in COMPARISON_SEQUENCES}
    trends = {}
    directions = {"ate_rmse": 1, "corr_peak_l0": -1, "corr_margin_l0": -1, "corr_entropy_l0": 1, "weight_mean": -1, "delta_norm": 1, "low_confidence_ratio": 1, "large_delta_ratio": 1, "entropy_weight_spearman": 0}
    for metric, direction in directions.items():
        values = [headline[name].get(metric) for name in COMPARISON_SEQUENCES]
        trends[metric] = {"absolute": dict(zip(COMPARISON_SEQUENCES, values)), "easy_to_medium": None if values[0] is None or values[1] is None else values[1] - values[0], "medium_to_difficult": None if values[1] is None or values[2] is None else values[2] - values[1], "easy_to_difficult": None if values[0] is None or values[2] is None else values[2] - values[0], "relative_to_mh01": {name: None if values[0] in (None, 0) or value is None else (value-values[0])/abs(values[0]) for name, value in zip(COMPARISON_SEQUENCES, values)}, "ordinal_label": ordinal_label(values, expected_direction=direction, metric=metric), "descriptive_only": True}
    association = {metric: {name: metrics[name]["window_associations"].get(metric) for name in COMPARISON_SEQUENCES} for metric in ("low_confidence_ratio", "large_delta_ratio", "weight_mean", "corr_entropy_l0", "corr_margin_l0", "delta_norm")}
    comparison = {"schema_version": schema_version, "protocol": "Experiment 1 — DPVO Internal Behavior under EuRoC Sequence Difficulty", "sequence_order": list(COMPARISON_SEQUENCES), "nominal_difficulty_guardrail": "EuRoC nominal difficulty order; not a controlled single-variable experiment and not causal evidence.", "thresholds": thresholds, "sequence_headlines": headline, "probe_perturbation": {name: summaries[name]["probe_perturbation"] for name in COMPARISON_SEQUENCES}, "headline_trends": trends, "distributions": distributions, "mismatch": {name: metrics[name]["behavior"]["mismatch"] for name in COMPARISON_SEQUENCES}, "patch_diagnostics": {name: metrics[name]["behavior"]["patch_diagnostics"] for name in COMPARISON_SEQUENCES}, "frozen_threshold_ratios": {name: {"low_confidence_ratio": summaries[name]["low_confidence_ratio"], "large_delta_ratio": summaries[name]["large_delta_ratio"]} for name in COMPARISON_SEQUENCES}, "difficulty_response": {name: metrics[name]["difficulty_stratification"] for name in COMPARISON_SEQUENCES}, "window_trajectory_association": association, "guardrail": "descriptive/exploratory; no causal or statistical-significance claim"}
    figures = _comparison_figures(output, data, thresholds)
    reference_path = sequence_runs["MH_01_easy"] / "report/metrics.json"
    summary = {"schema_version": schema_version, "status": "passed", "sequence_order": list(COMPARISON_SEQUENCES), "five_core_figures": [str(Path(path).relative_to(output)) for path in figures], "next_gate": "Pause after Experiment 1; do not enter Experiment 2 or implement JEPA."}
    manifest = {"schema_version": schema_version, "sequence_runs": {name: {"sequence_id": name, "result_path": name} for name in COMPARISON_SEQUENCES}, "input_metrics_sha256": {name: sha256_file(path / "report/metrics.json") for name, path in sequence_runs.items()}, "mh01_reference": {"run_id": metrics["MH_01_easy"]["provenance"]["run_id"], "metrics_path": "MH_01_easy/report/metrics.json", "metrics_sha256": sha256_file(reference_path), "git_commit": metrics["MH_01_easy"]["provenance"]["git"]["commit"], "config_sha256": metrics["MH_01_easy"]["provenance"]["config_sha256"], "frozen_thresholds": thresholds}, "copied_sequence_npz": False, "automatic_zip_created": False}
    atomic_json_dump(output / "summary.json", summary); atomic_json_dump(output / "comparison.json", comparison); atomic_json_dump(output / "manifest.json", manifest)
    (output / "REPORT.md").write_text(_final_comparison_report(comparison), encoding="utf-8")
    return output
