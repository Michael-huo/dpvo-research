"""Pure aggregation helpers for Exp6 H2 deployment profiling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def distribution_ms(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "total_ms": 0.0, "mean_ms": None,
                "p50_ms": None, "p95_ms": None, "max_ms": None}
    if not np.isfinite(array).all() or bool((array < 0).any()):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "count": int(array.size), "total_ms": float(array.sum()),
        "mean_ms": float(array.mean()), "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)), "max_ms": float(array.max()),
    }


@dataclass
class OnlineProfiler:
    samples: dict[str, list[float]] = field(default_factory=lambda: {
        name: [] for name in (
            "anchor_decode_preprocess", "jepa_encoder", "jepa_predictor",
            "bridge", "native_dpvo_frontend", "dpvo_graph_runtime",
        )
    })
    context_wait_ms: list[float] = field(default_factory=list)
    effective_hidden_delay_ms: list[float] = field(default_factory=list)

    def add(self, stage: str, milliseconds: float) -> None:
        if stage not in self.samples:
            raise KeyError(stage)
        value = float(milliseconds)
        if not np.isfinite(value) or value < 0:
            raise ValueError("latency must be finite and non-negative")
        self.samples[stage].append(value)

    @property
    def profiled_stage_subtotal_ms(self) -> float:
        return float(sum(sum(values) for values in self.samples.values()))

    def payload(self, *, peak_online_vram_bytes: int) -> dict[str, Any]:
        stages = {name: distribution_ms(values) for name, values in self.samples.items()}
        cloud = sum(float(row["total_ms"]) for row in stages.values())
        return {
            "schema": "h2_stage_profile_v2",
            "timing_scope": "profiled_online_stage_subtotal_not_wall_clock",
            "cuda_timing": "cuda_event_or_explicit_synchronize",
            "not_attributed_to_named_stage": [
                "model_load", "checkpoint_load", "warmup", "training",
                "online_temporary_npy_io", "online_ipc_wait", "artifact_serialization",
                "offline_reference", "offline_diagnostics",
            ],
            "stages": stages,
            "profiled_stage_subtotal_ms": float(cloud),
            "context_wait_ms": distribution_ms(self.context_wait_ms),
            "effective_hidden_delay_ms": distribution_ms(self.effective_hidden_delay_ms),
            "effective_hidden_delay_method": (
                "context_wait_plus_profiled_cloud_critical_path_excluding_artifact_io"
            ),
            "peak_online_vram_bytes": int(peak_online_vram_bytes),
        }


def transmission_payload(records: Sequence[Any], roles: dict[str, str]) -> dict[str, Any]:
    if not records:
        raise ValueError("transmission accounting requires records")
    total_raw = total_encoded = anchor_raw = anchor_encoded = 0
    for record in records:
        path = Path(record.rgb_path)
        total_encoded += path.stat().st_size
        try:
            import cv2
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(path)
            raw = int(image.nbytes)
        except ImportError:
            raise RuntimeError("OpenCV is required for decoded byte accounting")
        total_raw += raw
        if roles[record.identity.key] == "anchor":
            anchor_encoded += path.stat().st_size
            anchor_raw += raw
    anchor_count = sum(roles[row.identity.key] == "anchor" for row in records)
    return {
        "total_frame_count": len(records), "anchor_count": int(anchor_count),
        "anchor_ratio": anchor_count / len(records),
        "raw_full_bytes": int(total_raw), "raw_anchor_bytes": int(anchor_raw),
        "encoded_full_bytes": int(total_encoded), "encoded_anchor_bytes": int(anchor_encoded),
        "raw_byte_reduction": 1.0 - anchor_raw / total_raw,
        "encoded_byte_reduction": 1.0 - anchor_encoded / total_encoded,
    }


def matched_wall_clock_payload(*, full_rgb_seconds: float, h2_seconds: float,
                               sparse_rgb_seconds: float) -> dict[str, Any]:
    values = {
        "full_rgb_total_s": float(full_rgb_seconds),
        "h2_total_s": float(h2_seconds),
        "sparse_rgb_total_s": float(sparse_rgb_seconds),
    }
    if not all(np.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("matched wall-clock values must be finite and non-negative")
    full = values["full_rgb_total_s"]
    values["h2_over_full_rgb_ratio"] = values["h2_total_s"] / full if full > 0 else None
    values["extra_cloud_compute_s"] = values["h2_total_s"] - full
    values["sparse_rgb_role"] = "anchors_only_compute_reference"
    return values


def graph_workload_payload(runtime: Mapping[str, Any]) -> dict[str, Any]:
    processed = int(runtime["processed_observation_count"])
    total = float(runtime["dpvo_graph_runtime_total_ms"])
    if processed <= 0 or total < 0:
        raise ValueError("graph workload requires positive observations and non-negative time")
    return {
        "processed_observations": processed,
        "final_node_count": int(runtime["final_node_count_before_terminate"]),
        "final_patch_count": int(runtime["final_patch_count_before_terminate"]),
        "cumulative_factor_count": int(runtime["factor_count_allocated"]),
        "final_active_factor_count": int(runtime["final_active_factor_count"]),
        "dpvo_graph_total_ms": total,
        "dpvo_graph_mean_ms_per_processed_observation": total / processed,
    }


def break_even_payload(*, encoded_full_bytes: int, encoded_anchor_bytes: int,
                       extra_cloud_compute_s: float) -> dict[str, Any]:
    saved_bits = 8 * (int(encoded_full_bytes) - int(encoded_anchor_bytes))
    extra_seconds = float(extra_cloud_compute_s)
    if extra_seconds <= 0:
        return {"status": "no_extra_cloud_compute", "saved_upload_bits": saved_bits,
                "extra_cloud_compute_s": extra_seconds,
                "break_even_uplink_bandwidth_bps": None,
                "break_even_uplink_bandwidth_mbps": None}
    bandwidth = saved_bits / extra_seconds
    return {"status": "finite", "saved_upload_bits": saved_bits,
            "extra_cloud_compute_s": extra_seconds,
            "break_even_uplink_bandwidth_bps": bandwidth,
            "break_even_uplink_bandwidth_mbps": bandwidth / 1e6}


def context_wait_values(intervals: Iterable[Any]) -> list[float]:
    return [
        (interval.anchor1.timestamp_ns - query.identity.timestamp_ns) / 1e6
        for interval in intervals for query in interval.hidden
    ]
