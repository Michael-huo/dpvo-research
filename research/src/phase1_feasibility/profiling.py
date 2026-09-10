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
    stage_c_cpu_ms: dict[str, list[float]] = field(default_factory=lambda: {
        name: [] for name in (
            "stage_c_queue_wait", "stage_c_transfer_wait", "bridge_compute",
            "native_frontend_compute", "dpvo_graph_compute", "dpvo_sync_wait",
            "stage_c_python_other",
        )
    })
    stage_c_cuda_ms: dict[str, list[float]] = field(default_factory=lambda: {
        name: [] for name in (
            "stage_c_transfer_wait", "bridge_compute", "native_frontend_compute",
            "dpvo_graph_compute", "stage_c_python_other",
        )
    })
    stage_c_queue_detail_ms: dict[str, list[float]] = field(default_factory=lambda: {
        "consume_queue_wait": [], "prediction_ready_wait": [],
        "worker_flush_wait": [],
    })
    stage_c_matched_wall_ms: float | None = None

    def add(self, stage: str, milliseconds: float) -> None:
        if stage not in self.samples:
            raise KeyError(stage)
        value = float(milliseconds)
        if not np.isfinite(value) or value < 0:
            raise ValueError("latency must be finite and non-negative")
        self.samples[stage].append(value)

    @staticmethod
    def _validated_ms(milliseconds: float) -> float:
        value = float(milliseconds)
        if not np.isfinite(value) or value < 0:
            raise ValueError("latency must be finite and non-negative")
        return value

    def add_stage_c_cpu(self, category: str, milliseconds: float, *,
                        queue_detail: str | None = None) -> None:
        if category not in self.stage_c_cpu_ms:
            raise KeyError(category)
        value = self._validated_ms(milliseconds)
        self.stage_c_cpu_ms[category].append(value)
        if queue_detail is not None:
            if category != "stage_c_queue_wait" or queue_detail not in self.stage_c_queue_detail_ms:
                raise KeyError(queue_detail)
            self.stage_c_queue_detail_ms[queue_detail].append(value)

    def add_stage_c_cuda(self, category: str, milliseconds: float) -> None:
        if category not in self.stage_c_cuda_ms:
            raise KeyError(category)
        self.stage_c_cuda_ms[category].append(self._validated_ms(milliseconds))

    def finalize_stage_c(self, matched_wall_ms: float) -> None:
        if self.stage_c_matched_wall_ms is not None:
            raise RuntimeError("Stage C timing was finalized twice")
        total = self._validated_ms(matched_wall_ms)
        # Explicit packet conversion/control samples already live in
        # stage_c_python_other.  Include them before assigning the remaining
        # uninstrumented consumer-thread wall to that same category.
        accounted = sum(sum(values) for values in self.stage_c_cpu_ms.values())
        # Every explicit phase runs on the Stage C consumer thread.  Small negative
        # residuals can only be timer resolution error; larger values mean phases
        # overlapped or escaped the matched boundary.
        residual = total - accounted
        if residual < -0.1:
            raise RuntimeError(
                f"Stage C CPU timing does not reconcile: wall={total}, "
                f"accounted={accounted}"
            )
        self.stage_c_cpu_ms["stage_c_python_other"].append(max(0.0, residual))
        self.stage_c_matched_wall_ms = total

    @property
    def profiled_stage_subtotal_ms(self) -> float:
        return float(sum(sum(values) for values in self.samples.values()))

    def payload(self, *, peak_online_vram_bytes: int) -> dict[str, Any]:
        stages = {name: distribution_ms(values) for name, values in self.samples.items()}
        cloud = sum(float(row["total_ms"]) for row in stages.values())
        cpu = {
            name: distribution_ms(values)
            for name, values in self.stage_c_cpu_ms.items()
        }
        cuda = {
            name: distribution_ms(values)
            for name, values in self.stage_c_cuda_ms.items()
        }
        explicit_cpu = sum(
            float(row["total_ms"]) for row in cpu.values()
        )
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
            "stage_c_timing": {
                "schema": "h2_stage_c_timing_v1",
                "cpu_wall_exclusive": cpu,
                "cuda_event": cuda,
                "queue_wait_breakdown": {
                    name: distribution_ms(values)
                    for name, values in self.stage_c_queue_detail_ms.items()
                },
                "matched_wall_ms": self.stage_c_matched_wall_ms,
                "exclusive_cpu_total_ms": explicit_cpu,
                "reconciliation_error_ms": (
                    None if self.stage_c_matched_wall_ms is None else
                    self.stage_c_matched_wall_ms - explicit_cpu
                ),
                "cpu_and_cuda_domains_must_not_be_added": True,
                "cpu_category_definitions": {
                    "stage_c_queue_wait": (
                        "consumer dequeue, prediction-ready, and final worker flush waits"
                    ),
                    "stage_c_transfer_wait": (
                        "CPU staging plus H2D submission and completion wait"
                    ),
                    "bridge_compute": "bridge call wall on the Stage C consumer",
                    "native_frontend_compute": (
                        "native anchor frontend call wall on the Stage C process"
                    ),
                    "dpvo_graph_compute": (
                        "CPU call wall for slam.track_packet and terminate after input ready"
                    ),
                    "dpvo_sync_wait": "explicit CUDA synchronization wait outside graph calls",
                    "stage_c_python_other": (
                        "packet conversion, control logic, and reconciled residual wall"
                    ),
                },
                "dpvo_graph_compute_definition": (
                    "CUDA events around slam.track_packet and terminate only; "
                    "packet conversion and synchronization excluded"
                ),
            },
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
        "dpvo_graph_timing_definition": runtime.get(
            "dpvo_graph_runtime_definition",
            "legacy timing definition unavailable",
        ),
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
