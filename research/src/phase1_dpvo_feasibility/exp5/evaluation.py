"""Alignment, hook-state, runtime, and report evaluation for Exp5-0."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

def _identity(record: Any) -> tuple[int, int]:
    if hasattr(record, "frame_id"):
        return int(record.frame_id), int(record.timestamp)
    return int(record["frame_id"]), int(record["timestamp"])


def _duplicates(values: Iterable[tuple[int, int]]) -> int:
    rows = list(values)
    return len(rows) - len(set(rows))


def _latent_sanity(
    records: list[dict[str, Any]], *, expected_shape: tuple[int, ...], epsilon: float,
) -> dict[str, Any]:
    if not records:
        return {
            "passed": False, "finite": False, "shape": list(expected_shape),
            "norm_min": None, "norm_max": None, "mean_min": None, "mean_max": None,
            "std_min": None, "std_max": None,
        }
    norms = [float(record["latent_norm"]) for record in records]
    means = [float(record["latent_mean"]) for record in records]
    stds = [float(record["latent_std"]) for record in records]
    finite = bool(
        all(bool(record.get("finite_check")) for record in records)
        and all(math.isfinite(value) for value in (*norms, *means, *stds))
    )
    shapes_match = all(tuple(int(value) for value in record["jepa_shape"]) == expected_shape for record in records)
    nonzero = all(value > epsilon for value in norms) and all(value > epsilon for value in stds)
    return {
        "passed": bool(finite and shapes_match and nonzero),
        "finite": finite,
        "shape": list(expected_shape),
        "norm_min": min(norms),
        "norm_max": max(norms),
        "mean_min": min(means),
        "mean_max": max(means),
        "std_min": min(stds),
        "std_max": max(stds),
    }


def evaluate_sequence(
    *,
    expected: list[Any],
    baseline: dict[str, Any],
    sidecar: dict[str, Any],
    config: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    if not expected:
        raise ValueError("cannot evaluate an empty sequence")
    sidecar_frames = list(sidecar.get("frames", []))
    jepa_frames = list(sidecar.get("jepa_frames", []))
    expected_ids = [_identity(record) for record in expected]
    sidecar_ids = [_identity(record) for record in sidecar_frames]
    jepa_ids = [_identity(record) for record in jepa_frames]
    duplicate_count = sum(_duplicates(rows) for rows in (sidecar_ids, jepa_ids))
    expected_set = set(expected_ids)
    missing = (
        (expected_set - set(sidecar_ids))
        | (expected_set - set(jepa_ids))
    )
    extra = (set(sidecar_ids) | set(jepa_ids)) - expected_set
    ordered_alignment = bool(sidecar_ids == expected_ids and jepa_ids == expected_ids)
    frame_count_equal = bool(
        len(sidecar_frames) == len(jepa_frames) == len(expected)
    )

    timestamp_errors: list[int] = []
    expected_timestamp = {frame_id: timestamp for frame_id, timestamp in expected_ids}
    for records in (sidecar_frames, jepa_frames):
        for record in records:
            frame_id, timestamp = _identity(record)
            if frame_id in expected_timestamp:
                timestamp_errors.append(abs(timestamp - expected_timestamp[frame_id]))
    timestamp_error_max = max(timestamp_errors, default=0)
    timestamp_ok = timestamp_error_max <= int(config["validation"]["timestamp_error_max_ns"])

    raw_validation = dict(sidecar.get("trajectory_validation") or {})
    pose_threshold = float(config["validation"]["max_pose_error"])
    try:
        existing_pose_max_error = float(raw_validation["existing_pose_max_error"])
        existing_pose_count = int(raw_validation["existing_pose_count"])
        compared_frame_count = int(raw_validation["compared_frame_count"])
    except (KeyError, TypeError, ValueError):
        existing_pose_max_error = float("inf")
        existing_pose_count = 0
        compared_frame_count = 0
    trajectory_passed = bool(
        raw_validation.get("mode") == "single_run_hook_validation"
        and math.isfinite(existing_pose_max_error)
        and existing_pose_max_error < pose_threshold
        and existing_pose_count >= 0
        and compared_frame_count == len(expected)
        and bool(raw_validation.get("frame_index_unchanged"))
        and bool(raw_validation.get("pose_shape_unchanged"))
        and bool(raw_validation.get("finite_check"))
    )

    latent = _latent_sanity(
        jepa_frames,
        expected_shape=tuple(int(value) for value in config["jepa"]["expected_global_shape"]),
        epsilon=float(config["validation"]["latent_epsilon"]),
    )
    jepa_runtimes = [float(record["extraction_time"]) for record in jepa_frames]
    jepa_total = float(sum(jepa_runtimes))
    synchronization = bool(
        frame_count_equal and ordered_alignment and not missing and not extra
        and duplicate_count == 0 and timestamp_ok
    )
    # Worker completion is enforced before evaluation. The feasibility gate itself
    # contains only synchronization, latent sanity, and the single-run hook check.
    passed = bool(synchronization and trajectory_passed and latent["passed"])
    baseline_runtime = float(baseline["runtime"])
    sidecar_runtime = float(sidecar["runtime"])
    runtime_delta = sidecar_runtime - baseline_runtime
    return {
        "dataset": str(expected[0].dataset if hasattr(expected[0], "dataset") else expected[0]["dataset"]),
        "sequence": str(expected[0].sequence if hasattr(expected[0], "sequence") else expected[0]["sequence"]),
        "num_frames": len(expected),
        "timestamp_error": {
            "max_ns": int(timestamp_error_max),
            "passed": bool(timestamp_ok),
        },
        "baseline_runtime_total": baseline_runtime,
        "sidecar_runtime_total": sidecar_runtime,
        "runtime_overhead": {
            "seconds": runtime_delta,
            "relative": runtime_delta / baseline_runtime if baseline_runtime > 0 else None,
        },
        "jepa_runtime": {
            "mean_seconds": float(jepa_total / len(jepa_runtimes)) if jepa_runtimes else 0.0,
            "total_seconds": jepa_total,
        },
        "trajectory_validation": {
            "mode": "single_run_hook_validation",
            "existing_pose_max_error": (
                existing_pose_max_error if math.isfinite(existing_pose_max_error) else None
            ),
            "existing_pose_count": existing_pose_count,
            "passed": trajectory_passed,
        },
        "alignment": {
            "passed": synchronization,
            "frame_count_equal": frame_count_equal,
            "frame_id_and_timestamp_ordered": ordered_alignment,
            "missing_frame_count": len(missing),
            "extra_frame_count": len(extra),
            "duplicate_frame_count": duplicate_count,
        },
        "latent_sanity": latent,
        "status": "smoke_pass" if smoke and passed else ("pass" if passed else "fail"),
    }


def aggregate_metrics(
    sequence_metrics: list[dict[str, Any]], *, config: dict[str, Any], smoke: bool,
) -> dict[str, Any]:
    if not sequence_metrics:
        raise ValueError("at least one sequence metric is required")
    expected_status = "smoke_pass" if smoke else "pass"
    passed = all(row["status"] == expected_status for row in sequence_metrics)
    return {
        "schema_version": int(config["schema_version"]),
        "experiment": str(config["experiment"]["name"]),
        "scope": "smoke" if smoke else "formal",
        "mock_models": bool(smoke),
        "status": expected_status if passed else "fail",
        "protocol": {
            "dataset": str(config["dataset"]["dataset_type"]),
            "camera": str(config["dataset"]["camera"]),
            "stride": int(config["dataset"]["stride"]),
            "skip": int(config["dataset"]["skip"]),
            "sequences": [row["sequence"] for row in sequence_metrics],
            "existing_pose_max_error_gate": float(config["validation"]["max_pose_error"]),
            "runtime_gate": None,
        },
        "sequences": {row["sequence"]: row for row in sequence_metrics},
    }


def render_report(metrics: dict[str, Any]) -> str:
    smoke = metrics["scope"] == "smoke"
    rows = list(metrics["sequences"].values())
    alignment_pass = all(row["alignment"]["passed"] for row in rows)
    sidecar_pass = all(row["trajectory_validation"]["passed"] for row in rows)
    latent_pass = all(row["latent_sanity"]["passed"] for row in rows)
    lines = [
        "# Phase 1 / Experiment 5-0 — JEPA–DPVO Interface Validation",
        "",
        "## Result",
        "",
        f"**{metrics['status']}**",
        "",
    ]
    if smoke:
        lines += [
            "This is a CPU smoke result over real EuRoC files with mock DPVO/V-JEPA models. ",
            "It validates data flow and report generation only and is not model evidence.",
            "",
        ]
    lines += [
        "## Metrics",
        "",
        "| Sequence | Frames | Max timestamp error (ns) | Baseline (s) | Sidecar (s) | JEPA mean/frame (s) | JEPA total (s) | Existing-pose max error | Existing pose rows | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        validation = row["trajectory_validation"]
        pose_error = (
            "N/A" if validation["existing_pose_max_error"] is None
            else f"{validation['existing_pose_max_error']:.6g}"
        )
        lines.append(
            f"| `{row['sequence']}` | {row['num_frames']} | {row['timestamp_error']['max_ns']} | "
            f"{row['baseline_runtime_total']:.6g} | {row['sidecar_runtime_total']:.6g} | "
            f"{row['jepa_runtime']['mean_seconds']:.6g} | "
            f"{row['jepa_runtime']['total_seconds']:.6g} | {pose_error} | "
            f"{validation['existing_pose_count']} | {row['status']} |"
        )
    overhead_lines = []
    for row in rows:
        delta = float(row["runtime_overhead"]["seconds"])
        relative = row["runtime_overhead"]["relative"]
        relative_text = "N/A" if relative is None else f"{relative:.2%}"
        overhead_lines.append(
            f"- `{row['sequence']}`: sidecar overhead `{delta:.6g} s` (`{relative_text}`); "
            f"JEPA extraction mean `{row['jepa_runtime']['mean_seconds']:.6g} s/frame`."
        )
    lines += [
        "",
        "## Questions",
        "",
        "1. **Frame synchronization:** " + ("pass." if alignment_pass else "fail."),
        "2. **JEPA sidecar latent extraction:** " + ("pass." if latent_pass else "fail."),
        "3. **Non-intrusive sidecar integration:** "
        + ("pass." if sidecar_pass else "fail.")
        + " This checks whether DPVO state at the frame-dispatch boundary remains unchanged "
        "after JEPA sidecar execution.",
        "",
        "## Runtime diagnostic",
        "",
        "The measured overhead is:",
        "",
        *overhead_lines,
        "",
        "Runtime acceptability is intentionally a manual decision; Exp5-0 defines no speed threshold.",
        "",
        "## Interpretation boundary",
        "",
        "The hook check covers only existing DPVO pose entries exposed at the frame-dispatch boundary. "
        "It does not validate DPVO's complete internal computation graph, SLAM accuracy improvement, "
        "JEPA representation quality, or any fusion method.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    report_path = destination / "REPORT.md"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(metrics), encoding="utf-8")
