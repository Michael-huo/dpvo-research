"""Evaluation and visualization helpers for the final Experiment 3 protocol."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase1-exp3")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import evo.main_ape as main_ape
from evo.core import sync
from evo.core.geometry import umeyama_alignment
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface


SPARSE_TRAJECTORY_NOTE = "Sparse trajectory connects available RGB-anchor poses for visualization and is not a dense estimated trajectory."


def trajectory(poses: np.ndarray, timestamps_ns: np.ndarray) -> PoseTrajectory3D:
    poses = np.asarray(poses, dtype=np.float64)
    return PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=np.asarray(timestamps_ns, dtype=np.uint64).astype(np.float64),
    )


def select_timestamps(arrays: dict[str, np.ndarray], timestamps_ns: list[int] | np.ndarray) -> dict[str, np.ndarray]:
    available = {int(value): index for index, value in enumerate(arrays["timestamps_ns"])}
    wanted = [int(value) for value in timestamps_ns]
    if missing := [value for value in wanted if value not in available]:
        raise AssertionError(f"trajectory is missing {len(missing)} requested timestamps")
    indices = np.asarray([available[value] for value in wanted], dtype=np.int64)
    return {"poses": arrays["poses"][indices], "timestamps_ns": np.asarray(wanted, dtype=np.uint64)}


def ate(arrays: dict[str, np.ndarray], timestamps_ns: list[int] | np.ndarray, groundtruth_path: str) -> dict[str, Any]:
    selected = select_timestamps(arrays, timestamps_ns)
    reference = file_interface.read_tum_trajectory_file(groundtruth_path)
    reference, estimate = sync.associate_trajectories(reference, trajectory(selected["poses"], selected["timestamps_ns"]))
    result = main_ape.ape(reference, estimate, est_name="trajectory", pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    return {"translation_rmse": float(result.stats["rmse"]), "associated_pose_count": int(estimate.num_poses), "requested_pose_count": len(selected["poses"]), "alignment": "Sim(3)"}


def common_accepted(sparse: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    scheduled_sparse = [int(value) for value in sparse["scheduled_anchor_timestamps_ns"]]
    scheduled_oracle = [int(value) for value in oracle["scheduled_anchor_timestamps_ns"]]
    accepted_sparse = [int(value) for value in sparse["accepted_graph_timestamps_ns"]]
    accepted_oracle = [int(value) for value in oracle["accepted_graph_timestamps_ns"]]
    common = [value for value in accepted_sparse if value in set(accepted_oracle)]
    return {
        "scheduled_timestamps_exact": scheduled_sparse == scheduled_oracle,
        "accepted_timestamps_exact": accepted_sparse == accepted_oracle,
        "scheduled_anchor_count": len(scheduled_sparse),
        "sparse_accepted_anchor_count": len(accepted_sparse),
        "oracle_accepted_anchor_count": len(accepted_oracle),
        "common_accepted_anchor_count": len(common),
        "sparse_accepted_scheduled_coverage": len(accepted_sparse) / len(scheduled_sparse) if scheduled_sparse else None,
        "oracle_accepted_scheduled_coverage": len(accepted_oracle) / len(scheduled_oracle) if scheduled_oracle else None,
        "common_scheduled_coverage": len(common) / len(scheduled_sparse) if scheduled_sparse else None,
        "common_accepted_anchor_timestamps_ns": common,
    }


def recovery_metrics(full_ate: float, sparse_ate: float, oracle_ate: float) -> dict[str, float | None]:
    values = np.asarray([full_ate, sparse_ate, oracle_ate], dtype=np.float64)
    if not np.isfinite(values).all() or sparse_ate <= 0:
        raise ValueError("recovery metrics require finite ATE and positive Sparse ATE")
    degradation, recovery = float(sparse_ate - full_ate), float(sparse_ate - oracle_ate)
    return {
        "sparse_degradation_abs": degradation,
        "oracle_recovery_abs": recovery,
        "oracle_relative_improvement": float(recovery / sparse_ate),
        "gap_recovery": float(recovery / degradation) if degradation > 0 else None,
        "small_gap_interpretation_guardrail": "Report the raw denominator; do not over-interpret a large ratio when Sparse and Full ATE are close.",
    }


def _alignment(arrays: dict[str, np.ndarray], common: list[int], groundtruth_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    selected = select_timestamps(arrays, common)
    reference = file_interface.read_tum_trajectory_file(groundtruth_path)
    reference, estimate = sync.associate_trajectories(reference, trajectory(selected["poses"], selected["timestamps_ns"]))
    rotation, translation, scale = umeyama_alignment(estimate.positions_xyz.T, reference.positions_xyz.T, with_scale=True)
    return rotation, translation, float(scale)


def make_trajectory_figure(path: str | Path, arrays: dict[str, dict[str, np.ndarray]], common: list[int], groundtruth_path: str, *, sequence: str, actual_upload_ratio: float, full_anchor_ate: float, sparse_anchor_ate: float, oracle_anchor_ate: float, gap_recovery: float | None) -> None:
    if set(arrays) != {"full_rgb", "sparse_rgb", "oracle_fmap"}:
        raise ValueError("final trajectory figure requires Full, Sparse, and Oracle arrays")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference = file_interface.read_tum_trajectory_file(groundtruth_path)
    full_timestamps = arrays["full_rgb"]["timestamps_ns"].astype(np.float64)
    keep = (reference.timestamps >= full_timestamps.min()) & (reference.timestamps <= full_timestamps.max())
    figure, axis = plt.subplots(figsize=(8.2, 6.6))
    gt = reference.positions_xyz[keep]
    axis.plot(gt[:, 0], gt[:, 1], color="black", linewidth=2.0, label="GT")
    styles = {"full_rgb": ("#2a6fbb", "Full RGB (dense)", "-", 1.45), "sparse_rgb": ("#d9822b", "Sparse RGB anchors", "o-", 1.0), "oracle_fmap": ("#2f9e44", "Sparse + Oracle FMap (dense hybrid)", "-", 1.45)}
    for name, (color, label, style, width) in styles.items():
        rotation, translation, scale = _alignment(arrays[name], common, groundtruth_path)
        points = arrays[name]["poses"][:, :3]
        aligned = (scale * (rotation @ points.T) + translation[:, None]).T
        axis.plot(aligned[:, 0], aligned[:, 1], style, color=color, linewidth=width, markersize=2.5 if name == "sparse_rgb" else None, label=label)
    gap = "N/A" if gap_recovery is None else f"{gap_recovery:.3f}"
    axis.text(.01, .01, f"nominal upload: 12.5%   actual: {actual_upload_ratio:.2%}\nAnchor ATE [m] — Full: {full_anchor_ate:.4g}   Sparse: {sparse_anchor_ate:.4g}   Oracle: {oracle_anchor_ate:.4g}\nGap recovery: {gap}", transform=axis.transAxes, va="bottom", ha="left", fontsize=8.5, bbox={"boxstyle": "round", "facecolor": "white", "alpha": .86, "edgecolor": "#aaaaaa"})
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(f"{sequence} — Experiment 3 representative K=8")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    figure.text(.5, .012, SPARSE_TRAJECTORY_NOTE, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .035, 1, 1))
    figure.savefig(destination, dpi=160)
    plt.close(figure)
