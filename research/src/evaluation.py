"""Paired, timestamp-defined trajectory evaluation for canonical Exp6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .protocol import canonical_sha256


def select_timestamps(arrays: dict[str, np.ndarray], timestamps_ns: Sequence[int] | np.ndarray) -> dict[str, np.ndarray]:
    """Select an ordered timestamp population without depending on legacy Exp3."""
    available = {int(value): index for index, value in enumerate(arrays["timestamps_ns"])}
    wanted = [int(value) for value in timestamps_ns]
    missing = [value for value in wanted if value not in available]
    if missing:
        raise AssertionError(f"trajectory is missing {len(missing)} requested timestamps")
    indices = np.asarray([available[value] for value in wanted], dtype=np.int64)
    return {
        "poses": arrays["poses"][indices],
        "timestamps_ns": np.asarray(wanted, dtype=np.uint64),
    }


@dataclass(frozen=True)
class EvaluationPopulation:
    common_anchor_timestamps_ns: tuple[int, ...]
    rpe_pairs_ns: tuple[tuple[int, int], ...]
    horizon_ns: int
    tolerance_ns: int

    @property
    def population_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "source": "canonical_sparse_rgb_reference_k5_groundtruth_associable",
            "common_anchor_timestamps_ns": list(self.common_anchor_timestamps_ns),
            "sim3_alignment_timestamps_ns": list(self.common_anchor_timestamps_ns),
            "ate_timestamps_ns": list(self.common_anchor_timestamps_ns),
            "rpe_horizon_ns": self.horizon_ns,
            "rpe_pair_tolerance_ns": self.tolerance_ns,
            "rpe_pairs_ns": [list(item) for item in self.rpe_pairs_ns],
            "mode_local_pairing_forbidden": True,
        }
        if include_hash:
            payload["population_sha256"] = canonical_sha256(payload)
        return payload


def build_rpe_pairs(
    timestamps_ns: Sequence[int], *, horizon_seconds: float = 1.0,
    tolerance_ns: int = 1_000_000,
) -> tuple[tuple[int, int], ...]:
    """Build pairs by EuRoC time, never by frame/candidate offsets."""
    timestamps = np.asarray([int(value) for value in timestamps_ns], dtype=np.int64)
    if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("RPE timestamps must be a strictly increasing sequence")
    horizon_ns = int(round(float(horizon_seconds) * 1_000_000_000.0))
    if horizon_ns <= 0 or tolerance_ns < 0:
        raise ValueError("RPE horizon/tolerance is invalid")
    result: list[tuple[int, int]] = []
    for left_index, left in enumerate(timestamps):
        wanted = int(left) + horizon_ns
        right_index = int(np.searchsorted(timestamps, wanted, side="left"))
        options = [index for index in (right_index - 1, right_index)
                   if left_index < index < len(timestamps)]
        if not options:
            continue
        chosen = min(options, key=lambda index: (abs(int(timestamps[index]) - wanted), index))
        if abs(int(timestamps[chosen]) - wanted) <= int(tolerance_ns):
            result.append((int(left), int(timestamps[chosen])))
    if not result:
        raise ValueError("canonical timestamp population yielded no 1.0 s RPE pairs")
    return tuple(result)


def freeze_evaluation_population(
    timestamps_ns: Sequence[int], *, horizon_seconds: float, tolerance_ns: int,
) -> EvaluationPopulation:
    timestamps = tuple(int(value) for value in timestamps_ns)
    return EvaluationPopulation(
        timestamps,
        build_rpe_pairs(timestamps, horizon_seconds=horizon_seconds, tolerance_ns=tolerance_ns),
        int(round(float(horizon_seconds) * 1_000_000_000.0)),
        int(tolerance_ns),
    )


def _groundtruth(path: str | Path) -> dict[str, np.ndarray]:
    rows = np.loadtxt(path)
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError("EuRoC ground truth must contain timestamp, xyz, qwxyz")
    return {
        "timestamps_ns": np.rint(rows[:, 0]).astype(np.int64),
        "positions": rows[:, 1:4].astype(np.float64),
        "quaternions_xyzw": rows[:, [5, 6, 7, 4]].astype(np.float64),
    }


def filter_groundtruth_associable_timestamps(
    groundtruth_path: str | Path, timestamps_ns: Sequence[int], *, max_difference_ns: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Freeze the reference-derived evaluation subset before any mode is evaluated."""
    if max_difference_ns < 0:
        raise ValueError("ground-truth association tolerance must be non-negative")
    groundtruth = _groundtruth(groundtruth_path)
    gt_timestamps = groundtruth["timestamps_ns"]
    kept: list[int] = []
    excluded: list[int] = []
    for raw in timestamps_ns:
        timestamp = int(raw)
        position = int(np.searchsorted(gt_timestamps, timestamp))
        choices = [index for index in (position - 1, position) if 0 <= index < len(gt_timestamps)]
        if choices and min(abs(int(gt_timestamps[index]) - timestamp) for index in choices) <= max_difference_ns:
            kept.append(timestamp)
        else:
            excluded.append(timestamp)
    if len(kept) < 3:
        raise ValueError("fewer than three canonical reference timestamps have ground-truth support")
    return tuple(kept), tuple(excluded)


def _associate_groundtruth(
    groundtruth: dict[str, np.ndarray], timestamps_ns: Sequence[int], *, max_difference_ns: int = 20_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    gt_timestamps = groundtruth["timestamps_ns"]
    indices: list[int] = []
    for timestamp in timestamps_ns:
        position = int(np.searchsorted(gt_timestamps, int(timestamp)))
        choices = [index for index in (position - 1, position) if 0 <= index < len(gt_timestamps)]
        selected = min(choices, key=lambda index: abs(int(gt_timestamps[index]) - int(timestamp)))
        if abs(int(gt_timestamps[selected]) - int(timestamp)) > max_difference_ns:
            raise ValueError(f"ground-truth association failed at {timestamp}")
        indices.append(selected)
    return groundtruth["positions"][indices], groundtruth["quaternions_xyzw"][indices]


def _sim3(estimate_xyz: np.ndarray, reference_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    from evo.core.geometry import umeyama_alignment

    rotation, translation, scale = umeyama_alignment(
        estimate_xyz.T, reference_xyz.T, with_scale=True,
    )
    return np.asarray(rotation), np.asarray(translation), float(scale)


def associated_groundtruth_poses(
    groundtruth_path: str | Path, timestamps_ns: Sequence[int],
) -> np.ndarray:
    """Return associated ground-truth poses as xyz+xyzw without changing the contract."""
    xyz, quaternion = _associate_groundtruth(_groundtruth(groundtruth_path), timestamps_ns)
    return np.concatenate((xyz, quaternion), axis=1)


def fit_sim3_trajectory(
    arrays: Mapping[str, np.ndarray], population: EvaluationPopulation,
    groundtruth_path: str | Path,
) -> dict[str, Any]:
    """Fit this condition's Sim3 on the shared canonical fitting population."""
    selected = select_timestamps(dict(arrays), population.common_anchor_timestamps_ns)
    fitting_poses = np.asarray(selected["poses"], dtype=np.float64)
    gt_xyz, _ = _associate_groundtruth(
        _groundtruth(groundtruth_path), population.common_anchor_timestamps_ns,
    )
    rotation, translation, scale = _sim3(fitting_poses[:, :3], gt_xyz)
    raw = np.asarray(arrays["poses"], dtype=np.float64)
    aligned_xyz = (scale * (rotation @ raw[:, :3].T) + translation[:, None]).T
    aligned_rotation = Rotation.from_matrix(
        rotation[None] @ Rotation.from_quat(raw[:, 3:7]).as_matrix()
    ).as_quat()
    aligned = np.concatenate((aligned_xyz, aligned_rotation), axis=1)
    return {
        "scale": float(scale),
        "rotation": rotation,
        "translation": translation,
        "fitting_population_sha256": population.population_sha256,
        "aligned_poses": aligned,
    }


def evaluate_paired_trajectory(
    arrays: Mapping[str, np.ndarray], population: EvaluationPopulation,
    groundtruth_path: str | Path,
) -> dict[str, Any]:
    fitted = fit_sim3_trajectory(arrays, population, groundtruth_path)
    selected = select_timestamps({
        "poses": fitted["aligned_poses"],
        "timestamps_ns": np.asarray(arrays["timestamps_ns"]),
    }, population.common_anchor_timestamps_ns)
    poses = np.asarray(selected["poses"], dtype=np.float64)
    gt_xyz, gt_quaternion = _associate_groundtruth(
        _groundtruth(groundtruth_path), population.common_anchor_timestamps_ns,
    )
    aligned_xyz = poses[:, :3]
    ate_errors = np.linalg.norm(aligned_xyz - gt_xyz, axis=1)
    estimate_rotation = Rotation.from_quat(poses[:, 3:7])
    gt_rotation = Rotation.from_quat(gt_quaternion)
    index_by_timestamp = {
        timestamp: index for index, timestamp in enumerate(population.common_anchor_timestamps_ns)
    }
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for left, right in population.rpe_pairs_ns:
        i, j = index_by_timestamp[left], index_by_timestamp[right]
        estimate_delta = estimate_rotation[i].inv() * estimate_rotation[j]
        gt_delta = gt_rotation[i].inv() * gt_rotation[j]
        estimate_translation = estimate_rotation[i].inv().apply(aligned_xyz[j] - aligned_xyz[i])
        gt_translation = gt_rotation[i].inv().apply(gt_xyz[j] - gt_xyz[i])
        translation_errors.append(float(np.linalg.norm(estimate_translation - gt_translation)))
        rotation_errors.append(float((gt_delta.inv() * estimate_delta).magnitude() * 180.0 / np.pi))
    return {
        "population_sha256": population.population_sha256,
        "alignment": "Sim(3)",
        "alignment_pose_count": len(poses),
        "ate_population_count": len(ate_errors),
        "ate_rmse_m": float(np.sqrt(np.mean(np.square(ate_errors)))),
        "rpe_horizon_seconds": population.horizon_ns / 1e9,
        "rpe_pair_count": len(translation_errors),
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(np.square(translation_errors)))),
        "rotation_rpe_rmse_deg": float(np.sqrt(np.mean(np.square(rotation_errors)))),
        "sim3": {
            "scale": fitted["scale"],
            "rotation": np.asarray(fitted["rotation"]).tolist(),
            "translation": np.asarray(fitted["translation"]).tolist(),
            "fitting_population_sha256": fitted["fitting_population_sha256"],
            "convention": "aligned_xyz=scale*(rotation@raw_xyz)+translation; aligned_R=rotation@raw_R",
        },
    }




def population_coverage(
    arrays: Mapping[str, np.ndarray], timestamps_ns: Sequence[int],
) -> dict[str, Any]:
    available = {int(value) for value in arrays["timestamps_ns"]}
    wanted = [int(value) for value in timestamps_ns]
    missing = [value for value in wanted if value not in available]
    return {
        "required_pose_count": len(wanted),
        "available_pose_count": len(wanted) - len(missing),
        "canonical_pose_coverage": (len(wanted) - len(missing)) / len(wanted) if wanted else 0.0,
        "missing_timestamp_count": len(missing),
        "missing_timestamps_ns": missing,
        "population_was_not_shrunk": True,
    }


def dense_hidden_ate(
    arrays: Mapping[str, np.ndarray], dense_timestamps_ns: Sequence[int],
    hidden_timestamps_ns: Sequence[int], groundtruth_path: str | Path,
) -> dict[str, Any]:
    selected = select_timestamps(dict(arrays), dense_timestamps_ns)
    poses = np.asarray(selected["poses"], dtype=np.float64)
    gt_xyz, _ = _associate_groundtruth(_groundtruth(groundtruth_path), dense_timestamps_ns)
    rotation, translation, scale = _sim3(poses[:, :3], gt_xyz)
    aligned = (scale * (rotation @ poses[:, :3].T) + translation[:, None]).T
    errors = np.linalg.norm(aligned - gt_xyz, axis=1)
    index = {int(timestamp): offset for offset, timestamp in enumerate(dense_timestamps_ns)}
    hidden_indices = [index[int(timestamp)] for timestamp in hidden_timestamps_ns]
    hidden_errors = errors[hidden_indices]
    return {
        "alignment": "independent_dense_Sim3",
        "dense_population_count": len(errors),
        "dense_ate_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "hidden_population_count": len(hidden_errors),
        "hidden_only_ate_rmse_m": float(np.sqrt(np.mean(np.square(hidden_errors)))),
        "hidden_uses_dense_alignment_without_refit": True,
    }














def plot_canonical_trajectories(
    path: str | Path, arrays: Mapping[str, Mapping[str, np.ndarray]],
    population: EvaluationPopulation, groundtruth_path: str | Path, *,
    sequence: str, labels: Mapping[str, str],
) -> None:
    """Generic single-panel plot used by the final Exp6 modules."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase1-exp6")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if set(arrays) != set(labels):
        raise ValueError("trajectory label population mismatch")
    colors = ("#2a6fbb", "#d9822b", "#2f9e44", "#7b2cbf", "#d63384", "#6c757d")
    groundtruth = _groundtruth(groundtruth_path)
    timestamps = np.concatenate([
        np.asarray(value["timestamps_ns"], dtype=np.int64) for value in arrays.values()
    ])
    keep = ((groundtruth["timestamps_ns"] >= timestamps.min())
            & (groundtruth["timestamps_ns"] <= timestamps.max()))
    figure, axis = plt.subplots(figsize=(8.4, 6.6))
    gt = groundtruth["positions"][keep]
    axis.plot(gt[:, 0], gt[:, 1], color="black", linewidth=2.0, label="GT")
    for color, (name, values) in zip(colors, arrays.items()):
        aligned = fit_sim3_trajectory(values, population, groundtruth_path)["aligned_poses"][:, :3]
        axis.plot(aligned[:, 0], aligned[:, 1], color=color, linewidth=1.35,
                  label=labels[name])
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x [m]"); axis.set_ylabel("y [m]")
    axis.set_title(sequence); axis.grid(alpha=.25); axis.legend(fontsize=8.2)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(destination, dpi=180); plt.close(figure)
